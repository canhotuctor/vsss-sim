"""Batched Gymnasium environment using vectorized JAX physics.

Truncation resets each match on the next step; goals trigger kickoffs inside
the episode. Built-in opponents stay batched in JAX. Custom callables use a
per-environment Python loop and are intended for development.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from .. import config, physics
from ..config import InitMode, PhysicsBackend, RefereeMode, RewardProfile
from .match_rules import (
    MatchPhase,
    MatchRules,
    MatchStepInput,
    build_observation,
    initial_referee_state,
)


def _build_obs_batched(state) -> jnp.ndarray:
    """Build a normalised (num_envs, 46) observation array from a batched SimState."""
    return build_observation(state)


def _export_obs(state) -> np.ndarray:
    """Return writable NumPy observations at the Gymnasium interface."""
    return np.array(_build_obs_batched(state), dtype=np.float32, copy=True)


# VSSVecEnv

class VSSVecEnv(VectorEnv):
    """Batched JAX-backed Gymnasium VectorEnv for IEEE VSSS 3 v 3."""

    metadata = {
        "render_modes": [],
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 8,
        opponent_policy: str | Callable = "stationary",
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        init_mode: InitMode | str = InitMode.KICKOFF,
        reward_profile: RewardProfile | str = RewardProfile.DEFAULT,
        referee_mode: RefereeMode | str = RefereeMode.FULL,
        physics_backend: PhysicsBackend | str = PhysicsBackend.JAX,
    ) -> None:
        super().__init__()

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not callable(opponent_policy) and opponent_policy not in ("stationary", "random"):
            raise ValueError(
                f"Unknown opponent_policy '{opponent_policy}'. "
                "Choose 'stationary', 'random', or pass a callable."
            )

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self._opponent_policy_spec: str | Callable = opponent_policy
        self._init_mode = InitMode(init_mode)
        self._match_rules = MatchRules(
            reward_profile=reward_profile,
            max_episode_steps=max_episode_steps,
            init_mode=self._init_mode,
            referee_mode=referee_mode,
        )

        obs_dim = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        self.single_observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.N_ROBOTS * 2,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(num_envs, obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, config.N_ROBOTS * 2), dtype=np.float32,
        )

        engine = physics.get_backend(physics_backend)
        _reset_fn = (
            engine.reset_kickoff
            if self._init_mode == InitMode.KICKOFF
            else engine.reset_random
        )
        self._batched_reset = jax.jit(jax.vmap(_reset_fn))
        self._batched_step = jax.jit(jax.vmap(engine.step))
        self._batched_match_step = jax.jit(jax.vmap(self._match_rules.step))

        self._rng = np.random.default_rng()
        self._state = None        # batched SimState
        self._referee_state = None
        self._step_count = np.zeros(num_envs, dtype=np.int32)
        self._needs_reset = np.zeros(num_envs, dtype=bool)

    # Internal helpers

    def _split_keys(self, n: int) -> jnp.ndarray:
        """Return ``n`` PRNG keys derived from the env's master RNG."""
        seed = int(self._rng.integers(0, 2**31 - 1))
        return jax.random.split(jax.random.PRNGKey(seed), n)

    def _fresh_batched_state(self):
        return self._batched_reset(self._split_keys(self.num_envs))

    def _fresh_referee_state(self, count: int):
        return jax.vmap(lambda _: initial_referee_state())(
            jnp.arange(count, dtype=jnp.int32)
        )

    def _kickoff_subset(self, mask: np.ndarray) -> None:
        """Replace the SimState slice for envs whose ``mask`` is True.

        Does NOT touch ``_step_count`` — used for both episode resets and
        in-episode kickoffs after goals.
        """
        if not mask.any():
            return
        n = int(mask.sum())
        idx = jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)
        fresh = self._batched_reset(self._split_keys(n))

        def splice(full, part):
            return full.at[idx].set(part)
        self._state = jax.tree_util.tree_map(splice, self._state, fresh)
        fresh_referee = self._fresh_referee_state(n)
        self._referee_state = jax.tree_util.tree_map(
            splice, self._referee_state, fresh_referee
        )

    def _reset_subset(self, mask: np.ndarray) -> None:
        """Full episode reset: kickoff + zero step counter for affected envs."""
        self._kickoff_subset(mask)
        self._step_count[mask] = 0

    # Public state inspection helpers

    def current_obs(self) -> np.ndarray:
        """Return the current batched observation without stepping."""
        if self._state is None:
            raise RuntimeError("call reset() before current_obs()")
        return _export_obs(self._state)

    def _opponent_actions(self) -> np.ndarray:
        """Build a (num_envs, N_ROBOTS, 2) opponent action array.

        Built-in policies (stationary, random) do not observe the state, so no
        obs is built for them. The callable path builds obs lazily only when a
        user-supplied policy actually needs it.
        """
        if self._opponent_policy_spec == "stationary":
            return np.zeros((self.num_envs, config.N_ROBOTS, 2), dtype=np.float32)
        if self._opponent_policy_spec == "random":
            return self._rng.uniform(
                -1.0, 1.0, size=(self.num_envs, config.N_ROBOTS, 2),
            ).astype(np.float32)
        # callable: per-env Python loop (slow path, documented)
        obs_np = _export_obs(self._state)
        out = np.empty((self.num_envs, config.N_ROBOTS, 2), dtype=np.float32)
        for i in range(self.num_envs):
            out[i] = np.asarray(
                self._opponent_policy_spec(obs_np[i]),
                dtype=np.float32,
            ).reshape(config.N_ROBOTS, 2)
        return out

    # Gymnasium VectorEnv interface

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._fresh_batched_state()
        self._referee_state = self._fresh_referee_state(self.num_envs)
        self._step_count[:] = 0
        self._needs_reset[:] = False

        obs = _export_obs(self._state)
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._needs_reset.any():
            self._reset_subset(self._needs_reset)
            self._needs_reset[:] = False

        blue = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, config.N_ROBOTS, 2,
        )
        yellow = self._opponent_actions()
        all_actions = np.stack([blue, yellow], axis=1).astype(np.float32)  # (B, 2, 3, 2)
        all_actions_j = jnp.asarray(all_actions)
        stopped = self._referee_state.phase == MatchPhase.STOPPED
        all_actions_j = jnp.where(
            stopped[:, None, None, None], 0.0, all_actions_j
        )

        simulation_before = self._state
        simulation_after_physics, info_phys = self._batched_step(
            simulation_before, all_actions_j
        )
        goals = np.where(np.asarray(stopped), 0, np.asarray(info_phys["goal"], dtype=np.int32))  # (B,)
        self._step_count += 1
        # Independent seeded keys are also needed when both teams foul.
        kickoff_keys = self._split_keys(self.num_envs)
        transition = self._batched_match_step(
            MatchStepInput(
                simulation_before=simulation_before,
                simulation_after_physics=simulation_after_physics,
                goal=jnp.asarray(goals),
                step_count=jnp.asarray(self._step_count),
                kickoff_key=kickoff_keys,
                referee_state=self._referee_state,
                dt=jnp.full((self.num_envs,), config.DT, dtype=jnp.float32),
            )
        )
        self._state = transition.simulation
        self._referee_state = transition.referee_state
        rewards = np.asarray(transition.reward, dtype=np.float32)

        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = self._step_count >= self.max_episode_steps

        # Mark envs to auto-reset on the *next* step (NEXT_STEP autoreset semantics)
        self._needs_reset = truncations.copy()

        obs = _export_obs(self._state)
        info: dict[str, Any] = {
            "goal": goals,
            "event": np.asarray(transition.event, dtype=np.int32),
            "offending_team": np.asarray(transition.offending_team, dtype=np.int32),
            "restart": np.asarray(transition.restart, dtype=np.int32),
            "restart_team": np.asarray(transition.restart_team, dtype=np.int32),
            "quadrant": np.asarray(transition.quadrant, dtype=np.int32),
            "match_phase": np.asarray(
                transition.referee_state.phase, dtype=np.int32
            ),
        }
        return obs, rewards, terminations, truncations, info

    def close(self, **kwargs: Any) -> None:
        # Nothing device-side to free explicitly; JAX manages its own buffers.
        self._state = None
