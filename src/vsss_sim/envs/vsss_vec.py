"""
Batched IEEE VSSS 3 v 3 vector environment (Gymnasium ``VectorEnv``).

Runs ``num_envs`` independent matches in parallel by ``jax.vmap``-ing the JAX
physics backend. A single batched ``SimState`` PyTree is held on the active
JAX device; ``step`` is ``jit``-compiled and operates on the leading
``num_envs`` axis.

Auto-reset semantics
--------------------
- ``terminated`` is always ``False`` (no terminal state mid-match).
- ``truncated`` becomes ``True`` for an env when its step counter reaches
  ``max_episode_steps``.
- Goals trigger a kickoff *inside the same episode* (does not flip ``truncated``).
- Envs whose ``truncated`` is ``True`` are auto-reset *on the next step* (the
  default Gymnasium ``AutoresetMode.NEXT_STEP`` behaviour).

Opponent policies
-----------------
- ``"stationary"`` and ``"random"`` are fully batched in JAX (no Python loop).
- A user-supplied callable falls back to a per-env Python loop and is the slow
  path; acceptable for development, not for training throughput.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from .. import config
from .. import physics as _physics_pkg


# ---------------------------------------------------------------------------
# Normalisation constants (mirror envs.base)
# ---------------------------------------------------------------------------
_NORM_POS_X = config.FIELD_LENGTH / 2.0
_NORM_POS_Y = config.FIELD_WIDTH / 2.0
_NORM_VEL = config.ROBOT_MAX_WHEEL_SPEED * config.VELOCITY_NORM_HEADROOM
_NORM_OMEGA = _NORM_VEL / (config.ROBOT_WHEELBASE / 2.0)


def _build_obs_batched(state) -> jnp.ndarray:
    """Build a normalised (num_envs, 46) observation array from a batched SimState."""
    ball = state.ball       # (B, 4)
    robots = state.robots   # (B, N_TEAMS, N_ROBOTS, 6)
    B = ball.shape[0]

    obs_ball = jnp.stack([
        ball[:, 0] / _NORM_POS_X,
        ball[:, 1] / _NORM_POS_Y,
        ball[:, 2] / _NORM_VEL,
        ball[:, 3] / _NORM_VEL,
    ], axis=-1)  # (B, 4)

    flat = robots.reshape(B, config.N_TEAMS * config.N_ROBOTS, 6)  # (B, 6, 6)
    x = flat[:, :, 0] / _NORM_POS_X
    y = flat[:, :, 1] / _NORM_POS_Y
    theta = flat[:, :, 2]
    vx = flat[:, :, 3] / _NORM_VEL
    vy = flat[:, :, 4] / _NORM_VEL
    omega = flat[:, :, 5] / _NORM_OMEGA
    per_robot = jnp.stack([x, y, jnp.sin(theta), jnp.cos(theta), vx, vy, omega], axis=-1)
    # (B, 6, 7) → (B, 42)
    obs_robots = per_robot.reshape(B, -1)
    return jnp.concatenate([obs_ball, obs_robots], axis=-1).astype(jnp.float32)


def _export_obs(state) -> np.ndarray:
    """JAX → numpy obs at the wrapper boundary.

    Forces a **writable** copy. ``np.asarray`` on a JAX-on-CPU array returns
    a read-only view, which makes downstream ``torch.as_tensor`` warn about
    undefined behaviour. The copy is cheap (~num_envs × 46 × 4 bytes).
    """
    return np.array(_build_obs_batched(state), dtype=np.float32, copy=True)


# ---------------------------------------------------------------------------
# VSSVecEnv
# ---------------------------------------------------------------------------

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
        backend: str = "jax",
    ) -> None:
        super().__init__()

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        backend_lc = (backend or "jax").lower()
        if backend_lc != "jax":
            raise ValueError(
                f"VSSVecEnv currently only supports backend='jax', got '{backend}'. "
                "For the numpy backend, use SyncVectorEnv around VSSEnv instead."
            )
        if not callable(opponent_policy) and opponent_policy not in ("stationary", "random"):
            raise ValueError(
                f"Unknown opponent_policy '{opponent_policy}'. "
                "Choose 'stationary', 'random', or pass a callable."
            )

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self._backend = _physics_pkg.get_backend(backend_lc)
        self._opponent_policy_spec: str | Callable = opponent_policy

        # Spaces
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

        # JIT-compiled batched primitives
        self._batched_reset = jax.jit(jax.vmap(self._backend.reset_kickoff))
        self._batched_step = jax.jit(jax.vmap(self._backend.step))

        # Per-env Python-side bookkeeping (kept on host)
        self._rng = np.random.default_rng()
        self._state = None        # batched SimState
        self._step_count = np.zeros(num_envs, dtype=np.int32)
        # Per-env "needs reset on next step" flags (from last truncation)
        self._needs_reset = np.zeros(num_envs, dtype=bool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_keys(self, n: int) -> jnp.ndarray:
        """Return ``n`` PRNG keys derived from the env's master RNG."""
        seed = int(self._rng.integers(0, 2**31 - 1))
        return jax.random.split(jax.random.PRNGKey(seed), n)

    def _fresh_batched_state(self):
        return self._batched_reset(self._split_keys(self.num_envs))

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

    def _reset_subset(self, mask: np.ndarray) -> None:
        """Full episode reset: kickoff + zero step counter for affected envs."""
        self._kickoff_subset(mask)
        self._step_count[mask] = 0

    # ------------------------------------------------------------------
    # Public helpers for wrappers (e.g. SB3 adapter)
    # ------------------------------------------------------------------

    def reset_envs(self, mask: np.ndarray) -> np.ndarray:
        """Reset (kickoff + zero step counter) the envs where ``mask`` is True.

        Returns the full batched observation after the reset. Wrappers that
        want eager auto-reset semantics (e.g. SB3's VecEnv) call this after
        ``step`` to bring done envs to their post-reset state immediately,
        rather than waiting for the next ``step`` call.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError(
                f"mask shape {mask.shape} != (num_envs,) = ({self.num_envs},)"
            )
        if mask.any():
            self._reset_subset(mask)
            # Cancel any pending NEXT_STEP auto-reset for these envs — they
            # were just reset eagerly.
            self._needs_reset[mask] = False
        return _export_obs(self._state)

    def current_obs(self) -> np.ndarray:
        """Return the current batched observation without stepping."""
        if self._state is None:
            raise RuntimeError("call reset() before current_obs()")
        return _export_obs(self._state)

    def _opponent_actions(self, obs_np: np.ndarray) -> np.ndarray:
        """Build a (num_envs, N_ROBOTS, 2) opponent action array."""
        if self._opponent_policy_spec == "stationary":
            return np.zeros((self.num_envs, config.N_ROBOTS, 2), dtype=np.float32)
        if self._opponent_policy_spec == "random":
            return self._rng.uniform(
                -1.0, 1.0, size=(self.num_envs, config.N_ROBOTS, 2),
            ).astype(np.float32)
        # callable: per-env Python loop (slow path, documented)
        out = np.empty((self.num_envs, config.N_ROBOTS, 2), dtype=np.float32)
        for i in range(self.num_envs):
            out[i] = np.asarray(
                self._opponent_policy_spec(obs_np[i]),
                dtype=np.float32,
            ).reshape(config.N_ROBOTS, 2)
        return out

    # ------------------------------------------------------------------
    # Gymnasium VectorEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._fresh_batched_state()
        self._step_count[:] = 0
        self._needs_reset[:] = False

        obs = _export_obs(self._state)
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        # --- 1) Apply any pending auto-resets from the previous step ---
        if self._needs_reset.any():
            self._reset_subset(self._needs_reset)
            self._needs_reset[:] = False

        # --- 2) Build batched (B, N_TEAMS, N_ROBOTS, 2) action tensor ---
        blue = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, config.N_ROBOTS, 2,
        )
        obs_now = np.asarray(_build_obs_batched(self._state))
        yellow = self._opponent_actions(obs_now)
        all_actions = np.stack([blue, yellow], axis=1).astype(np.float32)  # (B, 2, 3, 2)
        all_actions_j = jnp.asarray(all_actions)

        # --- 3) One JIT'd vmap'd physics step ---
        ball_x_pre = np.asarray(self._state.ball[:, 0], dtype=np.float32)
        self._state, info_phys = self._batched_step(self._state, all_actions_j)
        ball_x_post = np.asarray(self._state.ball[:, 0], dtype=np.float32)
        goals = np.asarray(info_phys["goal"], dtype=np.int32)  # (B,)

        # --- 4) Score bookkeeping + in-episode kickoff on goal ---
        scored_mask = goals != 0
        if scored_mask.any():
            blue_inc = (goals == 1).astype(np.int32)
            yellow_inc = (goals == -1).astype(np.int32)
            inc = np.stack([blue_inc, yellow_inc], axis=-1)  # (B, 2)
            new_score = self._state.score + jnp.asarray(inc, dtype=jnp.int32)
            self._state = self._state._replace(score=new_score)
            # Goals trigger a kickoff but DO NOT end the episode (no step-counter reset).
            self._kickoff_subset(scored_mask)

        # --- 5) Step counter + truncation ---
        self._step_count += 1
        rewards = (
            goals.astype(np.float32)
            + np.float32(config.BALL_FORWARD_REWARD_COEF) * (ball_x_post - ball_x_pre)
        )
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = self._step_count >= self.max_episode_steps

        # Mark envs to auto-reset on the *next* step (NEXT_STEP autoreset semantics)
        self._needs_reset = truncations.copy()

        obs = _export_obs(self._state)
        info: dict[str, Any] = {"goal": goals}
        return obs, rewards, terminations, truncations, info

    def close(self, **kwargs: Any) -> None:
        # Nothing device-side to free explicitly; JAX manages its own buffers.
        self._state = None
