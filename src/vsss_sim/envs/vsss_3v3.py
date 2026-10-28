"""Gymnasium adapter for a single IEEE VSSS 3v3 match."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, SupportsFloat

import jax
import jax.numpy as jnp
import numpy as np

from .. import config, physics
from ..agents import random_policy, stationary_policy
from ..config import InitMode, RefereeMode, RewardProfile
from .base import VSSBaseEnv
from .match_rules import (
    MatchPhase,
    MatchRules,
    MatchStepInput,
    initial_referee_state,
)


class VSSEnv(VSSBaseEnv):
    """Gymnasium environment for one VSSS match.

    The blue action contains six normalized wheel speeds. Yellow uses
    stationary, random, or a callable returning an array of shape (3, 2).
    """

    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        control_rate_hz: float = config.FPS,
        physics_substeps: int = config.SUB_STEPS,
        speedup_rate: float = 1.0,
        init_mode: InitMode | str = InitMode.KICKOFF,
        reward_profile: RewardProfile | str = RewardProfile.DEFAULT,
        referee_mode: RefereeMode | str = RefereeMode.FULL,
    ) -> None:
        if isinstance(control_rate_hz, bool):
            raise ValueError("control_rate_hz must be a positive finite number")
        if isinstance(speedup_rate, bool):
            raise ValueError("speedup_rate must be a positive finite number")
        try:
            control_rate_hz = float(control_rate_hz)
            substeps_value = float(physics_substeps)
            speedup_rate = float(speedup_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("control rate, substeps, and speedup must be numeric") from error
        if not math.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be a positive finite number")
        if (
            isinstance(physics_substeps, bool)
            or not math.isfinite(substeps_value)
            or not substeps_value.is_integer()
            or substeps_value < 1
        ):
            raise ValueError("physics_substeps must be a positive integer")
        if not math.isfinite(speedup_rate) or speedup_rate <= 0.0:
            raise ValueError("speedup_rate must be a positive finite number")

        self._control_rate_hz = control_rate_hz
        self._control_dt = 1.0 / self._control_rate_hz
        self._physics_substeps = int(substeps_value)
        self._base_render_fps = self._control_rate_hz * self._physics_substeps
        self._speedup_rate = speedup_rate
        # Human mode paces frames by simulated time: sub-steps per wall second.
        # At >1x, frames group several sub-steps so display refresh cannot pin
        # simulated time to one physics sub-step per monitor refresh.
        target_substep_rate = self._base_render_fps * self._speedup_rate
        self._render_substep_accumulator = 0.0
        self._render_substeps_since_frame = 0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            render_fps=target_substep_rate if render_mode == "human" else None,
        )

        self._init_mode = InitMode(init_mode)
        self._match_rules = MatchRules(
            reward_profile=reward_profile,
            max_episode_steps=max_episode_steps,
            init_mode=self._init_mode,
            referee_mode=referee_mode,
        )
        self._compiled_match_step = jax.jit(self._match_rules.step)
        self._referee_state = initial_referee_state()

        if callable(opponent_policy):
            self._opponent_policy: Callable = opponent_policy
        elif opponent_policy == "stationary":
            self._opponent_policy = stationary_policy
        elif opponent_policy == "random":
            self._opponent_policy = random_policy(self._rng)
        else:
            raise ValueError(
                f"Unknown opponent_policy '{opponent_policy}'. "
                "Choose 'stationary', 'random', or pass a callable."
            )

    # Physics helpers

    def _reset_state(self) -> None:
        """Replace ``self._state`` using the configured init_mode."""
        reset_fn_name = (
            "reset_kickoff" if self._init_mode == InitMode.KICKOFF else "reset_random"
        )
        key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
        self._state = getattr(physics, reset_fn_name)(key)
        self._referee_state = initial_referee_state()

    def _step_physics(self, all_actions: np.ndarray) -> int:
        """Advance the physics by one control step. Returns the goal event."""
        j_actions = jnp.asarray(all_actions, dtype=jnp.float32)
        self._state, info_phys = physics.step(
            self._state,
            j_actions,
            dt=self._control_dt,
            sub_steps=self._physics_substeps,
        )
        return int(info_phys["goal"])

    def _step_physics_substeps(self, all_actions: np.ndarray) -> int:
        """Advance one control interval and draw its physics sub-steps.

        A single engine call returns every intermediate state, so rendered
        matches follow exactly the physics of ``_step_physics``. At normal
        speed every sub-step is drawn; at higher speeds frames are sampled.
        """
        self._state, info_phys, frames = physics.step_with_frames(
            self._state,
            jnp.asarray(all_actions, dtype=jnp.float32),
            dt=self._control_dt,
            sub_steps=self._physics_substeps,
        )
        balls, robots = np.asarray(frames.ball), np.asarray(frames.robots)
        goals = np.asarray(info_phys["goals"])
        for index in range(self._physics_substeps):
            self._render_after_physics(1, int(goals[index]), (balls[index], robots[index]))
        return int(info_phys["goal"])

    def _render_after_physics(
        self,
        simulated_substeps: int,
        goal: int,
        frame: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Render sparsely while preserving the simulation-time speed target."""
        self._render_substeps_since_frame += simulated_substeps
        should_render = self._speedup_rate <= 1.0
        if self._speedup_rate > 1.0:
            self._render_substep_accumulator += simulated_substeps
            should_render = (
                self._render_substep_accumulator >= self._speedup_rate or goal != 0
            )
            if should_render:
                if self._render_substep_accumulator >= self._speedup_rate:
                    self._render_substep_accumulator -= self._speedup_rate
                else:
                    self._render_substep_accumulator = 0.0
        if should_render:
            self._render_current_state(self._render_substeps_since_frame, frame)
            self._render_substeps_since_frame = 0

    # Gymnasium interface

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_state()
        self._step_count = 0
        self._episode_count += 1
        self._render_substep_accumulator = 0.0
        self._render_substeps_since_frame = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance one control timestep.

        Parameters
        ----------
        action : ndarray (6,)
            Normalised wheel speeds for the blue team:
            ``[vl_0, vr_0, vl_1, vr_1, vl_2, vr_2]``.
        """
        simulation_before = self._state
        blue_actions = np.asarray(action, dtype=np.float32).reshape(config.N_ROBOTS, 2)
        obs_current = self._get_obs()
        yellow_actions = self._opponent_policy(obs_current).reshape(config.N_ROBOTS, 2)
        all_actions = np.stack([blue_actions, yellow_actions], axis=0)
        stopped = int(np.asarray(self._referee_state.phase)) == MatchPhase.STOPPED
        if stopped:
            all_actions = np.zeros_like(all_actions)

        if self.render_mode == "human":
            goal = self._step_physics_substeps(all_actions)
        else:
            goal = self._step_physics(all_actions)
        goal = 0 if stopped else goal
        self._step_count += 1
        # Also supplies unbiased tie-breaking for simultaneous referee calls.
        kickoff_key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
        transition = self._compiled_match_step(
            MatchStepInput(
                simulation_before=simulation_before,
                simulation_after_physics=self._state,
                goal=jnp.asarray(goal, dtype=jnp.int32),
                step_count=jnp.asarray(self._step_count, dtype=jnp.int32),
                kickoff_key=kickoff_key,
                referee_state=self._referee_state,
                dt=jnp.float32(self._control_dt),
            )
        )
        self._state = transition.simulation
        self._referee_state = transition.referee_state
        reward = float(np.asarray(transition.reward))

        terminated = False  # VSSS has no terminal state mid-match
        truncated = self._step_count >= self.max_episode_steps

        obs = np.asarray(transition.observation, dtype=np.float32)
        info = self._get_info()
        info["goal"] = goal
        info["event"] = int(np.asarray(transition.event))
        info["offending_team"] = int(np.asarray(transition.offending_team))
        info["restart"] = int(np.asarray(transition.restart))
        info["restart_team"] = int(np.asarray(transition.restart_team))
        info["quadrant"] = int(np.asarray(transition.quadrant))
        info["match_phase"] = int(np.asarray(self._referee_state.phase))

        if self.render_mode == "rgb_array":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        return self._render_current_state(simulated_substeps=1)

    def _render_current_state(
        self,
        simulated_substeps: int,
        frame: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """Draw ``frame`` (ball, robots) from a sub-step, or the current state."""
        if self._renderer is None:
            from ..rendering import VSSRenderer
            self._renderer = VSSRenderer(
                render_mode=self.render_mode,
                fps=self._render_fps,
                baseline_fps=self._base_render_fps,
                control_dt=self._control_dt,
            )

        ball, robots = (
            (np.asarray(self._state.ball), np.asarray(self._state.robots))
            if frame is None
            else frame
        )
        return self._renderer.render(
            ball,
            robots,
            np.asarray(self._state.score),
            step=self._step_count,
            episode=self._episode_count,
            simulated_substeps=simulated_substeps,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
