"""
IEEE VSSS 3 v 3 Gymnasium environment.

Observation space (46-dimensional ``Box``):
    [ ball_x/norm, ball_y/norm, ball_vx/norm, ball_vy/norm,          # 4
      (per robot × 6 robots):                                         # 6×7 = 42
          x/norm, y/norm, sin θ, cos θ, vx/norm, vy/norm, ω/norm ]

Action space for the controlled team (``Box`` shape ``(6,)``):
    [ vl_0, vr_0, vl_1, vr_1, vl_2, vr_2 ]  in [-1, 1]

The **controlled team** is ``config.TEAM_BLUE`` by default.
The **opponent** follows a pluggable policy (default: stationary zeros).

Physics runs through the project's pure-functional JAX engine.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat

import jax
import jax.numpy as jnp
import numpy as np

from .. import config, physics
from ..agents import random_policy, stationary_policy
from ..config import InitMode
from .base import VSSBaseEnv


class VSSEnv(VSSBaseEnv):
    """
    IEEE VSSS 3 v 3 Gymnasium environment.

    Parameters
    ----------
    opponent_policy : str or callable, optional
        ``"stationary"`` (default) – yellow robots do not move.
        ``"random"``                – yellow robots use uniformly random actions.
        callable                    – called with the current observation and
                                      must return an ndarray of shape
                                      ``(N_ROBOTS, 2)`` with values in [-1, 1].
    render_mode : ``"human"`` | ``"rgb_array"`` | ``None``
    max_episode_steps : int
        Episode length in simulation steps (default: ``config.MAX_EPISODE_STEPS``).
    init_mode : ``InitMode`` | ``"kickoff"`` | ``"random"``
        Placement strategy used at every episode reset and in-episode kickoff.
        ``"kickoff"`` (default) uses the standard formation with small jitter.
        ``"random"`` places robots uniformly in their respective halves with
        random headings. Future modes (e.g. a learned selector) can be added
        by extending :class:`~vsss_sim.config.InitMode`.
    """

    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        render_fps: Optional[float] = None,
        init_mode: InitMode | str = InitMode.KICKOFF,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            render_fps=render_fps,
        )

        self._init_mode = InitMode(init_mode)

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

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Replace ``self._state`` using the configured init_mode."""
        reset_fn_name = (
            "reset_kickoff" if self._init_mode == InitMode.KICKOFF else "reset_random"
        )
        key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
        self._state = getattr(physics, reset_fn_name)(key)

    def _step_physics(self, all_actions: np.ndarray) -> int:
        """Advance the physics by one control step. Returns the goal event."""
        j_actions = jnp.asarray(all_actions, dtype=jnp.float32)
        self._state, info_phys = physics.step(self._state, j_actions)
        return int(info_phys["goal"])

    def _step_physics_substeps(self, all_actions: np.ndarray) -> int:
        """Run each physics sub-step individually, rendering after each one.

        Used only in ``render_mode="human"`` so the window shows smooth
        intermediate motion rather than one jump per control step.
        """
        sub_dt = config.DT / config.SUB_STEPS
        goal = 0

        j_actions = jnp.asarray(all_actions, dtype=jnp.float32)
        for _ in range(config.SUB_STEPS):
            self._state, info_phys = physics.step(
                self._state, j_actions, dt=sub_dt, sub_steps=1
            )
            g = int(info_phys["goal"])
            if goal == 0:
                goal = g
            self.render()

        return goal

    def _bump_score(self, team_idx: int) -> None:
        """Increment the score of one team."""
        self._state = self._state._replace(
            score=self._state.score.at[team_idx].add(1)
        )

    def _crowding_penalty(self) -> float:
        """Return a negative reward when 2+ allied robots crowd a goal area.

        Both goal areas are checked.
        """
        blue_pos = np.asarray(self._state.robots)[config.TEAM_BLUE, :, 0:2]
        half_l = config.FIELD_LENGTH / 2.0
        ga_half_y = config.GOAL_AREA_LENGTH_Y / 2.0
        ga_x = config.GOAL_AREA_LENGTH_X

        penalty = 0.0
        for x_min, x_max in (
            (-half_l, -half_l + ga_x),   # allied (blue) goal area
            ( half_l - ga_x,  half_l),   # opponent (yellow) goal area
        ):
            in_area = (
                (blue_pos[:, 0] >= x_min) & (blue_pos[:, 0] <= x_max)
                & (np.abs(blue_pos[:, 1]) <= ga_half_y)
            )
            if int(np.sum(in_area)) >= 2:
                penalty += config.GOAL_AREA_CROWDING_PENALTY
        return penalty

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

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
        blue_actions = np.asarray(action, dtype=np.float32).reshape(config.N_ROBOTS, 2)
        obs_current = self._get_obs()
        yellow_actions = self._opponent_policy(obs_current).reshape(config.N_ROBOTS, 2)
        all_actions = np.stack([blue_actions, yellow_actions], axis=0)

        ball_x_pre = float(np.asarray(self._state.ball)[0])
        if self.render_mode == "human":
            goal = self._step_physics_substeps(all_actions)
        else:
            goal = self._step_physics(all_actions)
        ball_x_post = float(np.asarray(self._state.ball)[0])
        self._step_count += 1

        if goal == 1:
            self._bump_score(config.TEAM_BLUE)
        elif goal == -1:
            self._bump_score(config.TEAM_YELLOW)

        # Reward: sparse ±1 on goal + small dense ball-forward-progress shaping
        # + penalty for packing 2+ allied robots into any goal area
        reward = (
            float(goal)
            + config.BALL_FORWARD_REWARD_COEF * (ball_x_post - ball_x_pre)
            + self._crowding_penalty()
        )

        terminated = False  # VSSS has no terminal state mid-match
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        info["goal"] = goal

        if goal != 0:
            self._reset_state()

        if self.render_mode == "rgb_array":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from ..rendering import VSSRenderer
            # In human mode, render() is called SUB_STEPS times per control
            # step, so the clock rate must be scaled up so that the wall-clock
            # rate of *control steps* (not sub-steps) matches render_fps.
            clock_fps = (
                self._render_fps * config.SUB_STEPS
                if self.render_mode == "human" and self._render_fps is not None
                else self._render_fps
            )
            self._renderer = VSSRenderer(render_mode=self.render_mode, fps=clock_fps)

        return self._renderer.render(
            np.asarray(self._state.ball),
            np.asarray(self._state.robots),
            np.asarray(self._state.score),
            step=self._step_count,
            episode=self._episode_count,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
