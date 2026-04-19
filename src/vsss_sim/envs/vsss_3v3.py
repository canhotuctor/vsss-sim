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
"""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat

import numpy as np

from .. import config
from ..agents import random_policy, stationary_policy
from ..physics import SimState, reset_kickoff
from ..physics import step as physics_step
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
    """

    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        render_fps: Optional[float] = None,
    ) -> None:
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps, render_fps=render_fps)

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

        self._state = SimState()
        reset_kickoff(self._state, rng=self._rng)
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Parameters
        ----------
        action : ndarray (6,)
            Normalised wheel speeds for the blue team:
            ``[vl_0, vr_0, vl_1, vr_1, vl_2, vr_2]``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        blue_actions = np.array(action, dtype=np.float64).reshape(config.N_ROBOTS, 2)
        obs_current = self._get_obs()
        yellow_actions = self._opponent_policy(obs_current).reshape(config.N_ROBOTS, 2)
        all_actions = np.stack([blue_actions, yellow_actions], axis=0)

        info_phys = physics_step(self._state, all_actions)
        self._step_count += 1

        # Scoring
        goal = info_phys["goal"]
        if goal == 1:
            self._state.score[config.TEAM_BLUE] += 1
        elif goal == -1:
            self._state.score[config.TEAM_YELLOW] += 1

        # Reward: simple sparse ±1 on goal, 0 otherwise
        reward = float(goal)

        terminated = False  # VSSS has no terminal state mid-match
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        info["goal"] = goal

        if goal != 0:
            reset_kickoff(self._state, rng=self._rng)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from ..rendering import VSSRenderer
            self._renderer = VSSRenderer(render_mode=self.render_mode, fps=self._render_fps)

        return self._renderer.render(
            self._state.ball,
            self._state.robots,
            self._state.score,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
