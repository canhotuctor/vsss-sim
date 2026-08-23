"""
Base Gymnasium environment for VSSS.

Defines the observation/action spaces, normalisation constants, and
the core helper methods (_get_obs, _get_info) shared by all env variants.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import config, physics

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
_NORM_POS_X = config.FIELD_LENGTH / 2.0
_NORM_POS_Y = config.FIELD_WIDTH / 2.0
_NORM_VEL = config.ROBOT_MAX_WHEEL_SPEED * config.VELOCITY_NORM_HEADROOM
_NORM_OMEGA = _NORM_VEL / (config.ROBOT_WHEELBASE / 2.0)

# Observation index helpers
OBS_BALL_SLICE = slice(0, 4)


class VSSBaseEnv(gym.Env):
    """
    Base environment: spaces, state container, and observation builder.

    Subclasses implement ``reset``, ``step``, ``render``, and ``close``.

    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(config.FPS),
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        render_fps: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._render_fps = render_fps
        self._rng = np.random.default_rng()
        self._state = self._initial_state()
        self._step_count = 0
        self._episode_count = 0
        self._renderer = None

        # Observation space: 46-dimensional continuous. All components are
        # normalised in `_get_obs`; their worst-case magnitudes (including
        # collision spikes and ball-in-goal-pocket overshoot) sit well under
        # 2.0. Bound at ±5.0 for a generous safety margin while still being
        # finite (Gymnasium wrappers like NormalizeObservation require it).
        obs_dim = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: 6 wheel speeds in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.N_ROBOTS * 2,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _initial_state(self):
        """Create an empty JAX simulation state."""
        return physics.empty_state()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return a normalised flat observation vector."""
        s = self._state
        ball = np.asarray(s.ball)
        robots = np.asarray(s.robots)
        obs = np.empty(4 + config.N_TEAMS * config.N_ROBOTS * 7, dtype=np.float32)

        # Ball
        obs[0] = ball[0] / _NORM_POS_X
        obs[1] = ball[1] / _NORM_POS_Y
        obs[2] = ball[2] / _NORM_VEL
        obs[3] = ball[3] / _NORM_VEL

        # Robots (both teams, blue first)
        idx = 4
        for team in range(config.N_TEAMS):
            for r in range(config.N_ROBOTS):
                x, y, theta, vx, vy, omega = robots[team, r]
                obs[idx + 0] = x / _NORM_POS_X
                obs[idx + 1] = y / _NORM_POS_Y
                obs[idx + 2] = float(np.sin(theta))
                obs[idx + 3] = float(np.cos(theta))
                obs[idx + 4] = vx / _NORM_VEL
                obs[idx + 5] = vy / _NORM_VEL
                obs[idx + 6] = omega / _NORM_OMEGA
                idx += 7

        return obs

    def _get_info(self) -> dict[str, Any]:
        score = np.asarray(self._state.score)
        return {
            "score_blue": int(score[config.TEAM_BLUE]),
            "score_yellow": int(score[config.TEAM_YELLOW]),
            "sim_time": float(np.asarray(self._state.t)),
        }
