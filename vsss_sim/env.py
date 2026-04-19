"""
Gymnasium environment for the IEEE VSSS 3 v 3 simulator.

Observation space (46-dimensional ``Box``):
    [ ball_x/norm, ball_y/norm, ball_vx/norm, ball_vy/norm,          # 4
      (per robot × 6 robots):                                         # 6×7 = 42
          x/norm, y/norm, sin(θ), cos(θ), vx/norm, vy/norm, ω/norm ]

Action space for the controlled team (``Box`` shape ``(6,)``):
    [ vl_0, vr_0, vl_1, vr_1, vl_2, vr_2 ]  in [-1, 1]

The **controlled team** is ``config.TEAM_BLUE`` by default.
The **opponent** follows a pluggable policy (default: stationary zeros).

GPU / vectorisation note
------------------------
For batch-environment GPU training, wrap multiple ``VSSEnv`` instances with
``gymnasium.vector.SyncVectorEnv`` / ``AsyncVectorEnv``, or replace the
``physics`` module's NumPy back-end with PyTorch tensors (all physics
operations are compatible with ``torch`` broadcasting semantics).
Isaac Gym / PhysX can be integrated by providing a custom ``physics.step``
callable that delegates to the GPU engine while preserving the
``SimState`` array layout.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .physics import SimState, reset_kickoff
from .physics import step as physics_step

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
_NORM_POS_X = config.FIELD_LENGTH / 2.0
_NORM_POS_Y = config.FIELD_WIDTH / 2.0
_NORM_VEL = config.ROBOT_MAX_WHEEL_SPEED * config.VELOCITY_NORM_HEADROOM
_NORM_OMEGA = _NORM_VEL / (config.ROBOT_WHEELBASE / 2.0)

# Observation indices for convenience
OBS_BALL_SLICE = slice(0, 4)


# ---------------------------------------------------------------------------
# Built-in opponent policies
# ---------------------------------------------------------------------------

def _policy_stationary(obs: np.ndarray) -> np.ndarray:
    """Opponent always produces zero wheel speeds."""
    return np.zeros((config.N_ROBOTS, 2), dtype=np.float32)


def _policy_random(rng: np.random.Generator) -> Callable:
    """Opponent sends uniformly random wheel speeds."""
    def _inner(obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(config.N_ROBOTS, 2)).astype(np.float32)
    return _inner


# ---------------------------------------------------------------------------
# VSSEnv
# ---------------------------------------------------------------------------

class VSSEnv(gym.Env):
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

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(config.FPS),
    }

    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng()
        self._state = SimState()
        self._step_count = 0
        self._renderer = None

        # Opponent policy
        if callable(opponent_policy):
            self._opponent_policy: Callable = opponent_policy
        elif opponent_policy == "stationary":
            self._opponent_policy = _policy_stationary
        elif opponent_policy == "random":
            self._opponent_policy = _policy_random(self._rng)
        else:
            raise ValueError(
                f"Unknown opponent_policy '{opponent_policy}'. "
                "Choose 'stationary', 'random', or pass a callable."
            )

        # Observation space: 46-dimensional continuous [-inf, inf]
        obs_dim = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
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
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return a normalised flat observation vector."""
        s = self._state
        obs = np.empty(4 + config.N_TEAMS * config.N_ROBOTS * 7, dtype=np.float32)

        # Ball
        obs[0] = s.ball[0] / _NORM_POS_X
        obs[1] = s.ball[1] / _NORM_POS_Y
        obs[2] = s.ball[2] / _NORM_VEL
        obs[3] = s.ball[3] / _NORM_VEL

        # Robots (both teams, blue first)
        idx = 4
        for team in range(config.N_TEAMS):
            for r in range(config.N_ROBOTS):
                x, y, theta, vx, vy, omega = s.robots[team, r]
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
        return {
            "score_blue": int(self._state.score[config.TEAM_BLUE]),
            "score_yellow": int(self._state.score[config.TEAM_YELLOW]),
            "sim_time": float(self._state.t),
        }

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
        # Build full (N_TEAMS, N_ROBOTS, 2) action tensor
        blue_actions = np.array(action, dtype=np.float64).reshape(
            config.N_ROBOTS, 2
        )
        obs_current = self._get_obs()
        yellow_actions = self._opponent_policy(obs_current).reshape(
            config.N_ROBOTS, 2
        )
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
            # Reset to kickoff after a goal
            reset_kickoff(self._state, rng=self._rng)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from .rendering import VSSRenderer
            self._renderer = VSSRenderer(render_mode=self.render_mode)

        return self._renderer.render(
            self._state.ball,
            self._state.robots,
            self._state.score,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
