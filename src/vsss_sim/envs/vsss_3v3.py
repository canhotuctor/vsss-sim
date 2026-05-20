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

Physics backend is selectable via the ``backend`` kwarg (``"numpy"`` or
``"jax"``) or the ``VSSS_PHYSICS_BACKEND`` environment variable.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat

import numpy as np

from .. import config
from ..agents import random_policy, stationary_policy
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
    backend : ``"numpy"`` | ``"jax"`` | ``None``
        Physics backend. ``None`` defers to ``VSSS_PHYSICS_BACKEND``, then ``"numpy"``.
    """

    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        render_fps: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            render_fps=render_fps,
            backend=backend,
        )

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
    # Backend dispatch helpers
    # ------------------------------------------------------------------

    def _is_jax(self) -> bool:
        return self._backend_name == "jax_backend"

    def _reset_state(self) -> None:
        """Replace ``self._state`` with a kickoff configuration."""
        if self._is_jax():
            import jax
            key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
            self._state = self._backend.reset_kickoff(key)
        else:
            self._state = self._backend.SimState()
            self._backend.reset_kickoff(self._state, rng=self._rng)

    def _step_physics(self, all_actions: np.ndarray) -> int:
        """Advance the physics by one control step. Returns the goal event."""
        if self._is_jax():
            import jax.numpy as jnp
            j_actions = jnp.asarray(all_actions, dtype=jnp.float32)
            self._state, info_phys = self._backend.step(self._state, j_actions)
            return int(info_phys["goal"])
        info_phys = self._backend.step(self._state, all_actions)
        return int(info_phys["goal"])

    def _bump_score(self, team_idx: int) -> None:
        """Increment the score of one team (backend-agnostic)."""
        if self._is_jax():
            self._state = self._state._replace(
                score=self._state.score.at[team_idx].add(1)
            )
        else:
            self._state.score[team_idx] += 1

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
        blue_actions = np.array(action, dtype=np.float64).reshape(config.N_ROBOTS, 2)
        obs_current = self._get_obs()
        yellow_actions = self._opponent_policy(obs_current).reshape(config.N_ROBOTS, 2)
        all_actions = np.stack([blue_actions, yellow_actions], axis=0)

        goal = self._step_physics(all_actions)
        self._step_count += 1

        if goal == 1:
            self._bump_score(config.TEAM_BLUE)
        elif goal == -1:
            self._bump_score(config.TEAM_YELLOW)

        # Reward: simple sparse ±1 on goal, 0 otherwise
        reward = float(goal)

        terminated = False  # VSSS has no terminal state mid-match
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        info["goal"] = goal

        if goal != 0:
            self._reset_state()

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
            np.asarray(self._state.ball),
            np.asarray(self._state.robots),
            np.asarray(self._state.score),
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
