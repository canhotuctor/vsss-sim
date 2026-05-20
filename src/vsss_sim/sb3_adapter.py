"""
Adapter: expose :class:`VSSVecEnv` (Gymnasium ``VectorEnv``) as an SB3 ``VecEnv``.

Why this exists
---------------
Stable Baselines 3 (≤ 2.8) has its own ``VecEnv`` interface that is *not* the
same as ``gymnasium.vector.VectorEnv``. The two differ in three load-bearing
ways:

1. **`reset()`** returns *only* the observation (no info tuple).
2. **`step()`** returns ``(obs, rewards, dones, list[dict])`` — one boolean
   ``dones`` array (= ``terminated | truncated``) instead of two, and a list
   of per-env dicts instead of a dict-of-arrays.
3. **Auto-reset is eager**: when ``dones[i]`` is True, the observation
   returned by ``step_wait()`` is *already* the post-reset observation, and
   the pre-reset (terminal) observation lives in
   ``info["terminal_observation"]``. Gymnasium's standard behaviour is
   "next-step" autoreset, which ``VSSVecEnv`` follows internally.

This adapter handles all three differences.

Episode statistics
------------------
SB3's logger reads ``info["episode"] = {"r": cumulative_reward, "l": length,
"t": wall_clock_seconds}`` on done. We track these per-env in the adapter
(SB3's ``Monitor`` wrapper does the same job for single envs).

Example
-------

>>> from vsss_sim.envs import VSSVecEnv
>>> from vsss_sim.sb3_adapter import VSSVecEnvToSB3
>>> from stable_baselines3 import PPO
>>>
>>> env = VSSVecEnvToSB3(VSSVecEnv(num_envs=256, opponent_policy="stationary"))
>>> model = PPO("MlpPolicy", env, n_steps=128, verbose=1)
>>> model.learn(total_timesteps=1_000_000)
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .envs import VSSVecEnv


def _normalise_indices(indices, num_envs: int) -> list[int]:
    if indices is None:
        return list(range(num_envs))
    if isinstance(indices, int):
        return [indices]
    return list(indices)


class VSSVecEnvToSB3(VecEnv):
    """SB3-compatible wrapper around a :class:`VSSVecEnv`.

    The wrapped env *is* the source of truth — this adapter only translates
    SB3-shaped calls to/from Gymnasium-shaped calls and handles eager
    auto-reset semantics + episode statistics.

    Parameters
    ----------
    vec_env : VSSVecEnv
        The batched JAX env to wrap. Its ``num_envs`` decides the SB3
        ``VecEnv`` size.
    """

    def __init__(self, vec_env: VSSVecEnv) -> None:
        self._vec = vec_env
        self._actions: Optional[np.ndarray] = None
        # Episode trackers (host-side, per-env)
        self._ep_returns = np.zeros(vec_env.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(vec_env.num_envs, dtype=np.int64)
        self._ep_start_time = np.full(vec_env.num_envs, time.time(), dtype=np.float64)
        self.render_mode = None  # set BEFORE super().__init__ — it queries get_attr

        super().__init__(
            num_envs=vec_env.num_envs,
            observation_space=vec_env.single_observation_space,
            action_space=vec_env.single_action_space,
        )

    # ------------------------------------------------------------------
    # Core SB3 VecEnv abstract methods
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        # SB3 reset honours seeds set via env.seed(seed). Pull the master seed
        # (env 0) if one was set; ignore per-env seeds since VSSVecEnv only
        # accepts a single master seed.
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        obs, _info = self._vec.reset(seed=seed)
        self._seeds = [None] * self.num_envs
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        self._ep_start_time[:] = time.time()
        # Per SB3 contract, reset_infos is the per-env info from reset.
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        assert self._actions is not None, "step_async must be called before step_wait"
        obs, rewards, terminations, truncations, _info_dict = self._vec.step(self._actions)
        self._actions = None

        dones = np.logical_or(terminations, truncations)

        # Update episode trackers with this step's reward + length.
        self._ep_returns += rewards.astype(np.float64)
        self._ep_lengths += 1

        # Build SB3-style per-env infos. Default to empty dicts.
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        if dones.any():
            # 1) Stash terminal obs BEFORE we replace it with the post-reset obs.
            terminal_obs = obs.copy()

            # 2) Emit episode info for done envs (SB3 logger reads this).
            now = time.time()
            for i in range(self.num_envs):
                if not dones[i]:
                    continue
                infos[i]["terminal_observation"] = terminal_obs[i]
                infos[i]["TimeLimit.truncated"] = bool(
                    truncations[i] and not terminations[i]
                )
                infos[i]["episode"] = {
                    "r": float(self._ep_returns[i]),
                    "l": int(self._ep_lengths[i]),
                    "t": float(now - self._ep_start_time[i]),
                }

            # 3) Eagerly reset the done envs so the returned obs is post-reset.
            obs = self._vec.reset_envs(dones)

            # 4) Zero trackers for the reset envs; restart their clocks.
            self._ep_returns[dones] = 0.0
            self._ep_lengths[dones] = 0
            self._ep_start_time[dones] = now

        return obs, rewards.astype(np.float32), dones, infos

    def close(self) -> None:
        self._vec.close()

    # ------------------------------------------------------------------
    # Introspection methods (SB3 calls these for logging / wrapper detection)
    # ------------------------------------------------------------------

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        idx = _normalise_indices(indices, self.num_envs)
        if attr_name == "render_mode":
            return [None] * len(idx)
        # Forward to the underlying VSSVecEnv for anything we don't special-case.
        if hasattr(self._vec, attr_name):
            value = getattr(self._vec, attr_name)
            return [value] * len(idx)
        raise AttributeError(
            f"'VSSVecEnvToSB3' / 'VSSVecEnv' have no attribute {attr_name!r}"
        )

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        idx = _normalise_indices(indices, self.num_envs)
        if len(idx) != self.num_envs:
            raise NotImplementedError(
                "Per-env set_attr is not supported (VSSVecEnv holds shared state)."
            )
        setattr(self._vec, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        idx = _normalise_indices(indices, self.num_envs)
        if not hasattr(self._vec, method_name):
            raise AttributeError(
                f"'VSSVecEnv' has no method {method_name!r}"
            )
        method = getattr(self._vec, method_name)
        # A single call on the underlying batched env covers all "envs"; we
        # replicate the return value so SB3 sees one entry per requested idx.
        result = method(*method_args, **method_kwargs)
        return [result] * len(idx)

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices=None
    ) -> list[bool]:
        # The underlying VSSVecEnv handles truncation itself; no wrappers in
        # the pipeline.
        idx = _normalise_indices(indices, self.num_envs)
        return [False] * len(idx)

    # ------------------------------------------------------------------
    # Optional but useful: rendering (returns env-0 frame if available)
    # ------------------------------------------------------------------

    def get_images(self) -> list:
        # VSSVecEnv doesn't render today. Return None per env so SB3's render
        # default-impl is a no-op.
        return [None] * self.num_envs
