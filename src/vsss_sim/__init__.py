"""
vsss_sim – IEEE VSSS 3 v 3 simulator for Reinforcement Learning.

Quick start
-----------
>>> import gymnasium as gym
>>> import vsss_sim  # noqa: F401 – registers the environment
>>> env = gym.make("VSSS-v0")
>>> obs, info = env.reset(seed=42)
>>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import gymnasium as gym

from . import config  # noqa: F401 – expose constants at package level
from .envs import VSSEnv, VSSVecEnv

__version__ = "0.1.0"
__all__ = ["VSSEnv", "VSSVecEnv", "config"]

# ---------------------------------------------------------------------------
# Register Gymnasium entry points
# ---------------------------------------------------------------------------
# Single-env: gym.make("VSSS-v0")
# Vector env: gym.make_vec("VSSS-v0", num_envs=N, vectorization_mode="custom")
gym.register(
    id="VSSS-v0",
    entry_point="vsss_sim.envs:VSSEnv",
    vector_entry_point="vsss_sim.envs:VSSVecEnv",
    max_episode_steps=config.MAX_EPISODE_STEPS,
)
