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
from .env import VSSEnv

__version__ = "0.1.0"
__all__ = ["VSSEnv", "config"]

# ---------------------------------------------------------------------------
# Register Gymnasium entry point
# ---------------------------------------------------------------------------
gym.register(
    id="VSSS-v0",
    entry_point="vsss_sim.env:VSSEnv",
    max_episode_steps=config.MAX_EPISODE_STEPS,
)
