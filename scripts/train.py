#!/usr/bin/env python3
"""
Training entry point – scaffold for PPO + MLflow.

Usage
-----
    python scripts/train.py

Requires
--------
    pip install vsss-sim[dev] mlflow stable-baselines3
"""

from __future__ import annotations

import mlflow
import gymnasium as gym
import vsss_sim  # registers VSSS-v0

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
PARAMS = {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_envs": 8,
    "opponent": "stationary",
    "total_timesteps": 1_000_000,
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
with mlflow.start_run():
    mlflow.log_params(PARAMS)

    env = gym.make("VSSS-v0", opponent_policy=PARAMS["opponent"])

    # --- your training loop here ---
    # e.g. with Stable-Baselines3:
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, learning_rate=PARAMS["learning_rate"])
    model.learn(total_timesteps=PARAMS["total_timesteps"])
    mlflow.pytorch.log_model(model.policy, "policy")

    env.close()

if __name__ == "__main__":
    print("Training scaffold – uncomment and fill in the sections above.")
