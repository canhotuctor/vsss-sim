#!/usr/bin/env python3
"""
Training run — PPO against stationary opponents, tracked with MLflow.

Runs 300 000 timesteps across 4 parallel envs. Saves the trained policy
as an MLflow artifact at the end of the run.

Usage
-----
    python scripts/train.py [--seed SEED] [--run-name NAME]

Requires
--------
    pip install -e ".[dev]" mlflow stable-baselines3
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

PARAMS = {
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "opponent": "stationary",
    "n_envs": 4,
    "total_timesteps": 300_000,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.01,
}


class _MLflowCallback(BaseCallback):
    """Log per-episode reward/length to MLflow as each episode finishes."""

    def __init__(self):
        super().__init__()
        self._episode = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self._episode += 1
                mlflow.log_metrics(
                    {
                        "ep_reward": info["episode"]["r"],
                        "ep_length": info["episode"]["l"],
                    },
                    step=self._episode,
                )
        return True


def main(seed: int, run_name: str) -> None:
    mlflow.set_experiment("vsss-train")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**PARAMS, "seed": seed})

        env = make_vec_env(
            "VSSS-v0",
            n_envs=PARAMS["n_envs"],
            seed=seed,
            env_kwargs={"opponent_policy": PARAMS["opponent"]},
        )

        eval_env = make_vec_env(
            "VSSS-v0",
            n_envs=1,
            seed=seed + 1000,
            env_kwargs={"opponent_policy": PARAMS["opponent"]},
        )

        eval_callback = EvalCallback(
            eval_env,
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        )

        model = PPO(
            PARAMS["policy"],
            env,
            learning_rate=PARAMS["learning_rate"],
            n_steps=PARAMS["n_steps"],
            batch_size=PARAMS["batch_size"],
            n_epochs=PARAMS["n_epochs"],
            gamma=PARAMS["gamma"],
            ent_coef=PARAMS["ent_coef"],
            seed=seed,
            verbose=1,
        )

        model.learn(
            total_timesteps=PARAMS["total_timesteps"],
            callback=[_MLflowCallback(), eval_callback],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "policy.zip"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")

        env.close()
        eval_env.close()

    print("Training complete. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSS training run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="ppo-stationary")
    args = parser.parse_args()
    main(args.seed, args.run_name)
