#!/usr/bin/env python3
"""
Smoke test — quick sanity check that the simulator and MLflow are wired up.

Runs 10 000 timesteps with a single env and stationary opponents.
Finishes in under a minute; inspect the MLflow UI to confirm metrics flow.

Usage
-----
    python scripts/smoke.py [--seed SEED]

Requires
--------
    pip install -e ".[dev]" mlflow stable-baselines3
"""

from __future__ import annotations

import argparse

import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

PARAMS = {
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "opponent": "stationary",
    "n_envs": 1,
    "total_timesteps": 10_000,
    "learning_rate": 3e-4,
    "n_steps": 512,
}


class _MLflowCallback(BaseCallback):
    """Log SB3 rollout metrics to MLflow after each rollout buffer flush."""

    def _on_rollout_end(self) -> None:
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            mlflow.log_metrics(
                {
                    "ep_rew_mean": self.logger.name_to_value["rollout/ep_rew_mean"],
                    "ep_len_mean": self.logger.name_to_value["rollout/ep_len_mean"],
                },
                step=self.num_timesteps,
            )

    def _on_step(self) -> bool:
        return True


def main(seed: int) -> None:
    mlflow.set_experiment("vsss-smoke")

    with mlflow.start_run(run_name=f"smoke-seed{seed}"):
        mlflow.log_params({**PARAMS, "seed": seed})

        env = make_vec_env(
            "VSSS-v0",
            n_envs=PARAMS["n_envs"],
            seed=seed,
            env_kwargs={"opponent_policy": PARAMS["opponent"]},
        )

        model = PPO(
            PARAMS["policy"],
            env,
            learning_rate=PARAMS["learning_rate"],
            n_steps=PARAMS["n_steps"],
            seed=seed,
            verbose=1,
        )

        model.learn(
            total_timesteps=PARAMS["total_timesteps"],
            callback=_MLflowCallback(),
        )

        env.close()

    print("Smoke test complete. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSS smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
