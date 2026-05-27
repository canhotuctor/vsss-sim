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


class _EpisodeDumpCallback(BaseCallback):
    """Log per-episode reward/length to MLflow and dump SB3's training table
    once every ``dump_every_episodes`` finished episodes.

    With ``num_envs > 1`` the natural default is ``dump_every_episodes = n_envs``
    so each dump summarises one "batch" of parallel episodes. SB3's own
    per-iteration auto-dump should be disabled (``PPO(..., verbose=0)``) to
    avoid duplicate tables.
    """

    def __init__(self, dump_every_episodes: int = 1):
        super().__init__()
        self._dump_every = max(1, int(dump_every_episodes))
        self._episode = 0
        self._since_last_dump = 0
        self._dumps = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            self._episode += 1
            self._since_last_dump += 1
            ep = info["episode"]
            mlflow.log_metrics(
                {"ep_reward": ep["r"], "ep_length": ep["l"]},
                step=self._episode,
            )
            self.logger.record("rollout/episode", self._episode)
            self.logger.record("rollout/ep_reward_last", float(ep["r"]))
            self.logger.record("rollout/ep_length_last", int(ep["l"]))

            if self._since_last_dump >= self._dump_every:
                self._dumps += 1
                self.model.dump_logs(iteration=self._dumps)
                self._since_last_dump = 0
        return True


def main(seed: int, run_name: str, save_path: Path | None) -> None:
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
            verbose=1,  # keeps stdout logger configured; log_interval=None disables auto-dump
        )

        model.learn(
            total_timesteps=PARAMS["total_timesteps"],
            log_interval=None,  # callback drives dumps per-episode-batch
            callback=[
                _EpisodeDumpCallback(dump_every_episodes=PARAMS["n_envs"]),
                eval_callback,
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "policy.zip"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            print(f"saved policy: {save_path}")

        env.close()
        eval_env.close()

    print("Training complete. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSS training run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="ppo-stationary")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Also save policy.zip to this path on disk (parent dirs auto-created). "
             "Independent of the MLflow artifact, which is always written.",
    )
    args = parser.parse_args()
    main(args.seed, args.run_name, args.save_path)
