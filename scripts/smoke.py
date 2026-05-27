#!/usr/bin/env python3
"""
Smoke test — quick sanity check that the simulator, MLflow, and SB3 are wired up.

Single-env mode (default, ``--num-envs 1``)
    Uses SB3's ``DummyVecEnv`` around one VSSEnv. ``--backend numpy|jax``
    selects the physics backend. ``--render`` opens a pygame window.

Batched mode (``--num-envs N`` for N > 1)
    Uses ``VSSVecEnvToSB3(VSSVecEnv(num_envs=N))`` — N matches in parallel via
    ``jit(vmap(step))``. Always uses the JAX physics backend. No render
    window (would only show env 0 anyway).

Usage
-----
    python scripts/smoke.py --backend jax --timesteps 5000
    python scripts/smoke.py --num-envs 64 --timesteps 50000

Requires
--------
    pip install -e ".[dev,jax]" mlflow stable-baselines3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3

PARAMS = {
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "opponent": "stationary",
    "learning_rate": 3e-4,
    "n_steps": 512,
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


def _build_env(num_envs: int, seed: int, backend: str | None,
               render: bool, fps: float | None,
               max_episode_steps: int | None):
    """Pick the right env based on `num_envs`."""
    if num_envs > 1:
        if render:
            print("warning: --render ignored in batched mode (num_envs > 1)")
        if backend is not None and backend != "jax":
            raise ValueError(
                f"batched mode requires backend='jax', got '{backend}'"
            )
        kwargs = {"num_envs": num_envs, "opponent_policy": PARAMS["opponent"]}
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        return VSSVecEnvToSB3(VSSVecEnv(**kwargs))

    # Single-env path
    env_kwargs = {"opponent_policy": PARAMS["opponent"]}
    if backend is not None:
        env_kwargs["backend"] = backend
    if render:
        env_kwargs["render_mode"] = "human"
        env_kwargs["render_fps"] = fps
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    return make_vec_env(
        "VSSS-v0",
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def main(seed: int, render: bool, fps: float | None, timesteps: int,
         backend: str | None, num_envs: int,
         save_path: Path | None,
         max_episode_steps: int | None) -> None:
    print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")
    mlflow.set_experiment("vsss-smoke")

    with mlflow.start_run(run_name=f"smoke-seed{seed}-n{num_envs}"):
        mlflow.log_params({
            **PARAMS, "seed": seed, "total_timesteps": timesteps,
            "backend": backend or "default", "n_envs": num_envs,
            "max_episode_steps": max_episode_steps or "default",
        })

        env = _build_env(num_envs, seed, backend, render, fps, max_episode_steps)

        model = PPO(
            PARAMS["policy"],
            env,
            learning_rate=PARAMS["learning_rate"],
            n_steps=PARAMS["n_steps"],
            seed=seed,
            verbose=1,  # keeps stdout logger configured; log_interval=None disables auto-dump
        )

        model.learn(
            total_timesteps=timesteps,
            log_interval=None,  # callback drives dumps per-episode-batch
            callback=_EpisodeDumpCallback(dump_every_episodes=num_envs),
        )

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            mlflow.log_artifact(str(save_path), artifact_path="model")
            print(f"saved policy: {save_path}")

        env.close()

    print("Smoke test complete. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSS smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Open a live pygame window (single-env only)")
    parser.add_argument("--fps", type=float, default=None, help="Cap render FPS (e.g. 30). Uncapped by default.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total env steps to train PPO for (default: 10 000).")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["numpy", "jax"],
        help="Physics backend (single-env only; batched always uses jax).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel envs. >1 uses VSSVecEnv via the SB3 adapter (jax backend).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Save policy.zip to this path on disk after training (parent dirs auto-created). "
             "Also logged as an MLflow artifact.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Episode length cap in steps (e.g. 300 = 5 s at 60 Hz). "
             f"Defaults to config.MAX_EPISODE_STEPS (currently 1200 = 20 s).",
    )
    args = parser.parse_args()
    main(args.seed, args.render, args.fps, args.timesteps, args.backend, args.num_envs,
         args.save_path, args.max_episode_steps)
