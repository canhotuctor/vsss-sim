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


def _build_env(num_envs: int, seed: int, backend: str | None,
               render: bool, fps: float | None):
    """Pick the right env based on `num_envs`."""
    if num_envs > 1:
        if render:
            print("warning: --render ignored in batched mode (num_envs > 1)")
        if backend is not None and backend != "jax":
            raise ValueError(
                f"batched mode requires backend='jax', got '{backend}'"
            )
        return VSSVecEnvToSB3(
            VSSVecEnv(num_envs=num_envs, opponent_policy=PARAMS["opponent"])
        )

    # Single-env path
    env_kwargs = {"opponent_policy": PARAMS["opponent"]}
    if backend is not None:
        env_kwargs["backend"] = backend
    if render:
        env_kwargs["render_mode"] = "human"
        env_kwargs["render_fps"] = fps
    return make_vec_env(
        "VSSS-v0",
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def main(seed: int, render: bool, fps: float | None, timesteps: int,
         backend: str | None, num_envs: int) -> None:
    mlflow.set_experiment("vsss-smoke")

    with mlflow.start_run(run_name=f"smoke-seed{seed}-n{num_envs}"):
        mlflow.log_params({
            **PARAMS, "seed": seed, "total_timesteps": timesteps,
            "backend": backend or "default", "n_envs": num_envs,
        })

        env = _build_env(num_envs, seed, backend, render, fps)

        model = PPO(
            PARAMS["policy"],
            env,
            learning_rate=PARAMS["learning_rate"],
            n_steps=PARAMS["n_steps"],
            seed=seed,
            verbose=1,
        )

        model.learn(
            total_timesteps=timesteps,
            callback=_MLflowCallback(),
        )

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
    args = parser.parse_args()
    main(args.seed, args.render, args.fps, args.timesteps, args.backend, args.num_envs)
