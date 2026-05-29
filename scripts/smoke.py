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
import copy
import threading
from pathlib import Path

import jax
import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from vsss_sim.config import InitMode
from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3

PARAMS = {
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "opponent": "stationary",
    "learning_rate": 3e-4,
    "n_steps": 512,  # overridden at runtime if --n-steps or --max-episode-steps is set
}


class _EpisodeDumpCallback(BaseCallback):
    """Log episode metrics to MLflow once per PPO iteration (generation).

    Accumulates all episodes that finish during a rollout in _on_step, then
    flushes mean reward, last-episode reward, mean length, and max length to
    MLflow and SB3's logger in _on_rollout_end — one data point per iteration.

    When ``save_every_gen=True``, snapshots the policy via deepcopy after each
    generation and serializes it to ``save_path`` in a background thread so the
    main training loop is not blocked by disk I/O.
    """

    def __init__(self, save_path: Path | None = None, save_every_gen: bool = False):
        super().__init__()
        self._iteration = 0
        self._episode = 0
        self._rollout_rewards: list[float] = []
        self._rollout_lengths: list[int] = []
        self._save_path = save_path
        self._save_every_gen = save_every_gen and save_path is not None
        self._save_thread: threading.Thread | None = None

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            self._episode += 1
            ep = info["episode"]
            self._rollout_rewards.append(float(ep["r"]))
            self._rollout_lengths.append(int(ep["l"]))
        return True

    def _on_rollout_end(self) -> None:
        self._iteration += 1
        rewards = self._rollout_rewards
        lengths = self._rollout_lengths

        if rewards:
            mean_r = sum(rewards) / len(rewards)
            last_r = rewards[-1]
            mean_l = sum(lengths) / len(lengths)
            max_l = max(lengths)

            mlflow.log_metrics(
                {
                    "ep_reward/mean": mean_r,
                    "ep_reward/last": last_r,
                    "ep_length/mean": mean_l,
                    "ep_length/max": max_l,
                    "episodes": self._episode,
                    "gen_n_episodes": len(rewards),
                },
                step=self._iteration,
            )
            self.logger.record("rollout/iteration", self._iteration)
            self.logger.record("rollout/episode", self._episode)
            self.logger.record("rollout/ep_reward_mean", mean_r)
            self.logger.record("rollout/ep_reward_last", last_r)
            self.logger.record("rollout/ep_length_mean", mean_l)
            self.logger.record("rollout/ep_length_max", max_l)
            self.model.dump_logs(iteration=self._iteration)

        self._rollout_rewards = []
        self._rollout_lengths = []

        if self._save_every_gen:
            # Block only if the previous save is still writing (shouldn't happen
            # in practice — disk I/O finishes well before the next generation ends).
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_thread.join()
            # deepcopy snapshots all tensors independently before handing off,
            # so the gradient update that follows cannot race with serialization.
            snapshot = copy.deepcopy(self.model)
            path = str(self._save_path)
            self._save_thread = threading.Thread(
                target=snapshot.save, args=(path,), daemon=True
            )
            self._save_thread.start()

    def _on_training_end(self) -> None:
        if self._save_thread is not None:
            self._save_thread.join()


def _build_env(num_envs: int, seed: int, backend: str | None,
               render: bool, fps: float | None,
               max_episode_steps: int | None,
               init_mode: str | None = None):
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
        if init_mode is not None:
            kwargs["init_mode"] = init_mode
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
    if init_mode is not None:
        env_kwargs["init_mode"] = init_mode
    return make_vec_env(
        "VSSS-v0",
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def main(seed: int, render: bool, fps: float | None,
         timesteps: int | None, generations: int | None, forever: bool,
         backend: str | None, num_envs: int,
         save_path: Path | None,
         max_episode_steps: int | None,
         n_steps: int | None,
         init_mode: str | None) -> None:
    print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")

    # Auto-align n_steps to max_episode_steps when not explicitly set,
    # so each generation contains exactly num_envs episodes.
    if n_steps is None:
        n_steps = max_episode_steps if max_episode_steps is not None else PARAMS["n_steps"]

    if forever:
        total_timesteps = int(1e15)
        save_every_gen = True
    elif generations is not None:
        total_timesteps = generations * n_steps * num_envs
        save_every_gen = False
    else:
        total_timesteps = timesteps if timesteps is not None else 10_000
        save_every_gen = False

    mlflow.set_experiment("vsss-smoke")

    with mlflow.start_run(run_name=f"smoke-seed{seed}-n{num_envs}"):
        mlflow.log_params({
            **PARAMS, "n_steps": n_steps, "seed": seed,
            "total_timesteps": total_timesteps,
            "backend": backend or "default", "n_envs": num_envs,
            "max_episode_steps": max_episode_steps or "default",
            "init_mode": init_mode or "kickoff",
            "mode": "forever" if forever else ("generations" if generations is not None else "timesteps"),
        })

        env = _build_env(num_envs, seed, backend, render, fps, max_episode_steps, init_mode)

        print(f"n_steps={n_steps}  num_envs={num_envs}  → {n_steps * num_envs} env-steps/generation")

        model = PPO(
            PARAMS["policy"],
            env,
            learning_rate=PARAMS["learning_rate"],
            n_steps=n_steps,
            seed=seed,
            verbose=1,
        )

        try:
            model.learn(
                total_timesteps=total_timesteps,
                log_interval=None,
                callback=_EpisodeDumpCallback(
                    save_path=save_path,
                    save_every_gen=save_every_gen,
                ),
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        if save_path is not None and not save_every_gen:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            mlflow.log_artifact(str(save_path), artifact_path="model")
            print(f"saved policy: {save_path}")

        env.close()

    print("Done. Run `mlflow ui` to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSS smoke / training script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Open a live pygame window (single-env only)")
    parser.add_argument("--fps", type=float, default=None, help="Cap render FPS (e.g. 30). Uncapped by default.")

    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--timesteps", type=int, default=None,
        help="Total env steps to run (default: 10 000 when no other duration flag is set).",
    )
    duration.add_argument(
        "--generations", type=int, default=None,
        help="Number of PPO generations (iterations) to run. "
             "total_timesteps = generations × n_steps × num_envs.",
    )
    duration.add_argument(
        "--forever", action="store_true",
        help="Run indefinitely (Ctrl-C to stop). Saves policy to --save-path after every "
             "generation in a background thread. Requires --save-path.",
    )

    parser.add_argument(
        "--backend", type=str, default=None, choices=["numpy", "jax"],
        help="Physics backend (single-env only; batched always uses jax).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel envs. >1 uses VSSVecEnv via the SB3 adapter (jax backend).",
    )
    parser.add_argument(
        "--save-path", type=Path, default=None,
        help="Where to save policy.zip. In normal/generations mode: saved once at the end "
             "(also as an MLflow artifact). In --forever mode: overwritten after every generation.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None,
        help="Episode length cap in steps (e.g. 300 = 5 s at 60 Hz). "
             "Defaults to config.MAX_EPISODE_STEPS (currently 1200 = 20 s).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="PPO rollout length per env per generation (default: auto-aligned to "
             "--max-episode-steps when set, otherwise 512). Setting this equal to "
             "--max-episode-steps gives exactly num-envs episodes per generation.",
    )
    parser.add_argument(
        "--init-mode", type=str, default=None, choices=[m.value for m in InitMode],
        help="Robot/ball placement strategy at episode reset. "
             "'kickoff' (default) uses the standard formation; "
             "'random' places robots uniformly in their respective halves.",
    )
    args = parser.parse_args()

    if args.forever and args.save_path is None:
        parser.error("--forever requires --save-path")

    main(
        args.seed, args.render, args.fps,
        args.timesteps, args.generations, args.forever,
        args.backend, args.num_envs,
        args.save_path, args.max_episode_steps, args.n_steps, args.init_mode,
    )
