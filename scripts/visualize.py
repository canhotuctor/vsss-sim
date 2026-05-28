#!/usr/bin/env python3
"""
Visualize a trained policy — load a saved SB3 model and roll out a few episodes
with the pygame renderer. No training happens here.

Usage
-----
    python scripts/visualize.py path/to/policy.zip
    python scripts/visualize.py path/to/policy.zip --episodes 5 --fps 30
    python scripts/visualize.py path/to/policy.zip --opponent random --seed 7

The model file is the ``policy.zip`` produced by ``scripts/train.py`` (logged
to MLflow under ``model/policy.zip``). Pass the local path to that artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import vsss_sim  # noqa: F401 — registers "VSSS-v0"
from stable_baselines3 import PPO
from vsss_sim import config
from vsss_sim.config import InitMode


def main(
    model_path: Path,
    episodes: int,
    fps: float | None,
    opponent: str,
    backend: str | None,
    seed: int,
    deterministic: bool,
    max_episode_steps: int | None,
    init_mode: str | None,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")

    print(f"loading model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    env_kwargs: dict = {
        "opponent_policy": opponent,
        "render_mode": "human",
        "render_fps": fps,
    }
    if backend is not None:
        env_kwargs["backend"] = backend
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    if init_mode is not None:
        env_kwargs["init_mode"] = init_mode

    env = gym.make("VSSS-v0", **env_kwargs)
    print(
        f"running {episodes} episode(s) — opponent={opponent}, "
        f"backend={backend or 'default'}, init={init_mode or 'kickoff'}, "
        f"deterministic={deterministic}"
    )

    try:
        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        ep_goals: list[tuple[int, int]] = []  # (blue_goals, yellow_goals)

        for ep in range(1, episodes + 1):
            obs, _info = env.reset(seed=seed + ep)
            total_reward = 0.0
            steps = 0
            blue_goals = 0
            yellow_goals = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                g = int(info.get("goal", 0))
                if g == 1:
                    blue_goals += 1
                elif g == -1:
                    yellow_goals += 1
                done = bool(terminated or truncated)

            ep_rewards.append(total_reward)
            ep_lengths.append(steps)
            ep_goals.append((blue_goals, yellow_goals))
            print(
                f"  episode {ep:>2}: reward={total_reward:+.3f}  "
                f"length={steps}  score(blue:yellow)={blue_goals}:{yellow_goals}"
            )

        print(
            f"\nsummary over {episodes} episode(s): "
            f"mean reward={np.mean(ep_rewards):+.3f}  "
            f"mean length={np.mean(ep_lengths):.0f}  "
            f"goals blue:yellow="
            f"{sum(b for b, _ in ep_goals)}:{sum(y for _, y in ep_goals)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained VSSS policy")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the saved SB3 model (.zip), e.g. mlruns/<exp>/<run>/artifacts/model/policy.zip",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to roll out (default: 10).")
    parser.add_argument(
        "--fps",
        type=float,
        default=config.FPS,
        help=f"Render FPS cap (default: {config.FPS:g} = sim rate, real-time playback; "
             "pass 0 for uncapped).",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="stationary",
        choices=["stationary", "random"],
        help="Opponent policy for the yellow team (default: stationary).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["numpy", "jax"],
        help="Physics backend (default: env default, usually numpy).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (offset per episode).")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy distribution instead of taking the deterministic action.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Episode length cap in steps (e.g. 300 = 5 s at 60 Hz). "
             f"Defaults to config.MAX_EPISODE_STEPS ({config.MAX_EPISODE_STEPS} = "
             f"{config.MAX_EPISODE_STEPS / config.FPS:.0f} s).",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default=None,
        choices=[m.value for m in InitMode],
        help="Robot/ball placement strategy at episode reset. "
             "'kickoff' (default) uses the standard formation; "
             "'random' places robots uniformly in their respective halves.",
    )
    args = parser.parse_args()

    fps = None if args.fps == 0 else args.fps
    main(
        model_path=args.model,
        episodes=args.episodes,
        fps=fps,
        opponent=args.opponent,
        backend=args.backend,
        seed=args.seed,
        deterministic=not args.stochastic,
        max_episode_steps=args.max_episode_steps,
        init_mode=args.init_mode,
    )
