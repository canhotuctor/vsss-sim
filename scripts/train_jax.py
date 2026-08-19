#!/usr/bin/env python3
"""Train PPO end-to-end in JAX, without Gymnasium, NumPy, SB3, or PyTorch."""
from __future__ import annotations

import argparse
import time

import jax

from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig


def train(args: argparse.Namespace) -> None:
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )
    steps_per_update = config.batch_size
    num_updates = args.total_timesteps // steps_per_update
    if num_updates < 1:
        raise ValueError(
            f"total_timesteps must be at least one batch ({steps_per_update:,})"
        )

    env = VSSJumanjiEnv(
        opponent_policy=args.opponent,
        init_mode=args.init_mode,
    )
    trainer = PPO(env, config)

    print(f"JAX devices : {jax.devices()}")
    print(f"Batch       : {config.num_envs} envs × {config.rollout_length} steps")
    print(f"Updates     : {num_updates} ({num_updates * steps_per_update:,} env steps)")

    compile_start = time.perf_counter()
    runner = trainer.initialize(jax.random.PRNGKey(args.seed))
    runner, metrics = trainer.update(runner)
    jax.block_until_ready(metrics)
    compile_seconds = time.perf_counter() - compile_start
    print(f"Compile + first update: {compile_seconds:.2f}s")

    start = time.perf_counter()
    completed_updates = 1
    for update in range(1, num_updates):
        runner, metrics = trainer.update(runner)
        if (update + 1) % args.log_interval == 0 or update + 1 == num_updates:
            host_metrics = jax.device_get(metrics)
            elapsed = time.perf_counter() - start
            measured_steps = update * steps_per_update
            fps = measured_steps / elapsed if elapsed else 0.0
            print(
                f"update={update + 1:>5}/{num_updates} "
                f"steps={int(host_metrics['env_steps']):>10,} "
                f"fps={fps:>10,.0f} "
                f"loss={float(host_metrics['loss']):>9.4f} "
                f"return={float(host_metrics['mean_episode_return']):>8.3f} "
                f"episodes={int(host_metrics['episodes']):>5}"
            )
        completed_updates = update + 1

    if completed_updates == 1:
        host_metrics = jax.device_get(metrics)
        print(
            f"steps={int(host_metrics['env_steps']):,} "
            f"loss={float(host_metrics['loss']):.4f} "
            f"mean_reward={float(host_metrics['mean_reward']):.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--opponent", choices=("stationary", "random"), default="stationary")
    parser.add_argument("--init-mode", choices=("kickoff", "random"), default="kickoff")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error("--log-interval must be at least 1")
    return args


if __name__ == "__main__":
    train(parse_args())
