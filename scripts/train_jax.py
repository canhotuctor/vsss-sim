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
    num_updates = (
        args.generations
        if args.generations is not None
        else args.total_timesteps // steps_per_update
    )
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
    print(f"Generations : {num_updates} ({num_updates * steps_per_update:,} env steps)")

    init_start = time.perf_counter()
    runner = trainer.initialize(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(runner)
    init_seconds = time.perf_counter() - init_start

    compile_start = time.perf_counter()
    train_executable = trainer.compile_train(runner, num_updates)
    compile_seconds = time.perf_counter() - compile_start

    execution_start = time.perf_counter()
    runner, metrics_history = train_executable(runner)
    jax.block_until_ready((runner, metrics_history))
    execution_seconds = time.perf_counter() - execution_start

    host_metrics = jax.device_get(
        jax.tree_util.tree_map(lambda values: values[-1], metrics_history)
    )
    total_steps = num_updates * steps_per_update
    fps = total_steps / execution_seconds if execution_seconds else 0.0
    print(f"Initialize   : {init_seconds:.3f}s")
    print(f"XLA compile  : {compile_seconds:.3f}s")
    print(f"Execute      : {execution_seconds:.3f}s")
    print(f"Throughput   : {fps:,.0f} env-steps/s")
    print(
        f"Final        : steps={int(host_metrics['env_steps']):,} "
        f"loss={float(host_metrics['loss']):.4f} "
        f"return={float(host_metrics['mean_episode_return']):.3f} "
        f"episodes={int(host_metrics['episodes'])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Fixed PPO update count; overrides --total-timesteps.",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--opponent", choices=("stationary", "random"), default="stationary")
    parser.add_argument("--init-mode", choices=("kickoff", "random"), default="kickoff")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.generations is not None and args.generations < 1:
        parser.error("--generations must be at least 1")
    return args


if __name__ == "__main__":
    train(parse_args())
