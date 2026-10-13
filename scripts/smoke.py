#!/usr/bin/env python3
"""Run a small native-JAX PPO training pass to smoke-check the training stack."""
from __future__ import annotations

import argparse
import time

import jax

from vsss_sim.config import InitMode, RewardProfile
from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=("stationary", "random"), default="stationary")
    parser.add_argument(
        "--init-mode", choices=tuple(mode.value for mode in InitMode), default=InitMode.KICKOFF.value
    )
    parser.add_argument(
        "--reward-profile",
        choices=tuple(profile.value for profile in RewardProfile),
        default=RewardProfile.DEFAULT.value,
    )
    args = parser.parse_args()

    if args.generations < 1 or args.num_envs < 1 or args.rollout_length < 1:
        parser.error("generations, num-envs, and rollout-length must be positive")

    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        update_epochs=1,
        num_minibatches=1,
    )
    env = VSSJumanjiEnv(
        opponent_policy=args.opponent,
        init_mode=args.init_mode,
        reward_profile=args.reward_profile,
    )
    trainer = PPO(env, ppo_config)

    initialize_started = time.perf_counter()
    runner = trainer.initialize(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(runner)
    initialize_seconds = time.perf_counter() - initialize_started

    compile_started = time.perf_counter()
    executable = trainer.compile_train(runner, args.generations)
    compile_seconds = time.perf_counter() - compile_started

    execute_started = time.perf_counter()
    runner, metrics = executable(runner)
    jax.block_until_ready((runner, metrics))
    execute_seconds = time.perf_counter() - execute_started
    final = jax.device_get(jax.tree_util.tree_map(lambda values: values[-1], metrics))

    print(f"JAX devices  : {jax.devices()}")
    print(f"Batch        : {args.num_envs} envs × {args.rollout_length} steps")
    print(f"Initialize   : {initialize_seconds:.3f}s")
    print(f"XLA compile  : {compile_seconds:.3f}s")
    print(f"Execute      : {execute_seconds:.3f}s")
    print(
        f"Final        : steps={int(final['env_steps']):,} "
        f"loss={float(final['loss']):.4f} "
        f"return={float(final['mean_episode_return']):.3f} "
        f"episodes={int(final['episodes'])}"
    )


if __name__ == "__main__":
    main()
