"""Command-line entry points for generic and legacy JAX PPO training."""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import jax

from ..config import RewardProfile
from ..envs.jumanji import VSSJumanjiEnv
from .ppo import PPO, PPOConfig
from .self_play import ExperimentConfig, run_experiment


def _resolve_experiment_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    repository_root = Path(__file__).resolve().parents[3]
    candidate = repository_root / "configs" / "experiments" / candidate.name
    if not candidate.is_file():
        raise FileNotFoundError(f"experiment config not found: {value}")
    return candidate


def _legacy_train(args: argparse.Namespace) -> None:
    policy_architecture = args.policy_architecture or "joint"
    mirror_augmentation = (
        args.mirror_augmentation
        if args.mirror_augmentation is not None
        else policy_architecture == "brandao_sets"
    )
    reward_profile = args.reward_profile or (
        RewardProfile.BRANDAO_REWARD.value
        if policy_architecture in ("brandao_shared", "brandao_sets")
        else RewardProfile.DEFAULT.value
    )
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        mirror_augmentation=mirror_augmentation,
    )
    steps_per_update = config.batch_size
    num_updates = (
        args.generations
        if args.generations is not None
        else args.total_timesteps // steps_per_update
    )
    if num_updates < 1:
        raise ValueError(f"total_timesteps must be at least one batch ({steps_per_update:,})")

    env = VSSJumanjiEnv(
        opponent_policy=args.opponent,
        max_episode_steps=(
            args.max_episode_steps
            if args.max_episode_steps is not None
            else (400 if policy_architecture in ("brandao_shared", "brandao_sets") else 300)
        ),
        init_mode=args.init_mode,
        terminate_on_goal=policy_architecture in ("brandao_shared", "brandao_sets"),
        reward_profile=reward_profile,
        policy_architecture=policy_architecture,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xp",
        help="Experiment JSON path or name under configs/experiments (historical self-play).",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--num-lineages", type=int)
    parser.add_argument("--num-envs-per-lineage", type=int)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument(
        "--policy-architecture",
        choices=("joint", "brandao_shared", "brandao_sets"),
        help="Centralized joint, shared Brandão MLP, or permutation-invariant set policy.",
    )
    parser.add_argument(
        "--mirror-augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use x-axis reflection augmentation (enabled by default for brandao_sets).",
    )
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--opponent", choices=("stationary", "random"), default="stationary")
    parser.add_argument("--init-mode", choices=("kickoff", "random"), default="kickoff")
    parser.add_argument(
        "--reward-profile",
        choices=tuple(profile.value for profile in RewardProfile),
        help="Team reward rules; an override for experiment JSON when using --xp.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.xp is None:
        _legacy_train(args)
        return

    config_path = _resolve_experiment_config(args.xp)
    config = ExperimentConfig.from_json(config_path)
    overrides = {}
    if args.generations is not None:
        overrides["num_generations"] = args.generations
    if args.num_lineages is not None:
        overrides["num_lineages"] = args.num_lineages
    if args.num_envs_per_lineage is not None:
        overrides["num_envs_per_lineage"] = args.num_envs_per_lineage
    if args.reward_profile is not None:
        overrides["reward_profile"] = args.reward_profile
    if args.policy_architecture is not None:
        overrides["policy_architecture"] = args.policy_architecture
        if args.mirror_augmentation is None:
            overrides["mirror_augmentation"] = (
                args.policy_architecture == "brandao_sets"
            )
    if args.mirror_augmentation is not None:
        overrides["mirror_augmentation"] = args.mirror_augmentation
    if overrides:
        config = replace(config, **overrides)
    run_experiment(config, config_source=config_path)


if __name__ == "__main__":
    main()
