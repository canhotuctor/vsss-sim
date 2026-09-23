#!/usr/bin/env python3
"""Benchmark JAX self-play lineage, environment, and active-pool dimensions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig
from vsss_sim.rl.self_play import OpponentSamplingWeights


def _parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    return sizes


def _memory_stats() -> list[dict]:
    stats = []
    for device in jax.devices():
        memory = device.memory_stats() if hasattr(device, "memory_stats") else None
        stats.append(
            {
                "device": str(device),
                "memory_stats": memory,
            }
        )
    return stats


def _benchmark_case(
    lineages: int,
    envs: int,
    pool_size: int,
    rollout_length: int,
    update_epochs: int,
    num_minibatches: int,
) -> dict:
    config = PPOConfig(
        num_envs=envs,
        rollout_length=rollout_length,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
    )
    trainer = PPO(VSSJumanjiEnv(), config)
    keys = jax.vmap(lambda index: jax.random.fold_in(jax.random.PRNGKey(0), index))(
        jnp.arange(lineages, dtype=jnp.uint32)
    )
    initial = jax.jit(jax.vmap(trainer._initialize))(keys)
    jax.block_until_ready(initial)
    lineage_params = initial.train_state.params
    pools = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(
            value[:, None, ...], (lineages, pool_size) + value.shape[1:]
        ),
        lineage_params,
    )
    weights = OpponentSamplingWeights(
        all_history=100, recent_half=0, latest=0, random=0, stationary=0
    )
    probabilities = jnp.broadcast_to(
        jnp.asarray(weights.category_probabilities()), (lineages, 5)
    )
    generations = jnp.broadcast_to(
        jnp.arange(pool_size, dtype=jnp.int32), (lineages, pool_size)
    )
    runners = trainer.initialize_lineages(keys, pools, probabilities, generations)
    jax.block_until_ready(runners)

    compile_start = time.perf_counter()
    executable = trainer.compile_train_lineages(
        runners, pools, probabilities, generations, num_updates=1
    )
    compile_seconds = time.perf_counter() - compile_start

    execute_start = time.perf_counter()
    runners, _ = executable(runners, pools, probabilities, generations)
    jax.block_until_ready(runners)
    first_execution_seconds = time.perf_counter() - execute_start

    steady_start = time.perf_counter()
    runners, _ = executable(runners, pools, probabilities, generations)
    jax.block_until_ready(runners)
    steady_execution_seconds = time.perf_counter() - steady_start

    total_steps = lineages * envs * config.rollout_length
    return {
        "lineages": lineages,
        "envs_per_lineage": envs,
        "active_pool_size": pool_size,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "steady_execution_seconds": steady_execution_seconds,
        "steady_env_steps_per_second": total_steps / max(steady_execution_seconds, 1e-9),
        "device_memory": _memory_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineages", type=_parse_sizes, default=[1, 4, 16])
    parser.add_argument("--envs-per-lineage", type=_parse_sizes, default=[16, 32])
    parser.add_argument("--pool-sizes", type=_parse_sizes, default=[1, 5, 10])
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; must not exist.")
    args = parser.parse_args()

    report = {"devices": [str(device) for device in jax.devices()], "cases": []}
    for lineages in args.lineages:
        for envs in args.envs_per_lineage:
            for pool_size in args.pool_sizes:
                print(
                    f"Benchmarking lineages={lineages}, envs/lineage={envs}, "
                    f"pool={pool_size} ...",
                    flush=True,
                )
                case = _benchmark_case(
                    lineages,
                    envs,
                    pool_size,
                    args.rollout_length,
                    args.update_epochs,
                    args.num_minibatches,
                )
                report["cases"].append(case)
                print(
                    f"compile={case['compile_seconds']:.2f}s, "
                    f"steady={case['steady_execution_seconds']:.2f}s, "
                    f"{case['steady_env_steps_per_second']:,.0f} env-steps/s",
                    flush=True,
                )

    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"benchmark output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
