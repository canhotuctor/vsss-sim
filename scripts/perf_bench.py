#!/usr/bin/env python3
"""Benchmark JAX physics, Gymnasium vector stepping, and native PPO inference."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# JAX_PLATFORMS must be set before importing JAX.
for index, argument in enumerate(sys.argv):
    value = None
    if argument == "--device" and index + 1 < len(sys.argv):
        value = sys.argv[index + 1]
    elif argument.startswith("--device="):
        value = argument.split("=", 1)[1]
    if value in ("cpu", "gpu"):
        os.environ["JAX_PLATFORMS"] = "cpu" if value == "cpu" else "cuda"
        break

def _stats(times_ns: list[int], trim_top_pct: float = 0.05) -> dict[str, float]:
    ordered = sorted(times_ns)
    cut = int(trim_top_pct * len(ordered))
    trimmed = ordered[: len(ordered) - cut] if cut else ordered
    mean = statistics.mean(trimmed)
    return {
        "min_us": ordered[0] / 1e3,
        "mean_us": mean / 1e3,
        "std_us": statistics.pstdev(trimmed, mu=mean) / 1e3,
        "median_us": statistics.median(ordered) / 1e3,
        "p95_us": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)] / 1e3,
    }


def _measure_rows(steps: int, batch_sizes: list[int], operation) -> list[dict]:
    import jax

    rows = []
    for batch in batch_sizes:
        run = operation(batch)
        result = run()  # warm-up / compilation
        for _ in range(10):
            result = run()
        jax.block_until_ready(result)
        times_ns = []
        for _ in range(steps):
            started = time.perf_counter_ns()
            result = run()
            jax.block_until_ready(result)
            times_ns.append(time.perf_counter_ns() - started)
        stats = _stats(times_ns)
        rows.append(
            {
                "batch": batch,
                "total_fps": batch / (stats["median_us"] / 1e6),
                **stats,
            }
        )
    return rows


def bench_jax_vmap(steps: int, batch_sizes: list[int]) -> list[dict]:
    import jax
    import jax.numpy as jnp

    from vsss_sim import config
    from vsss_sim.physics import jax_backend as backend

    step = jax.jit(jax.vmap(backend.step))

    def operation(batch: int):
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        initial = jax.vmap(backend.reset_kickoff)(keys)
        actions = jnp.zeros((batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
        state = initial

        def run():
            nonlocal state
            state, _ = step(state, actions)
            return state

        return run

    return _measure_rows(steps, batch_sizes, operation)


def bench_vssvecenv(steps: int, batch_sizes: list[int]) -> list[dict]:
    import numpy as np

    from vsss_sim import config
    from vsss_sim.envs import VSSVecEnv

    rows = []
    for batch in batch_sizes:
        env = VSSVecEnv(num_envs=batch, opponent_policy="stationary")
        env.reset(seed=0)
        actions = np.zeros((batch, config.N_ROBOTS * 2), dtype=np.float32)
        for _ in range(20):
            env.step(actions)
        times_ns = []
        try:
            for _ in range(steps):
                started = time.perf_counter_ns()
                env.step(actions)
                times_ns.append(time.perf_counter_ns() - started)
            stats = _stats(times_ns)
            rows.append(
                {
                    "batch": batch,
                    "total_fps": batch / (stats["median_us"] / 1e6),
                    **stats,
                }
            )
        finally:
            env.close()
    return rows


def bench_ppo_inference(steps: int, batch_sizes: list[int]) -> list[dict]:
    import jax
    import jax.numpy as jnp

    from vsss_sim.rl.ppo import ActorCritic

    observation_size = 46
    model = ActorCritic(action_size=6)
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((observation_size,), dtype=jnp.float32)
    )
    apply = jax.jit(model.apply)

    def operation(batch: int):
        observations = jax.random.uniform(
            jax.random.PRNGKey(batch), (batch, observation_size), minval=-1.0, maxval=1.0
        )
        return lambda: apply(params, observations)

    return _measure_rows(steps, batch_sizes, operation)


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'batch':>7} {'median':>12} {'p95':>12} {'total_fps':>14}")
    for row in rows:
        print(
            f"{row['batch']:>7} {row['median_us']:>10.1f}us "
            f"{row['p95_us']:>10.1f}us {row['total_fps']:>14,.0f}"
        )


def main() -> None:
    import jax

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--device", choices=("cpu", "gpu", "default"), default="default")
    args = parser.parse_args()
    if args.steps < 1 or not args.batches or any(batch < 1 for batch in args.batches):
        parser.error("steps and all batch sizes must be positive")

    print(f"JAX devices: {jax.devices()} | backend: {jax.default_backend()}")
    print(f"Measurements per batch: {args.steps}")
    vmap_rows = bench_jax_vmap(args.steps, args.batches)
    vec_rows = bench_vssvecenv(args.steps, args.batches)
    inference_rows = bench_ppo_inference(args.steps, args.batches)
    _print_table("Raw jax.vmap physics", vmap_rows)
    _print_table("VSSVecEnv wrapper", vec_rows)
    _print_table("Native Flax PPO policy inference", inference_rows)


if __name__ == "__main__":
    main()
