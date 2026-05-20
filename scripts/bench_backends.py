#!/usr/bin/env python3
"""
Benchmark physics throughput: numpy vs JAX single-env vs JAX vmap'd batches.

This is the empirical answer to "will VSSVecEnv be faster?" — the last block
runs `jax.vmap(step)` at increasing batch sizes, which is exactly what
VSSVecEnv will use internally. The per-env step-rate should stay roughly flat
(or improve) as batch size grows; total throughput scales near-linearly.

Usage:
    PYTHONPATH=src python scripts/bench_backends.py [--steps 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim import config
from vsss_sim.envs import VSSVecEnv
from vsss_sim.physics import jax_backend as jb
from vsss_sim.physics import numpy_backend as nb


def _bench_numpy(steps: int, seed: int) -> float:
    s = nb.SimState()
    nb.reset_kickoff(s, rng=np.random.default_rng(seed))
    a = np.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=np.float64)
    # Warm-up
    for _ in range(50):
        nb.step(s, a)
    t0 = time.perf_counter()
    for _ in range(steps):
        nb.step(s, a)
    return steps / (time.perf_counter() - t0)


def _bench_jax_single(steps: int, seed: int) -> float:
    s = jb.reset_kickoff(jax.random.PRNGKey(seed))
    a = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
    step = jax.jit(jb.step)
    # Warm-up (forces JIT compile)
    s, _ = step(s, a)
    jax.block_until_ready(s.robots)
    for _ in range(50):
        s, _ = step(s, a)
    jax.block_until_ready(s.robots)

    t0 = time.perf_counter()
    for _ in range(steps):
        s, _ = step(s, a)
    jax.block_until_ready(s.robots)
    return steps / (time.perf_counter() - t0)


def _bench_jax_vmap(steps: int, batch: int, seed: int) -> tuple[float, float]:
    """Return (per-env fps, total fps) for `jax.vmap(step)` at given batch size."""
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    states = jax.vmap(jb.reset_kickoff)(keys)
    actions = jnp.zeros((batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
    vstep = jax.jit(jax.vmap(jb.step))

    # Warm-up
    states, _ = vstep(states, actions)
    jax.block_until_ready(states.robots)
    for _ in range(10):
        states, _ = vstep(states, actions)
    jax.block_until_ready(states.robots)

    t0 = time.perf_counter()
    for _ in range(steps):
        states, _ = vstep(states, actions)
    jax.block_until_ready(states.robots)
    elapsed = time.perf_counter() - t0
    return steps / elapsed, (steps * batch) / elapsed


def _bench_vssvecenv(steps: int, batch: int, seed: int) -> tuple[float, float]:
    """Return (per-env fps, total fps) for VSSVecEnv at given batch size.

    This is what RL libraries actually consume — it includes the Gymnasium
    wrapper overhead (numpy ↔ jax conversion per step, opponent policy,
    obs/reward shaping)."""
    env = VSSVecEnv(num_envs=batch, opponent_policy="stationary")
    env.reset(seed=seed)
    a = np.zeros((batch, config.N_ROBOTS * 2), dtype=np.float32)

    # Warm-up
    for _ in range(10):
        env.step(a)

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(a)
    elapsed = time.perf_counter() - t0
    env.close()
    return steps / elapsed, (steps * batch) / elapsed


def main(steps: int, seed: int) -> None:
    print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")
    print(f"Steps per measurement: {steps}\n")

    print("== Single env (raw physics calls) ==")
    np_fps = _bench_numpy(steps, seed)
    print(f"  numpy            : {np_fps:>10,.0f} fps")
    jx1 = _bench_jax_single(steps, seed)
    print(f"  jax  (no vmap)   : {jx1:>10,.0f} fps")
    print(f"  ratio jax/numpy  : {jx1 / np_fps:>10.2f}x\n")

    print("== Raw jax.vmap(step) — physics ceiling ==")
    print(f"  {'batch':>5}  {'per-env fps':>14}  {'total fps':>14}  {'vs numpy':>10}")
    for batch in (1, 8, 64, 256, 1024):
        per_env, total = _bench_jax_vmap(steps, batch, seed)
        print(f"  {batch:>5}  {per_env:>14,.0f}  {total:>14,.0f}  {total / np_fps:>9.1f}x")

    print("\n== VSSVecEnv — what your RL trainer actually sees ==")
    print(f"  {'batch':>5}  {'per-env fps':>14}  {'total fps':>14}  {'vs numpy':>10}")
    for batch in (1, 8, 64, 256, 1024):
        per_env, total = _bench_vssvecenv(steps, batch, seed)
        print(f"  {batch:>5}  {per_env:>14,.0f}  {total:>14,.0f}  {total / np_fps:>9.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark physics backends.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Steps per measurement (default: 2000).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.steps, args.seed)
