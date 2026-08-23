#!/usr/bin/env python3
"""
Benchmark JAX physics throughput: single-state, vectorized, and Gymnasium paths.

This is the empirical answer to "will VSSVecEnv be faster?" — the last block
runs `jax.vmap(step)` at increasing batch sizes, which is exactly what
VSSVecEnv will use internally. The per-env step-rate should stay roughly flat
(or improve) as batch size grows; total throughput scales near-linearly.

Usage:
    PYTHONPATH=src python scripts/bench_backends.py [--steps 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# --device pre-parse: must run before `import jax` so JAX_PLATFORMS takes effect.
# Handles both `--device VALUE` and `--device=VALUE` forms; argparse below re-validates.
for _i, _a in enumerate(sys.argv):
    _v = None
    if _a == "--device" and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
    elif _a.startswith("--device="):
        _v = _a.split("=", 1)[1]
    if _v in ("cpu", "gpu"):
        os.environ["JAX_PLATFORMS"] = "cpu" if _v == "cpu" else "cuda"
        break

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vsss_sim import config  # noqa: E402
from vsss_sim.envs import VSSVecEnv  # noqa: E402
from vsss_sim.physics import jax_backend as jb  # noqa: E402


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


def _bench_sb3_ppo(num_envs: int, total_timesteps: int, seed: int) -> float:
    """End-to-end SB3 PPO training fps with the VSSVecEnv → SB3 adapter.

    Includes everything: PyTorch policy forward, JAX physics, numpy↔jax
    conversion, rollout buffer, gradient updates. This is what the user
    actually experiences during training.
    """
    from stable_baselines3 import PPO

    from vsss_sim.envs import VSSVecEnv
    from vsss_sim.sb3_adapter import VSSVecEnvToSB3

    env = VSSVecEnvToSB3(VSSVecEnv(num_envs=num_envs, opponent_policy="stationary"))
    # Use a uniform rollout size across all num_envs so we measure the same
    # amount of work each time. 32 × num_envs ≈ one rollout per env per measure.
    n_steps = 32
    model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=min(64 * num_envs, 4096),
                seed=seed, verbose=0)
    # Warm-up: one short learn to JIT/trace through the policy + env.
    model.learn(total_timesteps=num_envs * n_steps)

    start_steps = model.num_timesteps
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    elapsed = time.perf_counter() - t0
    actual_steps = model.num_timesteps - start_steps
    env.close()
    return actual_steps / elapsed


def main(steps: int, seed: int, sb3: bool, sb3_timesteps: int) -> None:
    print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")
    print(f"Steps per measurement: {steps}\n")

    print("== Single state (raw physics calls) ==")
    jx1 = _bench_jax_single(steps, seed)
    print(f"  jax (no vmap)    : {jx1:>10,.0f} fps\n")

    print("== Raw jax.vmap(step) — physics ceiling ==")
    print(f"  {'batch':>5}  {'per-env fps':>14}  {'total fps':>14}")
    for batch in (1, 8, 64, 256, 1024):
        per_env, total = _bench_jax_vmap(steps, batch, seed)
        print(f"  {batch:>5}  {per_env:>14,.0f}  {total:>14,.0f}")

    print("\n== VSSVecEnv — what your RL trainer actually sees ==")
    print(f"  {'batch':>5}  {'per-env fps':>14}  {'total fps':>14}")
    for batch in (1, 8, 64, 256, 1024):
        per_env, total = _bench_vssvecenv(steps, batch, seed)
        print(f"  {batch:>5}  {per_env:>14,.0f}  {total:>14,.0f}")

    if sb3:
        print("\n== End-to-end SB3 PPO training fps ==")
        print("  (includes PyTorch policy, JAX physics, conversions, rollout, gradient)")
        baseline_fps = None
        print(f"  {'num_envs':>9}  {'fps':>10}  {'vs num_envs=1':>14}")
        for num_envs in (1, 8, 64, 256):
            fps = _bench_sb3_ppo(num_envs, sb3_timesteps, seed)
            if baseline_fps is None:
                baseline_fps = fps
            speedup = fps / baseline_fps
            print(f"  {num_envs:>9}  {fps:>10,.0f}  {speedup:>13.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark JAX physics throughput.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Raw-physics steps per measurement (default: 2000).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sb3", action="store_true",
        help="Also run end-to-end SB3 PPO throughput bench (slower; adds ~30-60s).",
    )
    parser.add_argument(
        "--sb3-timesteps", type=int, default=4096,
        help="Timesteps per SB3 measurement (default: 4096).",
    )
    parser.add_argument(
        "--device", type=str, default="default",
        choices=["cpu", "gpu", "default"],
        help=(
            "Force JAX onto a specific device. cpu=JAX_PLATFORMS=cpu, "
            "gpu=JAX_PLATFORMS=cuda, default=let JAX pick (env var honored)."
        ),
    )
    args = parser.parse_args()
    main(args.steps, args.seed, args.sb3, args.sb3_timesteps)
