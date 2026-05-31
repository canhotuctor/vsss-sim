#!/usr/bin/env python3
"""
GPU saturation sweep for vsss-sim JAX physics.

Sweeps raw jax.vmap(step) and VSSVecEnv from batch=1024 up to 65536,
looking for the throughput plateau (saturation knee) on the active device.

Reports per-call latency, total fps, and efficiency vs batch=1024.
Also estimates VRAM usage at each batch size and flags the PCIe bottleneck
in the VSSVecEnv path.

Designed for the Ubuntu RTX 3060 (12 GB VRAM, 360 GB/s BW, 3840 CUDA cores)
but works on any JAX device — pass --device cpu to profile CPU instead.

Usage:
    # GPU (primary target):
    PYTHONPATH=src python scripts/gpu_saturation.py --device gpu
    # CPU (for comparison):
    PYTHONPATH=src python scripts/gpu_saturation.py --device cpu
    # Custom batch range:
    PYTHONPATH=src python scripts/gpu_saturation.py --device gpu \\
        --batches 512 1024 2048 4096 8192 16384 32768 65536
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# --device must be parsed before any JAX import
for _i, _a in enumerate(sys.argv):
    _v = None
    if _a == "--device" and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
    elif _a.startswith("--device="):
        _v = _a.split("=", 1)[1]
    if _v in ("cpu", "gpu"):
        os.environ["JAX_PLATFORMS"] = "cpu" if _v == "cpu" else "cuda"
        break

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    return time.perf_counter_ns()


def _measure_vmap(vstep, states, actions, steps: int) -> dict:
    """Warm + timed run; returns timing stats dict."""
    # warm-up: 10 steps to settle JIT
    for _ in range(10):
        states, _ = vstep(states, actions)
    jax.block_until_ready(states)

    times_ns = []
    for _ in range(steps):
        t0 = _now_ns()
        states, _ = vstep(states, actions)
        jax.block_until_ready(states)
        times_ns.append(_now_ns() - t0)

    batch = states.ball.shape[0]
    med_us = statistics.median(times_ns) / 1e3
    min_us = min(times_ns) / 1e3
    p95_us = sorted(times_ns)[int(0.95 * len(times_ns))] / 1e3
    total_fps = batch / (med_us / 1e6)
    return {
        "batch": batch,
        "min_us": min_us,
        "median_us": med_us,
        "p95_us": p95_us,
        "total_fps": total_fps,
        "per_env_fps": total_fps / batch,
    }


def _measure_vssvecenv(env, a: np.ndarray, steps: int) -> dict:
    """Warm + timed run of VSSVecEnv.step."""
    batch = a.shape[0]
    for _ in range(10):
        env.step(a)

    times_ns = []
    for _ in range(steps):
        t0 = _now_ns()
        env.step(a)
        times_ns.append(_now_ns() - t0)

    med_us = statistics.median(times_ns) / 1e3
    min_us = min(times_ns) / 1e3
    p95_us = sorted(times_ns)[int(0.95 * len(times_ns))] / 1e3
    total_fps = batch / (med_us / 1e6)
    return {
        "batch": batch,
        "min_us": min_us,
        "median_us": med_us,
        "p95_us": p95_us,
        "total_fps": total_fps,
        "per_env_fps": total_fps / batch,
    }


def _state_bytes(batch: int) -> int:
    """Approximate device memory for a batched SimState (state + 2× XLA buffers)."""
    per_env = (
        4              # ball  (4,) f32
        + 2 * 3 * 6   # robots (2,3,6) f32
        + 2            # score (2,) i32
        + 1            # t () f32
        + 2 * 3 * 2   # wheel_speeds (2,3,2) f32
    ) * 4  # bytes per float32/int32
    return per_env * batch * 3  # ×3: XLA keeps input + output + gradient buffers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(batches: list[int], steps: int, skip_vecenv: bool) -> None:
    import platform

    print("=" * 72)
    print("  vsss-sim GPU Saturation Sweep")
    print("=" * 72)
    print(f"  Platform  : {platform.platform()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  JAX devs  : {jax.devices()}")
    print(f"  Backend   : {jax.default_backend()}")
    print(f"  Batches   : {batches}")
    print(f"  Steps/run : {steps}")
    print()

    # Pre-compile vmap(step) once at the smallest batch; subsequent batch sizes
    # retrace from scratch (different shape) — compile time is measured per batch.
    vmap_rows: list[dict] = []
    vec_rows: list[dict] = []

    print("-- Raw jax.vmap(step) --")
    print(f"  {'batch':>7}  {'compile':>10}  {'min':>10}  {'median':>10}  "
          f"{'p95':>10}  {'total_fps':>12}  {'eff_vs_1k':>10}  {'VRAM_est':>10}")

    ref_fps: float | None = None

    for batch in batches:
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, batch)
        states = jax.vmap(jb.reset_kickoff)(keys)
        actions = jnp.zeros(
            (batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32
        )
        vstep = jax.jit(jax.vmap(jb.step))

        # First call = compile
        t0 = _now_ns()
        states, _ = vstep(states, actions)
        jax.block_until_ready(states)
        compile_ms = (_now_ns() - t0) / 1e6

        r = _measure_vmap(vstep, states, actions, steps)
        vmap_rows.append({**r, "compile_ms": compile_ms})

        if ref_fps is None:
            ref_fps = r["total_fps"]
        eff = r["total_fps"] / ref_fps
        vram_mb = _state_bytes(batch) / 1024**2

        print(
            f"  {batch:>7}  {compile_ms:>8.0f}ms  "
            f"{r['min_us']:>8.1f}μs  {r['median_us']:>8.1f}μs  "
            f"{r['p95_us']:>8.1f}μs  {r['total_fps']:>12,.0f}  "
            f"{eff:>9.2f}x  {vram_mb:>8.1f}MB"
        )

    # Identify saturation knee: first batch where gain < 20% of prior
    print()
    print("  Saturation analysis (throughput gain vs previous batch):")
    for i in range(1, len(vmap_rows)):
        prev = vmap_rows[i - 1]["total_fps"]
        curr = vmap_rows[i]["total_fps"]
        gain_pct = 100 * (curr - prev) / prev
        marker = "  <-- PLATEAU" if gain_pct < 10 else ""
        print(f"    batch {vmap_rows[i-1]['batch']:>6} → {vmap_rows[i]['batch']:>6}: "
              f"{gain_pct:>+6.1f}%{marker}")

    if skip_vecenv:
        print("\n  (VSSVecEnv sweep skipped via --no-vecenv)")
        return

    print()
    print("-- VSSVecEnv (Gymnasium wrapper, what SB3 sees) --")
    print(f"  {'batch':>7}  {'min':>10}  {'median':>10}  {'p95':>10}  "
          f"{'total_fps':>12}  {'vs_vmap':>10}  {'overhead':>10}")

    from vsss_sim.envs import VSSVecEnv

    for r_vmap, batch in zip(vmap_rows, batches):
        env = VSSVecEnv(num_envs=batch, opponent_policy="stationary")
        env.reset(seed=0)
        a = np.zeros((batch, config.N_ROBOTS * 2), dtype=np.float32)

        r = _measure_vssvecenv(env, a, steps)
        vec_rows.append(r)
        env.close()

        vs_vmap = r["total_fps"] / r_vmap["total_fps"]
        overhead_us = r["median_us"] - r_vmap["median_us"]
        print(
            f"  {batch:>7}  "
            f"{r['min_us']:>8.1f}μs  {r['median_us']:>8.1f}μs  "
            f"{r['p95_us']:>8.1f}μs  {r['total_fps']:>12,.0f}  "
            f"{vs_vmap:>9.3f}x  {overhead_us:>8.1f}μs"
        )

    print()
    print("  PCIe/wrapper overhead per call (VSSVecEnv median - vmap median):")
    for v, e in zip(vmap_rows, vec_rows):
        oh = e["median_us"] - v["median_us"]
        obs_bytes = e["batch"] * 46 * 4  # (B, 46) float32
        print(f"    batch={v['batch']:>6}  overhead={oh:>8.1f}μs  "
              f"obs_array={obs_bytes/1024:.1f}KB")

    print()
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    best_vmap = max(vmap_rows, key=lambda x: x["total_fps"])
    print(f"  Physics ceiling    : {best_vmap['total_fps']:>12,.0f} fps  "
          f"(batch={best_vmap['batch']})")
    if vec_rows:
        best_vec = max(vec_rows, key=lambda x: x["total_fps"])
        print(f"  VSSVecEnv peak     : {best_vec['total_fps']:>12,.0f} fps  "
              f"(batch={best_vec['batch']})")
        gap = best_vmap["total_fps"] / best_vec["total_fps"]
        print(f"  Wrapper penalty    : {gap:.1f}x  "
              f"(physics ceiling / VSSVecEnv peak)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU saturation sweep: vmap(step) and VSSVecEnv up to 65536 envs."
    )
    parser.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu", "default"],
        help="JAX device to use (default: gpu).",
    )
    parser.add_argument(
        "--batches", type=int, nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="Batch sizes to sweep (default: 1024 → 65536).",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Timed steps per batch size (default: 200; GPU calls are slower to measure).",
    )
    parser.add_argument(
        "--no-vecenv", dest="skip_vecenv", action="store_true",
        help="Skip VSSVecEnv sweep (faster; only measures raw vmap physics).",
    )
    args = parser.parse_args()
    main(args.batches, args.steps, args.skip_vecenv)
