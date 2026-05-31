#!/usr/bin/env python3
"""
Comprehensive performance benchmark for vsss-sim.

Measures:
  1. Component-level timings (μs/call) — reset, step, collision sub-routines
  2. JAX JIT compile times (one-shot measurement)
  3. Integrated throughput: raw vmap vs VSSVecEnv at varying batch sizes
  4. End-to-end SB3 PPO at varying num_envs across multiple generations

All JAX timings use jax.block_until_ready() to avoid measuring asynchronous
dispatch instead of actual computation. Reports min/median/p95 for component
timings and per-generation stats for integrated runs.

Usage:
    source .venv/bin/activate
    # CPU (default):
    PYTHONPATH=src python scripts/perf_bench.py [--reps N] [--gens N]
    # GPU (CUDA):
    PYTHONPATH=src python scripts/perf_bench.py --device gpu [--batches 1 8 64 256 1024 4096 8192]
"""
from __future__ import annotations

import os
import sys

# --device pre-parse: must run before `import jax` so JAX_PLATFORMS takes effect.
# Handles both `--device VALUE` and `--device=VALUE` forms.
for _i, _a in enumerate(sys.argv):
    _v = None
    if _a == "--device" and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
    elif _a.startswith("--device="):
        _v = _a.split("=", 1)[1]
    if _v in ("cpu", "gpu"):
        os.environ["JAX_PLATFORMS"] = "cpu" if _v == "cpu" else "cuda"
        break

import argparse
import statistics
import sys
import time
from typing import Callable

# --------------------------------------------------------------------------- #
# Timer helpers
# --------------------------------------------------------------------------- #

def _now_ns() -> int:
    return time.perf_counter_ns()

def _measure_ns(fn: Callable, reps: int) -> list[int]:
    """Run fn() reps times; return list of elapsed times in nanoseconds."""
    times = []
    for _ in range(reps):
        t0 = _now_ns()
        fn()
        times.append(_now_ns() - t0)
    return times

def _measure_ns_jax(fn: Callable, reps: int) -> list[int]:
    """Like _measure_ns but calls block_until_ready on the result."""
    import jax
    times = []
    for _ in range(reps):
        t0 = _now_ns()
        result = fn()
        jax.block_until_ready(result)
        times.append(_now_ns() - t0)
    return times

def _stats(ns_list: list[int]) -> dict:
    s = sorted(ns_list)
    n = len(s)
    return {
        "min_us":    s[0] / 1e3,
        "median_us": statistics.median(s) / 1e3,
        "p95_us":    s[int(0.95 * n)] / 1e3,
        "mean_us":   statistics.mean(s) / 1e3,
    }

def _fmt(stats: dict) -> str:
    return (
        f"min={stats['min_us']:>9.2f} μs  "
        f"median={stats['median_us']:>9.2f} μs  "
        f"p95={stats['p95_us']:>9.2f} μs"
    )

def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def _sub(title: str) -> None:
    print(f"\n-- {title} --")

# --------------------------------------------------------------------------- #
# 1. NumPy backend component benchmarks
# --------------------------------------------------------------------------- #

def bench_numpy_components(reps: int) -> dict:
    import numpy as np
    from vsss_sim import config
    from vsss_sim.physics import numpy_backend as nb

    rng = np.random.default_rng(0)
    s = nb.SimState()
    nb.reset_kickoff(s, rng=rng)
    a = np.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=np.float64)

    # Warm-up
    for _ in range(200):
        nb.step(s, a)

    results = {}

    results["reset_kickoff"] = _stats(_measure_ns(
        lambda: nb.reset_kickoff(s, rng=np.random.default_rng(42)), reps
    ))

    results["reset_random"] = _stats(_measure_ns(
        lambda: nb.reset_random(s, rng=np.random.default_rng(42)), reps
    ))

    results["step_full"] = _stats(_measure_ns(
        lambda: nb.step(s, a), reps
    ))

    # Sub-routines (call internals directly)
    _vl = np.array([0.5, 0.3, 0.1])
    _vr = np.array([0.4, 0.6, 0.2])
    _theta = np.array([0.0, 1.0, -0.5])
    results["diff_drive"] = _stats(_measure_ns(
        lambda: nb._diff_drive(_vl, _vr, _theta),
        reps
    ))

    results["ball_wall_collisions"] = _stats(_measure_ns(
        lambda: nb._ball_wall_collisions(s), reps
    ))

    results["robot_wall_collisions"] = _stats(_measure_ns(
        lambda: nb._robot_wall_collisions(s), reps
    ))

    results["ball_robot_collisions"] = _stats(_measure_ns(
        lambda: nb._ball_robot_collisions(s), reps
    ))

    results["robot_robot_collisions"] = _stats(_measure_ns(
        lambda: nb._robot_robot_collisions(s), reps
    ))

    return results


# --------------------------------------------------------------------------- #
# 2. JAX backend component benchmarks
# --------------------------------------------------------------------------- #

def bench_jax_components(reps: int) -> dict:
    import jax
    import jax.numpy as jnp
    from vsss_sim import config
    from vsss_sim.physics import jax_backend as jb

    key = jax.random.PRNGKey(0)
    s = jb.reset_kickoff(key)
    a = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)

    results = {}

    # ---- JIT compile times (cold-start, one measurement each) ----
    # These are wall-clock times for the *first* call to a jit'd function.

    step_jit = jax.jit(jb.step)
    t0 = _now_ns()
    s_out, _ = step_jit(s, a)
    jax.block_until_ready(s_out)
    results["jit_compile_step_ns"] = _now_ns() - t0

    reset_kickoff_jit = jax.jit(jb.reset_kickoff)
    t0 = _now_ns()
    s_out2 = reset_kickoff_jit(key)
    jax.block_until_ready(s_out2)
    results["jit_compile_reset_kickoff_ns"] = _now_ns() - t0

    reset_random_jit = jax.jit(jb.reset_random)
    t0 = _now_ns()
    s_out3 = reset_random_jit(key)
    jax.block_until_ready(s_out3)
    results["jit_compile_reset_random_ns"] = _now_ns() - t0

    # ---- Warm-up (drain all async, ensure JIT is settled) ----
    for _ in range(500):
        s, _ = step_jit(s, a)
    jax.block_until_ready(s)

    # ---- Warm timings (block_until_ready per call) ----
    results["step_warm"] = _stats(_measure_ns_jax(
        lambda: step_jit(s, a)[0], reps
    ))

    results["reset_kickoff_warm"] = _stats(_measure_ns_jax(
        lambda: reset_kickoff_jit(key), reps
    ))

    results["reset_random_warm"] = _stats(_measure_ns_jax(
        lambda: reset_random_jit(key), reps
    ))

    # ---- Sub-routine timings (jit'd individually) ----
    diff_drive_jit = jax.jit(jb._diff_drive)
    import numpy as np
    vl = jnp.array([0.5, 0.3, 0.1])
    vr = jnp.array([0.4, 0.6, 0.2])
    theta = jnp.array([0.0, 1.0, -0.5])
    # warm
    for _ in range(100):
        out = diff_drive_jit(vl, vr, theta)
    jax.block_until_ready(out)
    results["diff_drive_warm"] = _stats(_measure_ns_jax(
        lambda: diff_drive_jit(vl, vr, theta), reps
    ))

    return results


# --------------------------------------------------------------------------- #
# 3. JAX vmap throughput at varying batch sizes
# --------------------------------------------------------------------------- #

def bench_jax_vmap(steps: int, batch_sizes: list[int]) -> list[dict]:
    import jax
    import jax.numpy as jnp
    from vsss_sim import config
    from vsss_sim.physics import jax_backend as jb

    rows = []
    for batch in batch_sizes:
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        states = jax.vmap(jb.reset_kickoff)(keys)
        actions = jnp.zeros((batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
        vstep = jax.jit(jax.vmap(jb.step))

        # First call (compile)
        t0 = _now_ns()
        states, _ = vstep(states, actions)
        jax.block_until_ready(states)
        compile_ms = (_now_ns() - t0) / 1e6

        # Additional warm-up
        for _ in range(20):
            states, _ = vstep(states, actions)
        jax.block_until_ready(states)

        # Measurement
        t0 = _now_ns()
        for _ in range(steps):
            states, _ = vstep(states, actions)
        jax.block_until_ready(states)
        elapsed_s = (_now_ns() - t0) / 1e9

        per_call_us = elapsed_s / steps * 1e6
        total_fps = (steps * batch) / elapsed_s
        per_env_fps = total_fps / batch

        rows.append({
            "batch": batch,
            "compile_ms": compile_ms,
            "per_call_us": per_call_us,
            "per_env_fps": per_env_fps,
            "total_fps": total_fps,
        })
    return rows


# --------------------------------------------------------------------------- #
# 4. VSSVecEnv throughput (what RL libs see)
# --------------------------------------------------------------------------- #

def bench_vssvecenv(steps: int, batch_sizes: list[int]) -> list[dict]:
    import numpy as np
    from vsss_sim import config
    from vsss_sim.envs import VSSVecEnv

    rows = []
    for batch in batch_sizes:
        env = VSSVecEnv(num_envs=batch, opponent_policy="stationary")
        env.reset(seed=0)
        a = np.zeros((batch, config.N_ROBOTS * 2), dtype=np.float32)

        # Warm-up
        for _ in range(20):
            env.step(a)

        # Measurement
        t0 = _now_ns()
        for _ in range(steps):
            env.step(a)
        elapsed_s = (_now_ns() - t0) / 1e9

        per_call_us = elapsed_s / steps * 1e6
        total_fps = (steps * batch) / elapsed_s
        per_env_fps = total_fps / batch

        env.close()
        rows.append({
            "batch": batch,
            "per_call_us": per_call_us,
            "per_env_fps": per_env_fps,
            "total_fps": total_fps,
        })
    return rows


# --------------------------------------------------------------------------- #
# 5. End-to-end SB3 PPO across multiple generations
# --------------------------------------------------------------------------- #

def bench_sb3_generations(num_envs: int, n_steps: int, n_gens: int) -> dict:
    """
    Run SB3 PPO for n_gens iterations, measuring per-generation wall time.
    Update time = gap between _on_rollout_end and the next _on_rollout_start,
    so we get one update entry per completed rollout-update cycle.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from vsss_sim.envs import VSSVecEnv
    from vsss_sim.sb3_adapter import VSSVecEnvToSB3

    class _GenTimingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._rollout_t0 = None
            self._rollout_t1 = None    # set in _on_rollout_end; update starts here
            self.gen_rollout_ms: list[float] = []
            self.gen_update_ms: list[float] = []

        def _on_rollout_start(self):
            now = _now_ns()
            # If there was a prior rollout, the gap since it ended is the update time
            if self._rollout_t1 is not None:
                self.gen_update_ms.append((now - self._rollout_t1) / 1e6)
            self._rollout_t0 = now

        def _on_rollout_end(self):
            now = _now_ns()
            if self._rollout_t0 is not None:
                self.gen_rollout_ms.append((now - self._rollout_t0) / 1e6)
            self._rollout_t1 = now

        def _on_step(self) -> bool:
            return True

    env = VSSVecEnvToSB3(VSSVecEnv(num_envs=num_envs, opponent_policy="stationary"))
    batch_size = min(64 * num_envs, 4096)
    model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch_size,
                n_epochs=10, seed=0, verbose=0)

    # 1 generation warm-up (JIT compile + PyTorch trace)
    model.learn(total_timesteps=num_envs * n_steps)

    cb = _GenTimingCallback()
    total_ts = num_envs * n_steps * n_gens

    t_total_start = _now_ns()
    model.learn(total_timesteps=total_ts, reset_num_timesteps=False, callback=cb)
    total_elapsed_ms = (_now_ns() - t_total_start) / 1e6

    env.close()

    rollouts = cb.gen_rollout_ms
    updates = cb.gen_update_ms
    # rollouts has n_gens entries; updates has n_gens-1 entries (last update not captured)
    n_r = len(rollouts)
    n_u = len(updates)
    n_complete = min(n_r, n_u)  # fully captured (rollout + update) cycles

    overall_fps = (num_envs * n_steps * n_r / (total_elapsed_ms / 1000)
                   if total_elapsed_ms > 0 else 0)

    result = {
        "num_envs": num_envs,
        "n_steps": n_steps,
        "n_gens_measured": n_r,
        "n_complete_cycles": n_complete,
        "total_elapsed_ms": total_elapsed_ms,
        "overall_fps": overall_fps,
        "rollout_ms": rollouts,
        "update_ms": updates,
    }

    if rollouts:
        result["rollout_median_ms"] = statistics.median(rollouts)
        result["rollout_min_ms"] = min(rollouts)
        result["rollout_p95_ms"] = sorted(rollouts)[int(max(0, 0.95 * n_r - 1))]
        result["rollout_fps"] = statistics.median(
            [(num_envs * n_steps) / (ms / 1000) for ms in rollouts]
        )
    if updates:
        result["update_median_ms"] = statistics.median(updates)
        result["update_min_ms"] = min(updates)
        result["update_p95_ms"] = sorted(updates)[int(max(0, 0.95 * n_u - 1))]

    return result


# --------------------------------------------------------------------------- #
# 6. Single-env numpy benchmark (reference baseline)
# --------------------------------------------------------------------------- #

def bench_numpy_single(steps: int) -> dict:
    import numpy as np
    from vsss_sim import config
    from vsss_sim.physics import numpy_backend as nb

    s = nb.SimState()
    nb.reset_kickoff(s, rng=np.random.default_rng(0))
    a = np.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=np.float64)
    for _ in range(200):
        nb.step(s, a)

    t0 = _now_ns()
    for _ in range(steps):
        nb.step(s, a)
    elapsed_s = (_now_ns() - t0) / 1e9

    return {
        "fps": steps / elapsed_s,
        "us_per_step": elapsed_s / steps * 1e6,
    }


def bench_jax_single(steps: int) -> dict:
    import jax
    import jax.numpy as jnp
    from vsss_sim import config
    from vsss_sim.physics import jax_backend as jb

    key = jax.random.PRNGKey(0)
    s = jb.reset_kickoff(key)
    a = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
    step_jit = jax.jit(jb.step)

    # compile
    t0 = _now_ns()
    s, _ = step_jit(s, a)
    jax.block_until_ready(s)
    compile_ms = (_now_ns() - t0) / 1e6

    for _ in range(500):
        s, _ = step_jit(s, a)
    jax.block_until_ready(s)

    t0 = _now_ns()
    for _ in range(steps):
        s, _ = step_jit(s, a)
    jax.block_until_ready(s)
    elapsed_s = (_now_ns() - t0) / 1e9

    return {
        "compile_ms": compile_ms,
        "fps": steps / elapsed_s,
        "us_per_step": elapsed_s / steps * 1e6,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(reps: int, steps: int, gens: int, batches: list[int]) -> None:
    import jax
    import platform

    print("=" * 70)
    print("  vsss-sim Performance Benchmark")
    print("=" * 70)
    print(f"  Platform     : {platform.platform()}")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  JAX devices  : {jax.devices()}")
    print(f"  JAX backend  : {jax.default_backend()}")
    print(f"  Component reps: {reps}  |  Throughput steps: {steps}  |  Generations: {gens}")

    # ------------------------------------------------------------------ #
    # 1. NumPy component timings
    # ------------------------------------------------------------------ #
    _header("1. NumPy Backend — Component Timings")
    np_comps = bench_numpy_components(reps)
    for name, st in np_comps.items():
        print(f"  {name:<30s}  {_fmt(st)}")

    # ------------------------------------------------------------------ #
    # 2. JAX component timings
    # ------------------------------------------------------------------ #
    _header("2. JAX Backend — Component Timings (warm, after JIT compile)")

    _sub("JIT compile times (one-shot wall-clock)")
    jax_comps = bench_jax_components(reps)
    for key in ("jit_compile_step_ns", "jit_compile_reset_kickoff_ns",
                "jit_compile_reset_random_ns"):
        ms = jax_comps[key] / 1e6
        print(f"  {key:<35s}  {ms:>8.1f} ms")

    _sub("Warm timings (block_until_ready per call)")
    for name in ("step_warm", "reset_kickoff_warm", "reset_random_warm", "diff_drive_warm"):
        if name in jax_comps:
            print(f"  {name:<30s}  {_fmt(jax_comps[name])}")

    # ------------------------------------------------------------------ #
    # 3. Single-env throughput reference
    # ------------------------------------------------------------------ #
    _header("3. Single-Env Throughput (reference)")
    np_ref = bench_numpy_single(steps)
    jx_ref = bench_jax_single(steps)
    print(f"  numpy  single env : {np_ref['fps']:>10,.0f} fps  "
          f"({np_ref['us_per_step']:>8.2f} μs/step)")
    print(f"  jax    single env : {jx_ref['fps']:>10,.0f} fps  "
          f"({jx_ref['us_per_step']:>8.2f} μs/step)  "
          f"[compile: {jx_ref['compile_ms']:.1f} ms]")
    print(f"  jax/numpy speedup : {jx_ref['fps'] / np_ref['fps']:.1f}x")
    np_fps_ref = np_ref["fps"]

    # ------------------------------------------------------------------ #
    # 4. Raw vmap physics ceiling
    # ------------------------------------------------------------------ #
    _header("4. Raw jax.vmap(step) — Physics Ceiling")
    print(f"  {'batch':>6}  {'compile':>10}  {'per_call':>12}  "
          f"{'per_env_fps':>14}  {'total_fps':>14}  {'vs_numpy':>9}")
    vmap_rows = bench_jax_vmap(steps, batches)
    for r in vmap_rows:
        print(
            f"  {r['batch']:>6}  "
            f"{r['compile_ms']:>8.0f}ms  "
            f"{r['per_call_us']:>10.1f}μs  "
            f"{r['per_env_fps']:>14,.0f}  "
            f"{r['total_fps']:>14,.0f}  "
            f"{r['total_fps'] / np_fps_ref:>8.1f}x"
        )

    # ------------------------------------------------------------------ #
    # 5. VSSVecEnv (Gymnasium wrapper overhead)
    # ------------------------------------------------------------------ #
    _header("5. VSSVecEnv — Gymnasium Wrapper (what RL libs see)")
    print(f"  {'batch':>6}  {'per_call':>12}  {'per_env_fps':>14}  "
          f"{'total_fps':>14}  {'vs_numpy':>9}")
    vec_rows = bench_vssvecenv(steps, batches)
    for r in vec_rows:
        print(
            f"  {r['batch']:>6}  "
            f"{r['per_call_us']:>10.1f}μs  "
            f"{r['per_env_fps']:>14,.0f}  "
            f"{r['total_fps']:>14,.0f}  "
            f"{r['total_fps'] / np_fps_ref:>8.1f}x"
        )

    # Overhead ratio: vmap vs vecenv at same batch
    _sub("Wrapper overhead: VSSVecEnv vs raw vmap at same batch size")
    print(f"  {'batch':>6}  {'vmap_fps':>14}  {'vecenv_fps':>14}  {'overhead_x':>12}")
    for v, e in zip(vmap_rows, vec_rows):
        overhead = v["total_fps"] / e["total_fps"]
        print(f"  {v['batch']:>6}  {v['total_fps']:>14,.0f}  {e['total_fps']:>14,.0f}  "
              f"{overhead:>11.2f}x")

    # ------------------------------------------------------------------ #
    # 6. End-to-end SB3 PPO — multi-generation
    # ------------------------------------------------------------------ #
    _header("6. End-to-End SB3 PPO — Per-Generation Timing")
    sb3_configs = [(1, 512), (8, 512), (32, 512), (64, 256), (128, 256), (256, 128)]
    print(f"  {'num_envs':>9}  {'n_steps':>8}  {'gens':>5}  "
          f"{'rollout_med':>12}  {'update_med':>12}  "
          f"{'overall_fps':>12}  {'vs_1env':>9}")
    all_sb3 = []
    baseline_fps = None
    for (ne, ns) in sb3_configs:
        r = bench_sb3_generations(ne, ns, gens)
        if baseline_fps is None:
            baseline_fps = r["overall_fps"]
        all_sb3.append(r)
        rmss = f"{r.get('rollout_median_ms', 0):.1f}ms"
        umss = f"{r.get('update_median_ms', 0):.1f}ms" if r.get("update_median_ms") else "  N/A"
        vs = r["overall_fps"] / baseline_fps if baseline_fps else 1.0
        print(
            f"  {ne:>9}  {ns:>8}  {r['n_gens_measured']:>5}  "
            f"{rmss:>12}  {umss:>12}  "
            f"{r['overall_fps']:>12,.0f}  {vs:>8.2f}x"
        )

    # Per-generation detail for largest config
    largest = all_sb3[-1]
    _sub(f"Per-generation detail: num_envs={largest['num_envs']} n_steps={largest['n_steps']}")
    if largest["rollout_ms"]:
        print(f"  Gen  {'rollout_ms':>12}  {'update_ms':>12}  {'rollout_fps':>12}")
        up = largest["update_ms"]
        for i, r_ms in enumerate(largest["rollout_ms"]):
            u_ms = up[i] if i < len(up) else float("nan")
            rfps = (largest["num_envs"] * largest["n_steps"]) / (r_ms / 1000)
            print(f"  {i+1:>3}  {r_ms:>12.1f}  {u_ms:>12.1f}  {rfps:>12,.0f}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _header("Summary — Key Numbers")
    print(f"  numpy step latency          : {np_comps['step_full']['median_us']:>9.1f} μs/call")
    print(f"  jax step latency (warm)     : {jax_comps['step_warm']['median_us']:>9.1f} μs/call")
    print(f"  jax step latency (min)      : {jax_comps['step_warm']['min_us']:>9.1f} μs/call")
    print(f"  jax step JIT compile        : {jax_comps['jit_compile_step_ns']/1e6:>9.1f} ms  (one-time)")
    if vmap_rows:
        best_vmap = max(vmap_rows, key=lambda x: x["total_fps"])
        print(f"  raw vmap ceiling (batch={best_vmap['batch']})   : "
              f"{best_vmap['total_fps']:>9,.0f} fps")
    if vec_rows:
        best_vec = max(vec_rows, key=lambda x: x["total_fps"])
        print(f"  VSSVecEnv peak   (batch={best_vec['batch']})   : "
              f"{best_vec['total_fps']:>9,.0f} fps")
    if all_sb3:
        best_sb3 = max(all_sb3, key=lambda x: x["overall_fps"])
        print(f"  SB3 PPO peak  (envs={best_sb3['num_envs']:>3},steps={best_sb3['n_steps']:>3}) : "
              f"{best_sb3['overall_fps']:>9,.0f} fps")
        if best_sb3.get("rollout_median_ms") and best_sb3.get("update_median_ms"):
            total_gen = best_sb3["rollout_median_ms"] + best_sb3["update_median_ms"]
            rollout_pct = 100 * best_sb3["rollout_median_ms"] / total_gen
            update_pct = 100 * best_sb3["update_median_ms"] / total_gen
            print(f"    rollout: {rollout_pct:.1f}%  update: {update_pct:.1f}% of wall time per gen")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2000,
                        help="Repetitions for component timings (default: 2000)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Steps for throughput measurements (default: 3000)")
    parser.add_argument("--gens", type=int, default=12,
                        help="PPO generations per SB3 config (default: 12)")
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1, 8, 32, 64, 128, 256, 512, 1024],
                        help="Batch sizes to test for vmap/VSSVecEnv (default CPU set; "
                             "for GPU consider adding 2048 4096 8192 16384)")
    parser.add_argument(
        "--device", type=str, default="default", choices=["cpu", "gpu", "default"],
        help="Force JAX device: cpu=JAX_PLATFORMS=cpu, gpu=JAX_PLATFORMS=cuda, "
             "default=let JAX pick (must be first arg parsed before JAX import).",
    )
    args = parser.parse_args()
    main(args.reps, args.steps, args.gens, args.batches)
