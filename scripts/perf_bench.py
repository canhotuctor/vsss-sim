#!/usr/bin/env python3
"""
Performance benchmark for vsss-sim — JAX physics and Gymnasium wrapper.

Measures:
  1. Raw jax.vmap(step) throughput at varying batch sizes
  2. VSSVecEnv (Gymnasium wrapper) throughput at varying batch sizes
  3. End-to-end SB3 PPO across multiple generations

All JAX timings use jax.block_until_ready() and per-call measurement,
so each data point is an individual step call. Reports min / mean ± std /
median / p95 over --steps runs (default 1000).

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/perf_bench.py
    PYTHONPATH=src python scripts/perf_bench.py --device gpu --batches 1024 4096 8192 16384
"""
from __future__ import annotations

import os
import sys

# --device must be parsed before any JAX import so JAX_PLATFORMS takes effect.
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
import time

# --------------------------------------------------------------------------- #
# Timer helpers
# --------------------------------------------------------------------------- #

def _now_ns() -> int:
    return time.perf_counter_ns()


def _stats(ns_list: list[int], trim_top_pct: float = 0.05) -> dict:
    """Compute latency stats; mean/std use a trimmed sample (top trim_top_pct excluded)
    to suppress GC pauses and OS jitter that otherwise inflate std by 10-100×."""
    s = sorted(ns_list)
    n = len(s)
    cut = max(0, int(trim_top_pct * n))
    trimmed = s[: n - cut] if cut > 0 else s
    nt = len(trimmed)
    mean = sum(trimmed) / nt
    variance = sum((x - mean) ** 2 for x in trimmed) / nt
    std = variance ** 0.5
    return {
        "min_us":    s[0] / 1e3,
        "mean_us":   mean / 1e3,
        "std_us":    std / 1e3,
        "median_us": statistics.median(s) / 1e3,
        "p95_us":    s[int(0.95 * n)] / 1e3,
    }


def _fmt(st: dict) -> str:
    return (
        f"min={st['min_us']:>8.1f}μs  "
        f"mean={st['mean_us']:>8.1f}±{st['std_us']:.1f}μs  "
        f"median={st['median_us']:>8.1f}μs  "
        f"p95={st['p95_us']:>8.1f}μs"
    )


def _header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _sub(title: str) -> None:
    print(f"\n-- {title} --")


# --------------------------------------------------------------------------- #
# 1. Raw jax.vmap(step) — physics ceiling
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

        # First call = JIT compile; measure it separately.
        t0 = _now_ns()
        states, _ = vstep(states, actions)
        jax.block_until_ready(states)
        compile_ms = (_now_ns() - t0) / 1e6

        # Warm-up: drain async queue.
        for _ in range(20):
            states, _ = vstep(states, actions)
        jax.block_until_ready(states)

        # Per-call measurement.
        times_ns: list[int] = []
        for _ in range(steps):
            t0 = _now_ns()
            states, _ = vstep(states, actions)
            jax.block_until_ready(states)
            times_ns.append(_now_ns() - t0)

        st = _stats(times_ns)
        total_fps = batch / (st["median_us"] / 1e6)
        rows.append({
            "batch": batch,
            "compile_ms": compile_ms,
            "total_fps": total_fps,
            **st,
        })

    return rows


# --------------------------------------------------------------------------- #
# 2. VSSVecEnv (Gymnasium wrapper) — what RL libs see
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

        # Warm-up.
        for _ in range(20):
            env.step(a)

        # Per-call measurement.
        times_ns: list[int] = []
        for _ in range(steps):
            t0 = _now_ns()
            env.step(a)
            times_ns.append(_now_ns() - t0)

        env.close()
        st = _stats(times_ns)
        total_fps = batch / (st["median_us"] / 1e6)
        rows.append({
            "batch": batch,
            "total_fps": total_fps,
            **st,
        })

    return rows


# --------------------------------------------------------------------------- #
# 3. PPO policy inference — pure forward pass (what runs during rollout)
# --------------------------------------------------------------------------- #

def bench_ppo_inference(steps: int, batch_sizes: list[int]) -> list[dict]:
    """Measure pure PyTorch forward pass of the PPO MlpPolicy.

    This is the neural-network-only cost during rollout collection:
        with torch.no_grad():
            actions, values, log_probs = policy(obs_tensor)
    No env step, no JAX — just the MLP inference on CPU.
    """
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from vsss_sim.envs import VSSVecEnv
    from vsss_sim.sb3_adapter import VSSVecEnvToSB3

    obs_dim = 4 + 2 * 3 * 7  # 46

    # Build a throwaway env just to get a correctly-shaped policy.
    env = VSSVecEnvToSB3(VSSVecEnv(num_envs=1, opponent_policy="stationary"))
    model = PPO("MlpPolicy", env, seed=0, verbose=0)
    policy = model.policy
    policy.set_training_mode(False)
    env.close()

    rows = []
    for batch in batch_sizes:
        obs_np = np.random.uniform(-1.0, 1.0, size=(batch, obs_dim)).astype(np.float32)
        obs_t = torch.as_tensor(obs_np, device=model.device)

        # Warm-up: let PyTorch trace through the graph.
        with torch.no_grad():
            for _ in range(50):
                policy(obs_t)

        times_ns: list[int] = []
        for _ in range(steps):
            t0 = _now_ns()
            with torch.no_grad():
                policy(obs_t)
            times_ns.append(_now_ns() - t0)

        st = _stats(times_ns)
        rows.append({
            "batch": batch,
            "total_fps": batch / (st["median_us"] / 1e6),
            **st,
        })

    return rows


# --------------------------------------------------------------------------- #
# 4. End-to-end SB3 PPO — per-generation timing
# --------------------------------------------------------------------------- #

def bench_sb3_generations(num_envs: int, n_steps: int, n_gens: int,
                          batch_size: int = 64) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from vsss_sim.envs import VSSVecEnv
    from vsss_sim.sb3_adapter import VSSVecEnvToSB3

    class _GenTimingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._rollout_t0 = None
            self._rollout_t1 = None
            self.gen_rollout_ms: list[float] = []
            self.gen_update_ms: list[float] = []

        def _on_rollout_start(self):
            now = _now_ns()
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
    model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch_size, seed=0, verbose=0)

    # 1-generation warm-up (JIT compile + PyTorch trace).
    model.learn(total_timesteps=num_envs * n_steps)

    cb = _GenTimingCallback()
    t_total = _now_ns()
    model.learn(total_timesteps=num_envs * n_steps * n_gens,
                reset_num_timesteps=False, callback=cb)
    total_elapsed_ms = (_now_ns() - t_total) / 1e6
    env.close()

    rollouts = cb.gen_rollout_ms
    updates  = cb.gen_update_ms
    n_r, n_u = len(rollouts), len(updates)
    overall_fps = (num_envs * n_steps * n_r / (total_elapsed_ms / 1000)
                   if total_elapsed_ms > 0 else 0)

    result: dict = {
        "num_envs": num_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_gens_measured": n_r,
        "total_elapsed_ms": total_elapsed_ms,
        "overall_fps": overall_fps,
        "rollout_ms": rollouts,
        "update_ms": updates,
    }
    if rollouts:
        result["rollout_median_ms"] = statistics.median(rollouts)
        result["rollout_mean_ms"]   = sum(rollouts) / n_r
        result["rollout_std_ms"]    = statistics.stdev(rollouts) if n_r > 1 else 0.0
        result["rollout_fps"]       = statistics.median(
            [(num_envs * n_steps) / (ms / 1000) for ms in rollouts]
        )
    if updates:
        result["update_median_ms"] = statistics.median(updates)
        result["update_mean_ms"]   = sum(updates) / n_u
        result["update_std_ms"]    = statistics.stdev(updates) if n_u > 1 else 0.0
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(steps: int, gens: int, batches: list[int]) -> None:
    import jax
    import platform

    print("=" * 72)
    print("  vsss-sim Performance Benchmark")
    print("=" * 72)
    print(f"  Platform  : {platform.platform()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  JAX devs  : {jax.devices()}")
    print(f"  Backend   : {jax.default_backend()}")
    print(f"  Steps/run : {steps}  |  Generations: {gens}")
    print(f"  Batches   : {batches}")

    # ---------------------------------------------------------------------- #
    # 1. Raw vmap physics ceiling
    # ---------------------------------------------------------------------- #
    _header("1. Raw jax.vmap(step) — Physics Ceiling")
    print("  (mean±std computed on trimmed sample, top 5% excluded)")
    print(f"  {'batch':>7}  {'compile':>10}  {'min':>9}  {'mean±std(t)':>22}  "
          f"{'median':>9}  {'p95':>9}  {'total_fps':>12}")
    vmap_rows = bench_jax_vmap(steps, batches)
    for r in vmap_rows:
        print(
            f"  {r['batch']:>7}  {r['compile_ms']:>8.0f}ms  "
            f"{r['min_us']:>7.1f}μs  "
            f"{r['mean_us']:>8.1f}±{r['std_us']:<7.1f}μs  "
            f"{r['median_us']:>7.1f}μs  "
            f"{r['p95_us']:>7.1f}μs  "
            f"{r['total_fps']:>12,.0f}"
        )

    # ---------------------------------------------------------------------- #
    # 2. VSSVecEnv wrapper
    # ---------------------------------------------------------------------- #
    _header("2. VSSVecEnv — Gymnasium Wrapper (what RL libs see)")
    print("  (mean±std computed on trimmed sample, top 5% excluded)")
    print(f"  {'batch':>7}  {'min':>9}  {'mean±std(t)':>22}  "
          f"{'median':>9}  {'p95':>9}  {'total_fps':>12}  {'vs_vmap':>8}")
    vec_rows = bench_vssvecenv(steps, batches)
    for r, v in zip(vec_rows, vmap_rows):
        ratio = r["total_fps"] / v["total_fps"]
        print(
            f"  {r['batch']:>7}  "
            f"{r['min_us']:>7.1f}μs  "
            f"{r['mean_us']:>8.1f}±{r['std_us']:<7.1f}μs  "
            f"{r['median_us']:>7.1f}μs  "
            f"{r['p95_us']:>7.1f}μs  "
            f"{r['total_fps']:>12,.0f}  "
            f"{ratio:>7.3f}x"
        )

    _sub("Wrapper overhead: VSSVecEnv median - vmap median")
    print(f"  {'batch':>7}  {'vmap_med':>10}  {'vec_med':>10}  {'overhead':>10}  {'obs_KB':>8}")
    for v, e in zip(vmap_rows, vec_rows):
        oh = e["median_us"] - v["median_us"]
        obs_kb = e["batch"] * 46 * 4 / 1024
        print(f"  {v['batch']:>7}  {v['median_us']:>8.1f}μs  {e['median_us']:>8.1f}μs  "
              f"{oh:>8.1f}μs  {obs_kb:>7.1f}KB")

    # ---------------------------------------------------------------------- #
    # 3. End-to-end SB3 PPO
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # 3. PPO inference speed
    # ---------------------------------------------------------------------- #
    _header("3. PPO Policy Inference — Pure Forward Pass (rollout cost)")
    print("  (mean±std computed on trimmed sample, top 5% excluded)")
    print(f"  {'batch':>7}  {'min':>9}  {'mean±std(t)':>22}  "
          f"{'median':>9}  {'p95':>9}  {'total_fps':>12}")
    infer_rows = bench_ppo_inference(steps, batches)
    for r in infer_rows:
        print(
            f"  {r['batch']:>7}  "
            f"{r['min_us']:>7.1f}μs  "
            f"{r['mean_us']:>8.1f}±{r['std_us']:<7.1f}μs  "
            f"{r['median_us']:>7.1f}μs  "
            f"{r['p95_us']:>7.1f}μs  "
            f"{r['total_fps']:>12,.0f}"
        )

    _sub("Inference vs physics: policy / vmap ratio (>1 = policy is the bottleneck)")
    print(f"  {'batch':>7}  {'policy_fps':>12}  {'vmap_fps':>12}  {'ratio':>8}")
    for ri, rv in zip(infer_rows, vmap_rows):
        ratio = ri["total_fps"] / rv["total_fps"]
        bottleneck = " <-- policy bottleneck" if ratio < 1.0 else " <-- physics bottleneck"
        print(f"  {ri['batch']:>7}  {ri['total_fps']:>12,.0f}  {rv['total_fps']:>12,.0f}  "
              f"{ratio:>7.3f}x{bottleneck}")

    # ---------------------------------------------------------------------- #
    # 4. End-to-End SB3 PPO — Per-Generation Timing
    # ---------------------------------------------------------------------- #
    _header("4. End-to-End SB3 PPO — Per-Generation Timing")
    ppo_batch_sizes = [64, 256, 1024, 4096]
    sb3_configs = [(1, 4096), (8, 512), (32, 512), (64, 256), (128, 256), (256, 128)]
    all_sb3: list[dict] = []

    for ppo_bs in ppo_batch_sizes:
        print(f"\n  PPO batch_size={ppo_bs}")
        print(f"  {'num_envs':>9}  {'n_steps':>8}  {'gens':>5}  "
              f"{'rollout_med':>14}  {'update_med':>14}  {'overall_fps':>12}")
        for ne, ns in sb3_configs:
            r = bench_sb3_generations(ne, ns, gens, ppo_bs)
            all_sb3.append(r)
            rmss = (f"{r['rollout_mean_ms']:.1f}±{r['rollout_std_ms']:.1f}ms"
                    if "rollout_mean_ms" in r else "  N/A")
            umss = (f"{r['update_mean_ms']:.1f}±{r['update_std_ms']:.1f}ms"
                    if "update_mean_ms" in r else "  N/A")
            print(
                f"  {ne:>9}  {ns:>8}  {r['n_gens_measured']:>5}  "
                f"{rmss:>14}  {umss:>14}  "
                f"{r['overall_fps']:>12,.0f}"
            )

    # Per-generation detail for the largest config (last ppo_bs).
    largest = all_sb3[-1]
    _sub(f"Per-generation detail: num_envs={largest['num_envs']} n_steps={largest['n_steps']} "
         f"ppo_batch_size={largest['batch_size']}")
    if largest["rollout_ms"]:
        up = largest["update_ms"]
        print(f"  {'gen':>4}  {'rollout_ms':>12}  {'update_ms':>12}  {'rollout_fps':>12}")
        for i, r_ms in enumerate(largest["rollout_ms"]):
            u_ms = up[i] if i < len(up) else float("nan")
            rfps = (largest["num_envs"] * largest["n_steps"]) / (r_ms / 1000)
            print(f"  {i+1:>4}  {r_ms:>12.1f}  {u_ms:>12.1f}  {rfps:>12,.0f}")

    # ---------------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------------- #
    _header("Summary — Key Numbers")
    if vmap_rows:
        best_vmap = max(vmap_rows, key=lambda x: x["total_fps"])
        print(f"  Physics ceiling (batch={best_vmap['batch']})  : "
              f"{best_vmap['total_fps']:>12,.0f} fps")
    if vec_rows:
        best_vec = max(vec_rows, key=lambda x: x["total_fps"])
        print(f"  VSSVecEnv peak  (batch={best_vec['batch']})  : "
              f"{best_vec['total_fps']:>12,.0f} fps")
        gap = best_vmap["total_fps"] / best_vec["total_fps"]
        print(f"  Wrapper penalty                   : {gap:.2f}x")
    if infer_rows:
        best_inf = max(infer_rows, key=lambda x: x["total_fps"])
        print(f"  Policy inference peak (batch={best_inf['batch']}) : "
              f"{best_inf['total_fps']:>12,.0f} fps")
        # cross-over: first batch where inference is slower than VSSVecEnv
        if vec_rows:
            for ri, rv in zip(infer_rows, vec_rows):
                if ri["total_fps"] < rv["total_fps"]:
                    print(f"  Inference bottleneck starts at batch={ri['batch']} "
                          f"({ri['total_fps']:,.0f} < {rv['total_fps']:,.0f} fps)")
                    break
    if all_sb3:
        best_sb3 = max(all_sb3, key=lambda x: x["overall_fps"])
        print(f"  SB3 PPO peak (envs={best_sb3['num_envs']}, steps={best_sb3['n_steps']}) : "
              f"{best_sb3['overall_fps']:>12,.0f} fps")
        if best_sb3.get("rollout_median_ms") and best_sb3.get("update_median_ms"):
            total_gen = best_sb3["rollout_median_ms"] + best_sb3["update_median_ms"]
            print(f"    rollout: {100*best_sb3['rollout_median_ms']/total_gen:.1f}%  "
                  f"update: {100*best_sb3['update_median_ms']/total_gen:.1f}% of wall time")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vsss-sim perf benchmark — vmap, VSSVecEnv, SB3 PPO."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Per-call measurement runs for vmap/VSSVecEnv (default: 1000).",
    )
    parser.add_argument(
        "--gens", type=int, default=12,
        help="PPO generations per SB3 config (default: 12).",
    )
    parser.add_argument(
        "--batches", type=int, nargs="+",
        default=[1, 8, 32, 64, 128, 256, 512, 1024],
        help="Batch sizes for vmap/VSSVecEnv sweep.",
    )
    parser.add_argument(
        "--device", type=str, default="default", choices=["cpu", "gpu", "default"],
        help="Force JAX device (must be the first flag parsed before JAX import).",
    )
    args = parser.parse_args()
    main(args.steps, args.gens, args.batches)
