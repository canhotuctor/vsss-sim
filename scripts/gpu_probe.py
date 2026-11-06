#!/usr/bin/env python3
"""Quick check of whether a GPU (e.g. a GTX 1080 Ti) works with JAX here, and how fast.

Reports the driver/GPU (via nvidia-smi), whether JAX sees CUDA, physics throughput
at a few batch sizes, and the largest batch that fits in memory. Prints one JSON
line at the end so two machines can be compared.

    JAX_PLATFORMS=cuda python scripts/gpu_probe.py
    JAX_PLATFORMS=cuda python scripts/gpu_probe.py --batches 1024 4096 16384 --max-batch 262144
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time


def nvidia_smi() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"error": "nvidia-smi not found (no NVIDIA driver on PATH)"}
    fields = "name,memory.total,driver_version,compute_cap,clocks.max.sm"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {"error": out.stderr.strip()}
    return dict(zip(fields.split(","), [v.strip() for v in out.stdout.splitlines()[0].split(",")]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--max-batch", type=int, default=262144, help="Upper limit for the memory search.")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    report: dict = {"nvidia_smi": nvidia_smi()}
    print("nvidia-smi:", report["nvidia_smi"])

    import jax
    import jax.numpy as jnp
    import jaxlib

    from vsss_sim import config
    from vsss_sim.physics import jax_backend as jb

    report.update(jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(),
                  devices=[str(d) for d in jax.devices()])
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("JAX is NOT on the GPU. Set JAX_PLATFORMS=cuda; if it then errors, the installed"
              " JAX CUDA wheel probably does not support this GPU's compute capability.")
        print(json.dumps(report))
        return 1

    step = jax.jit(jax.vmap(jb.step))

    def run(batch: int, steps: int) -> float:
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        state = jax.jit(jax.vmap(jb.reset_kickoff))(keys)
        action = jnp.zeros((batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
        state, _ = step(state, action)  # compile
        jax.block_until_ready(state)
        start = time.perf_counter()
        for _ in range(steps):
            state, _ = step(state, action)
        jax.block_until_ready(state)
        return batch * steps / (time.perf_counter() - start)

    report["throughput_env_steps_per_s"] = {}
    for batch in args.batches:
        rate = run(batch, args.steps)
        report["throughput_env_steps_per_s"][batch] = round(rate)
        print(f"batch {batch:>7}: {rate:12,.0f} env-steps/s")

    largest, batch = 0, args.batches[0]
    while batch <= args.max_batch:
        try:
            run(batch, 2)
            largest = batch
            batch *= 2
        except Exception as error:  # XLA raises RuntimeError / XlaRuntimeError on OOM
            print(f"batch {batch} failed: {str(error).splitlines()[0][:100]}")
            break
    report["largest_batch_that_fits"] = largest
    stats = jax.devices()[0].memory_stats() or {}
    report["peak_bytes_in_use"] = stats.get("peak_bytes_in_use")
    print(f"largest batch that fits: {largest}  peak bytes: {report['peak_bytes_in_use']}")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
