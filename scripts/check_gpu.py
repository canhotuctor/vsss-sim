#!/usr/bin/env python3
"""GPU diagnostic for the JAX physics backend.

Prints JAX/jaxlib versions, the default backend, the device list, and runs
one jit(jb.step) call to confirm the step compiles and places output on the
default device.

Exit code:
    0  if at least one CUDA GPU device is visible to JAX.
    1  otherwise.

Usage:
    python scripts/check_gpu.py
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import jaxlib

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb


def main() -> int:
    print(f"jax     {jax.__version__}")
    print(f"jaxlib  {jaxlib.__version__}")
    print(f"backend {jax.default_backend()}")
    print(f"devices {jax.devices()}")

    state = jb.reset_kickoff(jax.random.PRNGKey(0))
    action = jnp.zeros(
        (config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32,
    )
    step = jax.jit(jb.step)
    state, _ = step(state, action)
    jax.block_until_ready(state.robots)
    print(f"step ran on {state.robots.devices()}")

    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    print("GPU detected" if has_gpu else "no GPU detected")
    return 0 if has_gpu else 1


if __name__ == "__main__":
    sys.exit(main())
