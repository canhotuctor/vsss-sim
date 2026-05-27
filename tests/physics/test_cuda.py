"""CUDA-only tests for the JAX physics backend.

Skipped automatically on machines without a CUDA-capable JAX device.
On Ubuntu with `jax[cuda12]` installed, these confirm: GPU is visible,
the JIT'd step runs on GPU, CPU↔GPU outputs agree within atol=1e-3,
and the vmap'd batched step works on GPU.

Run only these tests with:
    pytest -m cuda
Skip them explicitly with:
    pytest -m "not cuda"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb

HAS_CUDA = any(d.platform == "gpu" for d in jax.devices())

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device available"),
]


def _zero_action() -> jnp.ndarray:
    return jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)


def test_jax_devices_includes_cuda():
    """At least one of jax.devices() reports platform == 'gpu'."""
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, f"expected at least one GPU device, got {jax.devices()!r}"


def test_step_runs_on_cuda():
    """jit(jb.step) runs and places its output on the GPU."""
    state = jb.reset_kickoff(jax.random.PRNGKey(0))
    step = jax.jit(jb.step)
    state, _ = step(state, _zero_action())
    jax.block_until_ready(state.robots)
    devices = state.robots.devices()
    assert any(d.platform == "gpu" for d in devices), (
        f"expected GPU placement, got {devices!r}"
    )


def test_step_parity_cpu_vs_gpu():
    """Same key, same zero action, 100 steps — CPU and GPU positions match."""
    cpu = jax.devices("cpu")[0]
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, "precondition: GPU available"
    gpu = gpus[0]

    state_cpu = jax.device_put(jb.reset_kickoff(jax.random.PRNGKey(0)), cpu)
    state_gpu = jax.device_put(jb.reset_kickoff(jax.random.PRNGKey(0)), gpu)

    step_cpu = jax.jit(jb.step, device=cpu)
    step_gpu = jax.jit(jb.step, device=gpu)

    a_cpu = jax.device_put(_zero_action(), cpu)
    a_gpu = jax.device_put(_zero_action(), gpu)

    for _ in range(100):
        state_cpu, _ = step_cpu(state_cpu, a_cpu)
        state_gpu, _ = step_gpu(state_gpu, a_gpu)
    jax.block_until_ready(state_cpu.robots)
    jax.block_until_ready(state_gpu.robots)

    # Positions are the first two columns of the robot state.
    pos_cpu = np.asarray(state_cpu.robots[:, :, 0:2])
    pos_gpu = np.asarray(state_gpu.robots[:, :, 0:2])
    assert np.allclose(pos_cpu, pos_gpu, atol=1e-3), (
        f"CPU/GPU position divergence: max abs delta = "
        f"{np.max(np.abs(pos_cpu - pos_gpu))}"
    )


def test_vmap_step_on_cuda():
    """jit(vmap(jb.step)) at batch=64 runs on GPU and returns the right shape."""
    batch = 64
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    states = jax.vmap(jb.reset_kickoff)(keys)
    actions = jnp.zeros(
        (batch, config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32,
    )
    vstep = jax.jit(jax.vmap(jb.step))
    states, _ = vstep(states, actions)
    jax.block_until_ready(states.robots)

    assert states.robots.shape[0] == batch, (
        f"expected leading batch dim {batch}, got shape {states.robots.shape!r}"
    )
    devices = states.robots.devices()
    assert any(d.platform == "gpu" for d in devices), (
        f"expected GPU placement, got {devices!r}"
    )
