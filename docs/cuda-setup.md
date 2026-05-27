# CUDA setup on Ubuntu (NVIDIA RTX 3060)

This recipe gets the JAX physics backend running on a CUDA GPU and captures
benchmark numbers for `docs/vsss-sim-progress-2026-05.md` and `CLAUDE.md`.

> **Scope:** Linux x86_64 with an NVIDIA GPU. On macOS the `[cuda]` extra
> intentionally fails to install — `jax[cuda12]` has no Mac wheel.

## Prerequisites

1. A recent NVIDIA driver (the JAX wheel bundles its own CUDA 12 runtime, so
   you only need the kernel driver — not the full CUDA toolkit).
   See <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu> for the
   current minimum driver version.
2. Confirm the driver is loaded:

   ```bash
   nvidia-smi
   ```

   Expected: a table listing the RTX 3060 and its current memory/utilisation.

## Install

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate   # if not already
pip install -e ".[dev,jax,cuda]"
```

`jax[cuda12]` pulls in the matching `jaxlib` build and the CUDA 12 PJRT
plugin. Wheel size is ~500 MB; expect the install to take a minute.

## Verify

```bash
python scripts/check_gpu.py
```

Expected output (RTX 3060 example):

```
jax     0.4.xx
jaxlib  0.4.xx
backend cuda
devices [CudaDevice(id=0)]
step ran on {CudaDevice(id=0)}
GPU detected
```

Exit code `0` on success. If you see `CpuDevice(0)` instead, JAX did not pick
up the GPU — see [Troubleshooting](#troubleshooting).

## Run the CUDA test suite

```bash
pytest -m cuda
```

Expected: `4 passed`. These tests verify GPU placement, the JIT'd step on
GPU, CPU↔GPU parity within `atol=1e-3`, and the `vmap`'d batched step.

A full `pytest` run on Ubuntu should report:

```
151 passed, 0 warnings
```

(147 pre-existing + 4 CUDA tests now active.)

## Capture benchmark numbers

```bash
python scripts/bench_backends.py --sb3 | tee bench-ubuntu-cuda.txt
```

Then fold the relevant rows into:

- `docs/vsss-sim-progress-2026-05.md` — the "Measured throughput" table gets
  new rows for `JAX vmap batch=256 (CUDA)`, `VSSVecEnv batch=256 (CUDA)`,
  and the SB3 PPO numbers at `num_envs=256` on CUDA.
- `CLAUDE.md` — strike through / replace the "Open threads → CUDA on Ubuntu"
  bullet with measured results.

## Troubleshooting

- **`nvidia-smi` works but `check_gpu.py` shows `CpuDevice`.** The driver may
  be too old for the JAX CUDA plugin. Compare against the minimum on
  <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>.
- **`pip install` fails on `jax[cuda12]`.** You're on macOS or on a Linux
  system without an x86_64 wheel match. The `cuda` extra is Linux x86_64
  only.
- **`JAX_PLATFORMS=cpu` was set in your environment** (e.g. left over from a
  Mac shell rc). `check_gpu.py` will report `backend cpu` regardless of the
  GPU. Unset it: `unset JAX_PLATFORMS`.
- **`pytest -m cuda` reports 4 skipped on Ubuntu.** `HAS_CUDA` in
  `tests/physics/test_cuda.py` came out `False`. Re-run `check_gpu.py` —
  fix whatever it reports first.
- **CPU↔GPU parity test (`test_step_parity_cpu_vs_gpu`) fails by a small
  margin.** Widen `atol` from `1e-3` to `1e-2` in that test and add a one-
  line docstring note explaining why. GPU reductions are non-deterministic
  by default; the user's spec accepts this as part of the Ubuntu-run review.
