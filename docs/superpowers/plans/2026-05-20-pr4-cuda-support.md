# PR 4 — CUDA Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the JAX physics backend usable on a CUDA GPU on the Ubuntu RTX 3060 box, with all scaffolding (extras, scripts, tests, docs) done from this Mac session — the user runs the actual GPU install + benchmarks on Ubuntu and the numbers are folded back into the docs.

**Architecture:** All changes are additive. No existing physics, env, or adapter code is modified — they are already device-agnostic. We add a `cuda` extras group to `pyproject.toml`, a `scripts/check_gpu.py` diagnostic, a `--device` flag + tiny `sys.argv` pre-parse in `scripts/bench_backends.py`, a device-info print in `scripts/smoke.py`, four conditional CUDA tests that auto-skip on Mac, and an Ubuntu install/verify recipe in `docs/cuda-setup.md`. A deferred final task captures real RTX 3060 numbers into the docs after the user runs the recipe on Ubuntu.

**Tech Stack:** Python 3.14, JAX (`jax[cuda12]` on Ubuntu), pytest, Gymnasium, SB3. No new runtime deps beyond `jax[cuda12]`.

**Working directory:** This plan is being executed inside the `pr4-cuda-support` worktree at `.claude/worktrees/pr4-cuda-support/`. All paths below are relative to that worktree root unless absolute.

**Reference spec:** `docs/superpowers/specs/2026-05-20-pr4-cuda-support-design.md`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `cuda` extras group; register the `cuda` pytest marker. |
| `tests/physics/test_cuda.py` | Create | 4 CUDA-only tests, module-level `pytestmark` with `skipif(not HAS_CUDA)`. |
| `scripts/check_gpu.py` | Create | Standalone diagnostic; prints JAX device info, exit 0 iff a GPU is visible. |
| `scripts/bench_backends.py` | Modify | Pre-parse `--device` from `sys.argv` (set `JAX_PLATFORMS` before `import jax`), add `--device` argparse flag. Existing device-print on line 134 is already in place. |
| `scripts/smoke.py` | Modify | Add `import jax` and print devices at start of `main()`. No flag — `JAX_PLATFORMS` env var is the override path. |
| `docs/cuda-setup.md` | Create | Ubuntu install + verify + benchmark recipe with troubleshooting section. |
| `docs/vsss-sim-progress-2026-05.md` *(outside repo: `~/dev/personal/docs/`)* | Modify (Task 7) | Add measured CUDA throughput rows after Ubuntu run. |
| `CLAUDE.md` | Modify (Task 7) | Replace the "CUDA on Ubuntu RTX 3060" open-thread bullet with measured results. |

---

## Task 1: Add `cuda` extras group and register `cuda` pytest marker

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current `pyproject.toml`**

Read the file to confirm current contents:

```bash
cat pyproject.toml
```

Expected (relevant excerpt):

```toml
[project.optional-dependencies]
render = ["pygame>=2.4"]
gpu   = ["torch>=2.0"]
jax   = ["jax>=0.4"]
dev   = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
```

- [ ] **Step 2: Add the `cuda` extras group**

Edit `pyproject.toml`, replacing the `[project.optional-dependencies]` block with:

```toml
[project.optional-dependencies]
render = ["pygame>=2.4"]
gpu    = ["torch>=2.0"]
jax    = ["jax>=0.4"]
cuda   = ["jax[cuda12]>=0.4"]
dev    = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.4"]
```

(Only the `cuda = ...` line is new; the others get realigned for column-equality, optional.)

- [ ] **Step 3: Register the `cuda` pytest marker**

Edit `pyproject.toml`, replacing the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
markers   = [
    "cuda: requires a CUDA-capable JAX device (skipped otherwise)",
]
```

- [ ] **Step 4: Verify the marker is registered**

Run:

```bash
pytest --markers | grep -A1 "cuda"
```

Expected output includes:

```
@pytest.mark.cuda: requires a CUDA-capable JAX device (skipped otherwise)
```

- [ ] **Step 5: Verify the test suite still passes**

Run:

```bash
pytest
```

Expected: previous count passing (147 per CLAUDE.md), zero warnings. If you see `PytestUnknownMarkWarning` from any pre-existing test, the marker registration is incomplete — re-read Step 3.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "feat(packaging): add [cuda] extra and register cuda pytest marker"
```

---

## Task 2: Write the conditional CUDA test suite

**Files:**
- Create: `tests/physics/test_cuda.py`

- [ ] **Step 1: Confirm there is no GPU on this machine**

Run:

```bash
python -c "import jax; print(jax.devices()); print(any(d.platform == 'gpu' for d in jax.devices()))"
```

Expected on Mac M4 Pro: a list of CPU devices and `False`. This is the precondition that makes the new tests skip cleanly.

- [ ] **Step 2: Create `tests/physics/test_cuda.py`**

Write the complete file:

```python
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
```

- [ ] **Step 3: Verify tests are collected and skip cleanly on Mac**

Run:

```bash
pytest tests/physics/test_cuda.py -v
```

Expected:

```
tests/physics/test_cuda.py::test_jax_devices_includes_cuda SKIPPED (no CUDA device available)
tests/physics/test_cuda.py::test_step_runs_on_cuda SKIPPED (no CUDA device available)
tests/physics/test_cuda.py::test_step_parity_cpu_vs_gpu SKIPPED (no CUDA device available)
tests/physics/test_cuda.py::test_vmap_step_on_cuda SKIPPED (no CUDA device available)
======================== 4 skipped in <time> ========================
```

If any test errors (e.g. import failure), fix and re-run. If the skip reason is missing or wrong, re-check `pytestmark` in Step 2.

- [ ] **Step 4: Verify `-m cuda` and `-m "not cuda"` filters work**

```bash
pytest -m cuda 2>&1 | tail -3
```

Expected: `4 skipped, 147 deselected` (or similar — only the 4 cuda-marked tests are selected, and they skip because there's no GPU).

```bash
pytest -m "not cuda" 2>&1 | tail -3
```

Expected: `147 passed, 4 deselected`, zero warnings.

- [ ] **Step 5: Verify full suite is clean**

```bash
pytest
```

Expected: 147 passed, 4 skipped, 0 warnings. If you see a `PytestUnknownMarkWarning`, Task 1 Step 3 is incomplete — go back.

- [ ] **Step 6: Commit**

```bash
git add tests/physics/test_cuda.py
git commit -m "test(physics): conditional CUDA test suite (skips when no GPU)"
```

---

## Task 3: Create the `scripts/check_gpu.py` diagnostic

**Files:**
- Create: `scripts/check_gpu.py`

- [ ] **Step 1: Create `scripts/check_gpu.py`**

Write the file:

```python
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
```

- [ ] **Step 2: Make the script executable and run it**

```bash
chmod +x scripts/check_gpu.py
python scripts/check_gpu.py
echo "exit code: $?"
```

Expected on Mac: prints versions, lists `[CpuDevice(...)]`, prints "step ran on {CpuDevice(...)}", prints "no GPU detected", exit code `1`.

- [ ] **Step 3: Confirm script imports cleanly with stale bytecode**

```bash
find . -name __pycache__ -type d -path '*/scripts/*' -prune -exec rm -rf {} + 2>/dev/null; python scripts/check_gpu.py >/dev/null; echo $?
```

Expected: `1` (no second-import surprises).

- [ ] **Step 4: Commit**

```bash
git add scripts/check_gpu.py
git commit -m "feat(scripts): add check_gpu.py diagnostic for JAX device visibility"
```

---

## Task 4: Add `--device` flag and `sys.argv` pre-parse to `scripts/bench_backends.py`

**Files:**
- Modify: `scripts/bench_backends.py`

> **Note:** The existing line `print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")` already exists at the top of `main()` (currently line 134). Do not duplicate it.

- [ ] **Step 1: Read the current top of the file**

```bash
sed -n '1,20p' scripts/bench_backends.py
```

Expected: module docstring followed by `from __future__ import annotations`, then `import argparse`, `import time`, `import jax`, etc.

- [ ] **Step 2: Insert the `JAX_PLATFORMS` pre-parse above the `import jax` line**

Open `scripts/bench_backends.py`. Find this block at the top of the module (after the docstring and `from __future__ import annotations`):

```python
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
```

Replace it with:

```python
import argparse
import os
import sys
import time

# --device pre-parse: must run before `import jax` so JAX_PLATFORMS takes effect.
# Only handles the `--device VALUE` form; argparse below re-validates.
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
        if _v in ("cpu", "gpu"):
            os.environ["JAX_PLATFORMS"] = "cpu" if _v == "cpu" else "cuda"
        break

import jax
import jax.numpy as jnp
import numpy as np
```

- [ ] **Step 3: Add the `--device` argparse flag**

Still in `scripts/bench_backends.py`, find the `if __name__ == "__main__":` block at the bottom (around line 169–183). It currently ends with:

```python
    parser.add_argument(
        "--sb3-timesteps", type=int, default=4096,
        help="Timesteps per SB3 measurement (default: 4096).",
    )
    args = parser.parse_args()
    main(args.steps, args.seed, args.sb3, args.sb3_timesteps)
```

Insert a new `--device` argument **before** `args = parser.parse_args()`:

```python
    parser.add_argument(
        "--sb3-timesteps", type=int, default=4096,
        help="Timesteps per SB3 measurement (default: 4096).",
    )
    parser.add_argument(
        "--device", type=str, default="default",
        choices=["cpu", "gpu", "default"],
        help=(
            "Force JAX onto a specific device. cpu=JAX_PLATFORMS=cpu, "
            "gpu=JAX_PLATFORMS=cuda, default=let JAX pick (env var honored). "
            "Must come before --sb3 / --steps when used."
        ),
    )
    args = parser.parse_args()
    main(args.steps, args.seed, args.sb3, args.sb3_timesteps)
```

`main()` does not need to take a new argument — the pre-parse already set `JAX_PLATFORMS`, and the existing device-print on line 134 (now shifted lower by 6–8 lines) reports the effective device.

- [ ] **Step 4: Verify the default invocation still runs**

Run a quick benchmark with very few steps:

```bash
python scripts/bench_backends.py --steps 50 2>&1 | head -5
```

Expected: starts with `JAX devices: [CpuDevice(...)]   default backend: cpu` on Mac. Exit code 0.

- [ ] **Step 5: Verify `--device cpu` runs and the preamble reports CPU**

```bash
python scripts/bench_backends.py --steps 50 --device cpu 2>&1 | head -2
```

Expected: first line contains `default backend: cpu`.

- [ ] **Step 6: Verify the `--device gpu` env-var mapping in isolation**

Run the pre-parse snippet directly (does not touch `bench_backends.py`), so we can check `JAX_PLATFORMS=cuda` mapping without triggering JAX initialisation on Mac:

```bash
python -c "
import os, sys
sys.argv = ['x', '--device', 'gpu']
for _i, _a in enumerate(sys.argv):
    if _a == '--device' and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
        if _v in ('cpu', 'gpu'):
            os.environ['JAX_PLATFORMS'] = 'cpu' if _v == 'cpu' else 'cuda'
        break
print('JAX_PLATFORMS:', os.environ.get('JAX_PLATFORMS'))
"
```

Expected: `JAX_PLATFORMS: cuda`.

(On Mac, running `python scripts/bench_backends.py --device gpu` would force JAX to look for CUDA, fail to find it, and either fall back to CPU with a warning or raise — that's the user's signal to switch to Ubuntu. We do not test that path on Mac.)

- [ ] **Step 7: Verify `--help` shows the new flag**

```bash
python scripts/bench_backends.py --help | grep -A2 "device"
```

Expected: the new `--device {cpu,gpu,default}` line appears.

- [ ] **Step 8: Run the test suite to confirm nothing broke**

```bash
pytest -q 2>&1 | tail -5
```

Expected: 147 passed, 4 skipped, 0 warnings.

- [ ] **Step 9: Commit**

```bash
git add scripts/bench_backends.py
git commit -m "feat(bench): add --device {cpu,gpu,default} flag for JAX_PLATFORMS"
```

---

## Task 5: Add device preamble to `scripts/smoke.py`

**Files:**
- Modify: `scripts/smoke.py`

- [ ] **Step 1: Add `import jax` to the imports block**

Open `scripts/smoke.py`. Find the imports block (around lines 24–35):

```python
from __future__ import annotations

import argparse

import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3
```

Replace with (add `import jax` in the third-party block):

```python
from __future__ import annotations

import argparse

import jax
import mlflow
import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3
```

- [ ] **Step 2: Print devices at the start of `main()`**

Find the `main()` function (around line 96). It currently starts with:

```python
def main(seed: int, render: bool, fps: float | None, timesteps: int,
         backend: str | None, num_envs: int) -> None:
    mlflow.set_experiment("vsss-smoke")
```

Insert the device print as the very first line of the body:

```python
def main(seed: int, render: bool, fps: float | None, timesteps: int,
         backend: str | None, num_envs: int) -> None:
    print(f"JAX devices: {jax.devices()}   default backend: {jax.default_backend()}")
    mlflow.set_experiment("vsss-smoke")
```

- [ ] **Step 3: Verify the print fires on a tiny smoke run**

```bash
python scripts/smoke.py --timesteps 200 --num-envs 1 2>&1 | head -3
```

Expected: first line of output is `JAX devices: [CpuDevice(...)]   default backend: cpu` on Mac. The rest of the smoke run continues as normal.

- [ ] **Step 4: Verify the test suite is still clean**

```bash
pytest -q 2>&1 | tail -3
```

Expected: 147 passed, 4 skipped, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke.py
git commit -m "feat(smoke): log JAX devices and default backend at startup"
```

---

## Task 6: Write `docs/cuda-setup.md`

**Files:**
- Create: `docs/cuda-setup.md`

- [ ] **Step 1: Create `docs/cuda-setup.md`**

Write the file:

````markdown
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
````

- [ ] **Step 2: Verify the doc renders and has no broken internal links**

```bash
grep -E "\(#[a-z-]+\)" docs/cuda-setup.md
```

Expected: only `(#troubleshooting)` appears (the one cross-reference inside the doc), and the corresponding `## Troubleshooting` heading exists.

- [ ] **Step 3: Commit**

```bash
git add docs/cuda-setup.md
git commit -m "docs(cuda): add Ubuntu RTX 3060 install + verify + bench recipe"
```

---

## Task 7 (deferred): capture Ubuntu RTX 3060 benchmark numbers

> **This task is performed after the user has run the recipe on the Ubuntu box and pasted back the output.** If you (the agentic worker) reach this task before that output is available, stop and prompt the user. Do not invent numbers.

**Files:**
- Modify: `~/dev/personal/docs/vsss-sim-progress-2026-05.md` (in the user's Obsidian vault, **outside this repo**)
- Modify: `CLAUDE.md` (repo root)

- [ ] **Step 1: Request the benchmark output from the user**

Tell the user explicitly:

> "PR 4 scaffolding is complete. Please run on the Ubuntu RTX 3060 box:
>
> ```bash
> cd <repo>
> pip install -e ".[dev,jax,cuda]"
> python scripts/check_gpu.py
> pytest -m cuda
> python scripts/bench_backends.py --sb3 | tee bench-ubuntu-cuda.txt
> ```
>
> Then paste the contents of `bench-ubuntu-cuda.txt` (or just the throughput tables and SB3 fps numbers) back here."

- [ ] **Step 2: Update the throughput table in the progress doc**

Open `~/dev/personal/docs/vsss-sim-progress-2026-05.md`. Find the "Measured throughput" section's first table. Add new rows underneath the existing CPU rows. Example structure:

```markdown
| Setup | fps | vs numpy |
|---|---|---|
| numpy single-env | 3.5k | 1× |
| JAX single-env (jit) | 32k | 9× |
| `jax.vmap` batch=256 (Mac CPU) | 300k | 88× |
| `VSSVecEnv` batch=256 (Mac CPU) | 120k | 34× |
| `jax.vmap` batch=256 (RTX 3060) | <FILL IN> | <FILL IN> |
| `VSSVecEnv` batch=256 (RTX 3060) | <FILL IN> | <FILL IN> |
```

Fill in `<FILL IN>` from the user's pasted output. Compute the `vs numpy` column as `<gpu_fps> / 3500`.

Repeat for the SB3 PPO table — add `num_envs=256 (RTX 3060)` rows for the `n_steps=32` and `n_steps=512` configs.

- [ ] **Step 3: Update the "Open threads" bullet in `CLAUDE.md`**

Find the bullet that begins `**CUDA on Ubuntu RTX 3060** — pip install -U "jax[cuda12]"…` in the "Open threads / next directions" section of `CLAUDE.md`. Replace it with a one-line measured result, e.g.:

```markdown
- **CUDA on Ubuntu RTX 3060** — landed in PR 4. RTX 3060 hits ~<X>M env-steps/sec at batch=256 (vs ~300k on Mac CPU). End-to-end SB3 PPO at num_envs=256: ~<Y> fps. See `docs/cuda-setup.md`.
```

- [ ] **Step 4: Commit both docs together**

```bash
# From inside the repo worktree:
git add CLAUDE.md
git commit -m "docs(claude): record measured CUDA throughput from RTX 3060"
```

The vault file lives outside the repo; just save it — the Obsidian vault has its own sync.

- [ ] **Step 5: Hand the worktree back to the user for PR creation**

Tell the user PR 4 is ready to merge: summarise the commits in this worktree (`git log --oneline master..HEAD`) and remind them the branch is `worktree-pr4-cuda-support`, off `master`.

---

## Acceptance criteria (mirrors the spec)

After all tasks complete:

1. `pip install -e ".[dev,jax]"` succeeds on Mac and Ubuntu unchanged.
2. `pip install -e ".[dev,jax,cuda]"` succeeds on Ubuntu, fails cleanly on Mac.
3. `pytest` on Mac reports `147 passed, 4 skipped`, zero warnings.
4. `pytest` on Ubuntu after the CUDA install reports `151 passed`, zero warnings.
5. `python scripts/check_gpu.py` exits 0 on Ubuntu, 1 on Mac.
6. `python scripts/bench_backends.py --sb3` on Ubuntu produces visibly higher fps than on Mac CPU at batch=256.
7. Measured numbers are written into `docs/vsss-sim-progress-2026-05.md` and `CLAUDE.md` (Task 7).
8. `docs/cuda-setup.md` exists and the user has followed it once end-to-end without edits.

---

## Out of scope (not in any task)

- No changes to `src/vsss_sim/physics/*`, `src/vsss_sim/envs/*`, `src/vsss_sim/sb3_adapter.py`, or any test other than the new `test_cuda.py`.
- No `--device` flag on `smoke.py` — `JAX_PLATFORMS` env var is the override.
- No CI GPU runner.
- No `jax-metal` (Mac GPU) work.
- No shared `_has_cuda()` helper module — the 3-line check is inlined in `check_gpu.py` and `test_cuda.py` only.
