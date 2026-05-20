# PR 4 — CUDA support on Ubuntu RTX 3060

**Date:** 2026-05-20
**Status:** Draft (awaiting user review)
**Scope:** Make the JAX backend usable on a CUDA GPU on the Ubuntu RTX 3060 box. All deliverables are Mac-side scaffolding; the user runs the actual install + benchmarks on Ubuntu and the numbers are folded back into the docs.

---

## Goals

1. Install path — `pip install -e ".[cuda]"` brings in `jax[cuda12]` on Ubuntu x86_64.
2. Verifiability — one command answers "is JAX seeing the GPU?".
3. Observability — every `bench_backends.py` / `smoke.py` run records which devices JAX actually used.
4. Test safety net — a small suite of CUDA-only tests that auto-skip on Mac and run on Ubuntu, catching install regressions and confirming numpy↔CPU-JAX↔GPU-JAX parity.
5. Documented recipe — `docs/cuda-setup.md` walks the user from clean Ubuntu through "benchmark numbers captured".

---

## Non-Goals

- No changes to `VSSEnv`, `VSSVecEnv`, the SB3 adapter, or the JAX backend itself. They are already device-agnostic by construction.
- No CI/GitHub Actions GPU runner — out of scope.
- No tests of training convergence on GPU — too slow for unit tests; the bench script is the answer.
- No automatic device selection logic in user code; we rely on JAX's default device choice and the `JAX_PLATFORMS` env var.
- No `jax-metal` work on Mac — `[[vsss-sim-progress-2026-05]]` records that path as parked.

---

## Hardware Targets

| Machine | GPU | What this PR does |
|---|---|---|
| Ubuntu laptop | NVIDIA RTX 3060 (CUDA 12) | Installs `jax[cuda12]`, runs CUDA tests, captures benchmark numbers. |
| Mac M4 Pro | Apple Silicon (Metal) | `[cuda]` extra refuses to install (intended); CUDA tests collected but skipped; everything else works on CPU JAX as today. |

---

## Deliverables

### 1. `pyproject.toml` — add `cuda` extra and register `cuda` marker

```toml
[project.optional-dependencies]
render = ["pygame>=2.4"]
gpu    = ["torch>=2.0"]
jax    = ["jax>=0.4"]
cuda   = ["jax[cuda12]>=0.4"]
dev    = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
markers   = [
    "cuda: requires a CUDA-capable JAX device (skipped otherwise)",
]
```

`cuda` is a sibling of `jax`, not a replacement. `pip install -e ".[jax]"` remains the cross-platform default; `pip install -e ".[jax,cuda]"` is the Ubuntu opt-in. On Mac, pip will fail with "no matching distribution" for `jax[cuda12]` — that is the intended signal that the extra is Linux-only.

### 2. `scripts/check_gpu.py` — diagnostic

Standalone script. No CLI args. Prints:

- `jax.__version__`, `jaxlib.__version__`
- `jax.default_backend()`
- `jax.devices()` (full list)
- One `jit(jb.step)` call on the default device, with `.devices()` of the result printed to confirm placement.

Exit code: `0` if at least one device has `platform == "gpu"`, else `1`. Useful both as a manual check and as a CI/dev sanity gate.

```python
import sys
import jax
import jax.numpy as jnp
from vsss_sim import config
from vsss_sim.physics import jax_backend as jb

def main() -> int:
    import jaxlib
    print(f"jax     {jax.__version__}")
    print(f"jaxlib  {jaxlib.__version__}")
    print(f"backend {jax.default_backend()}")
    print(f"devices {jax.devices()}")

    state = jb.reset_kickoff(jax.random.PRNGKey(0))
    action = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
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

### 3. `scripts/bench_backends.py` — device-aware preamble + `--device` flag

Two changes:

- **Preamble print** at the start of `main()`, after argparsing: `print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")`. Captured in stdout so any benchmark log records the hardware that produced it.
- **`--device {cpu,gpu,default}` flag**: when set to `cpu`, sets `os.environ["JAX_PLATFORMS"] = "cpu"`; when set to `gpu`, sets `os.environ["JAX_PLATFORMS"] = "cuda"` (the canonical JAX platform name for NVIDIA — `gpu` on the CLI is the user-friendly alias). Set *before* `import jax` runs. Default is `default` — leaves `JAX_PLATFORMS` untouched so JAX picks the first available device (currently the GPU when present).

Implementation note: `JAX_PLATFORMS` must be set before `jax` is imported, but the existing top-level `import jax` is used by module-scope helper functions (`_bench_jax_single`, `_bench_jax_vmap`). Rather than restructure those, the module will do a tiny pre-parse of `sys.argv` for `--device` at the very top of the file, set `os.environ["JAX_PLATFORMS"]` accordingly, then run the existing imports unchanged:

```python
import os, sys
for i, a in enumerate(sys.argv):
    if a == "--device" and i + 1 < len(sys.argv):
        v = sys.argv[i + 1]
        if v in ("cpu", "gpu"):
            os.environ["JAX_PLATFORMS"] = "cpu" if v == "cpu" else "cuda"
        break

import jax          # safe to import now
import jax.numpy as jnp
# … rest of file unchanged
```

The full `argparse` runs later in `main()` and re-validates `--device`; the pre-parse only exists to set the env var. This is six lines of code and avoids restructuring the rest of the module.

### 4. `scripts/smoke.py` — same preamble, no behavior change

Add the same `JAX backend / devices` print at startup. No CLI flag — `JAX_PLATFORMS=cuda python scripts/smoke.py …` is the Ubuntu-side override.

### 5. `tests/physics/test_cuda.py` — conditional CUDA tests

New file. Module-level detection:

```python
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
```

Four tests:

1. **`test_jax_devices_includes_cuda`** — at least one device with `platform == "gpu"`.
2. **`test_step_runs_on_cuda`** — call `jit(jb.step)` once; assert the output's `.robots.devices()` contains a `platform == "gpu"` device.
3. **`test_step_parity_cpu_vs_gpu`** — same `PRNGKey(0)` and same zero action, 100 steps; positions on CPU and GPU agree within `atol=1e-3` (mirrors the existing numpy↔JAX parity tolerance). CPU run uses `jax.device_put(state, jax.devices("cpu")[0])`.
4. **`test_vmap_step_on_cuda`** — `jit(vmap(jb.step))` at batch=64, one step; assert output shape `(64, …)` and that the output device is a GPU.

Mac behavior: all four collected, all four skipped. Ubuntu behavior: all four run and pass.

A `pytest -m cuda` run executes only these tests; `pytest -m "not cuda"` excludes them (useful in case the GPU is busy with training).

### 6. `docs/cuda-setup.md` — install and verify recipe

Sections:

- **Prerequisites** — NVIDIA driver and CUDA 12 runtime; link to NVIDIA's instructions, do not duplicate. Confirm with `nvidia-smi`.
- **Install** — `pip install -e ".[dev,jax,cuda]"` from the project root.
- **Verify** — `python scripts/check_gpu.py` should exit 0 and print the RTX 3060.
- **Run the CUDA test suite** — `pytest -m cuda` should report 4 passed.
- **Capture benchmarks** — `python scripts/bench_backends.py --sb3` and paste numbers into `docs/vsss-sim-progress-2026-05.md` and `CLAUDE.md`.
- **Troubleshooting** — common failure modes: wrong CUDA major version, missing driver, `JAX_PLATFORMS=cpu` set in the environment.

### 7. Numbers capture (deferred until Ubuntu run)

After the user runs the install + bench on the Ubuntu box and pastes the output back, a follow-up commit (still in this PR's branch) folds the measured fps into:

- `docs/vsss-sim-progress-2026-05.md` — the throughput table gains a `JAX vmap batch=256 (CUDA)` row and a `VSSVecEnv batch=256 (CUDA)` row, plus the corresponding SB3 row.
- `CLAUDE.md` — the "Open threads" bullet for CUDA gets struck through / replaced with a measured-result note.

---

## Out-of-scope choices, with reasoning

- **No shared `_has_cuda()` helper module.** The detection is `any(d.platform == "gpu" for d in jax.devices())` — a single line. It appears in `check_gpu.py` and `test_cuda.py` and nowhere else. Extracting it would add an import path without saving code.
- **No `--device` flag on `smoke.py`.** `JAX_PLATFORMS=cuda` env var already works at the process boundary and avoids the import-order workaround on a script that's mostly used interactively.
- **No CI step for CUDA tests.** GitHub-hosted runners don't have GPUs; self-hosted is out of scope for this PR. The test suite is for the developer machine.
- **No torch/`gpu` extra changes.** PR 4 is JAX-only; the `gpu = ["torch>=2.0"]` extra is left alone (used by SB3's PyTorch policy on whatever device PyTorch picks).

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `jax[cuda12]` resolves but `jax.devices()` returns only CPU (driver/CUDA mismatch). | `check_gpu.py` exit code is the contract; the recipe in `docs/cuda-setup.md` makes it the first step after install. |
| Test-time parity differs from CPU by more than `atol=1e-3` due to non-determinism in GPU reductions. | Use the same tolerance as the existing numpy↔JAX parity test. If GPU diverges further, widen to `atol=1e-2` and note it in the test docstring — accept this as part of the Ubuntu-run review. |
| `JAX_PLATFORMS` set in the env at script start interacts badly with `--device default`. | `--device default` does not touch the env var; user override wins. Documented in `cuda-setup.md` troubleshooting. |
| Pre-parsing `sys.argv` for `--device` ahead of argparse silently mishandles unusual invocations (e.g. `--device=gpu` with `=` syntax, or a flag value supplied via env-var-style). | The pre-parse only handles the `--device VALUE` form. If a value is missing or unrecognised it leaves `JAX_PLATFORMS` alone, and the real `argparse` in `main()` validates the flag and errors out properly. Documented in a code comment. |

---

## Acceptance criteria

PR 4 is done when, on a freshly checked-out branch:

1. `pip install -e ".[dev,jax]"` succeeds on Mac and Ubuntu unchanged.
2. `pip install -e ".[dev,jax,cuda]"` succeeds on Ubuntu, fails cleanly on Mac.
3. `pytest` on Mac reports 147 passed + 4 skipped (or current count + 4 skipped), zero warnings.
4. `pytest` on Ubuntu after the CUDA install reports 151 passed (current count + 4), zero warnings.
5. `python scripts/check_gpu.py` exits 0 on Ubuntu, 1 on Mac.
6. `python scripts/bench_backends.py --sb3` on Ubuntu produces visibly higher fps than on Mac CPU at batch=256.
7. The captured numbers are written into `docs/vsss-sim-progress-2026-05.md` and `CLAUDE.md`.
8. `docs/cuda-setup.md` exists and the user has followed it once end-to-end without edits.
