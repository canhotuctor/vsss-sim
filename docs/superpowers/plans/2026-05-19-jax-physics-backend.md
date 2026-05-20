# JAX Physics Backend — PR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JAX physics backend that mirrors `numpy_backend.py` semantically, and make `VSSEnv` switchable between backends via constructor kwarg or `VSSS_PHYSICS_BACKEND` env var. Single-env first; batched `VSSVecEnv` lands in a follow-up PR.

**Architecture:**
- `src/vsss_sim/physics/jax_backend.py` is a pure-functional, JIT-compiled mirror of the numpy backend. `SimState` is a `NamedTuple` PyTree (`ball`, `robots`, `score`, `t`). `step(state, actions, dt, sub_steps) -> (state, info)` is jitted. Substeps use `jax.lax.fori_loop`. Pairwise robot collisions iterate the 15 static pairs with `fori_loop`. Ball↔robot collisions iterate the 6 robots with `fori_loop` so impulses chain like in the numpy version. `_ball_wall_collisions` uses `jnp.where`-style branchless logic.
- `physics/__init__.py` exposes `get_backend(name=None) -> module` resolver: kwarg wins, then `VSSS_PHYSICS_BACKEND`, then `"numpy"`. Existing top-level re-exports (`SimState`, `step`, `reset_kickoff`, …) stay so old code is unbroken.
- `VSSEnv(backend=...)` resolves a backend module at construction, stores its `SimState` instance, and dispatches `reset` / `step` via the module. The two backends differ in mutability (numpy mutates in place, jax returns new state), so the env owns a tiny shim that reassigns `self._state` after each step regardless.
- Float precision: JAX uses float32 by default. Tests assert parity against the float64 numpy backend with `atol=1e-3` for positions, `atol=1e-2` for velocities — generous enough for float32 + per-step rounding.

**Tech Stack:** JAX ≥ 0.10 (already installed), NumPy, Gymnasium, pytest. Python 3.14 venv at `/Users/mario.bezerra/dev/personal/vsss-sim/.venv`.

---

## File map

**Create:**
- `src/vsss_sim/physics/jax_backend.py` — full backend mirror.
- `tests/physics/test_jax_backend.py` — semantic-parity tests + backend-specific edge cases.
- `tests/physics/test_backend_resolver.py` — resolver tests.

**Modify:**
- `src/vsss_sim/physics/__init__.py` — add `get_backend`, keep existing re-exports.
- `src/vsss_sim/envs/base.py` — store backend module, expose backend-agnostic state access (small refactor).
- `src/vsss_sim/envs/vsss_3v3.py` — accept `backend` kwarg, dispatch reset/step through the resolved backend.
- `tests/envs/test_env.py` — parametrise over both backends.
- `scripts/smoke.py` — add `--backend` flag (numpy default).
- `pyproject.toml` — add `[jax]` extra.
- `CLAUDE.md` — short note on backend selection.

**Leave alone:** `numpy_backend.py`, `rendering/pygame.py`, `agents/*` (they read numpy arrays; we'll `np.asarray()` at the boundary).

---

## Task 1: Add `[jax]` extra to pyproject

**Files:** Modify `pyproject.toml:24-31`.

- [ ] **Step 1: Add the extra.**

```toml
[project.optional-dependencies]
render = ["pygame>=2.4"]
gpu   = ["torch>=2.0"]
jax   = ["jax>=0.4"]
dev   = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
]
```

- [ ] **Step 2: Verify jax is importable (already installed in `.venv`).**

```bash
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
```
Expected: `0.10.0 [CpuDevice(id=0)]` (on Mac).

- [ ] **Step 3: Commit.**

```bash
git add pyproject.toml
git commit -m "build: add [jax] optional dependency group"
```

---

## Task 2: Backend resolver in `physics/__init__.py`

**Files:**
- Modify: `src/vsss_sim/physics/__init__.py`
- Create: `tests/physics/test_backend_resolver.py`

- [ ] **Step 1: Write failing tests.**

`tests/physics/test_backend_resolver.py`:

```python
"""Tests for the physics backend resolver."""
import os
import pytest

from vsss_sim.physics import get_backend


class TestGetBackend:
    def test_default_is_numpy(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend()
        assert backend.__name__.endswith("numpy_backend")

    def test_explicit_numpy(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend("numpy")
        assert backend.__name__.endswith("numpy_backend")

    def test_explicit_jax(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend("jax")
        assert backend.__name__.endswith("jax_backend")

    def test_env_var_jax(self, monkeypatch):
        monkeypatch.setenv("VSSS_PHYSICS_BACKEND", "jax")
        backend = get_backend()
        assert backend.__name__.endswith("jax_backend")

    def test_kwarg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("VSSS_PHYSICS_BACKEND", "jax")
        backend = get_backend("numpy")
        assert backend.__name__.endswith("numpy_backend")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("isaac")
```

- [ ] **Step 2: Run the tests — they should fail (`get_backend` not defined).**

```bash
.venv/bin/pytest tests/physics/test_backend_resolver.py -v
```
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement the resolver. Replace `src/vsss_sim/physics/__init__.py` with:**

```python
"""Physics package — backend resolver and default re-exports.

A backend is any module that defines:
    SimState, step, reset_kickoff,
    _diff_drive, _ball_wall_collisions, _robot_wall_collisions,
    _ball_robot_collisions, _robot_robot_collisions.

Default backend is "numpy". Override with the ``VSSS_PHYSICS_BACKEND`` env var
or by passing ``backend=...`` to ``get_backend()`` / ``VSSEnv``.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Optional

# Keep direct re-exports of the numpy backend so existing imports
# (`from vsss_sim.physics import SimState, step, ...`) keep working.
from .numpy_backend import (
    SimState,
    _ball_robot_collisions,
    _ball_wall_collisions,
    _diff_drive,
    _robot_robot_collisions,
    _robot_wall_collisions,
    reset_kickoff,
    step,
)

_BACKENDS = {
    "numpy": "vsss_sim.physics.numpy_backend",
    "jax": "vsss_sim.physics.jax_backend",
}


def get_backend(name: Optional[str] = None) -> ModuleType:
    """Resolve a physics backend module.

    Priority: explicit ``name`` kwarg > ``VSSS_PHYSICS_BACKEND`` env var > "numpy".
    """
    if name is None:
        name = os.environ.get("VSSS_PHYSICS_BACKEND", "numpy")
    name = name.lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}"
        )
    return importlib.import_module(_BACKENDS[name])


__all__ = [
    "SimState",
    "step",
    "reset_kickoff",
    "_diff_drive",
    "_ball_wall_collisions",
    "_robot_wall_collisions",
    "_ball_robot_collisions",
    "_robot_robot_collisions",
    "get_backend",
]
```

- [ ] **Step 4: Run tests — `test_explicit_jax` and `test_env_var_jax` still fail (no jax_backend module yet). The numpy tests should pass.**

```bash
.venv/bin/pytest tests/physics/test_backend_resolver.py -v
```
Expected: 4 pass, 2 fail with "ModuleNotFoundError: ...jax_backend".

- [ ] **Step 5: Stub `jax_backend.py` so the resolver tests pass.**

Create `src/vsss_sim/physics/jax_backend.py`:

```python
"""JAX physics backend for vsss-sim. See numpy_backend.py for semantics."""
from __future__ import annotations

# Real implementation lands in subsequent tasks.
# This module is import-resolvable now so the backend resolver can find it.
```

- [ ] **Step 6: Run tests — all six pass.**

```bash
.venv/bin/pytest tests/physics/test_backend_resolver.py -v
```

- [ ] **Step 7: Commit.**

```bash
git add src/vsss_sim/physics/__init__.py src/vsss_sim/physics/jax_backend.py tests/physics/test_backend_resolver.py
git commit -m "feat(physics): add backend resolver and jax_backend module stub"
```

---

## Task 3: JAX `SimState` PyTree + converters

**Files:** Modify `src/vsss_sim/physics/jax_backend.py`. Add tests to a new `tests/physics/test_jax_backend.py`.

- [ ] **Step 1: Write failing tests.**

`tests/physics/test_jax_backend.py`:

```python
"""Tests for the JAX physics backend."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb
from vsss_sim.physics.numpy_backend import SimState as NumpySimState


class TestSimState:
    def test_empty_state_shapes(self):
        s = jb.empty_state()
        assert s.ball.shape == (4,)
        assert s.robots.shape == (config.N_TEAMS, config.N_ROBOTS, 6)
        assert s.score.shape == (2,)
        assert s.t.shape == ()

    def test_empty_state_dtype(self):
        s = jb.empty_state()
        assert s.ball.dtype == jnp.float32
        assert s.robots.dtype == jnp.float32
        assert s.score.dtype == jnp.int32
        assert s.t.dtype == jnp.float32

    def test_empty_state_zeros(self):
        s = jb.empty_state()
        assert jnp.all(s.ball == 0)
        assert jnp.all(s.robots == 0)
        assert jnp.all(s.score == 0)
        assert s.t == 0

    def test_is_pytree(self):
        s = jb.empty_state()
        leaves = jax.tree_util.tree_leaves(s)
        # ball, robots, score, t
        assert len(leaves) == 4

    def test_from_numpy_round_trip(self):
        np_s = NumpySimState()
        np_s.ball[:] = [0.1, 0.2, 0.3, 0.4]
        np_s.robots[0, 0, :] = [0.5, -0.5, 1.0, 0.0, 0.0, 0.0]
        np_s.score[:] = [1, 2]
        np_s.t = 1.5

        j_s = jb.from_numpy(np_s)
        np_s2 = jb.to_numpy(j_s)

        assert np.allclose(np_s2.ball, np_s.ball, atol=1e-5)
        assert np.allclose(np_s2.robots, np_s.robots, atol=1e-5)
        assert np.all(np_s2.score == np_s.score)
        assert np_s2.t == pytest.approx(np_s.t, abs=1e-5)
```

- [ ] **Step 2: Run — should all fail with AttributeError.**

```bash
.venv/bin/pytest tests/physics/test_jax_backend.py::TestSimState -v
```

- [ ] **Step 3: Implement `SimState` + converters. Replace `src/vsss_sim/physics/jax_backend.py` with:**

```python
"""JAX physics backend for vsss-sim.

Pure-functional mirror of ``numpy_backend.py``. ``SimState`` is a
:class:`typing.NamedTuple` PyTree — register-free, immutable, ``vmap``-friendly.

Float dtype is ``float32`` (GPU default). Tests assert semantic parity with the
float64 numpy backend within a generous tolerance.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import config
from .numpy_backend import SimState as NumpySimState


# ---------------------------------------------------------------------------
# State container (PyTree)
# ---------------------------------------------------------------------------

class SimState(NamedTuple):
    """Immutable simulation state. Fields are JAX arrays."""

    ball: jnp.ndarray    # (4,)   float32  [x, y, vx, vy]
    robots: jnp.ndarray  # (N_TEAMS, N_ROBOTS, 6) float32 [x, y, theta, vx, vy, omega]
    score: jnp.ndarray   # (2,)   int32
    t: jnp.ndarray       # () scalar float32


def empty_state() -> SimState:
    """Return an all-zero ``SimState``."""
    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32),
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
    )


def from_numpy(np_state: NumpySimState) -> SimState:
    """Convert a NumPy ``SimState`` to a JAX ``SimState``."""
    return SimState(
        ball=jnp.asarray(np_state.ball, dtype=jnp.float32),
        robots=jnp.asarray(np_state.robots, dtype=jnp.float32),
        score=jnp.asarray(np_state.score, dtype=jnp.int32),
        t=jnp.asarray(np_state.t, dtype=jnp.float32),
    )


def to_numpy(state: SimState) -> NumpySimState:
    """Convert a JAX ``SimState`` to a NumPy ``SimState``."""
    return NumpySimState(
        ball=np.asarray(state.ball, dtype=np.float64),
        robots=np.asarray(state.robots, dtype=np.float64),
        score=np.asarray(state.score, dtype=np.int32),
        t=float(state.t),
    )
```

- [ ] **Step 4: Run — all four tests pass.**

```bash
.venv/bin/pytest tests/physics/test_jax_backend.py::TestSimState -v
```

- [ ] **Step 5: Commit.**

```bash
git add src/vsss_sim/physics/jax_backend.py tests/physics/test_jax_backend.py
git commit -m "feat(physics/jax): add SimState PyTree and numpy converters"
```

---

## Task 4: `_diff_drive`

**Files:** Modify `src/vsss_sim/physics/jax_backend.py`. Extend `tests/physics/test_jax_backend.py`.

- [ ] **Step 1: Add failing tests.**

Append to `tests/physics/test_jax_backend.py`:

```python
class TestDiffDrive:
    def test_straight_forward(self):
        vx, vy, omega = jb._diff_drive(jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.0]))
        assert float(vx[0]) == pytest.approx(1.0)
        assert float(vy[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(omega[0]) == pytest.approx(0.0, abs=1e-6)

    def test_rotate_in_place(self):
        v = 0.5
        vx, vy, omega = jb._diff_drive(jnp.array([-v]), jnp.array([v]), jnp.array([0.0]))
        assert float(vx[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(vy[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(omega[0]) == pytest.approx(2 * v / config.ROBOT_WHEELBASE, rel=1e-4)

    def test_vectorised_shape(self):
        v_l = jnp.ones((config.N_TEAMS, config.N_ROBOTS))
        v_r = jnp.ones((config.N_TEAMS, config.N_ROBOTS))
        theta = jnp.zeros((config.N_TEAMS, config.N_ROBOTS))
        vx, vy, omega = jb._diff_drive(v_l, v_r, theta)
        assert vx.shape == (config.N_TEAMS, config.N_ROBOTS)
```

- [ ] **Step 2: Run — fails (`_diff_drive` missing).**

```bash
.venv/bin/pytest tests/physics/test_jax_backend.py::TestDiffDrive -v
```

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Differential-drive kinematics
# ---------------------------------------------------------------------------

def _diff_drive(
    v_left: jnp.ndarray,
    v_right: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert wheel speeds to body velocities (vectorised)."""
    v = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / config.ROBOT_WHEELBASE
    vx = v * jnp.cos(theta)
    vy = v * jnp.sin(theta)
    return vx, vy, omega
```

- [ ] **Step 4: Run — all pass.**

- [ ] **Step 5: Commit.**

```bash
git add src/vsss_sim/physics/jax_backend.py tests/physics/test_jax_backend.py
git commit -m "feat(physics/jax): add _diff_drive"
```

---

## Task 5: `reset_kickoff`

**Files:** Modify `src/vsss_sim/physics/jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing tests.**

```python
class TestResetKickoff:
    def test_ball_at_centre(self):
        key = jax.random.PRNGKey(0)
        s = jb.reset_kickoff(key)
        assert float(s.ball[0]) == pytest.approx(0.0)
        assert float(s.ball[1]) == pytest.approx(0.0)
        assert jnp.all(s.ball[2:4] == 0.0)

    def test_blue_on_left_yellow_on_right(self):
        key = jax.random.PRNGKey(0)
        s = jb.reset_kickoff(key)
        assert jnp.all(s.robots[config.TEAM_BLUE, :, 0] < 0)
        assert jnp.all(s.robots[config.TEAM_YELLOW, :, 0] > 0)

    def test_robots_within_field(self):
        key = jax.random.PRNGKey(42)
        s = jb.reset_kickoff(key)
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert jnp.all(jnp.abs(s.robots[:, :, 0]) <= half_l)
        assert jnp.all(jnp.abs(s.robots[:, :, 1]) <= half_w)

    def test_score_and_velocities_zero(self):
        key = jax.random.PRNGKey(0)
        s = jb.reset_kickoff(key)
        assert jnp.all(s.score == 0)
        assert jnp.all(s.robots[:, :, 3:6] == 0)

    def test_different_keys_give_different_states(self):
        s0 = jb.reset_kickoff(jax.random.PRNGKey(0))
        s1 = jb.reset_kickoff(jax.random.PRNGKey(1))
        # Jitter should make robot positions differ
        assert not jnp.allclose(s0.robots[:, :, 0:2], s1.robots[:, :, 0:2])

    def test_deterministic_for_same_key(self):
        key = jax.random.PRNGKey(7)
        s0 = jb.reset_kickoff(key)
        s1 = jb.reset_kickoff(key)
        assert jnp.allclose(s0.robots, s1.robots)
        assert jnp.allclose(s0.ball, s1.ball)
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

_BLUE_STARTS = jnp.array(
    [[-0.55, 0.0], [-0.30, 0.30], [-0.30, -0.30]], dtype=jnp.float32
)
_YELLOW_STARTS = jnp.array(
    [[0.55, 0.0], [0.30, -0.30], [0.30, 0.30]], dtype=jnp.float32
)


def reset_kickoff(key: jnp.ndarray) -> SimState:
    """Place robots and ball for a standard kickoff (functional)."""
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0 - config.ROBOT_RADIUS)
    clear = jnp.float32(config.KICKOFF_CLEAR_DIST)

    key_b, key_y = jax.random.split(key)
    blue_jitter = jax.random.uniform(key_b, (3, 2), minval=-0.05, maxval=0.05)
    yellow_jitter = jax.random.uniform(key_y, (3, 2), minval=-0.05, maxval=0.05)

    blue = _BLUE_STARTS + blue_jitter
    blue = blue.at[:, 0].set(jnp.clip(blue[:, 0], -half_l, -clear))
    blue = blue.at[:, 1].set(jnp.clip(blue[:, 1], -half_l, half_l))

    yellow = _YELLOW_STARTS + yellow_jitter
    yellow = yellow.at[:, 0].set(jnp.clip(yellow[:, 0], clear, half_l))
    yellow = yellow.at[:, 1].set(jnp.clip(yellow[:, 1], -half_l, half_l))

    robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32)
    robots = robots.at[config.TEAM_BLUE, :, 0:2].set(blue)
    robots = robots.at[config.TEAM_YELLOW, :, 0:2].set(yellow)

    # Face toward the ball at origin
    theta = jnp.arctan2(-robots[:, :, 1], -robots[:, :, 0])
    robots = robots.at[:, :, 2].set(theta)

    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=robots,
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
    )
```

Make sure `import jax` is at the top of the file.

- [ ] **Step 4: Run — all pass.**

- [ ] **Step 5: Commit.**

```bash
git add src/vsss_sim/physics/jax_backend.py tests/physics/test_jax_backend.py
git commit -m "feat(physics/jax): add reset_kickoff (functional, PRNGKey)"
```

---

## Task 6: `_robot_wall_collisions`

**Files:** Modify `src/vsss_sim/physics/jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing tests.**

```python
class TestRobotWallCollisions:
    def test_clamped_inside_field(self):
        s = jb.empty_state()
        s = s._replace(robots=s.robots.at[:, :, 0].set(config.FIELD_LENGTH))
        s = s._replace(robots=s.robots.at[:, :, 1].set(config.FIELD_WIDTH))
        s = jb._robot_wall_collisions(s)
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert jnp.all(s.robots[:, :, 0] <= half_l + 1e-5)
        assert jnp.all(s.robots[:, :, 1] <= half_w + 1e-5)

    def test_velocity_zeroed_at_wall(self):
        s = jb.empty_state()
        s = s._replace(robots=s.robots.at[0, 0, 0].set(config.FIELD_LENGTH))
        s = s._replace(robots=s.robots.at[0, 0, 3].set(1.0))  # moving further out
        s = jb._robot_wall_collisions(s)
        assert float(s.robots[0, 0, 3]) == pytest.approx(0.0)
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Robot–wall collisions (vectorised)
# ---------------------------------------------------------------------------

def _robot_wall_collisions(state: SimState) -> SimState:
    """Clamp robots (OBB) inside field boundaries; zero outward velocity."""
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half = jnp.float32(config.ROBOT_SIZE / 2.0)

    theta = state.robots[:, :, 2]
    extent = half * (jnp.abs(jnp.cos(theta)) + jnp.abs(jnp.sin(theta)))

    lim_x = half_l - extent
    lim_y = half_w - extent

    x = state.robots[:, :, 0]
    y = state.robots[:, :, 1]
    vx = state.robots[:, :, 3]
    vy = state.robots[:, :, 4]

    exceeded_neg_x = x < -lim_x
    exceeded_pos_x = x > lim_x
    exceeded_neg_y = y < -lim_y
    exceeded_pos_y = y > lim_y

    new_x = jnp.clip(x, -lim_x, lim_x)
    new_y = jnp.clip(y, -lim_y, lim_y)

    new_vx = jnp.where(exceeded_neg_x & (vx < 0), 0.0, vx)
    new_vx = jnp.where(exceeded_pos_x & (new_vx > 0), 0.0, new_vx)
    new_vy = jnp.where(exceeded_neg_y & (vy < 0), 0.0, vy)
    new_vy = jnp.where(exceeded_pos_y & (new_vy > 0), 0.0, new_vy)

    robots = state.robots
    robots = robots.at[:, :, 0].set(new_x)
    robots = robots.at[:, :, 1].set(new_y)
    robots = robots.at[:, :, 3].set(new_vx)
    robots = robots.at[:, :, 4].set(new_vy)
    return state._replace(robots=robots)
```

- [ ] **Step 4: Run — pass.**

- [ ] **Step 5: Commit.**

---

## Task 7: `_ball_wall_collisions`

**Files:** Modify `jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing tests.**

```python
class TestBallWallCollisions:
    def test_bounce_top_wall(self):
        s = jb.empty_state()
        s = s._replace(ball=jnp.array(
            [0.0, config.FIELD_WIDTH / 2.0 + config.BALL_RADIUS, 0.0, 1.0],
            dtype=jnp.float32,
        ))
        s2, goal = jb._ball_wall_collisions(s)
        assert int(goal) == 0
        assert float(s2.ball[1]) <= config.FIELD_WIDTH / 2.0
        assert float(s2.ball[3]) < 0

    def test_bounce_bottom_wall(self):
        s = jb.empty_state()
        s = s._replace(ball=jnp.array(
            [0.0, -(config.FIELD_WIDTH / 2.0 + config.BALL_RADIUS), 0.0, -1.0],
            dtype=jnp.float32,
        ))
        s2, goal = jb._ball_wall_collisions(s)
        assert int(goal) == 0
        assert float(s2.ball[3]) > 0

    def test_blue_scores_right_goal(self):
        s = jb.empty_state()
        s = s._replace(ball=jnp.array(
            [config.FIELD_LENGTH / 2.0 + 0.01, 0.0, 0.1, 0.0],
            dtype=jnp.float32,
        ))
        _, goal = jb._ball_wall_collisions(s)
        assert int(goal) == 1

    def test_yellow_scores_left_goal(self):
        s = jb.empty_state()
        s = s._replace(ball=jnp.array(
            [-(config.FIELD_LENGTH / 2.0 + 0.01), 0.0, -0.1, 0.0],
            dtype=jnp.float32,
        ))
        _, goal = jb._ball_wall_collisions(s)
        assert int(goal) == -1

    def test_no_goal_outside_posts(self):
        s = jb.empty_state()
        s = s._replace(ball=jnp.array(
            [config.FIELD_LENGTH / 2.0 + 0.01, config.GOAL_WIDTH, 1.0, 0.0],
            dtype=jnp.float32,
        ))
        s2, goal = jb._ball_wall_collisions(s)
        assert int(goal) == 0
        assert float(s2.ball[2]) < 0
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Ball–wall collisions and goal detection
# ---------------------------------------------------------------------------

def _ball_wall_collisions(state: SimState) -> tuple[SimState, jnp.ndarray]:
    """Reflect ball off field walls and detect goals.

    Returns
    -------
    new_state : SimState
    goal : int32 scalar (+1 blue, -1 yellow, 0 none).
    """
    r = jnp.float32(config.BALL_RADIUS)
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half_goal = jnp.float32(config.GOAL_WIDTH / 2.0)
    goal_depth = jnp.float32(config.GOAL_DEPTH)
    e_wall = jnp.float32(config.BALL_WALL_RESTITUTION)

    bx, by, bvx, bvy = state.ball[0], state.ball[1], state.ball[2], state.ball[3]

    # --- y walls ---
    hit_bot = by - r < -half_w
    hit_top = by + r > half_w
    by = jnp.where(hit_bot, -half_w + r, by)
    by = jnp.where(hit_top, half_w - r, by)
    bvy = jnp.where(hit_bot, jnp.abs(bvy) * e_wall, bvy)
    bvy = jnp.where(hit_top, -jnp.abs(bvy) * e_wall, bvy)

    # --- x walls / goals ---
    in_left_goal_y = jnp.abs(by) <= half_goal
    hit_left = bx - r < -half_l
    hit_right = bx + r > half_l

    left_goal = hit_left & in_left_goal_y
    right_goal = hit_right & in_left_goal_y

    # Left side
    bx = jnp.where(
        left_goal,
        -half_l - goal_depth + r,
        jnp.where(hit_left, -half_l + r, bx),
    )
    bvx = jnp.where(
        hit_left,
        jnp.abs(bvx) * e_wall,
        bvx,
    )

    # Right side
    bx = jnp.where(
        right_goal,
        half_l + goal_depth - r,
        jnp.where(hit_right, half_l - r, bx),
    )
    bvx = jnp.where(
        hit_right,
        -jnp.abs(bvx) * e_wall,
        bvx,
    )

    goal = jnp.where(
        right_goal,
        jnp.int32(1),
        jnp.where(left_goal, jnp.int32(-1), jnp.int32(0)),
    )

    new_ball = jnp.stack([bx, by, bvx, bvy])
    return state._replace(ball=new_ball), goal
```

- [ ] **Step 4: Run — pass.**

- [ ] **Step 5: Commit.**

---

## Task 8: `_ball_robot_collisions`

**Files:** Modify `jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing tests.**

```python
class TestBallRobotCollisions:
    def test_ball_pushed_away(self):
        s = jb.empty_state()
        overlap = 0.001
        dist = config.BALL_RADIUS + config.ROBOT_RADIUS - overlap
        s = s._replace(
            ball=jnp.array([dist, 0.0, -1.0, 0.0], dtype=jnp.float32),
            robots=s.robots.at[0, 0, 0:2].set(jnp.array([0.0, 0.0])),
        )
        s2 = jb._ball_robot_collisions(s)
        # Ball ends up at or past collision distance (no longer penetrating)
        assert float(s2.ball[0]) >= config.BALL_RADIUS + config.ROBOT_RADIUS - 1e-4

    def test_no_collision_when_separated(self):
        s = jb.empty_state()
        s = s._replace(
            ball=jnp.array([0.5, 0.0, 0.0, 0.0], dtype=jnp.float32),
            robots=s.robots.at[0, 0, 0:2].set(jnp.array([-0.5, 0.0])),
        )
        s2 = jb._ball_robot_collisions(s)
        assert jnp.allclose(s2.ball, s.ball)
        assert jnp.allclose(s2.robots, s.robots)
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Ball–robot collisions (circle vs OBB) — fori_loop over 6 robots
# ---------------------------------------------------------------------------

_N_ROBOTS_TOTAL = config.N_TEAMS * config.N_ROBOTS


def _ball_obb_penetration(
    ball_pos: jnp.ndarray, rob_pos: jnp.ndarray, theta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (world-space normal, penetration) for a ball vs square OBB."""
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    r_ball = jnp.float32(config.BALL_RADIUS)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    dx = ball_pos[0] - rob_pos[0]
    dy = ball_pos[1] - rob_pos[1]
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    clamp_x = jnp.clip(local_x, -half, half)
    clamp_y = jnp.clip(local_y, -half, half)
    diff_x = local_x - clamp_x
    diff_y = local_y - clamp_y
    dist = jnp.sqrt(diff_x * diff_x + diff_y * diff_y)

    # Outside-face case
    safe_dist = jnp.where(dist < 1e-9, 1.0, dist)
    lnx_out = diff_x / safe_dist
    lny_out = diff_y / safe_dist
    pen_out = r_ball - dist

    # Inside case
    pen_x_in = half - jnp.abs(local_x)
    pen_y_in = half - jnp.abs(local_y)
    use_x = pen_x_in <= pen_y_in
    lnx_in = jnp.where(use_x, jnp.sign(local_x), 0.0)
    lny_in = jnp.where(use_x, 0.0, jnp.sign(local_y))
    pen_in = jnp.where(use_x, pen_x_in, pen_y_in) + r_ball

    inside = dist < 1e-9
    lnx = jnp.where(inside, lnx_in, lnx_out)
    lny = jnp.where(inside, lny_in, lny_out)
    penetration = jnp.where(inside, pen_in, pen_out)

    nx = cos_t * lnx - sin_t * lny
    ny = sin_t * lnx + cos_t * lny
    return jnp.stack([nx, ny]), penetration


def _resolve_ball_robot_pair(
    ball: jnp.ndarray,  # (4,)
    robot: jnp.ndarray,  # (6,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve one ball-robot collision; returns updated (ball, robot)."""
    ball_pos = ball[0:2]
    ball_vel = ball[2:4]
    rob_pos = robot[0:2]
    rob_vel = robot[3:5]
    theta = robot[2]

    normal, penetration = _ball_obb_penetration(ball_pos, rob_pos, theta)
    collide = penetration > 0.0

    m_b = jnp.float32(config.BALL_MASS)
    m_r = jnp.float32(config.ROBOT_MASS)
    total_m = m_b + m_r
    e = jnp.float32(config.BALL_ROBOT_RESTITUTION)

    # Positional correction (only if collide)
    bp_corr = ball_pos + normal * penetration * (m_r / total_m)
    rp_corr = rob_pos - normal * penetration * (m_b / total_m)

    rel_vel = ball_vel - rob_vel
    vel_along = jnp.dot(rel_vel, normal)
    do_impulse = collide & (vel_along < 0)

    j = -(1.0 + e) * vel_along / (1.0 / m_b + 1.0 / m_r)
    impulse = j * normal

    bv_new = jnp.where(do_impulse, ball_vel + impulse / m_b, ball_vel)
    rv_new = jnp.where(do_impulse, rob_vel - impulse / m_r, rob_vel)

    bp_new = jnp.where(collide, bp_corr, ball_pos)
    rp_new = jnp.where(collide, rp_corr, rob_pos)

    new_ball = jnp.concatenate([bp_new, bv_new])
    new_robot = robot.at[0:2].set(rp_new).at[3:5].set(rv_new)
    return new_ball, new_robot


def _ball_robot_collisions(state: SimState) -> SimState:
    """Resolve elastic collisions between the ball and all 6 robots (sequential)."""
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)

    def body(i, carry):
        ball, robots_flat = carry
        new_ball, new_robot = _resolve_ball_robot_pair(ball, robots_flat[i])
        robots_flat = robots_flat.at[i].set(new_robot)
        return new_ball, robots_flat

    new_ball, new_robots_flat = jax.lax.fori_loop(
        0, _N_ROBOTS_TOTAL, body, (state.ball, robots_flat)
    )
    return state._replace(
        ball=new_ball,
        robots=new_robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6),
    )
```

- [ ] **Step 4: Run — pass.**

- [ ] **Step 5: Commit.**

---

## Task 9: `_robot_robot_collisions`

**Files:** Modify `jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing test.**

```python
class TestRobotRobotCollisions:
    def test_separates_overlapping_pair(self):
        s = jb.empty_state()
        s = s._replace(
            robots=s.robots
                .at[0, 0, 0:2].set(jnp.array([0.0, 0.0]))
                .at[0, 1, 0:2].set(jnp.array([0.03, 0.0]))  # overlapping
        )
        s2 = jb._robot_robot_collisions(s)
        # Distance between centres should be >= ROBOT_SIZE (approx) after resolve.
        d = jnp.linalg.norm(s2.robots[0, 0, 0:2] - s2.robots[0, 1, 0:2])
        assert float(d) >= config.ROBOT_SIZE - 1e-3

    def test_no_change_when_separated(self):
        s = jb.empty_state()
        s = s._replace(
            robots=s.robots
                .at[0, 0, 0:2].set(jnp.array([-0.5, 0.0]))
                .at[0, 1, 0:2].set(jnp.array([0.5, 0.0]))
        )
        before = s.robots.copy()
        s2 = jb._robot_robot_collisions(s)
        assert jnp.allclose(s2.robots, before)
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Robot–robot collisions (OBB vs OBB via SAT) — fori_loop over 15 pairs
# ---------------------------------------------------------------------------

# Pre-compute the 15 unique (i, j) pairs for 6 robots.
_PAIR_I, _PAIR_J = jnp.triu_indices(_N_ROBOTS_TOTAL, k=1)
_N_PAIRS = int(_PAIR_I.shape[0])


def _sat_square_overlap(
    pos_a: jnp.ndarray, theta_a: jnp.ndarray,
    pos_b: jnp.ndarray, theta_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SAT for two square OBBs of side ROBOT_SIZE.

    Returns (overlapping_bool, normal_from_B_to_A, min_overlap).
    """
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    delta = pos_a - pos_b
    ca, sa = jnp.cos(theta_a), jnp.sin(theta_a)
    cb, sb = jnp.cos(theta_b), jnp.sin(theta_b)

    axes = jnp.stack([
        jnp.stack([ ca,  sa]),
        jnp.stack([-sa,  ca]),
        jnp.stack([ cb,  sb]),
        jnp.stack([-sb,  cb]),
    ])  # (4, 2)

    # Support of each square along each axis
    # sup = half * (|u1·axis| + |u2·axis|), u1/u2 = local axes
    def support(axis, c, s):
        ax, ay = axis[0], axis[1]
        return half * (jnp.abs(c * ax + s * ay) + jnp.abs(-s * ax + c * ay))

    sup_a = jax.vmap(support, in_axes=(0, None, None))(axes, ca, sa)  # (4,)
    sup_b = jax.vmap(support, in_axes=(0, None, None))(axes, cb, sb)
    proj = axes @ delta  # (4,)
    dist = jnp.abs(proj)
    overlaps = sup_a + sup_b - dist  # (4,)

    overlapping = jnp.all(overlaps > 0)

    # Pick the axis with the minimum overlap (only meaningful if overlapping).
    min_idx = jnp.argmin(overlaps)
    min_overlap = overlaps[min_idx]
    axis = axes[min_idx]
    sign = jnp.where(proj[min_idx] >= 0, 1.0, -1.0)
    normal = axis * sign

    # When not overlapping, return safe zeros.
    normal = jnp.where(overlapping, normal, jnp.zeros(2, dtype=jnp.float32))
    min_overlap = jnp.where(overlapping, min_overlap, jnp.float32(0.0))
    return overlapping, normal, min_overlap


def _resolve_robot_pair(
    robots_flat: jnp.ndarray, i: jnp.ndarray, j: jnp.ndarray
) -> jnp.ndarray:
    e = jnp.float32(config.ROBOT_WALL_RESTITUTION)
    a = robots_flat[i]
    b = robots_flat[j]
    pos_a, theta_a, vel_a = a[0:2], a[2], a[3:5]
    pos_b, theta_b, vel_b = b[0:2], b[2], b[3:5]

    overlapping, normal, overlap = _sat_square_overlap(pos_a, theta_a, pos_b, theta_b)

    pa_new = pos_a + normal * overlap * 0.5
    pb_new = pos_b - normal * overlap * 0.5

    rel_vel = vel_a - vel_b
    vel_along = jnp.dot(rel_vel, normal)
    do_impulse = overlapping & (vel_along < 0)
    j_imp = -(1.0 + e) * vel_along * 0.5
    va_new = jnp.where(do_impulse, vel_a + j_imp * normal, vel_a)
    vb_new = jnp.where(do_impulse, vel_b - j_imp * normal, vel_b)

    pa_final = jnp.where(overlapping, pa_new, pos_a)
    pb_final = jnp.where(overlapping, pb_new, pos_b)

    new_a = a.at[0:2].set(pa_final).at[3:5].set(va_new)
    new_b = b.at[0:2].set(pb_final).at[3:5].set(vb_new)
    robots_flat = robots_flat.at[i].set(new_a).at[j].set(new_b)
    return robots_flat


def _robot_robot_collisions(state: SimState) -> SimState:
    """Resolve inelastic collisions between robots (OBB vs OBB)."""
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)

    def body(k, robots_flat):
        i = _PAIR_I[k]
        j = _PAIR_J[k]
        return _resolve_robot_pair(robots_flat, i, j)

    new_robots_flat = jax.lax.fori_loop(0, _N_PAIRS, body, robots_flat)
    return state._replace(
        robots=new_robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6)
    )
```

- [ ] **Step 4: Run — pass.**

- [ ] **Step 5: Commit.**

---

## Task 10: Public `step()`

**Files:** Modify `jax_backend.py`, extend tests.

- [ ] **Step 1: Append failing tests.**

```python
class TestStep:
    def _make_state(self):
        return jb.reset_kickoff(jax.random.PRNGKey(0))

    def _zero_actions(self):
        return jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)

    def test_step_advances_time(self):
        s = self._make_state()
        t0 = float(s.t)
        s2, _ = jb.step(s, self._zero_actions())
        assert float(s2.t) == pytest.approx(t0 + config.DT, abs=1e-5)

    def test_step_stationary_zero_action(self):
        s = self._make_state()
        initial = s.robots[:, :, 0:2]
        s2, _ = jb.step(s, self._zero_actions())
        assert jnp.allclose(s2.robots[:, :, 0:2], initial, atol=1e-5)

    def test_robot_moves_forward(self):
        s = jb.reset_kickoff(jax.random.PRNGKey(1))
        initial_x = float(s.robots[config.TEAM_BLUE, 0, 0])
        actions = self._zero_actions().at[config.TEAM_BLUE, 0, :].set(1.0)
        for _ in range(5):
            s, _ = jb.step(s, actions)
        assert float(s.robots[config.TEAM_BLUE, 0, 0]) > initial_x

    def test_goal_registered(self):
        s = self._make_state()
        s = s._replace(ball=jnp.array(
            [config.FIELD_LENGTH / 2.0 + 0.01, 0.0, 0.1, 0.0],
            dtype=jnp.float32,
        ))
        _, info = jb.step(s, self._zero_actions())
        assert int(info["goal"]) == 1

    def test_no_goal_centre(self):
        s = self._make_state()
        for _ in range(30):
            s, _ = jb.step(s, self._zero_actions())
        # No score should have accumulated externally; "score" stays zero.
        assert int(s.score[0]) == 0
        assert int(s.score[1]) == 0

    def test_ball_stays_in_field(self):
        s = self._make_state()
        s = s._replace(ball=s.ball.at[2].set(config.ROBOT_MAX_WHEEL_SPEED * 10))
        for _ in range(10):
            s, _ = jb.step(s, self._zero_actions())
        half_l = config.FIELD_LENGTH / 2.0 + config.GOAL_DEPTH + 0.1
        half_w = config.FIELD_WIDTH / 2.0 + 0.1
        assert abs(float(s.ball[0])) <= half_l
        assert abs(float(s.ball[1])) <= half_w

    def test_jit_compiles(self):
        """step should be jittable end-to-end."""
        s = self._make_state()
        a = self._zero_actions()
        step_jit = jax.jit(jb.step)
        s2, info = step_jit(s, a)
        # forces compilation; assert returns a state with same shapes
        assert s2.robots.shape == s.robots.shape
        assert "goal" in info


class TestParityWithNumpy:
    """Trajectories from numpy and jax should match within tolerance."""

    def test_step_parity_zero_actions(self):
        from vsss_sim.physics import numpy_backend as nb

        # Identical initial state from numpy reset
        np_s = nb.SimState()
        nb.reset_kickoff(np_s, rng=np.random.default_rng(0))
        j_s = jb.from_numpy(np_s)

        np_a = np.zeros((config.N_TEAMS, config.N_ROBOTS, 2))
        j_a = jnp.asarray(np_a, dtype=jnp.float32)

        for _ in range(20):
            nb.step(np_s, np_a)
            j_s, _ = jb.step(j_s, j_a)

        np.testing.assert_allclose(
            np.asarray(j_s.robots[:, :, 0:2]), np_s.robots[:, :, 0:2],
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(j_s.ball[0:2]), np_s.ball[0:2], atol=1e-3,
        )

    def test_step_parity_forward_action(self):
        from vsss_sim.physics import numpy_backend as nb

        np_s = nb.SimState()
        nb.reset_kickoff(np_s, rng=np.random.default_rng(2))
        j_s = jb.from_numpy(np_s)

        np_a = np.zeros((config.N_TEAMS, config.N_ROBOTS, 2))
        np_a[config.TEAM_BLUE, 0, :] = 1.0
        j_a = jnp.asarray(np_a, dtype=jnp.float32)

        for _ in range(15):
            nb.step(np_s, np_a)
            j_s, _ = jb.step(j_s, j_a)

        # Positions of the moving robot
        np.testing.assert_allclose(
            float(j_s.robots[config.TEAM_BLUE, 0, 0]),
            np_s.robots[config.TEAM_BLUE, 0, 0],
            atol=5e-3,
        )
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement. Append to `jax_backend.py`:**

```python
# ---------------------------------------------------------------------------
# Public step (jit-compiled)
# ---------------------------------------------------------------------------

from functools import partial


def _substep(state: SimState, wheel_speeds: jnp.ndarray, sub_dt: jnp.ndarray
             ) -> tuple[SimState, jnp.ndarray]:
    """One physics sub-step. Returns (new_state, goal_event)."""
    v_l = wheel_speeds[:, :, 0]
    v_r = wheel_speeds[:, :, 1]
    theta = state.robots[:, :, 2]
    vx, vy, omega = _diff_drive(v_l, v_r, theta)

    robots = state.robots
    robots = robots.at[:, :, 3].set(vx)
    robots = robots.at[:, :, 4].set(vy)
    robots = robots.at[:, :, 5].set(omega)
    robots = robots.at[:, :, 0].add(vx * sub_dt)
    robots = robots.at[:, :, 1].add(vy * sub_dt)
    new_theta = robots[:, :, 2] + omega * sub_dt
    new_theta = (new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    robots = robots.at[:, :, 2].set(new_theta)
    state = state._replace(robots=robots)

    # Ball friction (rolling)
    ball_vel = state.ball[2:4]
    speed = jnp.linalg.norm(ball_vel)
    safe_speed = jnp.where(speed > 1e-6, speed, 1.0)
    decel = jnp.minimum(speed, config.BALL_FRICTION * 9.81 * sub_dt)
    new_vel = ball_vel - (ball_vel / safe_speed) * decel
    new_vel = jnp.where(speed > 1e-6, new_vel, ball_vel)
    ball = state.ball.at[2:4].set(new_vel)
    ball = ball.at[0:2].add(new_vel * sub_dt)
    state = state._replace(ball=ball)

    # Collisions
    state = _robot_wall_collisions(state)
    state, goal = _ball_wall_collisions(state)
    state = _ball_robot_collisions(state)
    state = _robot_robot_collisions(state)
    return state, goal


@partial(jax.jit, static_argnames=("sub_steps",))
def step(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = 4,
) -> tuple[SimState, dict]:
    """Advance the simulation by one control timestep (functional + jitted)."""
    actions = jnp.clip(actions, -1.0, 1.0).astype(jnp.float32)
    wheel_speeds = actions * jnp.float32(config.ROBOT_MAX_WHEEL_SPEED)
    sub_dt = jnp.float32(dt / sub_steps)

    def body(_, carry):
        state, goal_acc = carry
        state, g = _substep(state, wheel_speeds, sub_dt)
        goal_acc = jnp.where(goal_acc == 0, g, goal_acc)
        return state, goal_acc

    state, goal = jax.lax.fori_loop(
        0, sub_steps, body, (state, jnp.int32(0))
    )
    state = state._replace(t=state.t + jnp.float32(dt))
    return state, {"goal": goal}
```

- [ ] **Step 4: Run — all tests in `test_jax_backend.py` pass.**

```bash
.venv/bin/pytest tests/physics/test_jax_backend.py -v
```

- [ ] **Step 5: Commit.**

```bash
git add src/vsss_sim/physics/jax_backend.py tests/physics/test_jax_backend.py
git commit -m "feat(physics/jax): add jitted step + parity tests with numpy backend"
```

---

## Task 11: Wire backend into `VSSEnv`

**Files:** Modify `src/vsss_sim/envs/base.py`, `src/vsss_sim/envs/vsss_3v3.py`, `tests/envs/test_env.py`.

The plan: in `VSSEnv.__init__`, resolve the backend module once. Store it as `self._backend`. Use `self._backend.SimState` /  `self._backend.step` / `self._backend.reset_kickoff` instead of imports.

The numpy backend mutates the state in place; the jax backend returns a new state. The env handles this with a unified call: `self._state = self._backend_step(self._state, all_actions)` where `_backend_step` is a tiny method that branches once.

- [ ] **Step 1: Write a failing parametrised env test.**

Append to `tests/envs/test_env.py` (we'll inspect existing patterns and add):

```python
import pytest

@pytest.mark.parametrize("backend", ["numpy", "jax"])
class TestBackendParity:
    def test_env_reset_step_runs(self, backend):
        import gymnasium as gym
        import numpy as np
        import vsss_sim  # noqa: F401  registers VSSS-v0
        env = gym.make("VSSS-v0", backend=backend)
        obs, info = env.reset(seed=0)
        assert obs.shape == (46,)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (46,)
        env.close()

    def test_observation_finite(self, backend):
        import gymnasium as gym
        import numpy as np
        import vsss_sim  # noqa: F401
        env = gym.make("VSSS-v0", backend=backend)
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs))
        env.close()
```

- [ ] **Step 2: Run — failing (`backend` kwarg unknown).**

```bash
.venv/bin/pytest tests/envs/test_env.py::TestBackendParity -v
```

- [ ] **Step 3: Refactor `VSSBaseEnv` and `VSSEnv` to accept a backend.**

`src/vsss_sim/envs/base.py` — replace lines 17 and 53-55 to use the resolver:

```python
# Top of file (replace `from ..physics import SimState`)
from .. import physics as _physics_pkg

# In __init__ replace `self._state = SimState()` with:
self._backend = _physics_pkg.get_backend(backend)
self._state = self._backend.SimState() if backend in (None, "numpy") else self._backend.empty_state()
```

Specifically, here is the full revised `__init__` signature:

```python
def __init__(
    self,
    render_mode: Optional[str] = None,
    max_episode_steps: int = config.MAX_EPISODE_STEPS,
    render_fps: Optional[float] = None,
    backend: Optional[str] = None,
) -> None:
    super().__init__()
    self.render_mode = render_mode
    self.max_episode_steps = max_episode_steps
    self._render_fps = render_fps
    self._rng = np.random.default_rng()
    self._backend = _physics_pkg.get_backend(backend)
    self._backend_name = self._backend.__name__.rsplit(".", 1)[-1]
    self._state = self._initial_state()
    self._step_count = 0
    self._renderer = None
    # ... rest unchanged (spaces)


def _initial_state(self):
    """Create a zero state appropriate for the active backend."""
    if self._backend_name == "jax_backend":
        return self._backend.empty_state()
    return self._backend.SimState()
```

Also update `_get_obs` to handle both backends — read state values via `np.asarray` at the boundary:

```python
def _get_obs(self) -> np.ndarray:
    """Return a normalised flat observation vector."""
    s = self._state
    ball = np.asarray(s.ball)
    robots = np.asarray(s.robots)
    obs = np.empty(4 + config.N_TEAMS * config.N_ROBOTS * 7, dtype=np.float32)

    obs[0] = ball[0] / _NORM_POS_X
    obs[1] = ball[1] / _NORM_POS_Y
    obs[2] = ball[2] / _NORM_VEL
    obs[3] = ball[3] / _NORM_VEL

    idx = 4
    for team in range(config.N_TEAMS):
        for r in range(config.N_ROBOTS):
            x, y, theta, vx, vy, omega = robots[team, r]
            obs[idx + 0] = x / _NORM_POS_X
            obs[idx + 1] = y / _NORM_POS_Y
            obs[idx + 2] = float(np.sin(theta))
            obs[idx + 3] = float(np.cos(theta))
            obs[idx + 4] = vx / _NORM_VEL
            obs[idx + 5] = vy / _NORM_VEL
            obs[idx + 6] = omega / _NORM_OMEGA
            idx += 7
    return obs


def _get_info(self) -> dict[str, Any]:
    return {
        "score_blue": int(np.asarray(self._state.score)[config.TEAM_BLUE]),
        "score_yellow": int(np.asarray(self._state.score)[config.TEAM_YELLOW]),
        "sim_time": float(np.asarray(self._state.t)),
    }
```

- [ ] **Step 4: Update `VSSEnv` (`src/vsss_sim/envs/vsss_3v3.py`).**

Replace imports and `__init__`:

```python
from .base import VSSBaseEnv


class VSSEnv(VSSBaseEnv):
    def __init__(
        self,
        opponent_policy: str | Callable = "stationary",
        render_mode: Optional[str] = None,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        render_fps: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            render_fps=render_fps,
            backend=backend,
        )
        # ... existing opponent_policy resolution unchanged
```

Replace `reset` and `step` to dispatch through the backend:

```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
        self._rng = np.random.default_rng(seed)

    if self._backend_name == "jax_backend":
        import jax  # local import keeps numpy-only paths cheap
        key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
        self._state = self._backend.reset_kickoff(key)
    else:
        self._state = self._backend.SimState()
        self._backend.reset_kickoff(self._state, rng=self._rng)
    self._step_count = 0
    return self._get_obs(), self._get_info()


def step(self, action):
    blue_actions = np.array(action, dtype=np.float64).reshape(config.N_ROBOTS, 2)
    obs_current = self._get_obs()
    yellow_actions = self._opponent_policy(obs_current).reshape(config.N_ROBOTS, 2)
    all_actions = np.stack([blue_actions, yellow_actions], axis=0)

    if self._backend_name == "jax_backend":
        import jax.numpy as jnp
        all_actions_j = jnp.asarray(all_actions, dtype=jnp.float32)
        self._state, info_phys = self._backend.step(self._state, all_actions_j)
        goal = int(info_phys["goal"])
    else:
        info_phys = self._backend.step(self._state, all_actions)
        goal = int(info_phys["goal"])

    self._step_count += 1

    # Score bookkeeping at env level (both backends): update the SimState's score
    if goal == 1:
        if self._backend_name == "jax_backend":
            self._state = self._state._replace(
                score=self._state.score.at[config.TEAM_BLUE].add(1)
            )
        else:
            self._state.score[config.TEAM_BLUE] += 1
    elif goal == -1:
        if self._backend_name == "jax_backend":
            self._state = self._state._replace(
                score=self._state.score.at[config.TEAM_YELLOW].add(1)
            )
        else:
            self._state.score[config.TEAM_YELLOW] += 1

    reward = float(goal)
    terminated = False
    truncated = self._step_count >= self.max_episode_steps

    obs = self._get_obs()
    info = self._get_info()
    info["goal"] = goal

    if goal != 0:
        if self._backend_name == "jax_backend":
            import jax
            key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1)))
            self._state = self._backend.reset_kickoff(key)
        else:
            self._backend.reset_kickoff(self._state, rng=self._rng)

    if self.render_mode == "human":
        self.render()

    return obs, reward, terminated, truncated, info
```

- [ ] **Step 5: Run parametrised tests.**

```bash
.venv/bin/pytest tests/envs/test_env.py -v
```

Expected: both numpy and jax variants pass.

- [ ] **Step 6: Run the entire test suite.**

```bash
.venv/bin/pytest -v
```
Expected: existing 67 tests + new jax/resolver/env tests all pass.

- [ ] **Step 7: Commit.**

```bash
git add src/vsss_sim/envs/base.py src/vsss_sim/envs/vsss_3v3.py tests/envs/test_env.py
git commit -m "feat(env): switchable backend via VSSEnv(backend=...) and VSSS_PHYSICS_BACKEND"
```

---

## Task 12: `--backend` flag for `scripts/smoke.py`

**Files:** Modify `scripts/smoke.py`.

- [ ] **Step 1: Inspect current CLI to know where to insert.** Read the script first.

- [ ] **Step 2: Add `--backend` arg and pass it to `gym.make`.**

```python
parser.add_argument(
    "--backend",
    type=str,
    default=None,
    choices=["numpy", "jax"],
    help="Physics backend to use (default: numpy or VSSS_PHYSICS_BACKEND)",
)
# ...
env = gym.make("VSSS-v0", ..., backend=args.backend)
```

- [ ] **Step 3: Run smoke without render to check it runs on both backends.**

```bash
.venv/bin/python scripts/smoke.py --timesteps 100 --backend numpy
.venv/bin/python scripts/smoke.py --timesteps 100 --backend jax
```

- [ ] **Step 4: Commit.**

```bash
git add scripts/smoke.py
git commit -m "feat(smoke): add --backend flag (numpy|jax)"
```

---

## Task 13: Update `CLAUDE.md`

**Files:** Modify `CLAUDE.md` — under "Open threads / next directions", note that JAX is now landed.

- [ ] **Step 1: Tweak the section to say JAX backend is implemented; batched VSSVecEnv still pending.**

- [ ] **Step 2: Commit.**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): note JAX backend landed; batched env still TODO"
```

---

## Self-review checklist

- [ ] Spec covers: backend resolver (Task 2), JAX state (3), all numpy_backend functions (4-9), public step (10), env integration (11), CLI (12), docs (13). ✓
- [ ] No placeholders — every code block is complete. ✓
- [ ] Type consistency: `SimState` is the same NamedTuple across all tasks; `step` returns `(SimState, dict)` everywhere; `_ball_wall_collisions` returns `(SimState, jnp.ndarray)` everywhere. ✓
- [ ] Float dtype consistent: float32 throughout the jax backend; cast at the boundary via `from_numpy` / `to_numpy`. ✓
- [ ] PRNG: `reset_kickoff(key: PRNGKey) -> SimState`. Env derives keys from its `_rng`. ✓
