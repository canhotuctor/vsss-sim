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

    ball: jnp.ndarray    # (4,)   float32   [x, y, vx, vy]
    robots: jnp.ndarray  # (N_TEAMS, N_ROBOTS, 6) float32 [x, y, theta, vx, vy, omega]
    score: jnp.ndarray   # (2,)   int32
    t: jnp.ndarray       # ()     float32


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


# ---------------------------------------------------------------------------
# Differential-drive kinematics (vectorised over robots)
# ---------------------------------------------------------------------------

def _diff_drive(
    v_left: jnp.ndarray,
    v_right: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert wheel speeds to body velocities.

    Returns ``(vx, vy, omega)`` shaped like the inputs.
    """
    v = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / config.ROBOT_WHEELBASE
    vx = v * jnp.cos(theta)
    vy = v * jnp.sin(theta)
    return vx, vy, omega
