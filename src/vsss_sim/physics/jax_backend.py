"""JAX physics backend for vsss-sim.

Pure-functional mirror of ``numpy_backend.py``. ``SimState`` is a
:class:`typing.NamedTuple` PyTree — register-free, immutable, ``vmap``-friendly.

Float dtype is ``float32`` (GPU default). Tests assert semantic parity with the
float64 numpy backend within a generous tolerance.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
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

    # Face toward the ball at origin.
    theta = jnp.arctan2(-robots[:, :, 1], -robots[:, :, 0])
    robots = robots.at[:, :, 2].set(theta)

    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=robots,
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
    )


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
