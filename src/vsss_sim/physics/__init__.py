"""JAX physics engine for the VSSS simulator.

The public physics API is re-exported from :mod:`jax_backend`. Keeping the
implementation in a dedicated module makes it convenient for benchmarks and
tests to import the complete engine while package-level imports remain concise.
"""

from . import jax_backend
from .jax_backend import (
    SimState,
    _ball_robot_collisions,
    _ball_wall_collisions,
    _diff_drive,
    _robot_robot_collisions,
    _robot_wall_collisions,
    empty_state,
    reset_kickoff,
    reset_random,
    step,
    step_with_frames,
)

__all__ = [
    "SimState",
    "empty_state",
    "step",
    "step_with_frames",
    "reset_kickoff",
    "reset_random",
    "_diff_drive",
    "_ball_wall_collisions",
    "_robot_wall_collisions",
    "_ball_robot_collisions",
    "_robot_robot_collisions",
    "jax_backend",
]
