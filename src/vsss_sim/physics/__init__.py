"""Physics engines for the VSSS simulator.

The default JAX engine's public API is re-exported from :mod:`jax_backend`.
Keeping the implementation in a dedicated module makes it convenient for
benchmarks and tests to import the complete engine while package-level imports
remain concise. :func:`get_backend` selects an engine by name; the optional
MuJoCo MJX engine is imported only when requested.
"""

from __future__ import annotations

from types import ModuleType

from ..config import PhysicsBackend
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


def get_backend(name: PhysicsBackend | str = PhysicsBackend.JAX) -> ModuleType:
    """Return the engine module for ``name``.

    Every engine provides ``SimState``, ``empty_state``, ``reset_kickoff``,
    ``reset_random``, ``step``, and ``step_with_frames`` with the same
    semantics, so environments can switch engines without other changes.
    """
    backend = PhysicsBackend(name)
    if backend is PhysicsBackend.MJX:
        try:
            from . import mjx_backend
        except ImportError as error:
            raise ImportError(
                "the mjx physics backend requires mujoco-mjx; "
                "install it with: pip install -e '.[mjx]'"
            ) from error
        return mjx_backend
    return jax_backend


__all__ = [
    "SimState",
    "empty_state",
    "get_backend",
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
