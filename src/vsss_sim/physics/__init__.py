"""Physics package — backend resolver and default re-exports.

A backend is a module that defines:
    ``SimState``, ``step``, ``reset_kickoff``,
    ``_diff_drive``, ``_ball_wall_collisions``, ``_robot_wall_collisions``,
    ``_ball_robot_collisions``, ``_robot_robot_collisions``.

Default backend is ``"numpy"``. Override with the ``VSSS_PHYSICS_BACKEND``
environment variable or by passing ``backend=...`` to ``get_backend()`` /
``VSSEnv``.
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

    Priority: explicit ``name`` kwarg > ``VSSS_PHYSICS_BACKEND`` env var > ``"numpy"``.
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
