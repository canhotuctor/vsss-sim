"""Physics package – re-exports the NumPy backend by default."""

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

__all__ = [
    "SimState",
    "step",
    "reset_kickoff",
    "_diff_drive",
    "_ball_wall_collisions",
    "_robot_wall_collisions",
    "_ball_robot_collisions",
    "_robot_robot_collisions",
]
