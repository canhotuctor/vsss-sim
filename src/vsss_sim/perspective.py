"""Team-perspective conversion for the centralized VSSS policy."""
from __future__ import annotations

import chex
import jax.numpy as jnp

from . import config


def switch_team_perspective(observation: chex.Array) -> chex.Array:
    """Rotate the field by 180 degrees and swap blue/yellow robot slots.

    Wheel commands are robot-local, so they do not need to be transformed.
    Applying this operation twice returns the original observation.
    """
    observation = jnp.asarray(observation)
    ball = observation[:4] * jnp.asarray(-1.0, dtype=observation.dtype)
    robots = observation[4:].reshape(config.N_TEAMS, config.N_ROBOTS, 7)
    signs = jnp.asarray(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
        dtype=observation.dtype,
    )
    rotated = robots * signs
    swapped = rotated[jnp.asarray([config.TEAM_YELLOW, config.TEAM_BLUE])]
    return jnp.concatenate([ball, swapped.reshape(-1)])
