"""Symmetries of the centralized IEEE VSSS observation and action spaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from . import config


class VSSSSymmetryElement(NamedTuple):
    """One element of ``C2 x S3_blue x S3_yellow``."""

    reflected: jax.Array
    blue_permutation: jax.Array
    yellow_permutation: jax.Array


@dataclass(frozen=True)
class VSSSSymmetry:
    """Group action for the flat centralized VSSS policy interface.

    Reflection is across the field's longitudinal (x) axis.  Because VSSS
    robots are differential-drive, reflection swaps their left and right wheel
    commands.  Robot permutations relabel observation slots; blue permutations
    also relabel the corresponding controlled-action blocks.
    """

    reflect: bool = True
    permute_blue: bool = True
    permute_yellow: bool = True

    def sample(self, key: chex.PRNGKey) -> VSSSSymmetryElement:
        reflection_key, blue_key, yellow_key = jax.random.split(key, 3)
        identity = jnp.arange(config.N_ROBOTS, dtype=jnp.int32)
        reflected = (
            jax.random.bernoulli(reflection_key)
            if self.reflect
            else jnp.asarray(False)
        )
        blue = (
            jax.random.permutation(blue_key, identity)
            if self.permute_blue
            else identity
        )
        yellow = (
            jax.random.permutation(yellow_key, identity)
            if self.permute_yellow
            else identity
        )
        return VSSSSymmetryElement(reflected, blue, yellow)

    def transform_observation(
        self, observation: chex.Array, element: VSSSSymmetryElement
    ) -> chex.Array:
        observation = jnp.asarray(observation)
        ball = observation[:4]
        robots = observation[4:].reshape(config.N_TEAMS, config.N_ROBOTS, 7)

        # [x, y, vx, vy] -> [x, -y, vx, -vy]
        ball_sign = jnp.asarray([1.0, -1.0, 1.0, -1.0], dtype=observation.dtype)
        reflected_ball = ball * ball_sign

        # [x, y, sin(theta), cos(theta), vx, vy, omega]
        # -> [x, -y, -sin(theta), cos(theta), vx, -vy, -omega]
        robot_sign = jnp.asarray(
            [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            dtype=observation.dtype,
        )
        reflected_robots = robots * robot_sign
        ball = jnp.where(element.reflected, reflected_ball, ball)
        robots = jnp.where(element.reflected, reflected_robots, robots)

        blue = robots[config.TEAM_BLUE, element.blue_permutation]
        yellow = robots[config.TEAM_YELLOW, element.yellow_permutation]
        return jnp.concatenate([ball, blue.reshape(-1), yellow.reshape(-1)])

    def transform_action(
        self, action: chex.Array, element: VSSSSymmetryElement
    ) -> chex.Array:
        """Map physical blue wheel commands into the transformed policy frame."""
        wheels = jnp.asarray(action).reshape(config.N_ROBOTS, 2)
        wheels = wheels[element.blue_permutation]
        reflected_wheels = wheels[:, ::-1]
        wheels = jnp.where(element.reflected, reflected_wheels, wheels)
        return wheels.reshape(-1)

    def inverse_action(
        self, action: chex.Array, element: VSSSSymmetryElement
    ) -> chex.Array:
        """Map transformed policy commands back to physical robot wheel commands."""
        wheels = jnp.asarray(action).reshape(config.N_ROBOTS, 2)
        reflected_wheels = wheels[:, ::-1]
        wheels = jnp.where(element.reflected, reflected_wheels, wheels)
        physical = jnp.empty_like(wheels).at[element.blue_permutation].set(wheels)
        return physical.reshape(-1)


def switch_team_perspective(observation: chex.Array) -> chex.Array:
    """Rotate the field by 180 degrees and swap teams for the yellow policy.

    Wheel commands are robot-local, so the returned policy action can be
    applied directly to the yellow robots without changing its wheel order.
    The transformation is its own inverse.
    """
    observation = jnp.asarray(observation)
    ball = observation[:4] * jnp.asarray(-1.0, dtype=observation.dtype)
    robots = observation[4:].reshape(config.N_TEAMS, config.N_ROBOTS, 7)
    # [x, y, sin(theta), cos(theta), vx, vy, omega] under a half-turn.
    signs = jnp.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0], dtype=observation.dtype)
    rotated = robots * signs
    swapped = rotated[jnp.asarray([config.TEAM_YELLOW, config.TEAM_BLUE])]
    return jnp.concatenate([ball, swapped.reshape(-1)])
