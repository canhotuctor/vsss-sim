"""Robot-local observations used by Brandão et al.'s shared policy."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config

FRAME_FEATURES = 53
HISTORY_LENGTH = 8
OBSERVATION_SIZE = FRAME_FEATURES * HISTORY_LENGTH

_HALF_LENGTH = jnp.float32(config.FIELD_LENGTH / 2.0)
_HALF_WIDTH = jnp.float32(config.FIELD_WIDTH / 2.0)
_MAX_DISTANCE = jnp.float32((config.FIELD_LENGTH**2 + config.FIELD_WIDTH**2) ** 0.5)
_POSITION_NOISE = jnp.asarray([1.854e-3, 1.679e-3], dtype=jnp.float32)
_ANGLE_NOISE = jnp.float32(3.123e-2)


def build_agent_features(
    simulation,
    previous_actions: jax.Array,
    time_remaining: jax.Array,
    noise_key: jax.Array,
    entity_ordered_actions: bool = False,
) -> jax.Array:
    """Return ``[team, robot, 53]`` paper-style features for both teams.

    Positions are field-centered and normalized per axis. Each team gets its
    own attacking direction, while robot wheel commands remain robot-local.
    The paper's stated camera noise is applied to position and heading reads.
    """
    position_key, angle_key = jax.random.split(noise_key)
    robot_xy = simulation.robots[..., :2]
    robot_theta = simulation.robots[..., 2]
    ball_xy = simulation.ball[:2]
    robot_xy = robot_xy + jax.random.normal(
        position_key, robot_xy.shape, dtype=jnp.float32
    ) * _POSITION_NOISE
    ball_xy = ball_xy + jax.random.normal(
        jax.random.fold_in(position_key, 1), ball_xy.shape, dtype=jnp.float32
    ) * _POSITION_NOISE
    robot_theta = robot_theta + jax.random.normal(
        angle_key, robot_theta.shape, dtype=jnp.float32
    ) * _ANGLE_NOISE

    features_by_team = []
    for team in range(config.N_TEAMS):
        opponent = 1 - team
        attack_sign = jnp.float32(1.0 if team == config.TEAM_BLUE else -1.0)
        own_xy = robot_xy[team] * attack_sign
        other_xy = robot_xy[opponent] * attack_sign
        canonical_ball = ball_xy * attack_sign
        canonical_theta = robot_theta[team] + jnp.where(
            team == config.TEAM_BLUE, 0.0, jnp.pi
        )
        own_actions = previous_actions[team]

        robot_features = []
        for robot in range(config.N_ROBOTS):
            self_xy = own_xy[robot]
            heading = canonical_theta[robot]
            teammates = jnp.asarray(
                [index for index in range(config.N_ROBOTS) if index != robot],
                dtype=jnp.int32,
            )
            ally_xy = own_xy[teammates]

            object_positions = jnp.concatenate(
                [self_xy[None, :], ally_xy, other_xy, canonical_ball[None, :]], axis=0
            )
            normalized_positions = object_positions / jnp.asarray(
                [_HALF_LENGTH, _HALF_WIDTH], dtype=jnp.float32
            )

            heading_features = jnp.asarray(
                [jnp.sin(heading), jnp.cos(heading), jnp.arctan2(jnp.sin(heading), jnp.cos(heading)) / jnp.pi],
                dtype=jnp.float32,
            )

            opponent_goal = jnp.asarray([_HALF_LENGTH, 0.0], dtype=jnp.float32)
            own_goal = jnp.asarray([-_HALF_LENGTH, 0.0], dtype=jnp.float32)
            targets = jnp.concatenate(
                [ally_xy, other_xy, jnp.stack([own_goal, opponent_goal])], axis=0
            )
            origins = jnp.concatenate(
                [jnp.broadcast_to(self_xy, (config.N_ROBOTS - 1 + config.N_ROBOTS, 2)),
                 jnp.broadcast_to(canonical_ball, (2, 2))],
                axis=0,
            )
            deltas = targets - origins
            bearings = jnp.arctan2(deltas[:, 1], deltas[:, 0]) / jnp.pi
            angle_features = jnp.stack(
                [jnp.sin(bearings * jnp.pi), jnp.cos(bearings * jnp.pi), bearings],
                axis=-1,
            ).reshape(-1)

            distances = jnp.concatenate(
                [
                    jnp.linalg.norm(ally_xy - self_xy, axis=-1),
                    jnp.linalg.norm(other_xy - self_xy, axis=-1),
                    jnp.linalg.norm(canonical_ball - self_xy)[None],
                    jnp.linalg.norm(
                        jnp.stack([own_goal, opponent_goal]) - canonical_ball, axis=-1
                    ),
                ]
            ) / _MAX_DISTANCE
            if entity_ordered_actions:
                action_order = jnp.concatenate([jnp.asarray([robot]), teammates])
                previous_team_actions = own_actions[action_order].reshape(-1)
            else:
                previous_team_actions = own_actions.reshape(-1)
            features = jnp.concatenate(
                [
                    normalized_positions.reshape(-1),
                    heading_features,
                    angle_features,
                    distances,
                    previous_team_actions,
                    jnp.asarray([time_remaining], dtype=jnp.float32),
                ]
            )
            robot_features.append(features)
        features_by_team.append(jnp.stack(robot_features))

    return jnp.stack(features_by_team).astype(jnp.float32)


def initialize_history(features: jax.Array) -> jax.Array:
    """Fill the eight-frame stack with the reset observation."""
    return jnp.broadcast_to(
        features[..., None, :],
        features.shape[:-1] + (HISTORY_LENGTH, FRAME_FEATURES),
    )


def advance_history(
    history: jax.Array, features: jax.Array, sample_observation: jax.Array
) -> jax.Array:
    """Append a policy observation at each 10 Hz decision step."""
    shifted = jnp.concatenate([history[..., 1:, :], features[..., None, :]], axis=-2)
    return jax.lax.select(sample_observation, shifted, history)


def flatten_history(history: jax.Array, team: int) -> jax.Array:
    """Return a team's three stacked observations for policy inference."""
    return history[team].reshape((config.N_ROBOTS, OBSERVATION_SIZE))


def mirror_x_axis_observation(observation: jax.Array) -> jax.Array:
    """Reflect a stacked Brandao observation across the field's x-axis.

    Entity order is retained. Coordinates and signed angular components are
    reflected, while each robot's previous left/right wheel commands are
    swapped to preserve the mirrored differential-drive motion.
    """
    frames = observation.reshape(observation.shape[:-1] + (HISTORY_LENGTH, FRAME_FEATURES))
    positions = frames[..., :14].reshape(frames.shape[:-1] + (7, 2))
    positions = positions.at[..., 1].multiply(-1.0)

    heading = frames[..., 14:17].at[..., 0].multiply(-1.0)
    heading = heading.at[..., 2].multiply(-1.0)
    bearings = frames[..., 17:38].reshape(frames.shape[:-1] + (7, 3))
    bearings = bearings.at[..., 0].multiply(-1.0)
    bearings = bearings.at[..., 2].multiply(-1.0)
    previous_actions = frames[..., 46:52].reshape(frames.shape[:-1] + (3, 2))
    previous_actions = jnp.flip(previous_actions, axis=-1)

    reflected = frames.at[..., :14].set(positions.reshape(frames.shape[:-1] + (14,)))
    reflected = reflected.at[..., 14:17].set(heading)
    reflected = reflected.at[..., 17:38].set(
        bearings.reshape(frames.shape[:-1] + (21,))
    )
    reflected = reflected.at[..., 46:52].set(
        previous_actions.reshape(frames.shape[:-1] + (6,))
    )
    return reflected.reshape(observation.shape)
