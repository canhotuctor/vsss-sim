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


# Symmetry group of one robot's observation: x-axis mirror (2) x ally order (2)
# x opponent order (3! = 6) = 24 elements. Element ``g`` is encoded as
# ``mirror + 2 * (ally_swap + 2 * opponent_permutation)``, so elements 0 and 1
# are the identity and the plain mirror.
_OPPONENT_PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
)
SYMMETRY_GROUP_SIZE = 2 * 2 * len(_OPPONENT_PERMUTATIONS)
MIRROR_GROUP_SIZE = 2


def _frame_symmetry(
    robot: int, mirror: bool, ally_swap: bool, opponents: tuple[int, ...],
    entity_ordered_actions: bool,
) -> tuple[list[int], list[float]]:
    """Signed permutation of one 53-feature frame: ``out[j] = sign[j] * in[index[j]]``."""
    index = list(range(FRAME_FEATURES))
    allies = (1, 0) if ally_swap else (0, 1)
    # Positions: entity slots self, ally0, ally1, opp0..2, ball.
    entity_source = [0, 1 + allies[0], 1 + allies[1]] + [3 + o for o in opponents] + [6]
    for slot, source in enumerate(entity_source):
        index[2 * slot] = 2 * source
        index[2 * slot + 1] = 2 * source + 1
    # Bearings and distances: ally0, ally1, opp0..2, then fixed slots.
    relative_source = [allies[0], allies[1]] + [2 + o for o in opponents]
    for slot, source in enumerate(relative_source):
        for component in range(3):
            index[17 + 3 * slot + component] = 17 + 3 * source + component
        index[38 + slot] = 38 + source
    # Previous actions: blocks of (left, right) wheels per robot.
    if entity_ordered_actions:
        action_source = [0, 1 + allies[0], 1 + allies[1]]
    else:
        teammates = [other for other in range(config.N_ROBOTS) if other != robot]
        action_source = list(range(config.N_ROBOTS))
        if ally_swap:
            action_source[teammates[0]], action_source[teammates[1]] = (
                teammates[1], teammates[0]
            )
    for block, source in enumerate(action_source):
        left, right = 46 + 2 * source, 47 + 2 * source
        index[46 + 2 * block] = right if mirror else left
        index[47 + 2 * block] = left if mirror else right

    sign = [1.0] * FRAME_FEATURES
    if mirror:
        for slot in range(7):
            sign[2 * slot + 1] = -1.0  # y coordinates
        sign[14] = sign[16] = -1.0  # heading sin and normalized angle
        for slot in range(7):
            sign[17 + 3 * slot] = -1.0  # bearing sin
            sign[19 + 3 * slot] = -1.0  # bearing normalized angle
    return index, sign


def symmetry_tables(entity_ordered_actions: bool):
    """Return gather tables for the 24 observation symmetries.

    ``index[robot, g]`` and ``sign[robot, g]`` (shape ``[3, 24, 424]``) map an
    observation of robot row ``robot`` to its transformed copy via
    ``obs[index] * sign``. ``wheel_swap[g]`` says whether the action's left and
    right wheels swap. The robot row matters only for team-ordered previous
    actions, where an ally swap exchanges two non-self action blocks.
    """
    import numpy as np

    index = np.zeros((config.N_ROBOTS, SYMMETRY_GROUP_SIZE, OBSERVATION_SIZE), np.int32)
    sign = np.ones((config.N_ROBOTS, SYMMETRY_GROUP_SIZE, OBSERVATION_SIZE), np.float32)
    wheel_swap = np.zeros((SYMMETRY_GROUP_SIZE,), bool)
    offsets = np.arange(HISTORY_LENGTH)[:, None] * FRAME_FEATURES
    for robot in range(config.N_ROBOTS):
        for g in range(SYMMETRY_GROUP_SIZE):
            mirror, rest = g % 2, g // 2
            ally_swap, opponent_index = rest % 2, rest // 2
            frame_index, frame_sign = _frame_symmetry(
                robot,
                bool(mirror),
                bool(ally_swap),
                _OPPONENT_PERMUTATIONS[opponent_index],
                entity_ordered_actions,
            )
            index[robot, g] = (offsets + np.asarray(frame_index)[None, :]).reshape(-1)
            sign[robot, g] = np.tile(np.asarray(frame_sign, np.float32), HISTORY_LENGTH)
            wheel_swap[g] = bool(mirror)
    return index, sign, wheel_swap
