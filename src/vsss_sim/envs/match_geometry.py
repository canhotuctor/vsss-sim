"""JAX-compatible field and robot geometry used by match rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config

_PLAY_INVOLVEMENT_DISTANCE = 0.12
_ROBOT_CONTACT_DISTANCE = config.ROBOT_SIZE + 0.01
_POLYGON_CAPACITY = 12


def _append_vertex(
    polygon: jax.Array, count: jax.Array, point: jax.Array
) -> tuple[jax.Array, jax.Array]:
    safe_count = jnp.minimum(count, _POLYGON_CAPACITY - 1)
    polygon = polygon.at[safe_count].set(point)
    return polygon, jnp.minimum(count + 1, _POLYGON_CAPACITY)


def _clip_polygon(
    polygon: jax.Array,
    count: jax.Array,
    axis: int,
    boundary: jax.Array,
    keep_greater: bool,
) -> tuple[jax.Array, jax.Array]:
    """Clip a small convex polygon against one axis-aligned half plane."""
    output = jnp.zeros_like(polygon)

    def inside(point: jax.Array) -> jax.Array:
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    def clip_edge(index: int, carry: tuple[jax.Array, jax.Array]):
        clipped, clipped_count = carry
        previous_index = jnp.where(index == 0, count - 1, index - 1)
        previous = polygon[previous_index]
        current = polygon[index]
        previous_inside = inside(previous)
        current_inside = inside(current)
        crosses = previous_inside != current_inside
        denominator = current[axis] - previous[axis]
        safe_denominator = jnp.where(jnp.abs(denominator) < 1e-8, 1e-8, denominator)
        fraction = (boundary - previous[axis]) / safe_denominator
        intersection = previous + fraction * (current - previous)
        clipped, clipped_count = jax.lax.cond(
            crosses,
            lambda state: _append_vertex(state[0], state[1], intersection),
            lambda state: state,
            (clipped, clipped_count),
        )
        return jax.lax.cond(
            current_inside,
            lambda state: _append_vertex(state[0], state[1], current),
            lambda state: state,
            (clipped, clipped_count),
        )

    return jax.lax.fori_loop(0, count, clip_edge, (output, jnp.int32(0)))


def _polygon_area(polygon: jax.Array, count: jax.Array) -> jax.Array:
    indices = jnp.arange(_POLYGON_CAPACITY, dtype=jnp.int32)
    next_indices = jnp.where(indices + 1 < count, indices + 1, 0)
    next_indices = jnp.clip(next_indices, 0, _POLYGON_CAPACITY - 1)
    following = polygon[next_indices]
    cross = polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0]
    return 0.5 * jnp.abs(jnp.sum(jnp.where(indices < count, cross, 0.0)))


def _robot_goal_area_fraction(robot: jax.Array, side: jax.Array) -> jax.Array:
    """Return overlap with the simulator's rectangular goal-area model."""
    half_length = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_depth = jnp.float32(config.GOAL_AREA_LENGTH_X / 2.0)
    half_width = jnp.float32(config.GOAL_AREA_LENGTH_Y / 2.0)
    half_robot = jnp.float32(config.ROBOT_SIZE / 2.0)
    left = side == 0
    x_min = jnp.where(left, -half_length, half_length - 2.0 * half_depth)
    x_max = jnp.where(left, -half_length + 2.0 * half_depth, half_length)

    local = jnp.asarray(
        [
            [-half_robot, -half_robot],
            [half_robot, -half_robot],
            [half_robot, half_robot],
            [-half_robot, half_robot],
        ],
        dtype=jnp.float32,
    )
    cosine = jnp.cos(robot[2])
    sine = jnp.sin(robot[2])
    rotation = jnp.asarray([[cosine, -sine], [sine, cosine]])
    corners = local @ rotation.T + robot[None, 0:2]
    polygon = jnp.zeros((_POLYGON_CAPACITY, 2), dtype=jnp.float32).at[:4].set(corners)
    count = jnp.int32(4)
    polygon, count = _clip_polygon(polygon, count, 0, x_min, True)
    polygon, count = _clip_polygon(polygon, count, 0, x_max, False)
    polygon, count = _clip_polygon(polygon, count, 1, -half_width, True)
    polygon, count = _clip_polygon(polygon, count, 1, half_width, False)
    return _polygon_area(polygon, count) / jnp.float32(config.ROBOT_SIZE**2)


def _goal_area_membership(robots: jax.Array) -> jax.Array:
    """Count a robot in a goal area only when at least half its footprint is in."""
    flat_robots = robots.reshape((-1, 6))
    sides = jnp.arange(2, dtype=jnp.int32)
    fractions = jax.vmap(
        lambda robot: jax.vmap(lambda side: _robot_goal_area_fraction(robot, side))(sides)
    )(flat_robots)
    return (fractions >= 0.5 - 1e-6).reshape((config.N_TEAMS, config.N_ROBOTS, 2))


def _ball_goal_area_membership(ball: jax.Array) -> jax.Array:
    """Check ball-circle overlap with each rectangular goal area."""
    x, y = ball[0], ball[1]
    half_length = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_depth = jnp.float32(config.GOAL_AREA_LENGTH_X / 2.0)
    half_width = jnp.float32(config.GOAL_AREA_LENGTH_Y / 2.0)

    def overlap(center_x: jax.Array) -> jax.Array:
        outside_x = jnp.maximum(jnp.abs(x - center_x) - half_depth, 0.0)
        outside_y = jnp.maximum(jnp.abs(y) - half_width, 0.0)
        return jnp.hypot(outside_x, outside_y) <= config.BALL_RADIUS

    return jnp.stack((overlap(-half_length + half_depth), overlap(half_length - half_depth)))


def _robot_ball_distance(robots: jax.Array, ball: jax.Array) -> jax.Array:
    return jnp.linalg.norm(robots[..., 0:2] - ball[None, 0:2], axis=-1)


def _players_near_each_other(robots: jax.Array) -> jax.Array:
    flat = robots.reshape((-1, 6))
    distances = jnp.linalg.norm(flat[:, None, 0:2] - flat[None, :, 0:2], axis=-1)
    not_self = ~jnp.eye(config.N_TEAMS * config.N_ROBOTS, dtype=jnp.bool_)
    return ((distances <= jnp.float32(_PLAY_INVOLVEMENT_DISTANCE)) & not_self).reshape(
        (config.N_TEAMS, config.N_ROBOTS, config.N_TEAMS, config.N_ROBOTS)
    )


def _ball_between(first: jax.Array, second: jax.Array, ball: jax.Array) -> jax.Array:
    """Approximate the rulebook's legal push when the ball lies between robots."""
    segment = second[..., 0:2] - first[..., 0:2]
    offset = ball[0:2] - first[..., 0:2]
    length_squared = jnp.sum(segment * segment, axis=-1)
    projection = jnp.sum(offset * segment, axis=-1) / jnp.maximum(length_squared, 1e-8)
    closest = first[..., 0:2] + jnp.clip(projection, 0.0, 1.0)[..., None] * segment
    lateral_distance = jnp.linalg.norm(ball[0:2] - closest, axis=-1)
    return (
        (projection >= 0.0)
        & (projection <= 1.0)
        & (lateral_distance <= config.ROBOT_SIZE / 2.0 + config.BALL_RADIUS)
    )
