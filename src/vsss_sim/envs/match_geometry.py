"""JAX-compatible field and robot geometry used by match rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config

_PLAY_INVOLVEMENT_DISTANCE = 0.08
_ROBOT_CONTACT_DISTANCE = config.ROBOT_SIZE + 0.01
_POLYGON_CAPACITY = 8
_CONTACT_TOLERANCE = 1e-4  # collision solver separation, not a proximity rule


def _clip_polygon(
    polygon: jax.Array,
    count: jax.Array,
    axis: int,
    boundary: jax.Array,
    keep_greater: bool,
) -> tuple[jax.Array, jax.Array]:
    """Clip every fixed-capacity edge in parallel, masking inactive vertices.

    Clipping a convex polygon adds at most one vertex per half plane, including
    duplicate boundary endpoints. A square clipped by four rectangle edges
    therefore fits eight vertices. Prefix counts retain the Sutherland-Hodgman
    vertex order without a data-dependent loop.
    """
    indices = jnp.arange(_POLYGON_CAPACITY, dtype=jnp.int32)
    previous_indices = jnp.where(indices == 0, jnp.maximum(count - 1, 0), indices - 1)
    previous = polygon[previous_indices]
    previous_inside = (
        previous[:, axis] >= boundary if keep_greater else previous[:, axis] <= boundary
    )
    current_inside = (
        polygon[:, axis] >= boundary if keep_greater else polygon[:, axis] <= boundary
    )
    active = indices < count
    crosses = (previous_inside != current_inside) & active
    denominator = polygon[:, axis] - previous[:, axis]
    safe_denominator = jnp.where(jnp.abs(denominator) < 1e-8, 1e-8, denominator)
    fraction = (boundary - previous[:, axis]) / safe_denominator
    intersection = previous + fraction[:, None] * (polygon - previous)

    # Each edge emits its intersection first and its inside endpoint second.
    candidates = jnp.stack((intersection, polygon), axis=1).reshape((-1, 2))
    valid = jnp.stack((crosses, current_inside & active), axis=1).reshape(-1)
    destinations = jnp.cumsum(valid.astype(jnp.int32)) - 1
    destinations = jnp.where(valid, destinations, _POLYGON_CAPACITY)
    output = jnp.zeros_like(polygon).at[destinations].set(candidates, mode="drop")
    return output, jnp.sum(valid, dtype=jnp.int32)


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
    """Any part of a rotated footprint, including boundary contact, is inside."""
    flat_robots = robots.reshape((-1, 6))
    sides = jnp.arange(2, dtype=jnp.int32)
    membership = jax.vmap(
        lambda robot: jax.vmap(lambda side: _robot_in_goal_area(robot, side))(sides)
    )(flat_robots)
    return membership.reshape((config.N_TEAMS, config.N_ROBOTS, 2))


def _robot_vertices(robot: jax.Array) -> jax.Array:
    half = config.ROBOT_SIZE / 2.0
    local = jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]])
    cosine, sine = jnp.cos(robot[2]), jnp.sin(robot[2])
    rotation = jnp.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + robot[None, :2]


def _convex_polygons_contact(first, second, tolerance=1e-6):
    """Separating-axis test, including edges and corners that just touch."""
    edges = jnp.concatenate(
        (jnp.roll(first, -1, axis=0) - first, jnp.roll(second, -1, axis=0) - second)
    )
    axes = jnp.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    a, b = first @ axes.T, second @ axes.T
    margin = tolerance * jnp.linalg.norm(axes, axis=-1)
    return jnp.all(
        (jnp.max(a, axis=0) + margin >= jnp.min(b, axis=0))
        & (jnp.max(b, axis=0) + margin >= jnp.min(a, axis=0))
    )


def _area_polygons():
    """Local coordinates: distance from goal line toward midfield, then y.

    Thirty-two chords approximate the circular arc to less than 0.06 mm.
    The rectangle and arc are tested separately because their union is concave.
    """
    depth, width = config.GOAL_AREA_LENGTH_X, config.GOAL_AREA_LENGTH_Y / 2.0
    rectangle = jnp.asarray([[0, -width], [depth, -width], [depth, width], [0, width]])
    radius = config.GOAL_AREA_ARC_RADIUS
    extent = jnp.arcsin(config.GOAL_AREA_ARC_HALF_WIDTH / radius)
    angles = jnp.linspace(-extent, extent, 33)
    arc = jnp.stack(
        (
            depth + config.GOAL_AREA_ARC_DEPTH - radius + radius * jnp.cos(angles),
            radius * jnp.sin(angles),
        ),
        axis=-1,
    )
    return rectangle, arc


def _robot_in_goal_area(robot, side):
    vertices = _robot_vertices(robot)
    sign = jnp.where(side == 0, 1.0, -1.0)
    local = vertices.at[:, 0].set(config.FIELD_LENGTH / 2.0 + sign * vertices[:, 0])
    rectangle, arc = _area_polygons()
    return _convex_polygons_contact(local, rectangle) | _convex_polygons_contact(local, arc)


def _ball_goal_area_membership(ball: jax.Array) -> jax.Array:
    """Require the whole ball inside the rectangle/arc union (Rule 9.2)."""
    _, arc = _area_polygons()
    depth, width = config.GOAL_AREA_LENGTH_X, config.GOAL_AREA_LENGTH_Y / 2.0
    boundary = jnp.concatenate(
        (
            jnp.asarray([[0, -width], [depth, -width]]),
            arc,
            jnp.asarray([[depth, width], [0, width]]),
        )
    )
    ends = jnp.roll(boundary, -1, axis=0)
    edges = ends - boundary

    def contained(sign):
        point = jnp.asarray([config.FIELD_LENGTH / 2.0 + sign * ball[0], ball[1]])
        rectangle_inside = (point[0] >= 0) & (point[0] <= depth) & (jnp.abs(point[1]) <= width)
        arc_center = depth + config.GOAL_AREA_ARC_DEPTH - config.GOAL_AREA_ARC_RADIUS
        circle_distance = jnp.square(point[0] - arc_center) + jnp.square(point[1])
        inside = rectangle_inside | (
            (point[0] >= depth) & (circle_distance <= config.GOAL_AREA_ARC_RADIUS**2)
        )
        projection = jnp.clip(
            jnp.sum((point - boundary) * edges, axis=-1)
            / jnp.maximum(jnp.sum(edges * edges, axis=-1), 1e-12),
            0,
            1,
        )
        closest = boundary + projection[:, None] * edges
        clearance = jnp.min(jnp.linalg.norm(point - closest, axis=-1))
        return inside & (clearance + 1e-6 >= config.BALL_RADIUS)

    return jnp.stack((contained(1.0), contained(-1.0)))


def _robot_ball_contacts(robots: jax.Array, ball: jax.Array) -> jax.Array:
    """Circle versus oriented square contact, rather than centre proximity."""
    delta = ball[:2] - robots[..., :2]
    cosine, sine = jnp.cos(robots[..., 2]), jnp.sin(robots[..., 2])
    local = jnp.stack(
        (
            cosine * delta[..., 0] + sine * delta[..., 1],
            -sine * delta[..., 0] + cosine * delta[..., 1],
        ),
        axis=-1,
    )
    outside = jnp.maximum(jnp.abs(local) - config.ROBOT_SIZE / 2.0, 0.0)
    return jnp.sum(outside * outside, axis=-1) <= (config.BALL_RADIUS + _CONTACT_TOLERANCE) ** 2


def _robot_contacts(robots: jax.Array) -> jax.Array:
    vertices = jax.vmap(_robot_vertices)(robots.reshape((-1, 6)))
    contacts = jax.vmap(
        lambda a: jax.vmap(lambda b: _convex_polygons_contact(a, b, _CONTACT_TOLERANCE))(vertices)
    )(vertices)
    contacts &= ~jnp.eye(config.N_TEAMS * config.N_ROBOTS, dtype=jnp.bool_)
    return contacts.reshape((config.N_TEAMS, config.N_ROBOTS, config.N_TEAMS, config.N_ROBOTS))


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
