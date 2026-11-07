"""Bounded continuous collision checks for the ball's planar motion.

The field is a fixed set of segments. Robots are swept as translating squares
with the ball's path in the robot frame swept over a few chords; the residual
rotation error is handled discretely by the contact solver. The
bounded dynamic loops allow early exit but do not support JAX reverse-mode
differentiation through the physics step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config


def _field_segments() -> jnp.ndarray:
    """The field perimeter, including goal pockets, as finite segments."""
    x = config.FIELD_LENGTH / 2.0
    y = config.FIELD_WIDTH / 2.0
    g = config.GOAL_WIDTH / 2.0
    d = config.GOAL_DEPTH
    c = config.FIELD_CHAMFER
    return jnp.asarray([
        ((-x + c, y), (x - c, y)),
        ((-x + c, -y), (x - c, -y)),
        ((x - c, y), (x, y - c)),
        ((x - c, -y), (x, -y + c)),
        ((-x + c, y), (-x, y - c)),
        ((-x + c, -y), (-x, -y + c)),
        ((x, y - c), (x, g)),
        ((x, -g), (x, -y + c)),
        ((-x, y - c), (-x, g)),
        ((-x, -g), (-x, -y + c)),
        ((x, g), (x + d, g)),
        ((x, -g), (x + d, -g)),
        ((-x, g), (-x - d, g)),
        ((-x, -g), (-x - d, -g)),
        ((x + d, -g), (x + d, g)),
        ((-x - d, -g), (-x - d, g)),
    ], dtype=jnp.float32)


def _square_segments() -> jnp.ndarray:
    h = config.ROBOT_SIZE / 2.0
    return jnp.asarray([
        ((-h, -h), (h, -h)),
        ((h, -h), (h, h)),
        ((h, h), (-h, h)),
        ((-h, h), (-h, -h)),
    ], dtype=jnp.float32)


_FIELD_VERTEX_ENDS = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 1),
                      (4, 1), (5, 1), (6, 1), (7, 0), (8, 1), (9, 0),
                      (10, 1), (11, 1), (12, 1), (13, 1))
_FIELD_VERTEX_SEGMENTS = ((0, 4), (0, 2), (1, 5), (1, 3), (2, 6), (3, 7),
                          (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13),
                          (10, 14), (11, 14), (12, 15), (13, 15))
_SQUARE_VERTEX_ENDS = ((0, 0), (0, 1), (1, 1), (2, 1))


# A robot spins up to ~49 rad/s, nearly a radian per substep. The ball's path in
# the robot frame is an arc; each interval replaces it by its chord. The chord's
# sagitta is r * dtheta**2 / 8 (well under 1 mm at four intervals), and the
# contact normal uses the interval's mean orientation.
ROBOT_SWEEP_INTERVALS = 2

# Ball wall/robot/goal events resolved per substep; leftover travel is dropped.
# The first UNROLLED_BALL_EVENTS run without a data-dependent loop.
MAX_BALL_EVENTS = 4
UNROLLED_BALL_EVENTS = 2

# A swept impact leaves the ball exactly touching. Overlap below this is float
# roundoff, not contact: counting it makes a second impulse depend on noise.
CONTACT_SLOP = 1e-6


def _vertices(segments, ends):
    return segments[jnp.asarray([i for i, _ in ends]),
                    jnp.asarray([j for _, j in ends])]


def _field_inward_normals() -> jnp.ndarray:
    """Normal toward the playable field or pocket for each field segment."""
    k = 2.0 ** -0.5
    return jnp.asarray([
        (0, -1), (0, 1), (-k, -k), (-k, k), (k, -k), (k, k),
        (-1, 0), (-1, 0), (1, 0), (1, 0),
        (0, -1), (0, 1), (0, -1), (0, 1), (-1, 0), (1, 0),
    ], dtype=jnp.float32)


def _outside_wall_geometry(pos: jnp.ndarray, segments: jnp.ndarray):
    a, b = segments[:, 0], segments[:, 1]
    edge = b - a
    length2 = jnp.sum(edge * edge, axis=-1)
    projection = jnp.sum((pos[None, :] - a) * edge, axis=-1) / length2
    signed = jnp.sum((pos[None, :] - a) * _field_inward_normals(), axis=-1)
    outside = (projection >= 0.0) & (projection <= 1.0) & (signed < 0.0)
    return signed, outside


def _allow_field_reentry(times, pos, delta, segments):
    """Do not bounce an already-outside centre back to the outside."""
    _, outside = _outside_wall_geometry(pos, segments)
    toward_interior = jnp.sum(_field_inward_normals() * delta[None, :], axis=-1) > 0.0
    ignore = outside & toward_interior
    # The same vertex belongs to two segments. The original duplicated cast
    # still permits the vertex whenever either of its segments permits it.
    vertex_segments = jnp.asarray(_FIELD_VERTEX_SEGMENTS)
    ignore_vertices = ignore[vertex_segments[:, 0]] & ignore[vertex_segments[:, 1]]
    ignore_candidates = jnp.concatenate([ignore, ignore_vertices])
    return jnp.where(ignore_candidates, jnp.inf, times)


def _cast_segments(
    pos: jnp.ndarray, delta: jnp.ndarray, segments: jnp.ndarray,
    vertices: jnp.ndarray, inward: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cast a circle centre against finite segments expanded by its radius.

    Return candidate fractions and unit normals for faces and both endpoints.
    Initial overlaps are left for the discrete position solver, except that
    when ``inward`` (the playable-side normal of each segment) is given, a
    face the centre is still on the playable side of, and is moving into,
    is a t=0 hit: otherwise a ball a hair inside a wall escapes it in one
    substep. Candidates
    at contact moving away or parallel are rejected, avoiding repeated zero-time
    events. Fractions outside [0, 1] are returned as infinity.
    """
    radius = jnp.float32(config.BALL_RADIUS)
    a, b = segments[:, 0], segments[:, 1]
    edge = b - a
    length = jnp.linalg.norm(edge, axis=-1)
    tangent = edge / length[:, None]
    normals = jnp.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    relative = pos[None, :] - a
    signed = jnp.sum(relative * normals, axis=-1)
    side = jnp.where(signed >= 0.0, 1.0, -1.0)
    normal_speed = jnp.sum(delta[None, :] * normals, axis=-1)
    safe_speed = jnp.where(jnp.abs(normal_speed) > 1e-12, normal_speed, 1.0)
    face_t = (side * radius - signed) / safe_speed
    overlapping = jnp.abs(signed) < radius - 1e-6
    if inward is not None:
        playable_side = signed * jnp.sum(normals * inward, axis=-1) > 0.0
        face_t = jnp.where(overlapping & playable_side, 0.0, face_t)
        overlapping = overlapping & ~playable_side
    face_point = pos[None, :] + face_t[:, None] * delta[None, :]
    along = jnp.sum((face_point - a) * tangent, axis=-1)
    face_valid = (
        ~overlapping
        & (normal_speed * side < -1e-9)
        # Float32 may represent an exact contact just inside the radius. For
        # an approaching sweep treat that numerical overlap as a t=0 hit,
        # irrespective of the total displacement's magnitude.
        & ((face_t >= 0.0) | (jnp.abs(signed) <= radius + 1e-6))
        & (face_t <= 1.0)
        & (along >= 0.0) & (along <= length)
    )
    face_t = jnp.where(face_valid, jnp.maximum(face_t, 0.0), jnp.inf)
    face_n = normals * side[:, None]

    # The moving point enters each distinct endpoint's radius-sized circle.
    offset = pos[None, :] - vertices
    aa = jnp.dot(delta, delta)
    safe_aa = jnp.where(aa > 1e-16, aa, 1.0)
    bb = jnp.sum(offset * delta[None, :], axis=-1)
    cc = jnp.sum(offset * offset, axis=-1) - radius * radius
    disc = bb * bb - aa * cc
    corner_t = (-bb - jnp.sqrt(jnp.maximum(disc, 0.0))) / safe_aa
    corner_vec = offset + corner_t[:, None] * delta[None, :]
    # At a valid root the distance is exactly radius. Divide by that known
    # value here, avoiding a norm per candidate; normalize the selected
    # candidate once after argmin to absorb floating-point roundoff.
    corner_n = corner_vec / radius
    approaching = jnp.sum(corner_n * delta[None, :], axis=-1) < -1e-9
    corner_valid = (
        (aa > 1e-16) & (disc >= 0.0) & (cc >= -1e-7)
        & (corner_t >= -1e-6) & (corner_t <= 1.0) & approaching
    )
    corner_t = jnp.where(corner_valid, jnp.maximum(corner_t, 0.0), jnp.inf)
    return (
        jnp.concatenate([face_t, corner_t]),
        jnp.concatenate([face_n, corner_n]),
    )


def _to_local(value: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * value[..., 0] + s * value[..., 1],
                      -s * value[..., 0] + c * value[..., 1]], axis=-1)


def _to_world(value: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * value[..., 0] - s * value[..., 1],
                      s * value[..., 0] + c * value[..., 1]], axis=-1)


def clip_ball_displacement(start_xy: jnp.ndarray, proposed_xy: jnp.ndarray) -> jnp.ndarray:
    """Limit a positional correction to the first static-wall contact.

    If the starting point already overlaps a wall, a correction that would
    deepen any such overlap is rejected; a correction that improves it is
    permitted, subject to the remaining walls. Existing overlaps still require
    the caller's discrete recovery pass.
    """
    segments = _field_segments()
    vertices = _vertices(segments, _FIELD_VERTEX_ENDS)
    delta = proposed_xy - start_xy

    def slide(_, carry):
        position, remaining = carry
        times, normals = _cast_segments(position, remaining, segments, vertices)
        times = _allow_field_reentry(times, position, remaining, segments)
        index = jnp.argmin(times)
        hit = jnp.isfinite(times[index])
        fraction = jnp.minimum(times[index], 1.0)
        position = position + fraction * remaining
        rest = (1.0 - fraction) * remaining
        normal = normals[index]
        normal /= jnp.where(jnp.linalg.norm(normal) > 1e-12, jnp.linalg.norm(normal), 1.0)
        inward = jnp.minimum(jnp.dot(rest, normal), 0.0)
        rest = jnp.where(hit, rest - inward * normal, jnp.zeros_like(rest))
        return position, rest

    def slide_cond(carry):
        count, _, remaining = carry
        return (count < 3) & jnp.any(remaining != 0.0)

    def slide_body(carry):
        count, position, remaining = carry
        position, remaining = slide(count, (position, remaining))
        return count + 1, position, remaining

    _, end_xy, _ = jax.lax.while_loop(slide_cond, slide_body, (0, start_xy, delta))
    a, b = segments[:, 0], segments[:, 1]
    edge = b - a
    length2 = jnp.sum(edge * edge, axis=-1)

    def distance(pos):
        u = jnp.clip(jnp.sum((pos[None, :] - a) * edge, axis=-1) / length2, 0.0, 1.0)
        closest = a + u[:, None] * edge
        return jnp.linalg.norm(pos[None, :] - closest, axis=-1)

    initial_distance = distance(start_xy)
    final_distance = distance(end_xy)
    signed_start, outside = _outside_wall_geometry(start_xy, segments)
    worsens_overlap = jnp.any(
        (initial_distance < config.BALL_RADIUS - 1e-6)
        & (final_distance < initial_distance - 1e-7)
        & ~outside
    )
    signed_end = jnp.sum((end_xy[None, :] - a) * _field_inward_normals(), axis=-1)
    projection = jnp.sum((start_xy[None, :] - a) * edge, axis=-1) / length2
    near_face = (projection >= 0.0) & (projection <= 1.0) & (
        signed_start < config.BALL_RADIUS - 1e-6
    )
    farther_outside = jnp.any(near_face & (signed_end < signed_start - 1e-7))
    return jnp.where(worsens_overlap | farther_outside, start_xy, end_xy)


def advance_ball(state, robot_start, sub_dt, ball_face_impulse):
    """Advance ball to earliest field, translating-robot or goal event.

    ``state`` contains the pre-movement ball and predicted end-of-substep
    robots. ``robot_start`` contains robot poses at the beginning of that
    substep. A bounded ``MAX_BALL_EVENTS`` loop preserves the existing contact impulse
    (including ball spin and vertical velocity). If the event budget is fully
    consumed, residual travel is conservatively discarded for this substep.

    Robot paths use linear translation and linearly interpolated orientation,
    swept in ``ROBOT_SWEEP_INTERVALS`` chords of the ball's robot-frame path. Trajectory
    changes caused by impacts are not swept.
    """
    start = robot_start.reshape((-1, 6))
    end_pos = state.robots.reshape((-1, 6))[:, :2]
    robot_theta = start[:, 2]
    end_theta = state.robots.reshape((-1, 6))[:, 2]
    robot_dtheta = (end_theta - robot_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    field_segments = _field_segments()
    square_segments = _square_segments()
    field_vertices = _vertices(field_segments, _FIELD_VERTEX_ENDS)
    square_vertices = _vertices(square_segments, _SQUARE_VERTEX_ENDS)
    radius = jnp.float32(config.BALL_RADIUS)
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_goal = jnp.float32(config.GOAL_WIDTH / 2.0)
    position = state.ball[:2]
    already_goal = (jnp.abs(position[0]) > half_l + radius) & (
        jnp.abs(position[1]) + radius <= half_goal + 1e-7
    )
    initial_goal = jnp.where(already_goal, jnp.where(position[0] > 0, 1, -1), 0).astype(jnp.int32)

    def body(_, carry):
        current, remaining, elapsed, goal = carry
        pos = current.ball[:2]
        delta = current.ball[2:4] * remaining
        wall_t, wall_n = _cast_segments(pos, delta, field_segments, field_vertices, _field_inward_normals())
        wall_t = _allow_field_reentry(wall_t, pos, delta, field_segments)

        at_start = start[:, :2] + elapsed * (end_pos - start[:, :2])
        robot_delta = (1.0 - elapsed) * (end_pos - start[:, :2])
        # Cast the ball's path in the robot's own frame, one chord per interval
        # of the remaining window. Expressing both interval endpoints in the
        # robot frame at their own times captures the robot's rotation, so a
        # spinning corner can hit a stationary ball.
        interval_t, interval_n = [], []
        for k in range(ROBOT_SWEEP_INTERVALS):
            lead0 = k / ROBOT_SWEEP_INTERVALS
            lead1 = (k + 1) / ROBOT_SWEEP_INTERVALS
            theta0 = robot_theta + (elapsed + (1.0 - elapsed) * lead0) * robot_dtheta
            theta1 = robot_theta + (elapsed + (1.0 - elapsed) * lead1) * robot_dtheta
            local_pos = _to_local(pos[None, :] + lead0 * delta[None, :] - at_start - lead0 * robot_delta, theta0)
            local_end = _to_local(pos[None, :] + lead1 * delta[None, :] - at_start - lead1 * robot_delta, theta1)
            local_delta = local_end - local_pos
            theta_k = 0.5 * (theta0 + theta1)
            t_k, local_n = jax.vmap(_cast_segments, in_axes=(0, 0, None, None))(
                local_pos, local_delta, square_segments, square_vertices
            )
            # A segment cast is two-sided: without this mask a ball that starts
            # inside a robot can spuriously bounce from the robot's interior face.
            # Existing overlaps are resolved by the discrete contact solver.
            outside = jnp.maximum(jnp.abs(local_pos) - config.ROBOT_SIZE / 2.0, 0.0)
            initially_overlapping = jnp.linalg.norm(outside, axis=-1) < radius - 1e-6
            t_k = jnp.where(initially_overlapping[:, None], jnp.inf, t_k)
            interval_t.append((k + t_k) / ROBOT_SWEEP_INTERVALS)
            interval_n.append(_to_world(local_n, theta_k[:, None]))
        interval_t = jnp.stack(interval_t)
        first_interval = jnp.argmin(interval_t, axis=0)
        robot_t = jnp.min(interval_t, axis=0)
        robot_n = jnp.take_along_axis(
            jnp.stack(interval_n), first_interval[None, :, :, None], axis=0
        )[0]
        # The translating-square cast approximates rotation, while response
        # includes the robot's angular surface velocity. Discard candidates
        # that are separating under that same response model: they produce no
        # impulse and otherwise can recur at t=0 until the event budget is
        # exhausted, silently discarding the remainder of the substep.
        robots_at_end = current.robots.reshape((-1, 6))
        candidate_t = jnp.where(jnp.isfinite(robot_t), robot_t, 0.0)
        ball_at_hit = pos[None, None, :] + candidate_t[:, :, None] * delta[None, None, :]
        robot_at_hit = at_start[:, None, :] + candidate_t[:, :, None] * robot_delta[:, None, :]
        arm_at_hit = ball_at_hit - radius * robot_n - robot_at_hit
        angular_surface = robots_at_end[:, None, 5:6] * jnp.stack(
            [-arm_at_hit[..., 1], arm_at_hit[..., 0]], axis=-1
        )
        surface_at_hit = robots_at_end[:, None, 3:5] + angular_surface
        approach = jnp.sum((current.ball[None, None, 2:4] - surface_at_hit) * robot_n, axis=-1)
        robot_t = jnp.where(approach < -1e-7, robot_t, jnp.inf)
        robot_t = robot_t.reshape(-1)
        robot_n = robot_n.reshape((-1, 2))

        # The whole ball crosses the line before a goal is counted.
        sides = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
        safe_dx = jnp.where(jnp.abs(delta[0]) > 1e-12, delta[0], 1.0)
        goal_t = (sides * (half_l + radius) - pos[0]) / safe_dx
        y_at_goal = pos[1] + goal_t * delta[1]
        goal_valid = (
            (sides * delta[0] > 1e-9) & (sides * pos[0] <= half_l + radius)
            & (goal_t >= 0.0) & (goal_t <= 1.0)
            & (jnp.abs(y_at_goal) + radius <= half_goal + 1e-7)
        )
        goal_t = jnp.where(goal_valid, goal_t, jnp.inf)

        candidates = jnp.concatenate([wall_t, robot_t, goal_t])
        selected = jnp.argmin(candidates)
        fraction = jnp.minimum(candidates[selected], 1.0)
        active = (remaining > 0.0) & (goal == 0)
        wall_count = wall_t.shape[0]
        robot_count = robot_t.shape[0]
        hit_wall = active & (selected < wall_count) & jnp.isfinite(candidates[selected])
        hit_robot = active & (selected >= wall_count) & (selected < wall_count + robot_count) & jnp.isfinite(candidates[selected])
        hit_goal = active & (selected >= wall_count + robot_count) & jnp.isfinite(candidates[selected])
        fraction = jnp.where(active, fraction, 0.0)
        moved_pos = pos + fraction * delta
        # Keep the stored goal state strictly across the line, matching the
        # existing goal-state contract even when crossing occurs at t == 1.
        goal_direction = jnp.where(selected == wall_count + robot_count, -1.0, 1.0)
        moved_pos = moved_pos.at[0].add(jnp.where(hit_goal, goal_direction * 1e-7, 0.0))
        all_normals = jnp.concatenate([wall_n, robot_n, jnp.zeros((2, 2), jnp.float32)])
        normal = all_normals[selected]
        normal /= jnp.where(jnp.linalg.norm(normal) > 1e-12, jnp.linalg.norm(normal), 1.0)
        # Each square has four face and four unique-corner candidates.
        robot_index = jnp.clip((selected - wall_count) // (square_segments.shape[0] + square_vertices.shape[0]), 0, start.shape[0] - 1)
        robots = current.robots.reshape((-1, 6))
        robot = robots[robot_index]
        hit_elapsed = elapsed + (1.0 - elapsed) * fraction
        hit_robot_pos = start[robot_index, :2] + hit_elapsed * (end_pos[robot_index] - start[robot_index, :2])
        arm = moved_pos - radius * normal - hit_robot_pos
        surface_xy = robot[3:5] + robot[5] * jnp.stack([-arm[1], arm[0]])
        surface = jnp.concatenate([jnp.where(hit_robot, surface_xy, jnp.zeros(2)), jnp.zeros(1)])
        normal3 = jnp.concatenate([normal, jnp.zeros(1)])
        velocity = jnp.stack([current.ball[2], current.ball[3], current.ball_vertical[1]])
        mass_share = jnp.where(hit_robot, config.ROBOT_MASS / (config.ROBOT_MASS + config.BALL_MASS), 1.0)
        restitution = jnp.where(hit_robot, config.BALL_ROBOT_RESTITUTION, config.BALL_WALL_RESTITUTION)
        friction = jnp.where(hit_robot, config.BALL_ROBOT_FRICTION, config.BALL_WALL_FRICTION)
        dv, dspin = ball_face_impulse(velocity, current.ball_spin, normal3, surface, restitution, friction, mass_share)
        hit = hit_wall | hit_robot
        dv = jnp.where(hit, dv, 0.0)
        dspin = jnp.where(hit, dspin, 0.0)
        reaction = -normal * jnp.dot(dv, normal3) * (config.BALL_MASS / config.ROBOT_MASS)
        robots = robots.at[robot_index, 3:5].add(jnp.where(hit_robot, reaction, 0.0))
        velocity = velocity + dv
        new_goal = jnp.where(hit_goal, jnp.where(selected == wall_count + robot_count, -1, 1), goal).astype(jnp.int32)
        current = current._replace(
            ball=jnp.concatenate([moved_pos, velocity[:2]]),
            ball_spin=current.ball_spin + dspin,
            ball_vertical=current.ball_vertical.at[1].set(velocity[2]),
            robots=robots.reshape(current.robots.shape),
        )
        remaining = jnp.where(active & (hit_wall | hit_robot), remaining * (1.0 - fraction), 0.0)
        elapsed = jnp.where(active, hit_elapsed, elapsed)
        return current, remaining, elapsed, new_goal

    # Nearly every environment needs one event and a batch rarely more than
    # two, so run the first ``UNROLLED_BALL_EVENTS`` unconditionally (no loop
    # predicate, which a GPU while loop syncs to the host on every trip) and
    # loop only for the rare remainder. Once an environment has no remaining
    # travel or has scored, ``body`` is a no-op for it (zero fraction, no impulse).
    carry = (state, sub_dt, jnp.float32(0.0), initial_goal)
    for event in range(UNROLLED_BALL_EVENTS):
        carry = body(event, carry)

    def event_cond(loop_carry):
        count, (_, remaining, _, goal) = loop_carry
        return (count < MAX_BALL_EVENTS) & (remaining > 0.0) & (goal == 0)

    def event_body(loop_carry):
        count, inner = loop_carry
        return count + 1, body(count, inner)

    _, (current, _, _, goal) = jax.lax.while_loop(
        event_cond, event_body, (UNROLLED_BALL_EVENTS, carry)
    )
    return current, goal
