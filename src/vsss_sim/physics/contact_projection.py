"""Bounded, position-only projection for coupled robot/ball contacts.

The velocity-level collision pass remains responsible for impulses, restitution,
friction and spin. These iterations only repair geometry; their equal positional
weights are a numerical preconditioner, not a model of physical masses.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from .. import config
from . import swept_collisions

# Treat sub-micrometre numerical overlap as contact slop. Counting a tiny
# roundoff contact in a normalization denominator can otherwise halve the
# correction of another, deeply overlapping pair.
_CONTACT_SLOP = 1e-6
_N_ROBOTS = config.N_TEAMS * config.N_ROBOTS
_PAIR_A = jnp.asarray([i for i in range(_N_ROBOTS) for j in range(i + 1, _N_ROBOTS)])
_PAIR_B = jnp.asarray([j for i in range(_N_ROBOTS) for j in range(i + 1, _N_ROBOTS)])


def _first_wall_contact(start, proposed):
    """Conservative one-cast positional clip; it may stop tangential motion."""
    segments = swept_collisions._field_segments()
    delta = proposed - start
    vertices = swept_collisions._vertices(segments, swept_collisions._FIELD_VERTEX_ENDS)
    times, _ = swept_collisions._cast_segments(start, delta, segments, vertices)
    times = swept_collisions._allow_field_reentry(times, start, delta, segments)
    fraction = jnp.minimum(jnp.min(times), 1.0)
    end = start + fraction * delta
    a, b = segments[:, 0], segments[:, 1]
    edge = b - a
    edge2 = jnp.sum(edge * edge, axis=-1)
    projection = jnp.sum((start[None, :] - a) * edge, axis=-1) / edge2
    inward = swept_collisions._field_inward_normals()
    signed_start = jnp.sum((start[None, :] - a) * inward, axis=-1)
    signed_end = jnp.sum((end[None, :] - a) * inward, axis=-1)
    near_face = (projection >= 0.0) & (projection <= 1.0) & (
        signed_start < config.BALL_RADIUS - 1e-6
    )
    farther_outside = jnp.any(near_face & (signed_end < signed_start - 1e-7))

    def dist2(pos):
        u = jnp.clip(jnp.sum((pos[None, :] - a) * edge, axis=-1) / edge2, 0.0, 1.0)
        difference = pos[None, :] - (a + u[:, None] * edge)
        return jnp.sum(difference * difference, axis=-1)

    initial2 = dist2(start)
    final2 = dist2(end)
    _, outside = swept_collisions._outside_wall_geometry(start, segments)
    radius_min = config.BALL_RADIUS - 1e-6
    worsens = jnp.any(
        (initial2 < radius_min * radius_min)
        & (final2 < initial2 - 1e-9) & ~outside
    )
    return jnp.where(farther_outside | worsens, start, end)


def project_contacts(
    state: Any,
    ball_obb_penetration: Callable,
    sat_square_overlap: Callable,
    robot_wall: Callable,
    robot_chamfer: Callable,
    *,
    iterations: int = 4,
) -> Any:
    """Project coupled position constraints while preserving every velocity.

    Ball corrections stop at their first static-wall contact and do not impart
    an impulse or score a goal. The initial ball position must already be in
    the permitted region; continuous movement should precede this projection.

    Jacobi corrections are recomputed each iteration and damped for bodies with
    multiple simultaneous contacts. Wall projections run after every pass.
    The fixed iteration budget can leave residual penetration, especially in
    dense/infeasible configurations; it is not an intersection-free guarantee.
    """
    original = state

    def iteration(_, current):
        robots = current.robots.reshape((_N_ROBOTS, 6))
        ball_xy = current.ball[:2]
        ball_normals, ball_penetrations = jax.vmap(
            ball_obb_penetration, in_axes=(None, 0, 0)
        )(ball_xy, robots[:, :2], robots[:, 2])
        ball_contacts = ball_penetrations > _CONTACT_SLOP
        ball_depth = jnp.where(ball_contacts, ball_penetrations, 0.0)
        ball_separation = 0.5 * ball_normals * ball_depth[:, None]
        ball_count = jnp.maximum(jnp.sum(ball_contacts), 1)
        proposed_ball = ball_xy + jnp.sum(ball_separation, axis=0) / ball_count
        # Position projection may stop at the first obstacle. Time-integrated
        # motion still uses the full swept bounce/sliding implementation.
        safe_ball = _first_wall_contact(ball_xy, proposed_ball)

        first = robots[_PAIR_A]
        second = robots[_PAIR_B]
        overlapping, normals, overlaps = jax.vmap(sat_square_overlap)(
            first[:, :2], first[:, 2], second[:, :2], second[:, 2]
        )
        overlapping = overlapping & (overlaps > _CONTACT_SLOP)
        pair_separation = 0.5 * normals * overlaps[:, None]
        pair_separation = jnp.where(overlapping[:, None], pair_separation, 0.0)
        pair_delta = jnp.zeros((_N_ROBOTS, 2), dtype=robots.dtype)
        pair_delta = pair_delta.at[_PAIR_A].add(pair_separation)
        pair_delta = pair_delta.at[_PAIR_B].add(-pair_separation)

        # Estimate the fraction of each body's proposed pair correction that
        # a field wall blocks. If exactly one side of a pair is wall-pinned,
        # transfer its positional correction to the free side of the pair.
        probe_robots = robots.at[:, :2].add(pair_delta)
        probe = current._replace(robots=probe_robots.reshape(current.robots.shape))
        probe = robot_chamfer(robot_wall(probe))
        actual_delta = probe.robots.reshape((_N_ROBOTS, 6))[:, :2] - robots[:, :2]
        desired2 = jnp.sum(pair_delta * pair_delta, axis=-1)
        ratio = jnp.sum(actual_delta * pair_delta, axis=-1) / jnp.where(desired2 > 1e-16, desired2, 1.0)
        blocked = jnp.where(desired2 > 1e-16, jnp.clip(1.0 - ratio, 0.0, 1.0), 0.0)
        weight_a = jnp.clip(1.0 + blocked[_PAIR_B] - blocked[_PAIR_A], 0.0, 2.0)
        weight_b = 2.0 - weight_a
        pair_delta = jnp.zeros((_N_ROBOTS, 2), dtype=robots.dtype)
        pair_delta = pair_delta.at[_PAIR_A].add(pair_separation * weight_a[:, None])
        pair_delta = pair_delta.at[_PAIR_B].add(-pair_separation * weight_b[:, None])
        pair_count = jnp.zeros((_N_ROBOTS,), dtype=robots.dtype)
        pair_count = pair_count.at[_PAIR_A].add(overlapping.astype(robots.dtype))
        pair_count = pair_count.at[_PAIR_B].add(overlapping.astype(robots.dtype))

        # Absorb the portion of each separation that the ball could not make.
        # In particular a wall-pinned or symmetrically pinched ball remains
        # still, so the robot moves by the full penetration in this pass. This
        # is geometric preconditioning only; it does not change the impulses.
        ball_move = safe_ball - ball_xy
        achieved = jnp.sum(ball_normals * ball_move[None, :], axis=-1)
        robot_depth = jnp.clip(ball_depth - achieved, 0.0, 1.5 * ball_depth)
        robot_ball_delta = -ball_normals * robot_depth[:, None]
        robot_delta = pair_delta + robot_ball_delta
        contact_count = pair_count + ball_contacts.astype(robots.dtype)
        robot_delta /= jnp.maximum(contact_count[:, None], 1.0)
        robots = robots.at[:, :2].add(robot_delta)
        current = current._replace(
            ball=current.ball.at[:2].set(safe_ball),
            robots=robots.reshape(current.robots.shape),
        )
        # Existing wall helpers also touch velocity; retain only their positions.
        wall_state = robot_chamfer(robot_wall(current))
        return current._replace(robots=current.robots.at[:, :, :2].set(wall_state.robots[:, :, :2]))

    result = jax.lax.fori_loop(0, iterations, iteration, state)
    # Keep the velocity/friction/spin effects of the single collision impulse
    # pass; position relaxation is intentionally not an additional bounce.
    robots = result.robots.at[:, :, 3:].set(original.robots[:, :, 3:])
    ball = result.ball.at[2:].set(original.ball[2:])
    return result._replace(
        ball=ball,
        robots=robots,
        ball_spin=original.ball_spin,
        ball_vertical=original.ball_vertical,
    )
