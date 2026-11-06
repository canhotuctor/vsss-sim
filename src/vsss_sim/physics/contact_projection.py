"""Bounded, position-only projection for coupled robot/ball contacts.

The velocity-level collision pass remains responsible for impulses, restitution,
friction and spin. These iterations only repair geometry; their equal positional
weights are a numerical preconditioner, not a model of physical masses.

Robot pairs are relaxed Gauss-Seidel style: the complete graph on six robots
splits into five perfect matchings, so each round corrects three independent
pairs at once, then re-clamps against the field. Walls therefore act as
zero-inverse-mass bodies that every later round sees, rather than something
estimated once per pass.
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
_CONTACT_SLOP = swept_collisions.CONTACT_SLOP
_N_ROBOTS = config.N_TEAMS * config.N_ROBOTS
_PAIR_A = jnp.asarray([i for i in range(_N_ROBOTS) for j in range(i + 1, _N_ROBOTS)])
_PAIR_B = jnp.asarray([j for i in range(_N_ROBOTS) for j in range(i + 1, _N_ROBOTS)])


def _matching_rounds(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Round-robin schedule: ``n - 1`` rounds of ``n // 2`` disjoint pairs."""
    assert n % 2 == 0
    rounds = []
    for r in range(n - 1):
        pairs = [(r, n - 1)]
        for k in range(1, n // 2):
            pairs.append(((r + k) % (n - 1), (r - k) % (n - 1)))
        rounds.append(tuple(pairs))
    return tuple(rounds)


_ROUNDS = _matching_rounds(_N_ROBOTS)
_ROUND_A = jnp.asarray([[a for a, _ in rnd] for rnd in _ROUNDS])
_ROUND_B = jnp.asarray([[b for _, b in rnd] for rnd in _ROUNDS])


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

    Ball corrections are Jacobi, averaged over simultaneous contacts. Robot
    pairs are Gauss-Seidel over five matchings; a robot the field clamped on the
    previous iteration is treated as immovable, so its partner absorbs the whole
    separation. The fixed iteration budget can leave residual penetration,
    especially in dense/infeasible configurations; it is not an
    intersection-free guarantee.
    """
    original = state

    def iteration(_, carry):
        current, pinned = carry
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

        # Absorb the portion of each separation that the ball could not make.
        # In particular a wall-pinned or symmetrically pinched ball remains
        # still, so the robot moves by the full penetration in this pass. This
        # is geometric preconditioning only; it does not change the impulses.
        ball_move = safe_ball - ball_xy
        achieved = jnp.sum(ball_normals * ball_move[None, :], axis=-1)
        robot_depth = jnp.clip(ball_depth - achieved, 0.0, 1.5 * ball_depth)
        robot_ball_delta = -ball_normals * robot_depth[:, None]
        robots = robots.at[:, :2].add(robot_ball_delta)
        current = current._replace(
            ball=current.ball.at[:2].set(safe_ball),
            robots=robots.reshape(current.robots.shape),
        )

        def clamp_to_field(state_):
            # Existing wall helpers also touch velocity; retain only positions.
            walled = robot_chamfer(robot_wall(state_))
            return state_._replace(
                robots=state_.robots.at[:, :, :2].set(walled.robots[:, :, :2])
            )

        def pair_round(index, state_):
            flat = state_.robots.reshape((_N_ROBOTS, 6))
            a, b = _ROUND_A[index], _ROUND_B[index]
            overlapping, normals, overlaps = jax.vmap(sat_square_overlap)(
                flat[a, :2], flat[a, 2], flat[b, :2], flat[b, 2]
            )
            overlapping = overlapping & (overlaps > _CONTACT_SLOP)
            # Inverse-mass weights: a robot the field clamped last pass cannot
            # yield, so its partner absorbs the whole separation.
            free_a, free_b = 1.0 - pinned[a], 1.0 - pinned[b]
            total = free_a + free_b
            share_a = jnp.where(total > 0.0, free_a / jnp.where(total > 0.0, total, 1.0), 0.5)
            full = jnp.where(overlapping[:, None], normals * overlaps[:, None], 0.0)
            flat = flat.at[a, :2].add(full * share_a[:, None]).at[b, :2].add(-full * (1.0 - share_a)[:, None])
            return state_._replace(robots=flat.reshape(state_.robots.shape))

        current = clamp_to_field(current)
        moved = jax.lax.fori_loop(0, _ROUND_A.shape[0], pair_round, current)
        clamped = clamp_to_field(moved)
        shove = clamped.robots[:, :, :2] - moved.robots[:, :, :2]
        pinned = (jnp.linalg.norm(shove, axis=-1) > _CONTACT_SLOP).reshape(-1).astype(robots.dtype)
        return clamped, pinned

    result, _ = jax.lax.fori_loop(
        0, iterations, iteration, (state, jnp.zeros((_N_ROBOTS,), dtype=state.robots.dtype))
    )
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
