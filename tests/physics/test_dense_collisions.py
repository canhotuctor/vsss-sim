"""Regression coverage for dense robot contacts against the former pair path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb


def _reference_sat(pos_a, theta_a, pos_b, theta_b):
    """Original scalar SAT, including first-minimum axis selection."""
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    ca, sa = jnp.cos(theta_a), jnp.sin(theta_a)
    cb, sb = jnp.cos(theta_b), jnp.sin(theta_b)
    axes = jnp.stack(
        (jnp.stack((ca, sa)), jnp.stack((-sa, ca)),
         jnp.stack((cb, sb)), jnp.stack((-sb, cb)))
    )

    def support(axis, cosine, sine):
        ax, ay = axis
        return half * (
            jnp.abs(cosine * ax + sine * ay)
            + jnp.abs(-sine * ax + cosine * ay)
        )

    projections = axes @ (pos_a - pos_b)
    overlaps = (
        jax.vmap(support, in_axes=(0, None, None))(axes, ca, sa)
        + jax.vmap(support, in_axes=(0, None, None))(axes, cb, sb)
        - jnp.abs(projections)
    )
    overlapping = jnp.all(overlaps > 0)
    index = jnp.argmin(overlaps)
    normal = axes[index] * jnp.where(projections[index] >= 0, 1.0, -1.0)
    return (
        overlapping,
        jnp.where(overlapping, normal, jnp.zeros(2, jnp.float32)),
        jnp.where(overlapping, overlaps[index], jnp.float32(0.0)),
    )


def _reference_pair_collisions(state):
    """Former 15-pair gather/scatter response, using original SAT."""
    robots = state.robots.reshape((6, 6))
    pair_i, pair_j = jnp.triu_indices(6, k=1)
    overlapping, normals, overlaps = jax.vmap(_reference_sat)(
        robots[pair_i, :2], robots[pair_i, 2],
        robots[pair_j, :2], robots[pair_j, 2],
    )
    separation = jnp.where(overlapping[:, None], normals * overlaps[:, None] * 0.5, 0.0)
    relative_velocity = robots[pair_i, 3:5] - robots[pair_j, 3:5]
    velocity_along = jnp.sum(relative_velocity * normals, axis=-1)
    do_impulse = overlapping & (velocity_along < 0)
    magnitude = jnp.where(
        do_impulse,
        -(1.0 + jnp.float32(config.ROBOT_WALL_RESTITUTION)) * velocity_along * 0.5,
        0.0,
    )
    impulse = magnitude[:, None] * normals
    positions = robots[:, :2].at[pair_i].add(separation).at[pair_j].add(-separation)
    velocities = robots[:, 3:5].at[pair_i].add(impulse).at[pair_j].add(-impulse)
    robots = robots.at[:, :2].set(positions).at[:, 3:5].set(velocities)
    return state._replace(robots=robots.reshape((2, 3, 6)))


def _state_with_robots(robots):
    return jb.empty_state()._replace(
        robots=jnp.asarray(robots, dtype=jnp.float32).reshape((2, 3, 6)),
        ball=jnp.asarray((0.2, 0.3, -0.4, 0.5), dtype=jnp.float32),
        score=jnp.asarray((1, 2), dtype=jnp.int32),
        t=jnp.float32(3.5),
        wheel_speeds=jnp.arange(12, dtype=jnp.float32).reshape((2, 3, 2)),
    )


def test_sat_one_hot_axis_selection_preserves_ties_and_contact_direction():
    poses = jnp.asarray(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.03, 0.03, 0.0),
            (0.0, 0.0, 0.7, 0.0, 0.0, -0.2),
            (0.02, 0.01, 0.7, 0.0, 0.0, 0.7),
            (0.0, 0.0, 0.0, config.ROBOT_SIZE, 0.0, 0.0),
            (0.0, 0.0, 0.5, 1.0, 1.0, -0.5),
        ),
        dtype=jnp.float32,
    )

    def evaluate(sat):
        return jax.jit(jax.vmap(lambda p: sat(p[:2], p[2], p[3:5], p[5])))(poses)

    actual = evaluate(jb._sat_square_overlap)
    expected = evaluate(_reference_sat)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)


def test_sat_tied_overlap_keeps_first_axis_gradient():
    position = jnp.asarray((0.02, 0.02), dtype=jnp.float32)

    def overlap_gradient(sat):
        return jax.grad(
            lambda point: sat(point, jnp.float32(0.0), jnp.zeros(2), jnp.float32(0.0))[2]
        )(position)

    np.testing.assert_array_equal(
        overlap_gradient(jb._sat_square_overlap), overlap_gradient(_reference_sat)
    )


@pytest.mark.parametrize("spread", (0.03, 0.1, 0.8))
def test_dense_contacts_match_unique_pair_response_for_random_clusters(spread):
    rng = np.random.default_rng(139)
    robots = rng.normal(size=(32, 6, 6)).astype(np.float32)
    robots[..., :2] *= np.float32(spread)
    robots[..., 2] *= np.float32(np.pi)
    states = jax.vmap(_state_with_robots)(jnp.asarray(robots))

    actual = jax.jit(jax.vmap(jb._robot_robot_collisions))(states)
    expected = jax.jit(jax.vmap(_reference_pair_collisions))(states)

    np.testing.assert_allclose(actual.robots, expected.robots, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual.robots[..., 2], states.robots[..., 2])
    np.testing.assert_array_equal(actual.robots[..., 5], states.robots[..., 5])
    np.testing.assert_allclose(
        np.asarray(actual.robots[..., 3:5]).sum(axis=(1, 2)),
        np.asarray(states.robots[..., 3:5]).sum(axis=(1, 2)),
        rtol=2e-6,
        atol=2e-6,
    )
    for field in ("ball", "score", "t", "wheel_speeds"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(states, field))


def test_coincident_robots_are_resolved_once_per_pair_with_original_orientation():
    robots = np.zeros((6, 6), dtype=np.float32)
    robots[:, 3:5] = np.arange(12, dtype=np.float32).reshape((6, 2))
    state = _state_with_robots(robots)

    actual = jax.jit(jb._robot_robot_collisions)(state)
    expected = jax.jit(_reference_pair_collisions)(state)

    np.testing.assert_allclose(actual.robots, expected.robots, rtol=1e-6, atol=1e-6)
    assert np.isfinite(actual.robots).all()


def test_separated_or_just_touching_robots_are_unchanged():
    robots = np.zeros((6, 6), dtype=np.float32)
    robots[:, 0] = (0.0, config.ROBOT_SIZE, 0.4, 0.6, 0.8, 1.0)
    robots[:, 3:6] = np.arange(18, dtype=np.float32).reshape((6, 3))
    state = _state_with_robots(robots)

    actual = jax.jit(jb._robot_robot_collisions)(state)

    for observed, unchanged in zip(actual, state, strict=True):
        np.testing.assert_array_equal(observed, unchanged)


def test_single_approaching_pair_uses_configured_restitution_without_double_counting():
    robots = np.zeros((6, 6), dtype=np.float32)
    robots[:, 0] = (0.0, 0.03, 0.4, 0.6, 0.8, 1.0)
    robots[:2, 3] = (1.0, -1.0)
    state = _state_with_robots(robots)

    actual = jax.jit(jb._robot_robot_collisions)(state).robots.reshape((6, 6))

    np.testing.assert_allclose(actual[:2, 3], (-config.ROBOT_WALL_RESTITUTION,
                                             config.ROBOT_WALL_RESTITUTION), atol=1e-6)
    np.testing.assert_allclose(actual[1, 0] - actual[0, 0], config.ROBOT_SIZE, atol=1e-6)
    np.testing.assert_array_equal(actual[2:], state.robots.reshape((6, 6))[2:])
