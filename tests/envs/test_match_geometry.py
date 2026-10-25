"""Regression coverage for fixed-capacity goal-area polygon clipping."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.envs.match_geometry import (
    _POLYGON_CAPACITY,
    _clip_polygon,
    _polygon_area,
    _robot_goal_area_fraction,
)


def _reference_fraction(robot: np.ndarray, side: int) -> float:
    """Independent float64 polygon oracle with ordinary variable-length lists."""
    half_robot = config.ROBOT_SIZE / 2.0
    corners = np.asarray(
        [[-half_robot, -half_robot], [half_robot, -half_robot],
         [half_robot, half_robot], [-half_robot, half_robot]]
    )
    cosine, sine = np.cos(robot[2]), np.sin(robot[2])
    polygon = list(corners @ np.asarray([[cosine, sine], [-sine, cosine]]) + robot[:2])
    half_length = config.FIELD_LENGTH / 2.0
    x_min = -half_length if side == 0 else half_length - config.GOAL_AREA_LENGTH_X
    x_max = x_min + config.GOAL_AREA_LENGTH_X
    half_width = config.GOAL_AREA_LENGTH_Y / 2.0
    for axis, boundary, greater in (
        (0, x_min, True), (0, x_max, False),
        (1, -half_width, True), (1, half_width, False),
    ):
        clipped = []
        if polygon:
            previous = polygon[-1]
            previous_inside = previous[axis] >= boundary if greater else previous[axis] <= boundary
            for current in polygon:
                current_inside = current[axis] >= boundary if greater else current[axis] <= boundary
                if previous_inside != current_inside:
                    fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                    clipped.append(previous + fraction * (current - previous))
                if current_inside:
                    clipped.append(current)
                previous, previous_inside = current, current_inside
        polygon = clipped
    if not polygon:
        return 0.0
    points = np.asarray(polygon)
    following = np.roll(points, -1, axis=0)
    area = 0.5 * abs(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]))
    return float(area / config.ROBOT_SIZE**2)


@pytest.mark.parametrize("side", [0, 1])
def test_fixed_goal_area_clipping_matches_polygon_oracle(side: int):
    generator = np.random.default_rng(824)
    robots = np.zeros((256, 6), dtype=np.float32)
    sign = -1 if side == 0 else 1
    robots[:, 0] = sign * (config.FIELD_LENGTH / 2.0 - config.GOAL_AREA_LENGTH_X / 2.0)
    robots[:, 0] += generator.uniform(-0.15, 0.15, len(robots))
    robots[:, 1] = generator.uniform(-0.45, 0.45, len(robots))
    robots[:, 2] = generator.uniform(-np.pi, np.pi, len(robots))

    actual = jax.jit(jax.vmap(lambda robot: _robot_goal_area_fraction(robot, jnp.int32(side))))(
        jnp.asarray(robots)
    )
    expected = [_reference_fraction(robot.astype(np.float64), side) for robot in robots]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert np.isfinite(actual).all()


def test_clipping_an_empty_polygon_stays_empty_and_finite():
    polygon = jnp.zeros((_POLYGON_CAPACITY, 2), dtype=jnp.float32)
    clipped, count = jax.jit(lambda points: _clip_polygon(points, jnp.int32(0), 0, jnp.float32(0.0), True))(
        polygon
    )

    assert int(count) == 0
    np.testing.assert_array_equal(clipped, polygon)


@pytest.mark.parametrize("half_extent, expected_count, expected_area", [(0.75, 8, 1.75), (0.0, 4, 0.0)])
def test_four_halfplanes_fit_capacity_including_boundary_duplicates(
    half_extent: float, expected_count: int, expected_area: float
):
    polygon = jnp.zeros((_POLYGON_CAPACITY, 2), dtype=jnp.float32).at[:4].set(
        jnp.asarray([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    )
    count = jnp.int32(4)
    for clip_number, (axis, boundary, greater) in enumerate(
        ((0, -half_extent, True), (0, half_extent, False),
         (1, -half_extent, True), (1, half_extent, False)),
        start=1,
    ):
        polygon, count = _clip_polygon(polygon, count, axis, jnp.float32(boundary), greater)
        assert int(count) <= 4 + clip_number <= _POLYGON_CAPACITY

    assert int(count) == expected_count
    np.testing.assert_allclose(_polygon_area(polygon, count), expected_area)


def test_goal_area_clipping_has_no_compiled_control_flow_loops():
    executable = jax.jit(_robot_goal_area_fraction).lower(
        jnp.zeros((6,), dtype=jnp.float32), jnp.int32(0)
    ).compile()
    hlo = executable.as_text()

    assert " while(" not in hlo
    assert " conditional(" not in hlo
