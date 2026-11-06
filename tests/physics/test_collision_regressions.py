"""Regression scenarios for continuous movement and coupled contacts.

These exercise the public simulation step, including the intermediate physics
frames, rather than invoking individual collision helpers in isolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as physics

# 0.1 mm allows for float32 arithmetic but is small relative to the ball.
_CONTACT_TOLERANCE = 1e-4
# A bounded iteration budget leaves a few millimetres of overlap in the worst
# case (a full-speed driver pushing a robot train into a wall). 4 mm is ~5% of
# a robot; the equal-weight projection this replaced left ~8 mm.
_PUSHED_TRAIN_TOLERANCE = 4e-3


def _separated_state():
    """A deterministic state with the six unused robots out of the way."""
    state = physics.empty_state()
    positions = jnp.array(
        [[-0.5, -0.4], [-0.25, -0.4], [0.0, -0.4],
         [0.25, -0.4], [0.5, -0.4], [0.0, 0.4]],
        dtype=jnp.float32,
    )
    robots = state.robots.reshape((6, 6)).at[:, :2].set(positions)
    return state._replace(robots=robots.reshape(state.robots.shape))


def _zero_actions():
    return jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)


def _ball_square_clearance(ball_xy, robot):
    """Independent circle-to-oriented-square distance from reported poses."""
    theta = float(robot[2])
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    local = rotation @ (np.asarray(ball_xy) - np.asarray(robot[:2]))
    outside = np.maximum(np.abs(local) - config.ROBOT_SIZE / 2.0, 0.0)
    return float(np.linalg.norm(outside)) - config.BALL_RADIUS


@pytest.mark.parametrize("side", [-1, 1])
def test_fast_ball_beside_goal_bounces_without_false_goal(side):
    state = _separated_state()._replace(
        ball=jnp.array(
            [side * (config.FIELD_LENGTH / 2 - config.BALL_RADIUS - 0.0001),
             0.24, side * 1.4, 0.0],
            dtype=jnp.float32,
        )
    )
    final, info, frames = physics.step_with_frames(state, _zero_actions())

    assert int(info["goal"]) == 0
    assert np.all(side * np.asarray(frames.ball[:, 0]) <=
                  config.FIELD_LENGTH / 2 - config.BALL_RADIUS + _CONTACT_TOLERANCE)
    assert side * float(final.ball[2]) < 0.0


@pytest.mark.parametrize("side", [-1, 1])
def test_ball_fully_crossing_open_goal_is_scored(side):
    state = _separated_state()._replace(
        ball=jnp.array([side * (config.FIELD_LENGTH / 2 - 0.03), 0.0,
                        side * 1.4, 0.0], dtype=jnp.float32)
    )
    final, info = physics.step(state, _zero_actions())

    assert int(info["goal"]) == side
    assert side * float(final.ball[0]) > config.FIELD_LENGTH / 2 + config.BALL_RADIUS


@pytest.mark.parametrize("side", [-1, 1])
def test_robot_pressing_ball_against_side_wall_keeps_ball_inside(side):
    state = _separated_state()
    ball_limit = config.FIELD_WIDTH / 2 - config.BALL_RADIUS
    robot_y = ball_limit - config.ROBOT_SIZE / 2 - config.BALL_RADIUS + 0.002
    robot = jnp.array([0.0, side * robot_y, side * jnp.pi / 2], dtype=jnp.float32)
    state = state._replace(
        robots=state.robots.at[0, 0, :3].set(robot),
        ball=jnp.array([0.0, side * ball_limit, 0.0, 0.0], dtype=jnp.float32),
        wheel_speeds=state.wheel_speeds.at[0, 0, :].set(config.ROBOT_MAX_WHEEL_SPEED),
    )
    actions = _zero_actions().at[0, 0, :].set(1.0)
    _, _, frames = physics.step_with_frames(state, actions)

    assert np.all(side * np.asarray(frames.ball[:, 1]) <= ball_limit + _CONTACT_TOLERANCE)


@pytest.mark.parametrize("side", [-1, 1])
def test_robot_pushed_by_another_robot_stays_inside_side_wall(side):
    state = _separated_state()
    robot_limit = config.FIELD_WIDTH / 2 - config.ROBOT_SIZE / 2
    robots = state.robots.at[0, 0, :3].set(
        jnp.array([0.0, side * robot_limit, side * jnp.pi / 2])
    )
    robots = robots.at[0, 1, :3].set(
        jnp.array([0.0, side * (config.FIELD_WIDTH / 2 - 1.5 * config.ROBOT_SIZE + 0.001),
                   side * jnp.pi / 2])
    )
    state = state._replace(
        robots=robots,
        ball=jnp.array([-0.4, 0.2, 0.0, 0.0], dtype=jnp.float32),
        wheel_speeds=state.wheel_speeds.at[0, 1, :].set(config.ROBOT_MAX_WHEEL_SPEED),
    )
    actions = _zero_actions().at[0, 1, :].set(1.0)
    _, _, frames = physics.step_with_frames(state, actions)

    assert np.all(side * np.asarray(frames.robots[:, 0, 0, 1]) <=
                  robot_limit + _CONTACT_TOLERANCE)


def test_two_robots_cannot_drive_into_ball_from_opposite_sides():
    state = _separated_state()
    robots = state.robots.at[0, 0, :3].set(jnp.array([-0.06, 0.0, 0.0]))
    robots = robots.at[0, 1, :3].set(jnp.array([0.06, 0.0, jnp.pi]))
    state = state._replace(
        robots=robots,
        ball=jnp.zeros(4, dtype=jnp.float32),
        wheel_speeds=state.wheel_speeds.at[0, :2, :].set(config.ROBOT_MAX_WHEEL_SPEED),
    )
    actions = _zero_actions().at[0, :2, :].set(1.0)
    _, _, frames = physics.step_with_frames(state, actions)

    for ball, frame_robots in zip(np.asarray(frames.ball), np.asarray(frames.robots)):
        for robot in frame_robots[0, :2]:
            assert _ball_square_clearance(ball[:2], robot) >= -_CONTACT_TOLERANCE


@pytest.mark.parametrize("side", [-1, 1])
def test_fast_ball_cannot_pass_through_stationary_robot(side):
    state = _separated_state()
    state = state._replace(
        robots=state.robots.at[0, 0, :3].set(jnp.zeros(3, dtype=jnp.float32)),
        ball=jnp.array([-side * 0.07, 0.0, side * 8.0, 0.0], dtype=jnp.float32),
    )
    _, info, frames = physics.step_with_frames(state, _zero_actions())

    assert int(info["goal"]) == 0
    # Check the first substep, before a later wall rebound can change direction.
    assert side * float(frames.ball[0, 0]) < side * float(frames.robots[0, 0, 0, 0])
    assert side * float(frames.ball[0, 2]) < 0.0


def test_batched_step_agrees_with_individual_steps_for_collision_scenarios():
    state = _separated_state()
    balls = jnp.array(
        [[config.FIELD_LENGTH / 2 - config.BALL_RADIUS - 0.0001, 0.24, 1.4, 0.0],
         [-0.07, 0.0, 8.0, 0.0]],
        dtype=jnp.float32,
    )
    second = state._replace(robots=state.robots.at[0, 0, :3].set(jnp.zeros(3)))
    states = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), state, second)
    states = states._replace(ball=balls)
    actions = jnp.stack([_zero_actions(), _zero_actions()])

    batched, batched_info = jax.jit(jax.vmap(physics.step))(states, actions)
    for index, item in enumerate((state._replace(ball=balls[0]), second._replace(ball=balls[1]))):
        individual, info = physics.step(item, actions[index])
        np.testing.assert_allclose(batched.ball[index], individual.ball, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(batched.robots[index], individual.robots, rtol=1e-5, atol=1e-6)
        assert int(batched_info["goal"][index]) == int(info["goal"])


def test_batched_jumanji_step_distinguishes_post_wall_from_valid_goal():
    pytest.importorskip("jumanji")
    from vsss_sim.envs.jumanji import VSSJumanjiEnv

    env = VSSJumanjiEnv(terminate_on_goal=True)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(42), 3))
    simulation = _separated_state()
    simulations = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), simulation)
    post_x = config.FIELD_LENGTH / 2 - config.BALL_RADIUS - 0.0001
    balls = jnp.array([[post_x, 0.24, 1.4, 0.0],
                       [-post_x, 0.24, -1.4, 0.0],
                       [config.FIELD_LENGTH / 2 - 0.03, 0.0, 1.4, 0.0]],
                      dtype=jnp.float32)
    states = states._replace(simulation=simulations._replace(ball=balls))
    next_states, timesteps = jax.jit(jax.vmap(env.step))(
        states, jnp.zeros((3, 6), dtype=jnp.float32)
    )

    np.testing.assert_array_equal(timesteps.extras["goal"], [0, 0, 1])
    np.testing.assert_array_equal(timesteps.last(), [False, False, True])
    np.testing.assert_array_equal(next_states.simulation.score[:, 0], [0, 0, 1])


def _max_robot_overlap(frame_robots):
    robots = np.asarray(frame_robots).reshape((-1, 6))
    first, second = np.triu_indices(robots.shape[0], 1)
    overlapping, _, overlap = jax.vmap(physics._sat_square_overlap)(
        robots[first, :2], robots[first, 2], robots[second, :2], robots[second, 2]
    )
    return float(jnp.max(jnp.where(overlapping, overlap, 0.0)))


@pytest.mark.parametrize("spin", [-1, 1])
@pytest.mark.parametrize("ball_at", [(0.06, 0.0), (0.0, 0.06), (-0.065, 0.02)])
def test_spinning_robot_corner_kicks_resting_ball_without_penetration(spin, ball_at):
    state = _separated_state()
    state = state._replace(
        robots=state.robots.at[0, 0, :3].set(jnp.zeros(3, dtype=jnp.float32)),
        ball=jnp.array([*ball_at, 0.0, 0.0], dtype=jnp.float32),
        wheel_speeds=state.wheel_speeds.at[0, 0, :].set(
            jnp.array([-1.0, 1.0]) * spin * config.ROBOT_MAX_WHEEL_SPEED
        ),
    )
    actions = _zero_actions().at[0, 0, :].set(jnp.array([-1.0, 1.0]) * spin)
    final, _, frames = physics.step_with_frames(state, actions)

    for ball, frame_robots in zip(np.asarray(frames.ball), np.asarray(frames.robots)):
        assert _ball_square_clearance(ball[:2], frame_robots[0, 0]) >= -_CONTACT_TOLERANCE
    # A corner sweeping at ~2.6 m/s transfers momentum; a mere positional
    # push-out would leave the ball at rest.
    assert np.linalg.norm(np.asarray(final.ball[2:])) > 0.1


@pytest.mark.parametrize("side", [-1, 1])
def test_robot_train_pushed_into_side_wall_stays_separated_and_inside(side):
    state = _separated_state()
    limit = config.FIELD_WIDTH / 2 - config.ROBOT_SIZE / 2
    pitch = config.ROBOT_SIZE + 0.001
    robots = state.robots
    for index, offset in enumerate((0.0, 1.0, 2.0)):
        robots = robots.at[0, index, :3].set(
            jnp.array([0.0, side * (limit - offset * pitch), side * jnp.pi / 2])
        )
    state = state._replace(
        robots=robots,
        ball=jnp.array([-0.4, 0.2, 0.0, 0.0], dtype=jnp.float32),
        wheel_speeds=state.wheel_speeds.at[0, 2, :].set(config.ROBOT_MAX_WHEEL_SPEED),
    )
    actions = _zero_actions().at[0, 2, :].set(1.0)
    _, _, frames = physics.step_with_frames(state, actions)

    assert np.all(side * np.asarray(frames.robots[:, 0, :3, 1]) <= limit + _CONTACT_TOLERANCE)
    for frame_robots in frames.robots:
        assert _max_robot_overlap(frame_robots) <= _PUSHED_TRAIN_TOLERANCE


def test_robot_pair_projection_converges_for_a_crowd_in_open_field():
    state = _separated_state()
    pitch = config.ROBOT_SIZE * 1.02
    grid = jnp.array([[(i % 3 - 1) * pitch, (i // 3 - 0.5) * pitch] for i in range(6)])
    robots = state.robots.reshape((6, 6)).at[:, :2].set(grid)
    state = state._replace(
        robots=robots.reshape(state.robots.shape),
        wheel_speeds=jnp.full_like(state.wheel_speeds, config.ROBOT_MAX_WHEEL_SPEED),
        ball=jnp.array([0.5, 0.5, 0.0, 0.0], dtype=jnp.float32),
    )
    # Every robot drives forward (+x) into the one ahead of it.
    _, _, frames = physics.step_with_frames(state, jnp.ones_like(_zero_actions()))
    for frame_robots in frames.robots:
        assert _max_robot_overlap(frame_robots) <= _PUSHED_TRAIN_TOLERANCE
