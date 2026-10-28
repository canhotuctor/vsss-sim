"""Ball spin, skidding, rolling, and pop-ups in the JAX physics engine."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.envs.match_referee import RestartType
from vsss_sim.envs.match_restarts import _restart_positions
from vsss_sim.physics import jax_backend as jb

R = config.BALL_RADIUS
K = config.BALL_INERTIA_FACTOR
G = config.GRAVITY
IDLE = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
step = jax.jit(jb.step)


def _parked(ball=(-0.5, 0.45, 0.0, 0.0), spin=(0.0, 0.0, 0.0), vertical=(0.0, 0.0)):
    """Robots parked along the bottom of the field, clear of the ball."""
    robots = np.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=np.float32)
    spots = [(-0.3, -0.55), (-0.1, -0.55), (0.1, -0.55), (0.3, -0.55), (0.5, -0.55), (-0.5, -0.55)]
    for index, (x, y) in enumerate(spots):
        robots[index // config.N_ROBOTS, index % config.N_ROBOTS, :2] = (x, y)
    return jb.empty_state()._replace(
        robots=jnp.asarray(robots),
        ball=jnp.asarray(ball, dtype=jnp.float32),
        ball_spin=jnp.asarray(spin, dtype=jnp.float32),
        ball_vertical=jnp.asarray(vertical, dtype=jnp.float32),
    )


def _floor_slip(state):
    """Velocity of the ball's floor contact point; zero while rolling."""
    vx, vy = state.ball[2], state.ball[3]
    wx, wy = state.ball_spin[0], state.ball_spin[1]
    return np.asarray([vx - R * wy, vy + R * wx])


def _run(state, steps, actions=IDLE):
    trajectory = [state]
    for _ in range(steps):
        state, _ = step(state, actions)
        trajectory.append(state)
    return trajectory


def _energy(state):
    """Ball kinetic (linear + spin) and potential energy per unit mass."""
    linear = state.ball[2] ** 2 + state.ball[3] ** 2 + state.ball_vertical[1] ** 2
    spin = K * R**2 * jnp.sum(state.ball_spin**2)
    return 0.5 * (linear + spin) + G * state.ball_vertical[0]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_ball_spin_and_height_start_at_rest():
    for state in (
        jb.empty_state(),
        jb.reset_kickoff(jax.random.PRNGKey(0)),
        jb.reset_random(jax.random.PRNGKey(1)),
    ):
        assert state.ball_spin.shape == (3,) and state.ball_spin.dtype == jnp.float32
        assert state.ball_vertical.shape == (2,) and state.ball_vertical.dtype == jnp.float32
        assert not np.any(np.asarray(state.ball_spin))
        assert not np.any(np.asarray(state.ball_vertical))


def test_resting_ball_stays_exactly_at_rest():
    state = _parked()
    final = _run(state, 10)[-1]
    np.testing.assert_array_equal(final.ball, state.ball)
    np.testing.assert_array_equal(final.ball_spin, state.ball_spin)
    np.testing.assert_array_equal(final.ball_vertical, state.ball_vertical)


# ---------------------------------------------------------------------------
# Floor: skidding and rolling
# ---------------------------------------------------------------------------

def test_kicked_ball_skids_then_rolls_at_five_sevenths_of_its_speed():
    v0 = 2.0
    trajectory = _run(_parked(ball=(-0.6, 0.3, v0, 0.0)), 6)
    skid_seconds = v0 / ((1.0 + 1.0 / K) * config.BALL_SLIDING_FRICTION * G)

    assert np.linalg.norm(_floor_slip(trajectory[1])) > 0.5  # still skidding after 0.1 s
    final = trajectory[-1]
    np.testing.assert_allclose(_floor_slip(final), 0.0, atol=1e-4)
    rolled = config.DT * 6 - skid_seconds
    expected = v0 / (1.0 + K) - config.BALL_FRICTION * G * rolled
    assert float(final.ball[2]) == pytest.approx(expected, rel=0.02)
    assert abs(float(final.ball[3])) < 1e-6


def test_rolling_ball_decelerates_by_rolling_resistance_only():
    v0 = 1.0
    final = _run(_parked(ball=(-0.6, 0.3, v0, 0.0), spin=(0.0, v0 / R, 0.0)), 5)[-1]
    expected = v0 - config.BALL_FRICTION * G * 5 * config.DT
    assert float(final.ball[2]) == pytest.approx(expected, rel=1e-4)
    np.testing.assert_allclose(_floor_slip(final), 0.0, atol=1e-5)


def test_backspin_conserves_angular_momentum_about_the_floor_contact():
    v0, backspin = 1.0, -2.0 / R
    final = _run(_parked(ball=(-0.6, 0.3, v0, 0.0), spin=(0.0, backspin, 0.0)), 4)[-1]
    rolling_speed = (v0 + K * R * backspin) / (1.0 + K)
    skid_seconds = (v0 - R * backspin) / ((1.0 + 1.0 / K) * config.BALL_SLIDING_FRICTION * G)
    expected = rolling_speed - config.BALL_FRICTION * G * (4 * config.DT - skid_seconds)
    np.testing.assert_allclose(_floor_slip(final), 0.0, atol=1e-4)
    # Within one sub-step of rolling resistance of the continuous answer.
    assert float(final.ball[2]) == pytest.approx(expected, abs=0.01)


def test_vertical_spin_decays_on_the_floor():
    final = _run(_parked(spin=(0.0, 0.0, 10.0)), 5)[-1]
    assert float(final.ball_spin[2]) == pytest.approx(10.0 - config.BALL_SPIN_DECEL * 0.5)


# ---------------------------------------------------------------------------
# Walls and robots
# ---------------------------------------------------------------------------

def test_rolling_ball_pops_up_off_a_wall_and_settles():
    v0 = 1.5
    trajectory = _run(_parked(ball=(0.0, 0.55, 0.0, v0), spin=(-v0 / R, 0.0, 0.0)), 15)
    heights = np.asarray([float(state.ball_vertical[0]) for state in trajectory])
    frames = jb.step_with_frames(trajectory[0], IDLE)[2]

    assert float(trajectory[1].ball[3]) < 0.0
    assert 0.002 < heights.max() < 0.03
    assert np.asarray(frames.ball_vertical[:, 1]).max() > 0.2
    np.testing.assert_array_equal(trajectory[-1].ball_vertical, [0.0, 0.0])


def test_skidding_ball_hits_a_wall_without_popping_up():
    trajectory = _run(_parked(ball=(0.0, 0.62, 0.0, 1.5)), 4)
    assert float(trajectory[-1].ball[3]) < 0.0
    assert max(float(state.ball_vertical[0]) for state in trajectory) < 2e-4


def test_robot_kick_leaves_the_ball_skidding():
    state = _parked(ball=(-0.2 + config.ROBOT_SIZE / 2 + R + 0.005, 0.0, 0.0, 0.0))
    robots = state.robots.at[0, 0].set(
        jnp.asarray([-0.2, 0.0, 0.0, config.ROBOT_MAX_WHEEL_SPEED, 0.0, 0.0])
    )
    wheels = jnp.full_like(state.wheel_speeds, 0.0).at[0, 0].set(config.ROBOT_MAX_WHEEL_SPEED)
    kicked, _ = step(state._replace(robots=robots, wheel_speeds=wheels), IDLE.at[0, 0].set(1.0))

    assert float(kicked.ball[2]) > 1.5
    assert np.linalg.norm(_floor_slip(kicked)) > 0.3


def test_glancing_wall_friction_converts_sideways_speed_into_spin():
    # A skidding ball sliding along the wall while pressed into it loses
    # sideways speed and gains spin about the vertical axis.
    bounced = _run(_parked(ball=(0.0, 0.62, 1.0, 1.0)), 1)[-1]
    assert float(bounced.ball[3]) < 0.0
    assert abs(float(bounced.ball_spin[2])) > 1.0


def test_ball_contacts_never_add_energy():
    keys = jax.random.split(jax.random.PRNGKey(5), 64)

    def random_state(key):
        position_key, velocity_key, spin_key = jax.random.split(key, 3)
        position = jax.random.uniform(position_key, (2,), minval=-0.7, maxval=0.7)
        position = position.at[1].multiply(0.85)
        velocity = jax.random.uniform(velocity_key, (2,), minval=-2.0, maxval=2.0)
        spin = jax.random.uniform(spin_key, (3,), minval=-80.0, maxval=80.0)
        return _parked(ball=(0.0, 0.0, 0.0, 0.0))._replace(
            ball=jnp.concatenate([position, velocity]), ball_spin=spin,
        )

    states = jax.vmap(random_state)(keys)
    after, _ = jax.vmap(jb.step)(states, jnp.zeros((64, 2, 3, 2)))
    before_energy = jax.vmap(_energy)(states)
    after_energy = jax.vmap(_energy)(after)
    assert np.all(np.asarray(after_energy) <= np.asarray(before_energy) * (1 + 1e-4) + 1e-6)


def test_restarts_stop_ball_spin_and_flight():
    state = jb.reset_kickoff(jax.random.PRNGKey(3))._replace(
        ball_spin=jnp.asarray([5.0, -3.0, 2.0]), ball_vertical=jnp.asarray([0.01, 0.2])
    )
    placed = _restart_positions(
        state, jnp.int32(RestartType.FREE_KICK), jnp.int32(0), jnp.int32(-1)
    )
    np.testing.assert_array_equal(placed.ball_spin, 0.0)
    np.testing.assert_array_equal(placed.ball_vertical, 0.0)
