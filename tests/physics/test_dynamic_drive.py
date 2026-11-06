"""Dynamic robot drive: force-limited motors and momentum-carrying contacts.

The default ``"kinematic"`` model overwrites robot velocity from the wheel
commands every sub-step. ``"dynamic"`` keeps velocity as state, so these
scenarios check stall, momentum exchange and push contests.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as physics

_G = 9.81


def _state(**robot_poses):
    """Empty state; unused robots parked along the bottom wall, ball far away."""
    state = physics.empty_state()
    parking = jnp.array([[-0.6, -0.55], [-0.4, -0.55], [-0.2, -0.55],
                         [0.2, -0.55], [0.4, -0.55], [0.6, -0.55]], dtype=jnp.float32)
    robots = state.robots.reshape((6, 6)).at[:, :2].set(parking)
    state = state._replace(
        robots=robots.reshape(state.robots.shape),
        ball=jnp.array([-0.6, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    return state


def _place(state, index, x, y, theta=0.0, vx=0.0, vy=0.0):
    flat = state.robots.reshape((6, 6)).at[index, :].set(
        jnp.array([x, y, theta, vx, vy, 0.0], dtype=jnp.float32)
    )
    return state._replace(robots=flat.reshape(state.robots.shape))


def _command(index, left, right=None):
    right = left if right is None else right
    actions = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)
    return actions.at[index // 3, index % 3, :].set(jnp.array([left, right]))


def _run(state, actions, control_steps, drive_model="dynamic"):
    frames = []
    for _ in range(control_steps):
        state, _, fr = physics.step_with_frames(state, actions, drive_model=drive_model)
        frames.append(fr)
    robots = np.concatenate([np.asarray(f.robots).reshape((-1, 6, 6)) for f in frames])
    return state, robots


def test_default_drive_model_is_kinematic():
    assert config.ROBOT_DRIVE_MODEL == "kinematic"


def test_dynamic_drive_accelerates_within_traction_limit_and_saturates_at_top_speed():
    state = _place(_state(), 0, -0.5, 0.0)
    final, robots = _run(state, _command(0, 1.0), 10)
    speed = robots[:, 0, 3]
    first_step = speed[config.SUB_STEPS - 1]

    # Traction caps acceleration at mu*g (~6.9 m/s^2); the kinematic slew allows 1 g.
    traction = config.ROBOT_WHEEL_FLOOR_FRICTION * _G
    assert 0.3 < first_step <= traction * config.DT * 1.05
    assert np.all(np.diff(speed) >= -1e-6)
    assert speed[-1] == pytest.approx(config.ROBOT_MAX_WHEEL_SPEED, rel=0.03)
    assert speed.max() <= config.ROBOT_MAX_WHEEL_SPEED * 1.001


def test_dynamic_drive_slows_a_robot_instead_of_snapping_to_command():
    state = _place(_state(), 0, -0.5, 0.0, vx=1.0)
    _, robots = _run(state, _command(0, 0.0), 1)
    # Braking is also traction-limited: one control step sheds <= mu*g*dt.
    assert robots[-1, 0, 3] > 1.0 - config.ROBOT_WHEEL_FLOOR_FRICTION * _G * config.DT * 1.05


def test_sideways_slide_decays_at_wheel_friction():
    state = _place(_state(), 0, 0.0, 0.0, theta=0.0, vy=1.0)
    _, robots = _run(state, _command(0, 0.0), 1)
    expected = 1.0 - config.ROBOT_WHEEL_FLOOR_FRICTION * _G * config.DT
    assert robots[-1, 0, 4] == pytest.approx(expected, abs=0.03)


def test_robot_driving_into_wall_stalls_and_stays_inside():
    limit = config.FIELD_WIDTH / 2 - config.ROBOT_SIZE / 2
    state = _place(_state(), 0, 0.0, limit - 0.002, theta=np.pi / 2)
    _, robots = _run(state, _command(0, 1.0), 10)
    assert robots[:, 0, 1].max() <= limit + 1e-4
    assert abs(robots[-1, 0, 4]) < 0.02


def test_moving_robot_transfers_momentum_to_a_resting_one():
    # Equal masses, restitution 0.2: the struck robot should leave at ~0.6 v.
    state = _place(_state(), 0, -0.1, 0.0, vx=1.0)
    state = _place(state, 1, 0.0, 0.0)
    _, robots = _run(state, _command(0, 0.0), 1)
    struck_peak = robots[:, 1, 3].max()
    striker_after = robots[-1, 0, 3]
    assert struck_peak > 0.4
    assert striker_after < 0.7


def test_contact_does_not_create_energy_when_both_robots_coast():
    state = _place(_state(), 0, -0.1, 0.0, vx=1.0)
    state = _place(state, 1, 0.0, 0.0)
    _, robots = _run(state, jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2)), 2)
    kinetic = np.sum(robots[:, :2, 3] ** 2 + robots[:, :2, 4] ** 2, axis=1)
    assert kinetic.max() <= 1.0 + 1e-3


def test_equal_head_on_pushers_deadlock():
    state = _place(_state(), 0, -0.0385, 0.0, theta=0.0)
    state = _place(state, 1, 0.0385, 0.0, theta=np.pi)
    actions = _command(0, 1.0) + _command(1, 1.0)
    _, robots = _run(state, actions, 10)
    assert abs(robots[-1, 0, 0] - robots[0, 0, 0]) < 0.02
    assert abs(robots[-1, 1, 0] - robots[0, 1, 0]) < 0.02


def test_stronger_command_wins_a_push_contest():
    state = _place(_state(), 0, -0.0385, 0.0, theta=0.0)
    state = _place(state, 1, 0.0385, 0.0, theta=np.pi)
    actions = _command(0, 1.0) + _command(1, 0.3)
    _, robots = _run(state, actions, 10)
    assert robots[-1, 0, 0] - robots[0, 0, 0] > 0.05
    assert robots[-1, 1, 0] - robots[0, 1, 0] > 0.05


@pytest.mark.parametrize("side", [-1, 1])
def test_robot_train_into_wall_is_tighter_than_kinematic(side):
    limit = config.FIELD_WIDTH / 2 - config.ROBOT_SIZE / 2
    pitch = config.ROBOT_SIZE + 0.001
    state = _state()
    for index, offset in enumerate((0.0, 1.0, 2.0)):
        state = _place(state, index, 0.0, side * (limit - offset * pitch), theta=side * np.pi / 2)
    actions = _command(2, 1.0)
    # Start the driver at full wheel speed so the push is immediate.
    state = state._replace(
        wheel_speeds=state.wheel_speeds.at[0, 2, :].set(config.ROBOT_MAX_WHEEL_SPEED)
    )

    def worst_overlap(model):
        _, robots = _run(state, actions, 3, drive_model=model)
        worst = 0.0
        for frame in robots:
            ov, _, depth = physics._sat_square_overlap(frame[1, :2], frame[1, 2], frame[2, :2], frame[2, 2])
            ov2, _, depth2 = physics._sat_square_overlap(frame[0, :2], frame[0, 2], frame[1, :2], frame[1, 2])
            worst = max(worst, float(jnp.where(ov, depth, 0.0)), float(jnp.where(ov2, depth2, 0.0)))
        return worst

    assert worst_overlap("dynamic") < worst_overlap("kinematic")
    assert worst_overlap("dynamic") < 2e-3


def test_invalid_drive_model_is_rejected():
    with pytest.raises(ValueError):
        physics.step(_state(), _command(0, 0.0), drive_model="bogus")
