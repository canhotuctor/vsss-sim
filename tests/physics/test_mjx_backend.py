"""Tests for the optional MuJoCo MJX physics engine."""

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

with contextlib.redirect_stdout(io.StringIO()):
    mjx = pytest.importorskip("mujoco.mjx")

import mujoco  # noqa: E402

from vsss_sim import config  # noqa: E402
from vsss_sim.physics import get_backend  # noqa: E402
from vsss_sim.physics import jax_backend as jb  # noqa: E402
from vsss_sim.physics import mjx_backend as mb  # noqa: E402

HALF = config.ROBOT_SIZE / 2.0
HALF_L = config.FIELD_LENGTH / 2.0
HALF_W = config.FIELD_WIDTH / 2.0
IDLE = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32)


def _parked(overrides=None, ball=(-0.5, 0.45, 0.0, 0.0)) -> jb.SimState:
    """Robots parked apart along the bottom of the field, clear of each other."""
    robots = np.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=np.float32)
    spots = [(-0.3, -0.55), (-0.1, -0.55), (0.1, -0.55), (0.3, -0.55), (0.0, -0.4), (-0.3, -0.4)]
    for index, (x, y) in enumerate(spots):
        robots[index // config.N_ROBOTS, index % config.N_ROBOTS, :2] = (x, y)
    for (team, robot), values in (overrides or {}).items():
        robots[team, robot, : len(values)] = values
    return jb.empty_state()._replace(
        robots=jnp.asarray(robots), ball=jnp.asarray(ball, dtype=jnp.float32)
    )


def _run(engine, state, actions, steps):
    goals = []
    for _ in range(steps):
        state, info = engine.step(state, actions)
        goals.append(int(info["goal"]))
    return state, goals


# ---------------------------------------------------------------------------
# Selection and model
# ---------------------------------------------------------------------------

def test_get_backend_selects_engine_modules():
    assert get_backend() is jb
    assert get_backend("jax") is jb
    assert get_backend(config.PhysicsBackend.MJX) is mb
    with pytest.raises(ValueError):
        get_backend("bullet")


def test_model_has_planar_bodies_and_only_explicit_contact_pairs():
    model = mujoco.MjModel.from_xml_string(mb.build_mjcf())
    n_robots = config.N_TEAMS * config.N_ROBOTS
    assert model.nq == model.nv == 3 * n_robots + 2
    assert model.nu == 3 * n_robots
    assert not (model.geom_contype | model.geom_conaffinity).any()
    # Each robot touches 12 walls and later robots; the ball touches all of them.
    assert model.npair == n_robots * 12 + n_robots * (n_robots - 1) // 2 + n_robots + 12


def test_options_and_timestep_are_validated():
    with pytest.raises(ValueError, match="steps_per_substep"):
        mb.MJXOptions(steps_per_substep=0)
    with pytest.raises(ValueError, match="max_contact_points"):
        mb.MJXOptions(max_contact_points=0)
    coarse = mb.MJXOptions(steps_per_substep=1)
    with pytest.raises(ValueError, match="timestep"):
        mb.step(_parked(), IDLE, dt=0.1, sub_steps=1, options=coarse)


# ---------------------------------------------------------------------------
# State contract
# ---------------------------------------------------------------------------

def test_step_keeps_the_sim_state_contract():
    state = jb.reset_kickoff(jax.random.PRNGKey(0))._replace(
        score=jnp.asarray([2, 1], dtype=jnp.int32)
    )
    actions = jnp.full((config.N_TEAMS, config.N_ROBOTS, 2), 0.7, dtype=jnp.float32)
    after, info = mb.step(state, actions)

    for name in jb.SimState._fields:
        assert getattr(after, name).shape == getattr(state, name).shape
        assert getattr(after, name).dtype == getattr(state, name).dtype
    np.testing.assert_array_equal(after.score, state.score)
    assert float(after.t) == pytest.approx(config.DT)
    assert int(info["goal"]) == 0
    assert np.all(np.abs(np.asarray(after.robots[:, :, 2])) <= np.pi)
    # Wheel speeds keep the JAX engine's slewed command semantics.
    expected, _ = jb.step(state, actions)
    np.testing.assert_allclose(after.wheel_speeds, expected.wheel_speeds, atol=1e-6)


def test_resting_state_stays_at_rest():
    state = _parked()
    after, _ = _run(mb, state, IDLE, 3)
    np.testing.assert_allclose(after.robots, state.robots, atol=1e-6)
    np.testing.assert_allclose(after.ball, state.ball, atol=1e-6)


def test_step_with_frames_matches_step_and_exposes_each_substep():
    state = jb.reset_random(jax.random.PRNGKey(3))
    actions = jax.random.uniform(jax.random.PRNGKey(4), (2, 3, 2), minval=-1.0, maxval=1.0)
    final, info = mb.step(state, actions)
    framed, frame_info, frames = mb.step_with_frames(state, actions)

    for name in jb.SimState._fields:
        np.testing.assert_allclose(getattr(framed, name), getattr(final, name), atol=1e-6)
    assert frames.robots.shape == (config.SUB_STEPS, config.N_TEAMS, config.N_ROBOTS, 6)
    np.testing.assert_allclose(frames.robots[-1], final.robots, atol=1e-6)
    assert int(frame_info["goal"]) == int(info["goal"])


# ---------------------------------------------------------------------------
# Agreement with the JAX engine where no contacts are involved
# ---------------------------------------------------------------------------

def test_free_driving_follows_the_jax_engine():
    state = _parked({(0, 1): (-0.6, 0.25, 0.0)})
    actions = IDLE.at[0, 1].set(1.0)
    reference, _ = _run(jb, state, actions, 6)
    driven, _ = _run(mb, state, actions, 6)

    assert float(driven.robots[0, 1, 3]) == pytest.approx(config.ROBOT_MAX_WHEEL_SPEED, rel=1e-3)
    # Traction-limited actuators trail the kinematic engine slightly.
    assert float(reference.robots[0, 1, 0] - driven.robots[0, 1, 0]) == pytest.approx(
        0.0, abs=0.025
    )
    assert abs(float(driven.robots[0, 1, 1]) - 0.25) < 1e-4

    spin = IDLE.at[0, 1].set(jnp.asarray([-1.0, 1.0]))
    spun, _ = _run(mb, state, spin, 3)
    expected_omega = 2.0 * config.ROBOT_MAX_WHEEL_SPEED / config.ROBOT_WHEELBASE
    assert float(spun.robots[0, 1, 5]) == pytest.approx(expected_omega, rel=1e-3)
    np.testing.assert_allclose(spun.robots[0, 1, :2], state.robots[0, 1, :2], atol=1e-4)


def test_ball_rolling_friction_matches_the_jax_engine():
    state = _parked(ball=(-0.3, 0.3, 1.0, 0.4))
    reference, _ = _run(jb, state, IDLE, 5)
    rolled, _ = _run(mb, state, IDLE, 5)
    np.testing.assert_allclose(rolled.ball[2:], reference.ball[2:], atol=1e-3)
    np.testing.assert_allclose(rolled.ball[:2], reference.ball[:2], atol=3e-3)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("speed", [1.0, 2.0, 3.0])
def test_ball_bounces_off_the_side_wall_with_configured_restitution(speed):
    state = _parked(ball=(0.0, 0.45, 0.0, speed))
    reference, _ = _run(jb, state, IDLE, 3)
    bounced, _ = _run(mb, state, IDLE, 3)
    assert float(bounced.ball[3]) < 0.0
    assert float(bounced.ball[3]) == pytest.approx(float(reference.ball[3]), rel=0.05)


def _free_collision(qpos_updates, qvel_updates):
    """Simulate 0.25 s of unactuated bodies; return MuJoCo positions and velocities."""
    timestep = config.DT / (config.SUB_STEPS * mb.DEFAULT_OPTIONS.steps_per_substep)
    model, template = mb._compiled_model(mb.DEFAULT_OPTIONS, timestep)
    model = model.replace(
        actuator_gainprm=model.actuator_gainprm.at[:, 0].set(0.0),
        actuator_biasprm=model.actuator_biasprm.at[:, 2].set(0.0),
    )
    qpos = np.zeros(model.nq, dtype=np.float32)
    for index in range(config.N_TEAMS * config.N_ROBOTS):
        qpos[3 * index:3 * index + 2] = (-0.5 + index // 3, -0.4 + 0.4 * (index % 3))
    qpos[-2:] = (0.0, -0.3)
    qvel = np.zeros(model.nv, dtype=np.float32)
    for index, value in qpos_updates.items():
        qpos[index] = value
    for index, value in qvel_updates.items():
        qvel[index] = value
    data = template.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
    data = jax.lax.fori_loop(
        0, int(round(0.25 / timestep)), lambda _, d: mjx.step(model, d), data
    )
    return np.asarray(data.qpos), np.asarray(data.qvel)


@pytest.mark.parametrize("speed", [0.65, 1.3])
def test_box_contacts_reproduce_configured_restitution(speed):
    # blue0 and yellow0 meet head-on: e ≈ ROBOT_WALL_RESTITUTION.
    qpos, qvel = _free_collision({0: -0.1, 1: 0.0, 9: 0.1, 10: 0.0}, {0: speed, 9: -speed})
    restitution = (qvel[9] - qvel[0]) / (2.0 * speed)
    assert restitution == pytest.approx(config.ROBOT_WALL_RESTITUTION, abs=0.1)
    assert qpos[9] - qpos[0] >= config.ROBOT_SIZE - 1e-3

    # blue1 drives straight into the top wall: nearly inelastic, as in the JAX engine.
    qpos, qvel = _free_collision({3: 0.0, 4: 0.5}, {4: speed})
    assert -0.15 < -qvel[4] / speed < 0.15
    assert qpos[4] + HALF <= HALF_W + 1e-3


def test_goals_are_detected_once_the_ball_crosses_the_line():
    for y, speed, direction in ((0.05, 1.0, 1), (-0.17, 1.0, 1), (0.1, -1.5, -1)):
        state = _parked(ball=(direction * 0.65, y, speed, 0.0))
        reference, reference_goals = _run(jb, state, IDLE, 3)
        _, goals = _run(mb, state, IDLE, 3)
        assert goals == reference_goals
        assert direction in goals


def test_ball_off_the_end_wall_beside_the_mouth_is_not_a_goal():
    for y in (0.3, 0.2 + 0.01, -0.45):
        state = _parked(ball=(0.6, y, 1.0, 0.0))
        bounced, goals = _run(mb, state, IDLE, 4)
        assert goals == [0, 0, 0, 0]
        assert float(bounced.ball[2]) < 0.0


def test_state_freezes_after_a_goal_within_the_control_step():
    state = _parked(ball=(0.72, 0.0, 1.2, 0.0))
    final, info, frames = mb.step_with_frames(state, IDLE)
    goals = np.asarray(info["goals"])
    first = int(np.flatnonzero(goals)[0])

    assert int(info["goal"]) == 1
    for index in range(first + 1, config.SUB_STEPS):
        np.testing.assert_array_equal(frames.ball[index], frames.ball[first])
    np.testing.assert_array_equal(final.ball, frames.ball[first])


def test_robots_can_enter_goal_pockets_but_not_pass_walls():
    pocket, _ = _run(mb, _parked({(0, 1): (0.55, 0.0, 0.0)}), IDLE.at[0, 1].set(1.0), 10)
    assert float(pocket.robots[0, 1, 0]) + HALF == pytest.approx(
        HALF_L + config.GOAL_DEPTH, abs=1e-3
    )

    corner, _ = _run(mb, _parked({(0, 1): (0.5, 0.4, np.pi / 4)}), IDLE.at[0, 1].set(1.0), 10)
    x, y = float(corner.robots[0, 1, 0]), float(corner.robots[0, 1, 1])
    chamfer_limit = HALF_L + HALF_W - config.FIELD_CHAMFER
    assert x + y + HALF * np.sqrt(2.0) <= chamfer_limit + 1e-3


def test_random_batched_play_stays_inside_the_field_without_deep_overlaps():
    keys = jax.random.split(jax.random.PRNGKey(7), 8)
    states = jax.vmap(jb.reset_random)(keys)
    step = jax.jit(jax.vmap(mb.step))
    action_keys = jax.random.split(jax.random.PRNGKey(8), 30)
    for action_key in action_keys:
        actions = jax.random.uniform(action_key, (8, 2, 3, 2), minval=-1.0, maxval=1.0)
        states, _ = step(states, actions)

    robots = np.asarray(states.robots).reshape(8, -1, 6)
    ball = np.asarray(states.ball)
    assert np.isfinite(robots).all() and np.isfinite(ball).all()
    extent = HALF * (np.abs(np.cos(robots[..., 2])) + np.abs(np.sin(robots[..., 2])))
    assert (np.abs(robots[..., 1]) + extent <= HALF_W + 3e-3).all()
    assert (np.abs(robots[..., 0]) + extent <= HALF_L + config.GOAL_DEPTH + 3e-3).all()
    assert (np.abs(ball[:, 1]) + config.BALL_RADIUS <= HALF_W + 3e-3).all()
    first, second = np.triu_indices(robots.shape[1], k=1)
    a, b = robots[:, first].reshape(-1, 6), robots[:, second].reshape(-1, 6)
    _, _, overlaps = jax.vmap(jb._sat_square_overlap)(a[:, :2], a[:, 2], b[:, :2], b[:, 2])
    assert float(jnp.max(overlaps)) < 0.015


def test_max_contact_points_option_compiles_a_bounded_model():
    options = mb.MJXOptions(max_contact_points=48)
    state = jb.reset_kickoff(jax.random.PRNGKey(1))
    bounded, _ = mb.step(state, IDLE.at[0].set(0.5), options=options)
    full, _ = mb.step(state, IDLE.at[0].set(0.5))
    # With few simultaneous contacts, the bounded solve agrees with the full one.
    np.testing.assert_allclose(bounded.robots, full.robots, atol=1e-4)
