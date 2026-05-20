"""Tests for the JAX physics backend."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import jax_backend as jb
from vsss_sim.physics.numpy_backend import SimState as NumpySimState


class TestSimState:
    def test_empty_state_shapes(self):
        s = jb.empty_state()
        assert s.ball.shape == (4,)
        assert s.robots.shape == (config.N_TEAMS, config.N_ROBOTS, 6)
        assert s.score.shape == (2,)
        assert s.t.shape == ()

    def test_empty_state_dtype(self):
        s = jb.empty_state()
        assert s.ball.dtype == jnp.float32
        assert s.robots.dtype == jnp.float32
        assert s.score.dtype == jnp.int32
        assert s.t.dtype == jnp.float32

    def test_empty_state_zeros(self):
        s = jb.empty_state()
        assert jnp.all(s.ball == 0)
        assert jnp.all(s.robots == 0)
        assert jnp.all(s.score == 0)
        assert s.t == 0

    def test_is_pytree(self):
        s = jb.empty_state()
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == 4  # ball, robots, score, t

    def test_from_numpy_round_trip(self):
        np_s = NumpySimState()
        np_s.ball[:] = [0.1, 0.2, 0.3, 0.4]
        np_s.robots[0, 0, :] = [0.5, -0.5, 1.0, 0.0, 0.0, 0.0]
        np_s.score[:] = [1, 2]
        np_s.t = 1.5

        j_s = jb.from_numpy(np_s)
        np_s2 = jb.to_numpy(j_s)

        assert np.allclose(np_s2.ball, np_s.ball, atol=1e-5)
        assert np.allclose(np_s2.robots, np_s.robots, atol=1e-5)
        assert np.all(np_s2.score == np_s.score)
        assert np_s2.t == pytest.approx(np_s.t, abs=1e-5)


class TestDiffDrive:
    def test_straight_forward(self):
        vx, vy, omega = jb._diff_drive(jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.0]))
        assert float(vx[0]) == pytest.approx(1.0)
        assert float(vy[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(omega[0]) == pytest.approx(0.0, abs=1e-6)

    def test_rotate_in_place(self):
        v = 0.5
        vx, vy, omega = jb._diff_drive(jnp.array([-v]), jnp.array([v]), jnp.array([0.0]))
        assert float(vx[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(vy[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(omega[0]) == pytest.approx(2 * v / config.ROBOT_WHEELBASE, rel=1e-4)

    def test_vectorised_shape(self):
        v_l = jnp.ones((config.N_TEAMS, config.N_ROBOTS))
        v_r = jnp.ones((config.N_TEAMS, config.N_ROBOTS))
        theta = jnp.zeros((config.N_TEAMS, config.N_ROBOTS))
        vx, vy, omega = jb._diff_drive(v_l, v_r, theta)
        assert vx.shape == (config.N_TEAMS, config.N_ROBOTS)


class TestResetKickoff:
    def test_ball_at_centre(self):
        s = jb.reset_kickoff(jax.random.PRNGKey(0))
        assert float(s.ball[0]) == pytest.approx(0.0)
        assert float(s.ball[1]) == pytest.approx(0.0)
        assert jnp.all(s.ball[2:4] == 0.0)

    def test_blue_on_left_yellow_on_right(self):
        s = jb.reset_kickoff(jax.random.PRNGKey(0))
        assert jnp.all(s.robots[config.TEAM_BLUE, :, 0] < 0)
        assert jnp.all(s.robots[config.TEAM_YELLOW, :, 0] > 0)

    def test_robots_within_field(self):
        s = jb.reset_kickoff(jax.random.PRNGKey(42))
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert jnp.all(jnp.abs(s.robots[:, :, 0]) <= half_l)
        assert jnp.all(jnp.abs(s.robots[:, :, 1]) <= half_w)

    def test_score_and_velocities_zero(self):
        s = jb.reset_kickoff(jax.random.PRNGKey(0))
        assert jnp.all(s.score == 0)
        assert jnp.all(s.robots[:, :, 3:6] == 0)

    def test_different_keys_give_different_states(self):
        s0 = jb.reset_kickoff(jax.random.PRNGKey(0))
        s1 = jb.reset_kickoff(jax.random.PRNGKey(1))
        assert not jnp.allclose(s0.robots[:, :, 0:2], s1.robots[:, :, 0:2])

    def test_deterministic_for_same_key(self):
        key = jax.random.PRNGKey(7)
        s0 = jb.reset_kickoff(key)
        s1 = jb.reset_kickoff(key)
        assert jnp.allclose(s0.robots, s1.robots)
        assert jnp.allclose(s0.ball, s1.ball)


class TestRobotWallCollisions:
    def test_clamped_inside_field(self):
        s = jb.empty_state()
        s = s._replace(robots=s.robots.at[:, :, 0].set(config.FIELD_LENGTH))
        s = s._replace(robots=s.robots.at[:, :, 1].set(config.FIELD_WIDTH))
        s = jb._robot_wall_collisions(s)
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert jnp.all(s.robots[:, :, 0] <= half_l + 1e-5)
        assert jnp.all(s.robots[:, :, 1] <= half_w + 1e-5)

    def test_velocity_zeroed_at_wall(self):
        s = jb.empty_state()
        s = s._replace(robots=s.robots.at[0, 0, 0].set(config.FIELD_LENGTH))
        s = s._replace(robots=s.robots.at[0, 0, 3].set(1.0))  # moving further out
        s = jb._robot_wall_collisions(s)
        assert float(s.robots[0, 0, 3]) == pytest.approx(0.0)
