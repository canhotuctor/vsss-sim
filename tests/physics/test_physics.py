"""Tests for vsss_sim.physics."""

import math

import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.physics import (
    SimState,
    _ball_robot_collisions,
    _ball_wall_collisions,
    _diff_drive,
    _robot_wall_collisions,
    reset_kickoff,
    step,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state() -> SimState:
    s = SimState()
    reset_kickoff(s, rng=np.random.default_rng(0))
    return s


def zero_actions() -> np.ndarray:
    return np.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Differential-drive kinematics
# ---------------------------------------------------------------------------

class TestDiffDrive:
    def test_straight_forward(self):
        v = 1.0
        vx, vy, omega = _diff_drive(
            np.array([v]), np.array([v]), np.array([0.0])
        )
        assert vx[0] == pytest.approx(v)
        assert vy[0] == pytest.approx(0.0, abs=1e-9)
        assert omega[0] == pytest.approx(0.0, abs=1e-9)

    def test_rotate_in_place(self):
        v = 0.5
        vx, vy, omega = _diff_drive(
            np.array([-v]), np.array([v]), np.array([0.0])
        )
        assert vx[0] == pytest.approx(0.0, abs=1e-9)
        assert vy[0] == pytest.approx(0.0, abs=1e-9)
        expected_omega = 2 * v / config.ROBOT_WHEELBASE
        assert omega[0] == pytest.approx(expected_omega)

    def test_heading_90_deg(self):
        v = 1.0
        theta = math.pi / 2
        vx, vy, omega = _diff_drive(
            np.array([v]), np.array([v]), np.array([theta])
        )
        assert vx[0] == pytest.approx(0.0, abs=1e-9)
        assert vy[0] == pytest.approx(v)

    def test_vectorised_shapes(self):
        v_l = np.ones((config.N_TEAMS, config.N_ROBOTS))
        v_r = np.ones((config.N_TEAMS, config.N_ROBOTS))
        theta = np.zeros((config.N_TEAMS, config.N_ROBOTS))
        vx, vy, omega = _diff_drive(v_l, v_r, theta)
        assert vx.shape == (config.N_TEAMS, config.N_ROBOTS)


# ---------------------------------------------------------------------------
# Reset / kickoff
# ---------------------------------------------------------------------------

class TestResetKickoff:
    def test_ball_at_centre(self):
        s = make_state()
        assert s.ball[0] == pytest.approx(0.0)
        assert s.ball[1] == pytest.approx(0.0)
        assert s.ball[2] == pytest.approx(0.0)
        assert s.ball[3] == pytest.approx(0.0)

    def test_blue_robots_on_left(self):
        s = make_state()
        assert np.all(s.robots[config.TEAM_BLUE, :, 0] < 0)

    def test_yellow_robots_on_right(self):
        s = make_state()
        assert np.all(s.robots[config.TEAM_YELLOW, :, 0] > 0)

    def test_robots_within_field(self):
        s = make_state()
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert np.all(np.abs(s.robots[:, :, 0]) <= half_l)
        assert np.all(np.abs(s.robots[:, :, 1]) <= half_w)

    def test_velocities_zeroed(self):
        s = make_state()
        assert np.all(s.robots[:, :, 3:6] == 0.0)

    def test_score_zeroed(self):
        s = make_state()
        assert np.all(s.score == 0)


# ---------------------------------------------------------------------------
# Ball wall collisions
# ---------------------------------------------------------------------------

class TestBallWallCollisions:
    def test_bounce_top_wall(self):
        s = SimState()
        s.ball[1] = config.FIELD_WIDTH / 2.0 + config.BALL_RADIUS  # outside
        s.ball[3] = 1.0  # moving up
        result = _ball_wall_collisions(s)
        assert result == 0
        assert s.ball[1] <= config.FIELD_WIDTH / 2.0
        assert s.ball[3] < 0  # velocity reversed

    def test_bounce_bottom_wall(self):
        s = SimState()
        s.ball[1] = -(config.FIELD_WIDTH / 2.0 + config.BALL_RADIUS)
        s.ball[3] = -1.0
        result = _ball_wall_collisions(s)
        assert result == 0
        assert s.ball[1] >= -config.FIELD_WIDTH / 2.0
        assert s.ball[3] > 0

    def test_blue_scores_right_goal(self):
        """Ball past +x end-line inside goal → blue scores (+1)."""
        s = SimState()
        s.ball[0] = config.FIELD_LENGTH / 2.0 + 0.01
        s.ball[1] = 0.0  # centred on goal
        result = _ball_wall_collisions(s)
        assert result == 1

    def test_yellow_scores_left_goal(self):
        """Ball past -x end-line inside goal → yellow scores (-1)."""
        s = SimState()
        s.ball[0] = -(config.FIELD_LENGTH / 2.0 + 0.01)
        s.ball[1] = 0.0
        result = _ball_wall_collisions(s)
        assert result == -1

    def test_no_goal_outside_posts(self):
        """Ball past end-line outside goal width → no goal, bounce."""
        s = SimState()
        s.ball[0] = config.FIELD_LENGTH / 2.0 + 0.01
        s.ball[1] = config.GOAL_WIDTH  # wider than half-goal
        s.ball[2] = 1.0
        result = _ball_wall_collisions(s)
        assert result == 0
        assert s.ball[2] < 0  # bounced


# ---------------------------------------------------------------------------
# Robot wall collisions
# ---------------------------------------------------------------------------

class TestRobotWallCollisions:
    def test_robots_clamped_inside_field(self):
        s = SimState()
        # Push all robots outside boundaries
        s.robots[:, :, 0] = config.FIELD_LENGTH
        s.robots[:, :, 1] = config.FIELD_WIDTH
        _robot_wall_collisions(s)
        half_l = config.FIELD_LENGTH / 2.0 - config.ROBOT_RADIUS
        half_w = config.FIELD_WIDTH / 2.0 - config.ROBOT_RADIUS
        assert np.all(s.robots[:, :, 0] <= half_l + 1e-9)
        assert np.all(s.robots[:, :, 1] <= half_w + 1e-9)


# ---------------------------------------------------------------------------
# Ball–robot collision
# ---------------------------------------------------------------------------

class TestBallRobotCollision:
    def test_ball_pushed_away(self):
        s = SimState()
        # Place ball and robot in contact
        overlap = 0.001
        s.robots[0, 0, 0] = 0.0
        s.robots[0, 0, 1] = 0.0
        dist = config.BALL_RADIUS + config.ROBOT_RADIUS - overlap
        s.ball[0] = dist
        s.ball[1] = 0.0
        s.ball[2] = -1.0   # ball moving toward robot
        s.ball[3] = 0.0

        _ball_robot_collisions(s)
        # Ball should have bounced back (vx now positive or ball moved right)
        assert s.ball[0] >= config.ROBOT_RADIUS + config.BALL_RADIUS - 1e-6


# ---------------------------------------------------------------------------
# Full step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_advances_time(self):
        s = make_state()
        t0 = s.t
        step(s, zero_actions())
        assert s.t == pytest.approx(t0 + config.DT)

    def test_step_ball_stays_in_field(self):
        s = make_state()
        s.ball[2] = config.ROBOT_MAX_WHEEL_SPEED * 10  # huge velocity
        for _ in range(10):
            step(s, zero_actions())
        half_l = config.FIELD_LENGTH / 2.0 + config.GOAL_DEPTH + 0.1
        half_w = config.FIELD_WIDTH / 2.0 + 0.1
        assert abs(s.ball[0]) <= half_l
        assert abs(s.ball[1]) <= half_w

    def test_step_stationary_zero_action(self):
        """With zero actions, robots should not move (they start stationary)."""
        s = make_state()
        initial_positions = s.robots[:, :, 0:2].copy()
        step(s, zero_actions())
        assert np.allclose(s.robots[:, :, 0:2], initial_positions, atol=1e-9)

    def test_robot_moves_forward(self):
        """Full-forward action should move robot in its heading direction."""
        s = SimState()
        reset_kickoff(s, rng=np.random.default_rng(1))
        # Blue robot 0 is at some negative x, heading toward +x (≈ 0 rad)
        initial_x = s.robots[config.TEAM_BLUE, 0, 0]
        actions = zero_actions()
        actions[config.TEAM_BLUE, 0, :] = 1.0  # full forward
        for _ in range(5):
            step(s, actions)
        # Should have moved in the +x direction
        assert s.robots[config.TEAM_BLUE, 0, 0] > initial_x

    def test_goal_registered(self):
        """Ball placed near goal should produce a goal event."""
        s = SimState()
        reset_kickoff(s)
        # Put ball just inside yellow goal
        s.ball[0] = config.FIELD_LENGTH / 2.0 + 0.01
        s.ball[1] = 0.0
        s.ball[2] = 0.1  # moving right (already past end-line)
        info = step(s, zero_actions())
        assert info["goal"] == 1

    def test_no_goal_centre(self):
        s = make_state()
        for _ in range(30):
            step(s, zero_actions())
        # Ball shouldn't have scored from centre with no movement
        assert s.score[0] == 0
        assert s.score[1] == 0
