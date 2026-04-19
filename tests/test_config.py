"""Tests for vsss_sim.config – verify IEEE VSS specification compliance."""

import pytest

from vsss_sim import config


class TestFieldDimensions:
    def test_field_length(self):
        assert config.FIELD_LENGTH == pytest.approx(1.50)

    def test_field_width(self):
        assert config.FIELD_WIDTH == pytest.approx(1.30)

    def test_goal_width(self):
        assert config.GOAL_WIDTH == pytest.approx(0.40)

    def test_goal_depth_positive(self):
        assert config.GOAL_DEPTH > 0

    def test_goal_fits_in_field(self):
        assert config.GOAL_WIDTH < config.FIELD_WIDTH


class TestRobotSpec:
    def test_robot_size(self):
        # Max 7.5 cm × 7.5 cm per IEEE VSS rules
        assert config.ROBOT_SIZE == pytest.approx(0.075)

    def test_robot_radius(self):
        assert config.ROBOT_RADIUS == pytest.approx(config.ROBOT_SIZE / 2)

    def test_wheel_speed_positive(self):
        assert config.ROBOT_MAX_WHEEL_SPEED > 0

    def test_wheel_speed_from_motor(self):
        expected = config.ROBOT_MAX_MOTOR_SPEED * config.ROBOT_WHEEL_RADIUS
        assert config.ROBOT_MAX_WHEEL_SPEED == pytest.approx(expected)

    def test_robot_fits_in_field(self):
        assert config.ROBOT_SIZE < config.FIELD_WIDTH
        assert config.ROBOT_SIZE < config.FIELD_LENGTH


class TestBallSpec:
    def test_ball_radius(self):
        # Golf ball diameter ≈ 42.67 mm → radius ≈ 21.35 mm
        assert config.BALL_RADIUS == pytest.approx(0.02135, rel=1e-3)

    def test_ball_mass_positive(self):
        assert config.BALL_MASS > 0

    def test_ball_smaller_than_robot(self):
        assert config.BALL_RADIUS < config.ROBOT_RADIUS


class TestTeamConstants:
    def test_n_teams(self):
        assert config.N_TEAMS == 2

    def test_n_robots(self):
        # 3 v 3 format
        assert config.N_ROBOTS == 3

    def test_team_indices(self):
        assert config.TEAM_BLUE == 0
        assert config.TEAM_YELLOW == 1


class TestSimConstants:
    def test_dt(self):
        assert config.DT == pytest.approx(1.0 / config.FPS)

    def test_fps_positive(self):
        assert config.FPS > 0

    def test_max_steps_positive(self):
        assert config.MAX_EPISODE_STEPS > 0
