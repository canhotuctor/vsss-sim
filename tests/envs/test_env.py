"""Tests for vsss_sim.envs (VSSEnv Gymnasium interface)."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import vsss_sim  # noqa: F401 – registers "VSSS-v0"
from vsss_sim import config
from vsss_sim.envs import VSSEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = VSSEnv(opponent_policy="stationary", render_mode=None)
    yield e
    e.close()


@pytest.fixture
def env_random_opp():
    e = VSSEnv(opponent_policy="random", render_mode=None)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_human_mode_draws_every_substep_of_the_same_physics(self, monkeypatch):
        pytest.importorskip("pygame")
        from vsss_sim.rendering import VSSRenderer

        frames = []
        monkeypatch.setattr(
            VSSRenderer, "render",
            lambda self, ball, robots, score, **kwargs: frames.append(np.array(robots)),
        )
        human = VSSEnv(opponent_policy="stationary", render_mode="human")
        plain = VSSEnv(opponent_policy="stationary")
        action = np.full(6, 0.8, dtype=np.float32)
        human.reset(seed=3)
        plain.reset(seed=3)

        human.step(action)
        plain.step(action)

        assert len(frames) == config.SUB_STEPS
        assert all(not np.allclose(a, b) for a, b in zip(frames, frames[1:]))
        np.testing.assert_allclose(frames[-1], np.asarray(plain._state.robots), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(human._state.robots), np.asarray(plain._state.robots), atol=1e-6
        )
        human.close()
        plain.close()

    def test_frame_pacing_absorbs_slow_frames_and_restarts_after_stalls(self, monkeypatch):
        pytest.importorskip("pygame")
        from types import SimpleNamespace

        from vsss_sim.rendering import pygame as rendering

        now, sleeps = [0.0], []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        monkeypatch.setattr(
            rendering, "time", SimpleNamespace(perf_counter=lambda: now[0], sleep=sleep)
        )
        renderer = rendering.VSSRenderer(render_mode="human", fps=100.0)

        renderer._pace(1)
        now[0] += 0.015  # a slow frame
        renderer._pace(1)
        renderer._pace(1)
        assert sleeps == pytest.approx([0.01, 0.005])

        now[0] += 1.0  # a stall: restart the schedule rather than racing ahead
        renderer._pace(2)
        assert sleeps[-1] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_obs_space_shape(self, env):
        expected = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        assert env.observation_space.shape == (expected,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (config.N_ROBOTS * 2,)

    def test_action_space_bounds(self, env):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_obs_in_space_after_reset(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        result = env.reset(seed=42)
        assert len(result) == 2
        obs, info = result

    def test_reset_obs_finite(self, env):
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs))

    def test_reset_info_keys(self, env):
        _, info = env.reset()
        assert "score_blue" in info
        assert "score_yellow" in info
        assert "sim_time" in info

    def test_reset_score_zero(self, env):
        _, info = env.reset()
        assert info["score_blue"] == 0
        assert info["score_yellow"] == 0

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        assert np.allclose(obs1, obs2)

    def test_reset_different_seeds_differ(self, env):
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Positions should differ with different seeds
        assert not np.allclose(obs1, obs2)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        result = env.step(np.zeros(config.N_ROBOTS * 2))
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape

    def test_step_obs_finite(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))

    def test_step_reward_type(self, env):
        env.reset(seed=0)
        _, reward, *_ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_step_reward_no_goal_in_shaping_band(self, env):
        """With no goal, reward is the small ball-forward-progress shaping term,
        bounded by coef * FIELD_LENGTH in magnitude."""
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(config.N_ROBOTS * 2))
        assert info["goal"] == 0
        bound = config.BALL_FORWARD_REWARD_COEF * config.FIELD_LENGTH
        assert abs(reward) <= bound + 1e-6

    def test_step_reward_ball_forward_shaping(self):
        """Moving the ball in +x (toward yellow goal) gives positive shaping;
        -x gives negative shaping. No goal in either case."""
        e = VSSEnv(render_mode=None)
        e.reset(seed=0)
        # Park ball mid-field with a clear +x velocity; one step → +Δx.
        e._state = e._state._replace(
            ball=e._state.ball.at[:].set(np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        )
        _, reward_fwd, _, _, info_fwd = e.step(np.zeros(config.N_ROBOTS * 2))
        assert info_fwd["goal"] == 0
        assert reward_fwd > 0.0

        e.reset(seed=0)
        e._state = e._state._replace(
            ball=e._state.ball.at[:].set(np.array([0.0, 0.0, -0.5, 0.0], dtype=np.float32))
        )
        _, reward_back, _, _, info_back = e.step(np.zeros(config.N_ROBOTS * 2))
        assert info_back["goal"] == 0
        assert reward_back < 0.0
        e.close()

    def test_truncation_at_max_steps(self):
        e = VSSEnv(max_episode_steps=5)
        e.reset()
        truncated = False
        for _ in range(6):
            _, _, _, truncated, _ = e.step(np.zeros(config.N_ROBOTS * 2))
        assert truncated
        e.close()

    def test_zero_action_no_truncation_early(self, env):
        env.reset(seed=0)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(config.N_ROBOTS * 2)
            )
        assert not terminated
        assert not truncated

    def test_step_time_advances(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(config.N_ROBOTS * 2))
        assert info["sim_time"] > 0

    def test_score_updates_after_goal(self):
        """Forcibly place ball in goal and verify score update."""
        e = VSSEnv(render_mode=None)
        e.reset(seed=0)
        # Push ball into yellow goal
        e._state = e._state._replace(
            ball=e._state.ball.at[:].set(np.array(
                [config.FIELD_LENGTH / 2.0 + 0.02, 0.0, 0.05, 0.0], dtype=np.float32
            ))
        )

        obs, reward, terminated, truncated, info = e.step(
            np.zeros(config.N_ROBOTS * 2)
        )
        # +1 goal reward plus a small shaping contribution from Δx in the step.
        bound = config.BALL_FORWARD_REWARD_COEF * config.FIELD_LENGTH
        assert 1.0 - bound - 1e-6 <= reward <= 1.0 + bound + 1e-6
        assert info["goal"] == 1
        assert info["score_blue"] == 1
        assert info["score_yellow"] == 0
        np.testing.assert_allclose(obs, e._get_obs())
        e.close()

    def test_brandao_goal_reward_and_kickoff_observation(self):
        e = VSSEnv(
            render_mode=None,
            max_episode_steps=5,
            reward_profile="brandao_reward",
        )
        e.reset(seed=0)
        e._state = e._state._replace(
            ball=e._state.ball.at[:].set(np.array(
                [config.FIELD_LENGTH / 2.0 + 0.02, 0.0, 0.05, 0.0], dtype=np.float32
            ))
        )

        obs, reward, _, _, info = e.step(np.zeros(config.N_ROBOTS * 2))

        assert reward == pytest.approx(18.0)
        assert info["goal"] == 1
        assert info["score_blue"] == 1
        np.testing.assert_allclose(obs, e._get_obs())
        assert abs(float(e._state.ball[0])) < 0.1
        e.close()


# ---------------------------------------------------------------------------
# Opponent policies
# ---------------------------------------------------------------------------

class TestOpponentPolicies:
    def test_stationary_opponent(self, env):
        env.reset(seed=0)
        # Just check it runs without error
        for _ in range(10):
            env.step(env.action_space.sample())

    def test_random_opponent(self, env_random_opp):
        env_random_opp.reset(seed=0)
        for _ in range(10):
            env_random_opp.step(env_random_opp.action_space.sample())

    def test_callable_opponent(self):
        def my_policy(obs):
            return np.ones((config.N_ROBOTS, 2), dtype=np.float32) * 0.5

        e = VSSEnv(opponent_policy=my_policy)
        e.reset()
        for _ in range(5):
            e.step(e.action_space.sample())
        e.close()

    def test_invalid_opponent_raises(self):
        with pytest.raises(ValueError):
            VSSEnv(opponent_policy="unknown_policy")


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

class TestGymRegistration:
    def test_make_registered_env(self):
        e = gym.make("VSSS-v0")
        obs, _ = e.reset()
        assert obs.shape == (4 + config.N_TEAMS * config.N_ROBOTS * 7,)
        e.close()


# ---------------------------------------------------------------------------
# Gymnasium env checker (passive compliance check)
# ---------------------------------------------------------------------------

class TestGymCompliance:
    def test_env_checker(self):
        e = VSSEnv(render_mode=None)
        # check_env raises AssertionError on compliance failures
        check_env(e, skip_render_check=True)
        e.close()
