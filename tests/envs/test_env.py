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

    def test_step_reward_goal_values(self, env):
        """Reward should be ±1.0 on goal or 0.0 otherwise."""
        env.reset(seed=0)
        _, reward, *_ = env.step(np.zeros(config.N_ROBOTS * 2))
        assert reward in (-1.0, 0.0, 1.0)

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
        e._state.ball[0] = config.FIELD_LENGTH / 2.0 + 0.02
        e._state.ball[1] = 0.0
        e._state.ball[2] = 0.05

        obs, reward, terminated, truncated, info = e.step(
            np.zeros(config.N_ROBOTS * 2)
        )
        assert reward == 1.0
        assert info["goal"] == 1
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
        check_env(e, warn=True, skip_render_check=True)
        e.close()
