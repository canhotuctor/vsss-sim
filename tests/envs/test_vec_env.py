"""Tests for VSSVecEnv (batched JAX-backed Gymnasium VectorEnv)."""
import gymnasium as gym
import numpy as np
import pytest

import vsss_sim  # noqa: F401 — registers VSSS-v0
from vsss_sim import config
from vsss_sim.envs import VSSVecEnv


# ---------------------------------------------------------------------------
# Spaces & basic shape
# ---------------------------------------------------------------------------

@pytest.fixture
def env4():
    e = VSSVecEnv(num_envs=4, opponent_policy="stationary")
    yield e
    e.close()


class TestSpaces:
    def test_num_envs(self, env4):
        assert env4.num_envs == 4

    def test_single_observation_space(self, env4):
        expected = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        assert env4.single_observation_space.shape == (expected,)

    def test_batched_observation_space(self, env4):
        expected = 4 + config.N_TEAMS * config.N_ROBOTS * 7
        assert env4.observation_space.shape == (4, expected)

    def test_single_action_space(self, env4):
        assert env4.single_action_space.shape == (config.N_ROBOTS * 2,)

    def test_batched_action_space(self, env4):
        assert env4.action_space.shape == (4, config.N_ROBOTS * 2)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_batched_obs(self, env4):
        obs, info = env4.reset(seed=0)
        assert obs.shape == (4, 46)
        assert obs.dtype == np.float32

    def test_reset_obs_finite(self, env4):
        obs, _ = env4.reset(seed=0)
        assert np.all(np.isfinite(obs))

    def test_reset_envs_differ(self, env4):
        obs, _ = env4.reset(seed=42)
        # Different seeds across envs → different starting positions (kickoff jitter)
        assert not np.allclose(obs[0], obs[1])

    def test_reset_deterministic_with_seed(self, env4):
        obs1, _ = env4.reset(seed=7)
        obs2, _ = env4.reset(seed=7)
        np.testing.assert_allclose(obs1, obs2)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_5_tuple(self, env4):
        env4.reset(seed=0)
        out = env4.step(np.zeros((4, 6), dtype=np.float32))
        assert len(out) == 5

    def test_step_shapes(self, env4):
        env4.reset(seed=0)
        obs, rew, term, trunc, info = env4.step(
            np.zeros((4, 6), dtype=np.float32)
        )
        assert obs.shape == (4, 46)
        assert rew.shape == (4,)
        assert term.shape == (4,)
        assert trunc.shape == (4,)

    def test_step_obs_finite(self, env4):
        env4.reset(seed=0)
        obs, *_ = env4.step(np.zeros((4, 6), dtype=np.float32))
        assert np.all(np.isfinite(obs))

    def test_zero_action_no_termination(self, env4):
        env4.reset(seed=0)
        for _ in range(5):
            _, _, term, trunc, _ = env4.step(np.zeros((4, 6), dtype=np.float32))
        assert not np.any(term)
        assert not np.any(trunc)

    def test_truncation_per_env(self):
        e = VSSVecEnv(num_envs=2, max_episode_steps=3, opponent_policy="stationary")
        e.reset(seed=0)
        trunc_history = []
        for _ in range(4):
            _, _, _, trunc, _ = e.step(np.zeros((2, 6), dtype=np.float32))
            trunc_history.append(trunc.copy())
        # Both envs should hit truncation at step 3
        assert np.all(trunc_history[2])
        e.close()


# ---------------------------------------------------------------------------
# Opponent policies
# ---------------------------------------------------------------------------

class TestOpponents:
    def test_random_opponent(self):
        e = VSSVecEnv(num_envs=4, opponent_policy="random")
        e.reset(seed=0)
        for _ in range(3):
            e.step(np.zeros((4, 6), dtype=np.float32))
        e.close()

    def test_callable_opponent(self):
        def policy(obs):  # single-env obs, (46,)
            return np.ones((config.N_ROBOTS, 2), dtype=np.float32) * 0.3
        e = VSSVecEnv(num_envs=4, opponent_policy=policy)
        e.reset(seed=0)
        for _ in range(3):
            e.step(np.zeros((4, 6), dtype=np.float32))
        e.close()

    def test_invalid_opponent_raises(self):
        with pytest.raises(ValueError):
            VSSVecEnv(num_envs=2, opponent_policy="unknown")


# ---------------------------------------------------------------------------
# Plausibility: each env's kickoff matches the structural invariants of a
# single VSSEnv kickoff (ball at centre, robots split by team across midfield).
# ---------------------------------------------------------------------------

class TestKickoffStructure:
    def test_ball_at_centre_for_all_envs(self):
        vec = VSSVecEnv(num_envs=4, opponent_policy="stationary")
        obs, _ = vec.reset(seed=42)
        # First 2 obs entries are ball x/y normalised — both ~0 at kickoff.
        np.testing.assert_allclose(obs[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(obs[:, 1], 0.0, atol=1e-5)
        vec.close()

    def test_blue_robots_on_left_yellow_on_right(self):
        vec = VSSVecEnv(num_envs=3, opponent_policy="stationary")
        obs, _ = vec.reset(seed=0)
        # Robot block layout: obs[4 + team*N_ROBOTS*7 + r*7 + 0] is normalised x.
        # Blue is team 0 (left, x<0); yellow is team 1 (right, x>0).
        for team_idx in range(config.N_TEAMS):
            for r in range(config.N_ROBOTS):
                x = obs[:, 4 + (team_idx * config.N_ROBOTS + r) * 7 + 0]
                if team_idx == config.TEAM_BLUE:
                    assert (x < 0).all(), f"blue robot {r} not on left: {x}"
                else:
                    assert (x > 0).all(), f"yellow robot {r} not on right: {x}"
        vec.close()
