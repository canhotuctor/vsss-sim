"""Tests for VSSVecEnvToSB3 — SB3 VecEnv adapter around VSSVecEnv."""
import numpy as np
import pytest
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from vsss_sim import config
from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3


# ---------------------------------------------------------------------------
# Construction & SB3 contract surface
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_sb3_vec_env(self):
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            assert isinstance(env, VecEnv)
            assert env.num_envs == 4
        finally:
            env.close()

    def test_per_env_spaces_not_batched(self):
        """SB3 VecEnv exposes PER-ENV observation_space / action_space (not batched)."""
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            obs_dim = 4 + config.N_TEAMS * config.N_ROBOTS * 7
            assert env.observation_space.shape == (obs_dim,)
            assert env.action_space.shape == (config.N_ROBOTS * 2,)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Reset (SB3 contract: returns obs only, no info tuple)
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_only_obs(self):
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            out = env.reset()
            assert isinstance(out, np.ndarray)
            assert out.shape == (4, 4 + config.N_TEAMS * config.N_ROBOTS * 7)
            assert out.dtype == np.float32
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Step (SB3 contract: dones = term|trunc, infos = list[dict])
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_shapes_and_types(self):
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=3))
        try:
            env.reset()
            actions = np.zeros((3, config.N_ROBOTS * 2), dtype=np.float32)
            obs, rew, dones, infos = env.step(actions)
            assert obs.shape == (3, 4 + config.N_TEAMS * config.N_ROBOTS * 7)
            assert rew.shape == (3,)
            assert dones.shape == (3,)
            assert dones.dtype == bool
            assert isinstance(infos, list)
            assert len(infos) == 3
            assert all(isinstance(d, dict) for d in infos)
        finally:
            env.close()

    def test_zero_action_no_dones_short_horizon(self):
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=2))
        try:
            env.reset()
            actions = np.zeros((2, config.N_ROBOTS * 2), dtype=np.float32)
            for _ in range(5):
                _, _, dones, _ = env.step(actions)
            assert not dones.any()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Auto-reset (the load-bearing semantic): on done, obs is post-reset and
# info["terminal_observation"] holds the pre-reset obs.
# ---------------------------------------------------------------------------

class TestAutoReset:
    def test_eager_autoreset_on_truncation(self):
        """At max_episode_steps, the step returns dones=True AND post-reset obs."""
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=2, max_episode_steps=3))
        try:
            env.reset()
            actions = np.zeros((2, config.N_ROBOTS * 2), dtype=np.float32)
            # First 2 steps: no truncation
            for _ in range(2):
                _, _, dones, _ = env.step(actions)
                assert not dones.any()
            # 3rd step: truncated. obs returned should be post-reset (= kickoff).
            obs, _, dones, infos = env.step(actions)
            assert dones.all()
            # Ball is at centre on kickoff (obs[0:2] ≈ 0).
            np.testing.assert_allclose(obs[:, 0], 0.0, atol=1e-5)
            np.testing.assert_allclose(obs[:, 1], 0.0, atol=1e-5)
            # Each info dict should contain the pre-reset terminal observation.
            for d in infos:
                assert "terminal_observation" in d
                assert d["terminal_observation"].shape == obs.shape[1:]
                assert "TimeLimit.truncated" in d
                assert d["TimeLimit.truncated"] is True
        finally:
            env.close()

    def test_episode_info_emitted_on_done(self):
        """SB3's logger watches info['episode'] = {'r': ..., 'l': ..., 't': ...}."""
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=2, max_episode_steps=3))
        try:
            env.reset()
            actions = np.zeros((2, config.N_ROBOTS * 2), dtype=np.float32)
            for _ in range(2):
                env.step(actions)
            _, _, _, infos = env.step(actions)
            for d in infos:
                assert "episode" in d
                ep = d["episode"]
                assert ep["l"] == 3       # length in steps
                assert ep["r"] == 0.0     # cumulative reward (no goals from zero action)
                assert ep["t"] >= 0       # elapsed wall-clock seconds
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Introspection methods SB3 calls during __init__ and training
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_get_attr_render_mode(self):
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            modes = env.get_attr("render_mode")
            assert len(modes) == 4
            assert all(m is None for m in modes)
        finally:
            env.close()

    def test_env_is_wrapped_false(self):
        """VSSVecEnv isn't wrapped in TimeLimit (we handle truncation ourselves)."""
        import gymnasium as gym
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            wrapped = env.env_is_wrapped(gym.wrappers.TimeLimit)
            assert wrapped == [False] * 4
        finally:
            env.close()


# ---------------------------------------------------------------------------
# End-to-end: SB3 PPO can construct and learn briefly without errors.
# ---------------------------------------------------------------------------

class TestSB3Integration:
    def test_ppo_learn_smoke(self):
        from stable_baselines3 import PPO
        env = VSSVecEnvToSB3(VSSVecEnv(num_envs=4))
        try:
            model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, verbose=0)
            # 256 timesteps = 64 batched steps × 4 envs — runs through 2 rollouts.
            model.learn(total_timesteps=256)
        finally:
            env.close()
