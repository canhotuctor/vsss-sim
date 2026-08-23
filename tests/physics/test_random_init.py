"""Tests for random robot/ball initialization (InitMode.RANDOM)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.config import InitMode
from vsss_sim.physics import jax_backend as jb

_HALF_L = config.FIELD_LENGTH / 2.0
_HALF_W = config.FIELD_WIDTH / 2.0
_MARGIN = config.ROBOT_SIZE


class TestJaxResetRandom:
    def _reset(self, seed: int = 0):
        return jb.reset_random(jax.random.PRNGKey(seed))

    def test_velocities_zeroed(self):
        s = self._reset()
        assert jnp.all(s.robots[:, :, 3:] == 0.0)
        assert jnp.all(s.ball[2:] == 0.0)
        assert jnp.all(s.wheel_speeds == 0.0)

    def test_robots_within_field_x(self):
        for seed in range(10):
            s = self._reset(seed)
            xs = s.robots[:, :, 0]
            assert jnp.all(xs >= -_HALF_L + _MARGIN)
            assert jnp.all(xs <= _HALF_L - _MARGIN)

    def test_robots_within_field_y(self):
        for seed in range(10):
            s = self._reset(seed)
            ys = s.robots[:, :, 1]
            assert jnp.all(jnp.abs(ys) <= _HALF_W - _MARGIN)

    def test_robots_can_spawn_anywhere(self):
        # Over many seeds, both teams should appear in both halves
        keys = jax.random.split(jax.random.PRNGKey(0), 50)
        states = jax.vmap(jb.reset_random)(keys)
        blue_xs = states.robots[:, config.TEAM_BLUE, :, 0]
        yellow_xs = states.robots[:, config.TEAM_YELLOW, :, 0]
        assert float(blue_xs.min()) < 0 and float(blue_xs.max()) > 0
        assert float(yellow_xs.min()) < 0 and float(yellow_xs.max()) > 0

    def test_different_seeds_differ(self):
        s0 = self._reset(0)
        s1 = self._reset(1)
        assert not jnp.allclose(s0.robots, s1.robots)

    def test_vmap_compatible(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 64)
        states = jax.vmap(jb.reset_random)(keys)
        assert states.robots.shape == (64, config.N_TEAMS, config.N_ROBOTS, 6)
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert jnp.all(jnp.abs(states.robots[:, :, :, 0]) <= half_l)
        assert jnp.all(jnp.abs(states.robots[:, :, :, 1]) <= half_w)

    def test_ball_within_field(self):
        for seed in range(10):
            s = self._reset(seed)
            assert float(jnp.abs(s.ball[0])) <= _HALF_L - _MARGIN
            assert float(jnp.abs(s.ball[1])) <= _HALF_W - _MARGIN

    def test_ball_x_y_uncorrelated(self):
        # With the key bug fixed, x and y must not be equal across seeds
        keys = jax.random.split(jax.random.PRNGKey(0), 20)
        balls = jnp.stack([jb.reset_random(k).ball[:2] for k in keys])
        assert not jnp.allclose(balls[:, 0], balls[:, 1])


# ---------------------------------------------------------------------------
# Env-level: VSSEnv + VSSVecEnv
# ---------------------------------------------------------------------------

from vsss_sim.envs import VSSEnv, VSSVecEnv  # noqa: E402


class TestVSSEnvRandomInit:
    def test_random_init(self):
        env = VSSEnv(init_mode=InitMode.RANDOM)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (46,)

    def test_string_init_mode(self):
        env = VSSEnv(init_mode="random")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (46,)

    def test_invalid_init_mode(self):
        with pytest.raises(ValueError):
            VSSEnv(init_mode="banana")

    def test_step_after_random_init(self):
        env = VSSEnv(init_mode=InitMode.RANDOM)
        env.reset(seed=42)
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (46,)


class TestVSSVecEnvRandomInit:
    def test_basic(self):
        env = VSSVecEnv(num_envs=8, init_mode=InitMode.RANDOM)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8, 46)
        env.close()

    def test_string_init_mode(self):
        env = VSSVecEnv(num_envs=4, init_mode="random")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 46)
        env.close()

    def test_robots_within_field(self):
        env = VSSVecEnv(num_envs=32, init_mode=InitMode.RANDOM)
        env.reset(seed=0)
        xs = np.array(env._state.robots[:, :, :, 0])
        ys = np.array(env._state.robots[:, :, :, 1])
        half_l = config.FIELD_LENGTH / 2.0
        half_w = config.FIELD_WIDTH / 2.0
        assert np.all(np.abs(xs) <= half_l)
        assert np.all(np.abs(ys) <= half_w)
        env.close()

    def test_step_runs(self):
        env = VSSVecEnv(num_envs=8, init_mode=InitMode.RANDOM)
        env.reset(seed=0)
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (8, 46)
        env.close()
