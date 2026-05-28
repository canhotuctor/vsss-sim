"""Tests for random robot/ball initialization (InitMode.RANDOM)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.config import InitMode
from vsss_sim.physics import numpy_backend as nb

_HALF_L = config.FIELD_LENGTH / 2.0
_HALF_W = config.FIELD_WIDTH / 2.0
_MARGIN = config.ROBOT_SIZE


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

class TestNumpyResetRandom:
    def _reset(self, seed: int = 0) -> nb.SimState:
        state = nb.SimState()
        nb.reset_random(state, rng=np.random.default_rng(seed))
        return state

    def test_velocities_zeroed(self):
        s = self._reset()
        assert np.all(s.robots[:, :, 3:] == 0.0)
        assert np.all(s.ball[2:] == 0.0)
        assert np.all(s.wheel_speeds == 0.0)

    def test_robots_within_field_x(self):
        for seed in range(10):
            s = self._reset(seed)
            xs = s.robots[:, :, 0]
            assert np.all(xs >= -_HALF_L + _MARGIN)
            assert np.all(xs <= _HALF_L - _MARGIN)

    def test_robots_within_field_y(self):
        for seed in range(10):
            s = self._reset(seed)
            ys = s.robots[:, :, 1]
            assert np.all(np.abs(ys) <= _HALF_W - _MARGIN)

    def test_robots_can_spawn_anywhere(self):
        # Over many seeds, both teams should appear in both halves
        blue_xs = np.array([self._reset(s).robots[config.TEAM_BLUE, :, 0] for s in range(50)]).ravel()
        yellow_xs = np.array([self._reset(s).robots[config.TEAM_YELLOW, :, 0] for s in range(50)]).ravel()
        assert blue_xs.min() < 0 and blue_xs.max() > 0
        assert yellow_xs.min() < 0 and yellow_xs.max() > 0

    def test_headings_full_circle(self):
        # Over many seeds, headings should span at least 3 radians on each side
        thetas = np.array([
            self._reset(s).robots[:, :, 2].ravel() for s in range(50)
        ]).ravel()
        assert thetas.min() < -math.pi * 0.8
        assert thetas.max() > math.pi * 0.8

    def test_different_seeds_differ(self):
        s0 = self._reset(0)
        s1 = self._reset(1)
        assert not np.allclose(s0.robots, s1.robots)

    def test_ball_within_field(self):
        for seed in range(10):
            s = self._reset(seed)
            assert abs(s.ball[0]) < _HALF_L
            assert abs(s.ball[1]) < _HALF_W


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from vsss_sim.physics import jax_backend as jb  # noqa: E402


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
            assert float(jnp.abs(s.ball[0])) < _HALF_L
            assert float(jnp.abs(s.ball[1])) < _HALF_W


# ---------------------------------------------------------------------------
# Env-level: VSSEnv + VSSVecEnv
# ---------------------------------------------------------------------------

from vsss_sim.envs import VSSEnv, VSSVecEnv  # noqa: E402


class TestVSSEnvRandomInit:
    def test_numpy_backend_random(self):
        env = VSSEnv(backend="numpy", init_mode=InitMode.RANDOM)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (46,)

    def test_jax_backend_random(self):
        env = VSSEnv(backend="jax", init_mode=InitMode.RANDOM)
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
        env = VSSEnv(backend="numpy", init_mode=InitMode.RANDOM)
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
