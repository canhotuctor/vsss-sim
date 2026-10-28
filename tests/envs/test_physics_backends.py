"""Every environment adapter runs on either physics engine without other changes."""

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.envs import VSSEnv, VSSVecEnv
from vsss_sim.physics import jax_backend


def _mjx_backend():
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.importorskip("mujoco.mjx")
    from vsss_sim.physics import mjx_backend

    return mjx_backend


def test_adapters_default_to_the_jax_engine_and_reject_unknown_engines():
    assert VSSEnv()._physics is jax_backend
    with pytest.raises(ValueError):
        VSSEnv(physics_backend="bullet")
    with pytest.raises(ValueError):
        VSSVecEnv(num_envs=1, physics_backend="bullet")


def test_single_match_runs_on_mjx():
    mjx_backend = _mjx_backend()
    env = VSSEnv(opponent_policy="random", physics_backend="mjx")
    assert env._physics is mjx_backend
    observation, _ = env.reset(seed=0)
    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(
            np.full(6, 0.5, dtype=np.float32)
        )
    assert observation.shape == (46,)
    assert np.isfinite(observation).all() and np.isfinite(reward)
    assert not terminated and not truncated
    assert float(env._state.t) == pytest.approx(3 * config.DT)
    env.close()


def test_human_mode_draws_every_mjx_substep_of_the_same_physics(monkeypatch):
    _mjx_backend()
    pytest.importorskip("pygame")
    from vsss_sim.rendering import VSSRenderer

    frames = []
    monkeypatch.setattr(
        VSSRenderer, "render",
        lambda self, ball, robots, score, **kwargs: frames.append(np.array(robots)),
    )
    human = VSSEnv(render_mode="human", physics_backend="mjx")
    plain = VSSEnv(physics_backend="mjx")
    human.reset(seed=3)
    plain.reset(seed=3)
    action = np.full(6, 0.8, dtype=np.float32)
    human.step(action)
    plain.step(action)

    assert len(frames) == config.SUB_STEPS
    np.testing.assert_allclose(frames[-1], np.asarray(plain._state.robots), atol=1e-5)
    human.close()
    plain.close()


def test_vector_env_runs_on_mjx():
    _mjx_backend()
    env = VSSVecEnv(num_envs=3, opponent_policy="random", physics_backend="mjx")
    observations, _ = env.reset(seed=1)
    for _ in range(2):
        observations, rewards, terminations, truncations, info = env.step(
            np.full((3, 6), 0.5, dtype=np.float32)
        )
    assert observations.shape == (3, 46)
    assert np.isfinite(observations).all() and np.isfinite(rewards).all()
    env.close()


def test_jumanji_env_runs_on_mjx_under_jit_and_vmap():
    _mjx_backend()
    pytest.importorskip("jumanji")
    from vsss_sim.envs.jumanji import VSSJumanjiEnv

    env = VSSJumanjiEnv(physics_backend="mjx")
    assert env.physics_backend is config.PhysicsBackend.MJX
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states, _ = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.full((2, 6), 0.5, dtype=jnp.float32)
    states, timestep = jax.jit(jax.vmap(env.step))(states, actions)

    assert timestep.observation.shape == (2, 46)
    assert np.isfinite(np.asarray(timestep.observation)).all()
    np.testing.assert_allclose(states.simulation.t, config.DT)
