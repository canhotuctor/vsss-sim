"""Tests for the optional, fully JAX-resident Jumanji environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from vsss_sim import config  # noqa: E402
from vsss_sim.envs.jumanji import VSSJumanjiEnv  # noqa: E402


def test_specs_match_public_gymnasium_environment():
    env = VSSJumanjiEnv()
    assert env.observation_spec.shape == (46,)
    assert env.action_spec.shape == (6,)
    assert env.observation_spec.dtype == jnp.float32
    assert env.action_spec.dtype == jnp.float32


def test_reset_and_step_are_jittable():
    env = VSSJumanjiEnv()
    state, first = jax.jit(env.reset)(jax.random.PRNGKey(0))
    next_state, timestep = jax.jit(env.step)(state, jnp.zeros(6, dtype=jnp.float32))

    assert bool(first.first())
    assert bool(timestep.mid())
    assert np.asarray(timestep.observation).shape == (46,)
    assert np.asarray(timestep.reward).dtype == np.float32
    assert int(next_state.step_count) == 1


def test_reset_and_step_vectorize_without_host_transfer():
    env = VSSJumanjiEnv(opponent_policy="random", init_mode="random")
    keys = jax.random.split(jax.random.PRNGKey(7), 8)
    states, first = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((8, 6), dtype=jnp.float32)
    states, timestep = jax.jit(jax.vmap(env.step))(states, actions)

    assert first.observation.shape == (8, 46)
    assert timestep.observation.shape == (8, 46)
    assert timestep.reward.shape == (8,)
    assert states.simulation.robots.shape == (8, 2, 3, 6)


def test_time_limit_is_bootstrappable_truncation():
    env = VSSJumanjiEnv(max_episode_steps=1)
    state, _ = env.reset(jax.random.PRNGKey(0))
    _, timestep = env.step(state, jnp.zeros(6, dtype=jnp.float32))

    assert bool(timestep.last())
    assert float(timestep.discount) == 1.0


def test_goal_kickoff_preserves_score_and_match_time():
    env = VSSJumanjiEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    ball = state.simulation.ball.at[0].set(config.FIELD_LENGTH / 2.0 + 0.02)
    ball = ball.at[1].set(0.0).at[2].set(0.05)
    state = state._replace(simulation=state.simulation._replace(ball=ball))

    next_state, timestep = jax.jit(env.step)(state, jnp.zeros(6, dtype=jnp.float32))

    assert int(timestep.extras["goal"]) == 1
    assert int(next_state.simulation.score[config.TEAM_BLUE]) == 1
    assert float(next_state.simulation.t) > 0.0
    assert abs(float(next_state.simulation.ball[0])) < 0.1


def test_only_jax_native_opponents_are_accepted():
    with pytest.raises(ValueError, match="JAX-native"):
        VSSJumanjiEnv(opponent_policy="custom")
