"""Tests for the optional, fully JAX-resident Jumanji environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from vsss_sim import config  # noqa: E402
from vsss_sim.envs.brandao_observation import build_agent_features  # noqa: E402
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


def test_brandao_goal_reward_is_jittable_and_uses_remaining_horizon():
    env = VSSJumanjiEnv(max_episode_steps=5, reward_profile="brandao_reward")
    state, _ = env.reset(jax.random.PRNGKey(0))
    ball = state.simulation.ball.at[0].set(config.FIELD_LENGTH / 2.0 + 0.02)
    ball = ball.at[1].set(0.0).at[2].set(0.05)
    state = state._replace(simulation=state.simulation._replace(ball=ball))

    next_state, timestep = jax.jit(env.step)(state, jnp.zeros(6, dtype=jnp.float32))

    assert int(timestep.extras["goal"]) == 1
    np.testing.assert_allclose(timestep.reward, 18.0, atol=1e-6)
    assert int(next_state.simulation.score[config.TEAM_BLUE]) == 1
    assert abs(float(next_state.simulation.ball[0])) < 0.1
    np.testing.assert_allclose(
        timestep.observation,
        env._match_rules.observation(next_state.simulation),
    )


def test_brandao_shared_policy_builds_stacked_agent_views_and_exact_10hz_steps():
    env = VSSJumanjiEnv(
        max_episode_steps=10,
        reward_profile="brandao_reward",
        policy_architecture="brandao_shared",
        terminate_on_goal=True,
    )
    state, first = env.reset(jax.random.PRNGKey(8))
    assert first.observation.shape == (3, 424)
    assert env.action_spec.shape == (3, 2)
    assert state.observation_history.shape == (2, 3, 8, 53)

    action = jnp.asarray([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]])
    time_before = state.simulation.t
    state, first_step = jax.jit(env.step)(state, action)
    time_after_one_decision = state.simulation.t
    state, second_step = jax.jit(env.step)(state, action)

    np.testing.assert_allclose(
        time_after_one_decision - time_before, 0.1, atol=1e-7
    )
    np.testing.assert_allclose(
        state.simulation.t - time_after_one_decision, 0.1, atol=1e-7
    )
    assert first_step.reward.shape == (3,)
    assert second_step.observation.shape == (3, 424)
    assert env.opponent_observation(state).shape == (3, 424)
    np.testing.assert_allclose(
        state.observation_history[config.TEAM_BLUE, 0, -1, 46:52],
        action.reshape(-1),
    )


def test_brandao_set_observation_keeps_each_ally_action_with_its_entity():
    env = VSSJumanjiEnv(
        reward_profile="brandao_reward",
        policy_architecture="brandao_sets",
    )
    state, timestep = env.reset(jax.random.PRNGKey(18))
    previous_actions = jnp.arange(12, dtype=jnp.float32).reshape((2, 3, 2)) / 10.0
    features = build_agent_features(
        state.simulation,
        previous_actions,
        jnp.float32(1.0),
        jax.random.PRNGKey(19),
        entity_ordered_actions=True,
    )

    assert timestep.observation.shape == (3, 424)
    for robot in range(config.N_ROBOTS):
        allies = [index for index in range(config.N_ROBOTS) if index != robot]
        expected = previous_actions[config.TEAM_BLUE, [robot, *allies]].reshape(-1)
        np.testing.assert_array_equal(
            features[config.TEAM_BLUE, robot, 46:52], expected
        )


def test_only_jax_native_opponents_are_accepted():
    with pytest.raises(ValueError, match="JAX-native"):
        VSSJumanjiEnv(opponent_policy="custom")
