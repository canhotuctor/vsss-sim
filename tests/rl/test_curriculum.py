"""Behavioral coverage for online start-state replay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig
from vsss_sim.rl.ppo import _curriculum_scores


def test_reward_priority_masks_incomplete_episodes_and_handles_equal_returns():
    score = jax.jit(_curriculum_scores)
    weights, low, high = score(jnp.array([-2., 0., 2., 100.]),
                               jnp.array([True, True, True, False]))
    np.testing.assert_allclose(weights, [0., .25, 0., 0.])
    assert float(low) == -2. and float(high) == 2.
    weights, low, high = score(jnp.array([3., 3., 12., 12.]),
                               jnp.array([True, True, False, False]))
    np.testing.assert_allclose(weights, [.25, .25, 0., 0.])
    weights, low, high = score(jnp.zeros(4), jnp.zeros(4, dtype=bool))
    assert np.isfinite(np.asarray(weights)).all()
    np.testing.assert_array_equal(weights, 0.)
    assert float(low) == 0. and float(high) == 0.


def test_level_reset_reproduces_physics_with_fresh_episode_randomness():
    env = VSSJumanjiEnv(init_mode="random", policy_architecture="brandao_shared",
                        reward_profile="brandao_reward")
    level = jax.random.PRNGKey(123)
    reset = jax.jit(env.reset_level)
    a, ta = reset(level, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    b, tb = reset(level, jax.random.PRNGKey(3), jax.random.PRNGKey(4))
    for x, y in zip(jax.tree_util.tree_leaves(a.simulation),
                    jax.tree_util.tree_leaves(b.simulation), strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.key, b.key)
    assert int(a.step_count) == 0
    assert int(b.step_count) == 0
    np.testing.assert_array_equal(a.previous_actions, 0)
    assert np.isfinite(np.asarray(ta.observation)).all()
    assert np.isfinite(np.asarray(tb.observation)).all()
    assert ta.observation.shape == (3, 424)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_replay_fraction_must_be_a_probability(fraction):
    with pytest.raises(ValueError):
        PPOConfig(num_envs=2, rollout_length=2, num_minibatches=1,
                  curriculum_replay_fraction=fraction)


def test_completed_episodes_across_updates_enter_replay_buffer():
    env = VSSJumanjiEnv(init_mode="random", max_episode_steps=3)
    cfg = PPOConfig(num_envs=2, rollout_length=2, update_epochs=1,
                    num_minibatches=1, hidden_sizes=(8,),
                    curriculum_replay_fraction=1.0, curriculum_buffer_size=8)
    trainer = PPO(env, cfg)
    runner = trainer.initialize(jax.random.PRNGKey(8))
    initial_keys = np.asarray(runner.initial_level_keys).copy()
    runner, first = trainer.update(runner)
    assert int(first["episodes"]) == 0
    np.testing.assert_array_equal(runner.initial_level_keys, initial_keys)
    runner, second = trainer.update(runner)
    jax.block_until_ready((runner, second))
    assert int(second["episodes"]) == 2
    saved = np.asarray(runner.replay_level_keys)[np.asarray(runner.replay_valid)]
    assert len(saved) >= 2
    for key in initial_keys:
        assert any(np.array_equal(key, candidate) for candidate in saved)
    assert np.isfinite(np.asarray(runner.replay_weights)).all()
    # Buffer updates happen between PPO updates. The next episode reset
    # must select from the already-populated pool when replay fraction is 1.
    runner, third = trainer.update(runner)
    jax.block_until_ready((runner, third))
    assert int(third["episodes"]) == 2
    for key in np.asarray(runner.initial_level_keys):
        assert any(np.array_equal(key, candidate) for candidate in saved)
