"""Correctness and compiled smoke tests for the JAX-native PPO."""

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig, compute_gae


def test_config_rejects_non_divisible_minibatches():
    try:
        PPOConfig(num_envs=3, rollout_length=3, num_minibatches=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected invalid minibatch configuration to fail")


def test_gae_bootstraps_truncation_but_stops_episode_recursion():
    rewards = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)
    values = jnp.asarray([[0.5], [0.25]], dtype=jnp.float32)
    bootstraps = jnp.asarray([[0.25], [4.0]], dtype=jnp.float32)
    discounts = jnp.ones((2, 1), dtype=jnp.float32)
    episode_ends = jnp.asarray([[False], [True]])

    advantages, targets = compute_gae(
        rewards,
        values,
        bootstraps,
        discounts,
        episode_ends,
        gamma=0.9,
        gae_lambda=0.95,
    )

    final_delta = 2.0 + 0.9 * 4.0 - 0.25
    first_delta = 1.0 + 0.9 * 0.25 - 0.5
    np.testing.assert_allclose(advantages[1, 0], final_delta, rtol=1e-6)
    np.testing.assert_allclose(
        advantages[0, 0], first_delta + 0.9 * 0.95 * final_delta, rtol=1e-6
    )
    np.testing.assert_allclose(targets, advantages + values, rtol=1e-6)


def test_actions_are_bounded():
    env = VSSJumanjiEnv()
    trainer = PPO(env, PPOConfig(num_envs=2, rollout_length=2, num_minibatches=1))
    runner = trainer.initialize(jax.random.PRNGKey(0))

    deterministic = trainer.act(runner.train_state.params, runner.observation)
    stochastic = trainer.act(
        runner.train_state.params,
        runner.observation,
        key=jax.random.PRNGKey(1),
        deterministic=False,
    )

    assert deterministic.shape == (2, 6)
    assert stochastic.shape == (2, 6)
    assert bool(jnp.all(jnp.abs(deterministic) <= 1.0))
    assert bool(jnp.all(jnp.abs(stochastic) <= 1.0))


def test_one_compiled_update_changes_parameters_and_counts_steps():
    env = VSSJumanjiEnv(max_episode_steps=3)
    config = PPOConfig(
        num_envs=2,
        rollout_length=4,
        update_epochs=1,
        num_minibatches=2,
        hidden_sizes=(16,),
    )
    trainer = PPO(env, config)
    runner = trainer.initialize(jax.random.PRNGKey(3))
    params_before = runner.train_state.params

    runner, metrics = trainer.update(runner)
    jax.block_until_ready(metrics)

    changed = [
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(runner.train_state.params),
            strict=True,
        )
    ]
    assert any(changed)
    assert int(metrics["env_steps"]) == config.batch_size
    assert int(metrics["episodes"]) == config.num_envs
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
