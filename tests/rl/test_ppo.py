"""Correctness and compiled smoke tests for the JAX-native PPO."""

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim.envs.brandao_observation import mirror_x_axis_observation
from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig, compute_gae
from vsss_sim.rl.ppo import _TrainingBatch, create_policy_model


def test_config_rejects_non_divisible_minibatches():
    try:
        PPOConfig(num_envs=3, rollout_length=3, num_minibatches=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected invalid minibatch configuration to fail")



def test_ppo_loss_stays_finite_for_extreme_probability_ratios():
    env = VSSJumanjiEnv()
    trainer = PPO(
        env,
        PPOConfig(
            num_envs=2,
            rollout_length=2,
            update_epochs=1,
            num_minibatches=1,
            hidden_sizes=(8,),
        ),
    )
    runner = trainer.initialize(jax.random.PRNGKey(89))
    batch = _TrainingBatch(
        observation=runner.observation,
        raw_action=jnp.zeros((2, 6), dtype=jnp.float32),
        old_log_prob=jnp.full((2,), -1.0e6, dtype=jnp.float32),
        old_value=jnp.zeros((2,), dtype=jnp.float32),
        advantage=jnp.ones((2,), dtype=jnp.float32),
        target=jnp.zeros((2,), dtype=jnp.float32),
    )

    loss, metrics = trainer._loss(runner.train_state.params, batch)

    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["approx_kl"]))
    assert float(metrics["ratio_log_clipped_fraction"]) == 1.0

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


def test_brandao_shared_ppo_updates_from_three_local_transitions_per_tick():
    env = VSSJumanjiEnv(
        max_episode_steps=3,
        reward_profile="brandao_reward",
        policy_architecture="brandao_shared",
        terminate_on_goal=True,
    )
    config = PPOConfig(
        num_envs=1,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
    )
    trainer = PPO(env, config)
    runner = trainer.initialize(jax.random.PRNGKey(17))
    params_before = runner.train_state.params

    runner, metrics = trainer.update(runner)
    jax.block_until_ready((runner, metrics))

    changed = [
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(runner.train_state.params),
            strict=True,
        )
    ]
    assert any(changed)
    assert runner.observation.shape == (1, 3, 424)
    assert int(metrics["env_steps"]) == 2
    assert int(metrics["agent_transitions"]) == 6
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


def _permute_set_observation(
    observation: jax.Array,
    ally_order: tuple[int, int],
    opponent_order: tuple[int, int, int],
) -> jax.Array:
    frames = observation.reshape((8, 53))
    positions = frames[:, :14].reshape((8, 7, 2))
    positions = positions.at[:, 1:3].set(positions[:, 1:3][:, ally_order])
    positions = positions.at[:, 3:6].set(positions[:, 3:6][:, opponent_order])

    bearings = frames[:, 17:38].reshape((8, 7, 3))
    bearings = bearings.at[:, :2].set(bearings[:, :2][:, ally_order])
    bearings = bearings.at[:, 2:5].set(bearings[:, 2:5][:, opponent_order])

    distances = frames[:, 38:46]
    distances = distances.at[:, :2].set(distances[:, :2][:, ally_order])
    distances = distances.at[:, 2:5].set(distances[:, 2:5][:, opponent_order])
    actions = frames[:, 46:52].reshape((8, 3, 2))
    actions = actions.at[:, 1:3].set(actions[:, 1:3][:, ally_order])

    permuted = frames.at[:, :14].set(positions.reshape((8, 14)))
    permuted = permuted.at[:, 17:38].set(bearings.reshape((8, 21)))
    permuted = permuted.at[:, 38:46].set(distances)
    permuted = permuted.at[:, 46:52].set(actions.reshape((8, 6)))
    return permuted.reshape((424,))


def test_brandao_set_policy_is_invariant_to_all_ally_and_opponent_orders():
    from itertools import permutations

    observation = jax.random.normal(jax.random.PRNGKey(31), (424,))
    model = create_policy_model("brandao_sets", hidden_sizes=(16, 8))
    params = model.init(jax.random.PRNGKey(32), observation)
    expected = model.apply(params, observation)

    for ally_order in permutations(range(2)):
        for opponent_order in permutations(range(3)):
            permuted = _permute_set_observation(
                observation, ally_order, opponent_order
            )
            actual = model.apply(params, permuted)
            for expected_value, actual_value in zip(expected, actual, strict=True):
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=2e-6, atol=2e-6
                )


def test_x_axis_mirror_transform_is_an_involution():
    observation = jax.random.normal(jax.random.PRNGKey(41), (3, 424))
    reflected = mirror_x_axis_observation(observation)
    restored = mirror_x_axis_observation(reflected)

    np.testing.assert_array_equal(restored, observation)
    original_actions = observation.reshape((3, 8, 53))[..., 46:52].reshape(
        (3, 8, 3, 2)
    )
    reflected_actions = reflected.reshape((3, 8, 53))[..., 46:52].reshape(
        (3, 8, 3, 2)
    )
    np.testing.assert_array_equal(reflected_actions, original_actions[..., ::-1])


def test_brandao_set_ppo_trains_with_on_the_fly_mirror_augmentation():
    env = VSSJumanjiEnv(
        max_episode_steps=2,
        reward_profile="brandao_reward",
        policy_architecture="brandao_sets",
        terminate_on_goal=True,
    )
    config = PPOConfig(
        num_envs=1,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(16,),
        mirror_augmentation=True,
    )
    trainer = PPO(env, config)
    runner = trainer.initialize(jax.random.PRNGKey(51))
    params_before = runner.train_state.params
    runner, metrics = trainer.update(runner)
    jax.block_until_ready((runner, metrics))

    changed = [
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(runner.train_state.params),
            strict=True,
        )
    ]
    assert any(changed)
    assert runner.observation.shape == (1, 3, 424)
    assert int(metrics["env_steps"]) == 2
    assert int(metrics["agent_transitions"]) == 6
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


def test_multiple_generations_run_inside_compiled_scan():
    env = VSSJumanjiEnv(max_episode_steps=3)
    config = PPOConfig(
        num_envs=2,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(8,),
    )
    trainer = PPO(env, config)
    runner = trainer.initialize(jax.random.PRNGKey(9))

    runner, metrics_history = trainer.train(runner, num_updates=3)
    jax.block_until_ready((runner, metrics_history))

    assert int(runner.env_steps) == 3 * config.batch_size
    assert metrics_history["loss"].shape == (3,)
    assert int(metrics_history["env_steps"][-1]) == 3 * config.batch_size
    assert all(
        np.isfinite(np.asarray(values)).all() for values in metrics_history.values()
    )
