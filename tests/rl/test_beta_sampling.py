"""Distribution and rollout integration checks for batched Beta sampling."""

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl import PPO, PPOConfig
from vsss_sim.rl.ppo import _sample_beta


def test_batched_beta_moments_and_independent_policy_groups():
    draws = 32768
    alpha = np.asarray((0.2, 1.0, 2.0, 25.0), dtype=np.float32)
    beta = np.asarray((0.7, 1.0, 5.0, 40.0), dtype=np.float32)
    shape = (2, len(alpha), draws)
    samples = np.asarray(
        jax.jit(_sample_beta)(
            jax.random.PRNGKey(420),
            jnp.broadcast_to(jnp.asarray(alpha)[None, :, None], shape),
            jnp.broadcast_to(jnp.asarray(beta)[None, :, None], shape),
        )
    )

    expected_mean = alpha / (alpha + beta)
    expected_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    assert samples.shape == shape
    assert np.isfinite(samples).all()
    assert ((samples >= 0) & (samples <= 1)).all()
    assert np.all(
        np.abs(samples.mean(axis=-1) - expected_mean)
        < 6 * np.sqrt(expected_variance / draws)
    )
    np.testing.assert_allclose(
        samples.var(axis=-1),
        np.broadcast_to(expected_variance, shape[:2]),
        rtol=0.06,
        atol=1e-5,
    )
    centered = samples - samples.mean(axis=-1, keepdims=True)
    correlations = (centered[0] * centered[1]).mean(axis=-1) / expected_variance
    assert (np.abs(correlations) < 0.04).all()


def _count_gamma_draws(graph):
    if isinstance(graph, (tuple, list)):
        return sum(_count_gamma_draws(value) for value in graph)
    graph = getattr(graph, "jaxpr", graph)
    if not hasattr(graph, "eqns"):
        return 0
    return sum(
        int(equation.primitive.name == "random_gamma")
        + sum(_count_gamma_draws(value) for value in equation.params.values())
        for equation in graph.eqns
    )


def test_self_play_rollout_batches_learner_and_opponent_gamma_draws():
    trainer = PPO(
        VSSJumanjiEnv(
            max_episode_steps=2,
            reward_profile="brandao_reward",
            policy_architecture="brandao_shared",
        ),
        PPOConfig(num_envs=2, rollout_length=2, update_epochs=1, num_minibatches=1),
    )
    key = jax.random.PRNGKey(821)
    initial = trainer.initialize(key)
    pool = jax.tree_util.tree_map(lambda value: value[None], initial.train_state.params)
    probabilities = jnp.asarray((0.0, 0.0, 1.0, 0.0, 0.0), dtype=jnp.float32)
    generations = jnp.asarray((0,), dtype=jnp.int32)
    runner = trainer.initialize(key, pool, probabilities, generations)

    rollout = jax.make_jaxpr(trainer._collect_rollout)(
        runner, pool, probabilities, generations
    )

    assert _count_gamma_draws(rollout) == 1
