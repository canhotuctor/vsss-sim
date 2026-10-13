"""Tests for generic symmetry augmentation and the VSSS group action."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim import config
from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.rl.symmetry import SymmetryEnv
from vsss_sim.symmetry import VSSSSymmetry, VSSSSymmetryElement


def _element(reflected: bool, blue=(0, 1, 2), yellow=(0, 1, 2)):
    return VSSSSymmetryElement(
        jnp.asarray(reflected),
        jnp.asarray(blue, dtype=jnp.int32),
        jnp.asarray(yellow, dtype=jnp.int32),
    )


def test_reflection_flips_geometric_features_and_swaps_wheels():
    symmetry = VSSSSymmetry()
    observation = jnp.arange(46, dtype=jnp.float32) + 1
    action = jnp.arange(6, dtype=jnp.float32)
    element = _element(True)

    transformed = np.asarray(symmetry.transform_observation(observation, element))
    expected = np.asarray(observation).copy()
    expected[[1, 3]] *= -1
    for offset in range(4, 46, 7):
        expected[[offset + 1, offset + 2, offset + 5, offset + 6]] *= -1

    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(
        symmetry.transform_action(action, element),
        np.asarray([1, 0, 3, 2, 5, 4], dtype=np.float32),
    )


def test_action_transform_and_inverse_round_trip():
    symmetry = VSSSSymmetry()
    action = jnp.asarray([0.1, 0.2, 1.1, 1.2, 2.1, 2.2], dtype=jnp.float32)
    element = _element(True, blue=(2, 0, 1), yellow=(1, 2, 0))

    transformed = symmetry.transform_action(action, element)
    restored = symmetry.inverse_action(transformed, element)

    np.testing.assert_allclose(restored, action)


def test_robot_permutations_move_whole_feature_blocks():
    symmetry = VSSSSymmetry()
    observation = jnp.arange(46, dtype=jnp.float32)
    element = _element(False, blue=(2, 0, 1), yellow=(1, 2, 0))

    transformed = symmetry.transform_observation(observation, element)
    original_robots = np.asarray(observation[4:]).reshape(2, config.N_ROBOTS, 7)
    transformed_robots = np.asarray(transformed[4:]).reshape(2, config.N_ROBOTS, 7)

    np.testing.assert_array_equal(transformed_robots[0], original_robots[0, [2, 0, 1]])
    np.testing.assert_array_equal(transformed_robots[1], original_robots[1, [1, 2, 0]])


def test_sample_can_disable_each_symmetry_component():
    element = VSSSSymmetry(False, False, False).sample(jax.random.PRNGKey(0))

    assert not bool(element.reflected)
    np.testing.assert_array_equal(element.blue_permutation, np.arange(config.N_ROBOTS))
    np.testing.assert_array_equal(element.yellow_permutation, np.arange(config.N_ROBOTS))


def test_generic_wrapper_is_jittable_and_preserves_specs():
    base_env = VSSJumanjiEnv(max_episode_steps=2)
    env = SymmetryEnv(base_env, VSSSSymmetry())
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state, timestep = reset(jax.random.PRNGKey(0))
    state, timestep = step(state, jnp.zeros(env.action_spec.shape, dtype=jnp.float32))

    assert timestep.observation.shape == env.observation_spec.shape
    assert env.action_spec is base_env.action_spec
    assert env.observation_spec is base_env.observation_spec
