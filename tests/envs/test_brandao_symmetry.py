"""The 24-element observation symmetry group used by sampled augmentation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.envs import brandao_observation as observation
from vsss_sim.physics import jax_backend as physics

_OPPONENT_PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
)


def _apply(index, sign, features):
    return features[index] * sign


@pytest.mark.parametrize("entity_ordered_actions", [False, True])
def test_group_element_one_is_the_existing_x_axis_mirror(entity_ordered_actions):
    index, sign, wheel_swap = observation.symmetry_tables(entity_ordered_actions)
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (3, observation.OBSERVATION_SIZE)))
    expected = np.asarray(observation.mirror_x_axis_observation(jnp.asarray(obs)))
    for robot in range(3):
        np.testing.assert_array_equal(_apply(index[robot, 1], sign[robot, 1], obs[robot]), expected[robot])
        np.testing.assert_array_equal(_apply(index[robot, 0], sign[robot, 0], obs[robot]), obs[robot])
    assert not wheel_swap[0] and wheel_swap[1]


def _transform_world(simulation, previous_actions, team, robot, mirror, ally_swap, opponents):
    """Relabel and reflect the physical state the way group element g does for one robot."""
    robots = np.array(simulation.robots)
    ball = np.array(simulation.ball)
    actions = np.array(previous_actions)
    if ally_swap:
        a, b = [other for other in range(config.N_ROBOTS) if other != robot]
        robots[team, [a, b]] = robots[team, [b, a]]
        actions[team, [a, b]] = actions[team, [b, a]]
    robots[1 - team] = robots[1 - team][list(opponents)]
    actions[1 - team] = actions[1 - team][list(opponents)]
    if mirror:
        robots[..., 1] *= -1.0
        robots[..., 2] *= -1.0
        ball[1] *= -1.0
        actions = actions[..., ::-1]
    return simulation._replace(robots=jnp.asarray(robots), ball=jnp.asarray(ball)), jnp.asarray(actions)


@pytest.mark.parametrize("entity_ordered_actions", [False, True])
def test_every_group_element_matches_relabelling_the_physical_world(
    monkeypatch, entity_ordered_actions
):
    monkeypatch.setattr(observation, "_POSITION_NOISE", jnp.zeros(2, jnp.float32))
    monkeypatch.setattr(observation, "_ANGLE_NOISE", jnp.float32(0.0))
    simulation = physics.reset_random(jax.random.PRNGKey(17))
    previous_actions = jax.random.uniform(
        jax.random.PRNGKey(18), (config.N_TEAMS, config.N_ROBOTS, 2), minval=-1, maxval=1
    )
    noise_key = jax.random.PRNGKey(0)

    def features(sim, actions):
        return np.asarray(observation.build_agent_features(
            sim, actions, jnp.float32(0.5), noise_key, entity_ordered_actions
        ))

    original = features(simulation, previous_actions)
    index, sign, _ = observation.symmetry_tables(entity_ordered_actions)
    frame = observation.FRAME_FEATURES
    for g in range(observation.SYMMETRY_GROUP_SIZE):
        mirror, rest = g % 2, g // 2
        ally_swap, opponents = rest % 2, _OPPONENT_PERMUTATIONS[rest // 2]
        for team in range(config.N_TEAMS):
            for robot in range(config.N_ROBOTS):
                sim, actions = _transform_world(
                    simulation, previous_actions, team, robot, mirror, ally_swap, opponents
                )
                expected = features(sim, actions)[team, robot]
                actual = _apply(index[robot, g, :frame], sign[robot, g, :frame], original[team, robot])
                np.testing.assert_allclose(actual, expected, atol=2e-5, err_msg=f"g={g} team={team} robot={robot}")


def test_group_tables_are_signed_permutations_of_every_frame():
    index, sign, _ = observation.symmetry_tables(False)
    for robot in range(3):
        for g in range(observation.SYMMETRY_GROUP_SIZE):
            assert sorted(index[robot, g]) == list(range(observation.OBSERVATION_SIZE))
            assert set(np.unique(sign[robot, g])) <= {-1.0, 1.0}
