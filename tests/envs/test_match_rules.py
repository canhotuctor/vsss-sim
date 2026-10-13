"""Tests for the shared match rules and reward profiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.config import RewardProfile
from vsss_sim.envs.match_rules import (
    MatchRules,
    MatchStepInput,
    build_observation,
)
from vsss_sim.physics import jax_backend as physics


def _simulation(ball: tuple[float, float, float, float] = (0, 0, 0, 0)):
    simulation = physics.reset_kickoff(jax.random.PRNGKey(0))
    robots = jnp.zeros_like(simulation.robots)
    return simulation._replace(
        ball=jnp.asarray(ball, dtype=jnp.float32),
        robots=robots,
        score=jnp.asarray([2, 3], dtype=jnp.int32),
        t=jnp.float32(4.5),
    )


def _context(
    before,
    after,
    goal: int = 0,
    step_count: int = 1,
    seed: int = 1,
) -> MatchStepInput:
    return MatchStepInput(
        simulation_before=before,
        simulation_after_physics=after,
        goal=jnp.asarray(goal, dtype=jnp.int32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        kickoff_key=jax.random.PRNGKey(seed),
    )


def _angle_reward(point_a: np.ndarray, vertex: np.ndarray, point_c: np.ndarray) -> float:
    vector_a = point_a - vertex
    vector_c = point_c - vertex
    denominator = max(float(np.linalg.norm(vector_a) * np.linalg.norm(vector_c)), 1e-8)
    cosine = np.clip(float(np.dot(vector_a, vector_c)) / denominator, -1.0, 1.0)
    return float(np.arccos(cosine) / np.pi - 1.0)


def test_observation_builder_handles_scalar_and_batched_simulation():
    simulation = _simulation()
    batch = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), simulation)

    scalar_observation = jax.jit(build_observation)(simulation)
    batch_observation = jax.jit(build_observation)(batch)

    assert scalar_observation.shape == (46,)
    assert batch_observation.shape == (2, 46)
    np.testing.assert_array_equal(batch_observation[0], scalar_observation)
    np.testing.assert_array_equal(batch_observation[1], scalar_observation)


def test_default_profile_preserves_goal_progress_and_crowding_reward():
    before = _simulation()
    after = before._replace(ball=before.ball.at[0].set(0.1))
    robots = after.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.70, 0.0])
    robots = robots.at[config.TEAM_BLUE, 1, 0:2].set([-0.68, 0.0])
    after = after._replace(robots=robots)

    result = MatchRules().step(_context(before, after))

    expected = 0.1 * config.BALL_FORWARD_REWARD_COEF + config.GOAL_AREA_CROWDING_PENALTY
    np.testing.assert_allclose(result.reward, expected, atol=1e-7)
    assert result.simulation.score.tolist() == [2, 3]


def test_brandao_continuous_reward_matches_paper_terms_and_team_average():
    before = _simulation()
    after = before._replace(ball=before.ball.at[0].set(0.14))
    blue_positions = jnp.asarray(
        [[-0.4, -0.2], [0.2, -0.3], [0.1, 0.4]], dtype=jnp.float32
    )
    robots = after.robots.at[config.TEAM_BLUE, :, 0:2].set(blue_positions)
    after = after._replace(robots=robots)
    horizon = 20

    rules = MatchRules(
        reward_profile=RewardProfile.BRANDAO_REWARD,
        max_episode_steps=horizon,
    )
    reward = jax.jit(rules.step)(_context(before, after)).reward

    before_ball = np.asarray([0.0, 0.0])
    after_ball = np.asarray([0.14, 0.0])
    blue = np.asarray(blue_positions)
    opponent_goal = np.asarray([config.FIELD_LENGTH / 2.0, 0.0])
    own_goal = -opponent_goal
    speed = np.clip(
        (
            np.linalg.norm(before_ball - opponent_goal)
            - np.linalg.norm(after_ball - opponent_goal)
            - 0.05
        )
        / 0.14,
        -1.0,
        1.0,
    )
    proximity = -min(float(np.linalg.norm(blue - after_ball, axis=-1).min()), 1.0)
    offensive = np.mean(
        [_angle_reward(robot, after_ball, opponent_goal) for robot in blue]
    )
    defensive = np.mean(
        [_angle_reward(after_ball, robot, own_goal) for robot in blue]
    )
    expected = 0.7 * speed + 0.1 * proximity + 0.1 * offensive + 0.1 * defensive

    np.testing.assert_allclose(reward, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("goal, expected_reward", [(1, 14.0), (-1, -14.0)])
def test_brandao_goal_reward_uses_remaining_horizon_and_keeps_match_score(
    goal: int, expected_reward: float
):
    before = _simulation()
    after = before._replace(ball=before.ball.at[0].set(0.7))
    rules = MatchRules(reward_profile="brandao_reward", max_episode_steps=5)

    result = jax.jit(rules.step)(_context(before, after, goal=goal, step_count=3))

    np.testing.assert_allclose(result.reward, expected_reward, atol=1e-6)
    assert result.simulation.score.tolist() == [2 + (goal == 1), 3 + (goal == -1)]
    assert float(result.simulation.t) == float(after.t)
    assert abs(float(result.simulation.ball[0])) < 0.1
    np.testing.assert_allclose(result.observation, build_observation(result.simulation))


def test_goal_without_kickoff_returns_scored_physics_state():
    before = _simulation()
    after = before._replace(ball=before.ball.at[0].set(0.7))
    rules = MatchRules(kickoff_on_goal=False)

    result = rules.step(_context(before, after, goal=1))

    assert result.simulation.score.tolist() == [3, 3]
    np.testing.assert_array_equal(result.simulation.ball, after.ball)
    np.testing.assert_allclose(result.observation, build_observation(after))


def test_match_step_rules_jit_and_vmap():
    before = _simulation()
    after = before._replace(ball=before.ball.at[0].set(0.1))
    batch_before = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), before)
    batch_after = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), after)
    context = MatchStepInput(
        simulation_before=batch_before,
        simulation_after_physics=batch_after,
        goal=jnp.asarray([0, 1], dtype=jnp.int32),
        step_count=jnp.asarray([1, 2], dtype=jnp.int32),
        kickoff_key=jax.random.split(jax.random.PRNGKey(9), 2),
    )
    rules = MatchRules(reward_profile="brandao_reward", max_episode_steps=10)

    result = jax.jit(jax.vmap(rules.step))(context)
    jax.block_until_ready(result)

    assert result.observation.shape == (2, 46)
    assert result.reward.shape == (2,)
    np.testing.assert_array_equal(result.simulation.score[:, config.TEAM_BLUE], [2, 3])


@pytest.mark.parametrize("profile", ["unknown", "brandao-2022"])
def test_unknown_reward_profile_is_rejected(profile: str):
    with pytest.raises(ValueError, match="RewardProfile"):
        MatchRules(reward_profile=profile)
