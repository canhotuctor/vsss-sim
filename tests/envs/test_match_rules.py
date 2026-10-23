"""Tests for the shared match rules and reward profiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.config import RewardProfile
from vsss_sim.envs.match_geometry import (
    _goal_area_membership,
    _robot_goal_area_fraction,
)
from vsss_sim.envs.match_rules import (
    MatchRules,
    MatchStepInput,
    RefereeEvent,
    RestartType,
    build_observation,
    initial_referee_state,
)
from vsss_sim.envs.match_restarts import _restart_positions
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


@pytest.mark.parametrize("heading", [0.0, np.pi / 4.0])
def test_goal_area_corner_clips_robot_footprint_on_both_edges(heading: float):
    robots = _simulation().robots
    inner_x = -config.FIELD_LENGTH / 2.0 + config.GOAL_AREA_LENGTH_X
    upper_y = config.GOAL_AREA_LENGTH_Y / 2.0
    robot = jnp.asarray([inner_x, upper_y, heading, 0.0, 0.0, 0.0])
    robots = robots.at[config.TEAM_BLUE, 0].set(robot)

    fraction = jax.jit(_robot_goal_area_fraction)(robot, jnp.int32(0))
    membership = jax.jit(_goal_area_membership)(robots)

    np.testing.assert_allclose(fraction, 0.25, atol=1e-5)
    assert not bool(membership[config.TEAM_BLUE, 0, 0])


@pytest.mark.parametrize(
    "offset, expected_membership",
    [(-1e-3, True), (0.0, True), (1e-3, False)],
)
def test_goal_area_membership_uses_half_footprint_threshold(
    offset: float, expected_membership: bool
):
    robots = _simulation().robots
    inner_x = -config.FIELD_LENGTH / 2.0 + config.GOAL_AREA_LENGTH_X
    robot = jnp.asarray([inner_x + offset, 0.0, 0.0, 0.0, 0.0, 0.0])
    robots = robots.at[config.TEAM_BLUE, 0].set(robot)

    fraction = _robot_goal_area_fraction(robot, jnp.int32(0))
    membership = jax.jit(_goal_area_membership)(robots)

    np.testing.assert_allclose(fraction, 0.5 - offset / config.ROBOT_SIZE, atol=1e-5)
    assert bool(membership[config.TEAM_BLUE, 0, 0]) is expected_membership


@pytest.mark.parametrize("offending_team", [config.TEAM_BLUE, config.TEAM_YELLOW])
def test_personal_foul_identifies_the_team_that_advances_into_the_opponent(
    offending_team: int,
):
    before = _simulation()
    robots_before = before.robots
    goal_area_x = -0.66 if offending_team == config.TEAM_BLUE else 0.66
    direction = 1.0 if offending_team == config.TEAM_BLUE else -1.0
    victim_x = goal_area_x + direction * 0.08
    robots_before = robots_before.at[offending_team, 0, 0:2].set(
        [goal_area_x - direction * 0.09, 0.0]
    )
    robots_before = robots_before.at[1 - offending_team, 0, 0:2].set([victim_x, 0.0])
    robots_after = robots_before.at[offending_team, 0, 0:2].set([goal_area_x, 0.0])
    after = before._replace(robots=robots_after)

    result = MatchRules().step(_context(before._replace(robots=robots_before), after))

    assert int(result.event) == RefereeEvent.PERSONAL_FOUL
    assert int(result.offending_team) == offending_team
    assert int(result.restart_team) == 1 - offending_team


def test_foul_near_midfield_is_not_attributed_to_remote_pair_during_goal_area_play():
    before = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots_before = before.robots.at[config.TEAM_BLUE, 0, 0:2].set([0.15, 0.0])
    robots_before = robots_before.at[config.TEAM_YELLOW, 0, 0:2].set([0.12, 0.0])
    robots_after = robots_before.at[config.TEAM_BLUE, 0, 0:2].set([0.05, 0.0])
    after = before._replace(robots=robots_after)

    result = MatchRules().step(_context(before._replace(robots=robots_before), after))

    assert int(result.event) == RefereeEvent.NONE
    assert int(result.restart) == RestartType.NONE


@pytest.mark.parametrize(
    "ball, opponent_position, expected_event",
    [
        ((0.66, 0.0, 0.0, 0.0), (-0.2, 0.0), RefereeEvent.NONE),
        ((0.0, 0.0, 0.0, 0.0), (0.66, 0.0), RefereeEvent.NONE),
        ((0.66, 0.0, 0.0, 0.0), (0.66, 0.0), RefereeEvent.ATTACKER_GOAL_AREA),
    ],
)
def test_two_attackers_in_opponent_area_require_contested_ball_in_that_area(
    ball, opponent_position, expected_event
):
    simulation = _simulation(ball=ball)
    robots = simulation.robots
    robots = robots.at[config.TEAM_BLUE, 0, 0:2].set([0.66, 0.0])
    robots = robots.at[config.TEAM_BLUE, 1, 0:2].set([0.69, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, 0:2].set(opponent_position)
    simulation = simulation._replace(robots=robots)

    result = MatchRules().step(_context(simulation, simulation))

    assert int(result.event) == expected_event
    if expected_event == RefereeEvent.ATTACKER_GOAL_AREA:
        assert int(result.offending_team) == config.TEAM_BLUE


def test_foul_is_detected_near_goal_area_ball_when_robots_are_outside_area():
    before = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots_before = before.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.48, 0.0])
    robots_before = robots_before.at[config.TEAM_YELLOW, 0, 0:2].set([-0.58, 0.0])
    robots_after = robots_before.at[config.TEAM_BLUE, 0, 0:2].set([-0.56, 0.0])
    after = before._replace(robots=robots_after)
    before = before._replace(robots=robots_before)

    result = MatchRules().step(_context(before, after))

    assert int(result.event) == RefereeEvent.PERSONAL_FOUL
    assert int(result.offending_team) == config.TEAM_BLUE


def test_contact_with_ball_between_robots_is_not_a_personal_foul():
    before = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots_before = before.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.80, 0.0])
    robots_before = robots_before.at[config.TEAM_YELLOW, 0, 0:2].set([-0.62, 0.0])
    robots_after = robots_before.at[config.TEAM_BLUE, 0, 0:2].set([-0.70, 0.0])
    before = before._replace(robots=robots_before)
    after = before._replace(robots=robots_after)

    result = MatchRules().step(_context(before, after))

    assert int(result.event) == RefereeEvent.NONE
    assert int(result.restart) == RestartType.NONE


@pytest.mark.parametrize(
    "timer_name,ball,robots,expected_timer",
    [
        (
            "field_impasse_seconds",
            (-0.66, 0.0, 0.0, 0.0),
            [(config.TEAM_YELLOW, 0, -0.63, 0.0), (config.TEAM_BLUE, 0, -0.69, 0.0)],
            "goal_area_impasse_seconds",
        ),
        (
            "goal_area_impasse_seconds",
            (0.0, 0.0, 0.0, 0.0),
            [],
            "field_impasse_seconds",
        ),
    ],
)
def test_switching_impasse_context_does_not_reuse_other_timer(
    timer_name: str,
    ball: tuple[float, float, float, float],
    robots: list[tuple[int, int, float, float]],
    expected_timer: str,
):
    before = _simulation(ball=ball)
    after = before
    for team, robot, x, y in robots:
        after = after._replace(robots=after.robots.at[team, robot, 0:2].set([x, y]))
    referee_state = initial_referee_state()._replace(**{timer_name: jnp.float32(9.99)})
    context = _context(before, after)
    context = context._replace(
        referee_state=referee_state,
        dt=jnp.float32(0.02),
    )

    result = MatchRules().step(context)

    assert int(result.event) == RefereeEvent.NONE
    assert float(getattr(result.referee_state, expected_timer)) == pytest.approx(0.02)
    other_timer = (
        "field_impasse_seconds"
        if expected_timer == "goal_area_impasse_seconds"
        else "goal_area_impasse_seconds"
    )
    assert float(getattr(result.referee_state, other_timer)) == pytest.approx(0.0)


def test_defender_goal_area_violation_waits_until_ball_exits_without_contest():
    first = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots = first.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.70, 0.0])
    robots = robots.at[config.TEAM_BLUE, 1, 0:2].set([-0.62, 0.0])
    first = first._replace(robots=robots)

    step = jax.jit(MatchRules().step)
    first_context = _context(first, first)._replace(referee_state=initial_referee_state())
    pending = step(first_context)

    assert int(pending.event) == RefereeEvent.NONE
    assert bool(pending.referee_state.defender_area_violation_pending[config.TEAM_BLUE])

    exiting = pending.simulation._replace(ball=pending.simulation.ball.at[0].set(-0.55))
    call_context = _context(pending.simulation, exiting)._replace(
        referee_state=pending.referee_state
    )
    result = step(call_context)

    assert int(result.event) == RefereeEvent.DEFENDER_GOAL_AREA
    assert int(result.offending_team) == config.TEAM_BLUE
    assert int(result.restart) == RestartType.PENALTY_KICK


def test_defender_goal_area_violation_is_immediate_during_active_ball_contest():
    simulation = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots = simulation.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.70, 0.0])
    robots = robots.at[config.TEAM_BLUE, 1, 0:2].set([-0.62, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, 0:2].set([-0.56, 0.0])
    simulation = simulation._replace(robots=robots)

    result = MatchRules().step(_context(simulation, simulation))

    assert int(result.event) == RefereeEvent.DEFENDER_GOAL_AREA
    assert int(result.offending_team) == config.TEAM_BLUE
    assert not bool(result.referee_state.defender_area_violation_pending[config.TEAM_BLUE])


@pytest.mark.parametrize("reward_profile", list(RewardProfile))
def test_uncontested_stopped_ball_in_goal_area_awards_penalty_against_area_team(
    reward_profile: RewardProfile,
):
    simulation = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    context = _context(simulation, simulation)._replace(
        referee_state=initial_referee_state()._replace(
            goal_area_stopped_seconds=jnp.float32(4.99)
        ),
        dt=jnp.float32(0.02),
    )

    result = MatchRules(reward_profile=reward_profile).step(context)

    assert int(result.event) == RefereeEvent.GOAL_AREA_STOPPED_BALL
    assert int(result.offending_team) == config.TEAM_BLUE
    assert int(result.restart) == RestartType.PENALTY_KICK
    assert int(result.restart_team) == config.TEAM_YELLOW
    assert float(result.reward) == pytest.approx(-1.0)
    assert float(result.referee_state.goal_area_stopped_seconds) == 0.0


@pytest.mark.parametrize(
    "ball,robots",
    [
        # Rolling ball: not at rest.
        ((-0.66, 0.0, 0.05, 0.0), []),
        # Contested ball: the goal-area impasse rule applies instead.
        ((-0.66, 0.0, 0.0, 0.0), [(config.TEAM_YELLOW, 0, -0.63, 0.0), (config.TEAM_BLUE, 0, -0.69, 0.0)]),
        # Ball at rest outside the goal areas: the free-ball rule applies instead.
        ((0.0, 0.0, 0.0, 0.0), []),
    ],
)
def test_goal_area_stopped_ball_timer_only_runs_for_uncontested_ball_at_rest(
    ball: tuple[float, float, float, float],
    robots: list[tuple[int, int, float, float]],
):
    simulation = _simulation(ball=ball)
    for team, robot, x, y in robots:
        simulation = simulation._replace(
            robots=simulation.robots.at[team, robot, 0:2].set([x, y])
        )
    context = _context(simulation, simulation)._replace(
        referee_state=initial_referee_state()._replace(
            goal_area_stopped_seconds=jnp.float32(4.99)
        ),
        dt=jnp.float32(0.02),
    )

    result = MatchRules().step(context)

    assert int(result.event) != RefereeEvent.GOAL_AREA_STOPPED_BALL
    assert float(result.referee_state.goal_area_stopped_seconds) == 0.0


@pytest.mark.parametrize("restart_team", [config.TEAM_BLUE, config.TEAM_YELLOW])
def test_free_kick_formation_keeps_players_off_the_ball_and_each_other(restart_team: int):
    half_length = config.FIELD_LENGTH / 2.0 - config.ROBOT_RADIUS
    half_width = config.FIELD_WIDTH / 2.0 - config.ROBOT_RADIUS
    xs, ys = np.meshgrid(
        np.linspace(-half_length, half_length, 21), np.linspace(-half_width, half_width, 19)
    )
    balls = np.stack((xs.ravel(), ys.ravel()), axis=-1)
    # Keep ball spots the ball itself can reach past the corner chamfers.
    corner = config.FIELD_LENGTH / 2.0 + config.FIELD_WIDTH / 2.0 - config.FIELD_CHAMFER
    balls = balls[np.abs(balls).sum(axis=-1) <= corner - 2.0 * config.BALL_RADIUS]

    def place_and_step(ball_xy):
        simulation = physics.empty_state()._replace(
            ball=jnp.concatenate((ball_xy, jnp.zeros(2))).astype(jnp.float32)
        )
        restarted = _restart_positions(
            simulation,
            jnp.int32(RestartType.FREE_KICK),
            jnp.int32(restart_team),
            jnp.int32(-1),
        )
        stepped, _ = physics.step(restarted, jnp.zeros((2, 3, 2)), dt=0.1, sub_steps=6)
        return restarted, stepped

    restarted, stepped = jax.jit(jax.vmap(place_and_step))(jnp.asarray(balls, jnp.float32))
    positions = np.asarray(restarted.robots[..., 0:2]).reshape(len(balls), 6, 2)
    ball = np.asarray(restarted.ball[:, 0:2])
    np.testing.assert_allclose(ball, balls, atol=1e-6)

    ball_distance = np.linalg.norm(positions - ball[:, None, :], axis=-1)
    assert ball_distance.min() >= 0.10 - 1e-5
    pair_distance = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    pair_distance += np.eye(6)[None] * 10.0
    assert pair_distance.min() >= config.ROBOT_SIZE * 2.0**0.5
    # Players start clear of the ball and walls, so a frozen step leaves it in place.
    np.testing.assert_allclose(np.asarray(stepped.ball[:, 0:2]), ball, atol=1e-6)


@pytest.mark.parametrize(
    "attacker_moves, expected_event",
    [(False, RefereeEvent.NONE), (True, RefereeEvent.GOALKEEPER_BLOCK)],
)
def test_goalkeeper_block_requires_observed_closing_motion(
    attacker_moves: bool, expected_event: RefereeEvent
):
    before = _simulation()
    robots_before = before.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.66, 0.0])
    attacker_start = -0.44 if attacker_moves else -0.62
    robots_before = robots_before.at[config.TEAM_YELLOW, 0, 0:2].set([attacker_start, 0.0])
    robots_after = robots_before.at[config.TEAM_YELLOW, 0, 0:2].set([-0.62, 0.0])
    after = before._replace(robots=robots_after)
    context = _context(before._replace(robots=robots_before), after)

    result = MatchRules().step(context)

    assert int(result.event) == expected_event
    if attacker_moves:
        assert int(result.offending_team) == config.TEAM_YELLOW
        assert int(result.restart) == RestartType.GOAL_KICK


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
