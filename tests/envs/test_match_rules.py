"""Tests for the shared match rules and reward profiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.config import RewardProfile
from vsss_sim.envs.match_geometry import (
    _goal_area_membership,
    _robot_contacts,
    _robot_goal_area_fraction,
)
from vsss_sim.envs.match_restarts import _restart_positions
from vsss_sim.envs.match_rules import (
    MatchPhase,
    MatchRules,
    MatchStepInput,
    RefereeEvent,
    RestartType,
    build_observation,
    initial_referee_state,
)
from vsss_sim.physics import jax_backend as physics


def _simulation(ball: tuple[float, float, float, float] = (0, 0, 0, 0)):
    simulation = physics.reset_kickoff(jax.random.PRNGKey(0))
    robots = jnp.zeros_like(simulation.robots).at[..., :2].set(jnp.asarray([
        [[-.3, -.4], [-.3, 0], [-.3, .4]], [[.3, -.4], [.3, 0], [.3, .4]]]))
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
    assert bool(membership[config.TEAM_BLUE, 0, 0])


@pytest.mark.parametrize(
    "offset, expected_membership",
    [(-1e-3, True), (0.0, True), (1e-3, False)],
)
def test_goal_area_membership_includes_any_footprint_contact(
    offset: float, expected_membership: bool
):
    robots = _simulation().robots
    inner_x = -config.FIELD_LENGTH / 2.0 + config.GOAL_AREA_LENGTH_X
    robot = jnp.asarray([inner_x + config.ROBOT_SIZE / 2 + offset, 0.25, 0.0, 0.0, 0.0, 0.0])
    robots = robots.at[config.TEAM_BLUE, 0].set(robot)

    membership = jax.jit(_goal_area_membership)(robots)

    assert bool(membership[config.TEAM_BLUE, 0, 0]) is expected_membership


def _contact_scene(
    offending_team: int,
    offender_speed: float,
    victim_speed: float = 0.0,
    ball: tuple[float, float, float, float] = (0.0, 0.5, 0.0, 0.0),
    lane_seconds: float = 0.3,
    push_seconds: float = 0.0,
    start_gap: float = 0.09,
):
    """Robot 0 of ``offending_team`` drives along x into the opponent's robot 0.

    Speeds are wheel drives toward the other robot; the referee state carries
    how long the victim has been in the offender's lane.
    """
    direction = 1.0 if offending_team == config.TEAM_BLUE else -1.0
    before = _simulation(ball=ball)
    victim_x = direction * 0.10
    offender_x = victim_x - direction * (config.ROBOT_SIZE + start_gap)
    facing = 0.0 if direction > 0 else np.pi
    robots = before.robots.at[offending_team, 0, :3].set([offender_x, 0.0, facing])
    robots = robots.at[1 - offending_team, 0, :3].set([victim_x, 0.0, facing + np.pi])
    wheels = jnp.zeros_like(before.wheel_speeds)
    wheels = wheels.at[offending_team, 0].set(offender_speed)
    wheels = wheels.at[1 - offending_team, 0].set(victim_speed)
    before = before._replace(robots=robots, wheel_speeds=wheels)
    after = before._replace(
        robots=robots.at[offending_team, 0, 0].set(victim_x - direction * config.ROBOT_SIZE)
    )
    referee = initial_referee_state()
    referee = referee._replace(
        lane_seconds=referee.lane_seconds.at[offending_team, 0, 0].set(lane_seconds),
        push_seconds=referee.push_seconds.at[offending_team, 0, 0].set(push_seconds),
    )
    return before, after, referee


def _call(before, after, referee, **rules):
    return MatchRules(**rules).step(_context(before, after)._replace(referee_state=referee))


@pytest.mark.parametrize("offending_team", [config.TEAM_BLUE, config.TEAM_YELLOW])
def test_avoidable_hard_impact_is_a_personal_foul_anywhere(offending_team: int):
    result = _call(*_contact_scene(offending_team, offender_speed=1.2))

    assert int(result.event) == RefereeEvent.PERSONAL_FOUL
    assert int(result.offending_team) == offending_team
    assert int(result.restart_team) == 1 - offending_team
    assert int(result.restart) == RestartType.FREE_KICK


@pytest.mark.parametrize("offending_team", [config.TEAM_BLUE, config.TEAM_YELLOW])
@pytest.mark.parametrize("ball_near_victim, expected", [
    (True, RefereeEvent.PERSONAL_FOUL), (False, RefereeEvent.NONE),
])
def test_moderate_impact_is_a_foul_only_on_a_victim_involved_in_play(
    offending_team: int, ball_near_victim: bool, expected: RefereeEvent
):
    direction = 1.0 if offending_team == config.TEAM_BLUE else -1.0
    ball = (direction * 0.20, 0.0, 0.0, 0.0) if ball_near_victim else (0.0, 0.5, 0.0, 0.0)

    result = _call(*_contact_scene(offending_team, offender_speed=0.7, ball=ball))

    assert int(result.event) == expected


@pytest.mark.parametrize(
    "scene",
    [
        # The victim only just entered the lane: the offender could not stop.
        dict(offender_speed=1.2, lane_seconds=0.1),
        # Both drive into each other: no clear fault.
        dict(offender_speed=1.2, victim_speed=1.0),
        # Leaning into a resting contact.
        dict(offender_speed=0.05, start_gap=0.0),
        # Both robots dispute a ball beside them.
        dict(offender_speed=1.2, ball=(0.0625, 0.065, 0.0, 0.0)),
    ],
    ids=["lane-cut", "mutual", "creep", "dispute"],
)
def test_unavoidable_mutual_slow_or_disputed_contact_plays_on(scene: dict):
    result = _call(*_contact_scene(config.TEAM_BLUE, **scene))

    assert int(result.event) == RefereeEvent.NONE


@pytest.mark.parametrize("pushed, expected", [
    (0.95, RefereeEvent.PERSONAL_FOUL), (0.85, RefereeEvent.NONE),
])
def test_sustained_push_becomes_a_foul_after_one_second(pushed: float, expected: RefereeEvent):
    before, after, referee = _contact_scene(
        config.TEAM_BLUE, offender_speed=0.5, start_gap=0.0, push_seconds=pushed
    )
    after = after._replace(robots=before.robots)

    result = _call(before, after, referee)

    assert int(result.event) == expected
    if expected == RefereeEvent.NONE:
        assert float(result.referee_state.push_seconds[0, 0, 0]) == pytest.approx(0.95)


def _drive_until_call(simulation, actions, steps=6):
    rules = MatchRules()
    referee = initial_referee_state()
    for index in range(steps):
        after, info = physics.step(simulation, actions)
        result = rules.step(
            _context(simulation, after, goal=int(info["goal"]))._replace(referee_state=referee)
        )
        if int(result.event) != RefereeEvent.NONE:
            return result
        simulation, referee = result.simulation, result.referee_state
    return result


def test_ramming_a_robot_that_stood_in_the_lane_is_a_foul():
    simulation = _simulation(ball=(0.5, -0.5, 0.0, 0.0))
    robots = simulation.robots.at[config.TEAM_BLUE, 0, :3].set([-0.30, 0.2, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, :3].set([0.0, 0.2, np.pi / 2])
    actions = jnp.zeros((2, 3, 2)).at[config.TEAM_BLUE, 0].set(1.0)

    result = _drive_until_call(simulation._replace(robots=robots), actions)

    assert int(result.event) == RefereeEvent.PERSONAL_FOUL
    assert int(result.offending_team) == config.TEAM_BLUE


def test_cutting_across_a_driving_robot_does_not_draw_a_foul():
    simulation = _simulation(ball=(0.5, -0.5, 0.0, 0.0))
    robots = simulation.robots.at[config.TEAM_BLUE, 0, :3].set([-0.30, 0.2, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, :3].set([-0.05, 0.05, np.pi / 2])
    actions = jnp.zeros((2, 3, 2)).at[:, 0].set(1.0)
    simulation = simulation._replace(robots=robots)

    rules = MatchRules()
    referee = initial_referee_state()
    touched = False
    for _ in range(4):
        after, info = physics.step(simulation, actions)
        touched |= bool(_robot_contacts(after.robots)[0, 0, 1, 0])
        result = rules.step(_context(simulation, after)._replace(referee_state=referee))
        assert int(result.event) != RefereeEvent.PERSONAL_FOUL
        simulation, referee = result.simulation, result.referee_state
    assert touched


@pytest.mark.parametrize(
    "ball, opponent_position, expected_event",
    [
        ((0.66, 0.0, 0.0, 0.0), (-0.2, 0.0), RefereeEvent.ATTACKER_GOAL_AREA),
        ((0.0, 0.0, 0.0, 0.0), (0.66, 0.0), RefereeEvent.NONE),
        ((0.66, 0.0, 0.0, 0.0), (0.66, 0.0), RefereeEvent.ATTACKER_GOAL_AREA),
    ],
)
def test_two_attackers_require_an_attack_but_not_an_opposing_defender(
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


def test_two_defenders_without_an_attack_do_not_create_a_delayed_penalty():
    first = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots = first.robots.at[0, 0, :2].set(jnp.asarray([-.70, 0.0]))
    robots = robots.at[0, 1, :2].set(jnp.asarray([-.62, 0.0]))
    first = first._replace(robots=robots)
    step = jax.jit(MatchRules().step)
    initial = step(_context(first, first))
    assert int(initial.event) == RefereeEvent.NONE
    exiting = initial.simulation._replace(ball=initial.simulation.ball.at[0].set(-.50))
    result = step(_context(initial.simulation, exiting)._replace(referee_state=initial.referee_state))
    assert int(result.event) == RefereeEvent.NONE


def _defender_foul_scene():
    """A second blue defender touches the ball while yellow attacks blue's area."""
    simulation = _simulation(ball=(-0.66, 0.0, 0.0, 0.0))
    robots = simulation.robots.at[config.TEAM_BLUE, 0, 0:2].set([-0.70, 0.0])
    robots = robots.at[config.TEAM_BLUE, 1, 0:2].set([-0.62, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, 0:2].set([-0.60, 0.0])
    return simulation._replace(robots=robots)


def test_defender_foul_plays_advantage_until_the_attack_ends():
    simulation = _defender_foul_scene()
    step = jax.jit(MatchRules().step)

    foul = step(_context(simulation, simulation))
    assert int(foul.event) == RefereeEvent.NONE
    assert int(foul.referee_state.advantage_team) == config.TEAM_YELLOW
    assert int(foul.referee_state.phase) == MatchPhase.PLAYING

    cleared = simulation._replace(ball=simulation.ball.at[0].set(-0.40))
    result = step(_context(simulation, cleared)._replace(referee_state=foul.referee_state))

    assert int(result.event) == RefereeEvent.DEFENDER_GOAL_AREA
    assert int(result.offending_team) == config.TEAM_BLUE
    assert int(result.restart) == RestartType.PENALTY_KICK
    assert int(result.restart_team) == config.TEAM_YELLOW
    assert int(result.referee_state.advantage_team) == -1


def test_goal_during_defender_foul_advantage_cancels_the_penalty():
    simulation = _defender_foul_scene()
    foul = MatchRules().step(_context(simulation, simulation))

    result = MatchRules().step(
        _context(simulation, simulation, goal=-1)._replace(referee_state=foul.referee_state)
    )

    assert int(result.event) == RefereeEvent.GOAL
    assert result.simulation.score.tolist() == [2, 4]
    assert int(result.referee_state.advantage_team) == -1


def test_defender_foul_advantage_expires_after_five_seconds():
    simulation = _defender_foul_scene()
    referee = initial_referee_state()._replace(
        advantage_team=jnp.int32(config.TEAM_YELLOW), advantage_seconds=jnp.float32(4.85)
    )
    almost = MatchRules().step(_context(simulation, simulation)._replace(referee_state=referee))
    assert int(almost.event) == RefereeEvent.NONE

    result = MatchRules().step(
        _context(simulation, simulation)._replace(referee_state=almost.referee_state)
    )

    assert int(result.event) == RefereeEvent.DEFENDER_GOAL_AREA


@pytest.mark.parametrize("pending, expected", [
    (False, RefereeEvent.GOAL_AREA_IMPASSE), (True, RefereeEvent.DEFENDER_GOAL_AREA),
])
def test_pending_defender_foul_takes_precedence_over_a_later_call(
    pending: bool, expected: RefereeEvent
):
    # Keeper and attacker contest a stuck ball; the second defender has left.
    simulation = _defender_foul_scene()
    simulation = simulation._replace(
        robots=simulation.robots.at[config.TEAM_BLUE, 1, 0:2].set([-0.3, 0.3])
    )
    referee = initial_referee_state()._replace(goal_area_impasse_seconds=jnp.float32(4.99))
    if pending:
        referee = referee._replace(
            advantage_team=jnp.int32(config.TEAM_YELLOW), advantage_seconds=jnp.float32(1.0)
        )

    result = MatchRules().step(_context(simulation, simulation)._replace(referee_state=referee))

    assert int(result.event) == expected


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
    "reward_profile, individual_agent_rewards",
    [
        (RewardProfile.DEFAULT, False),
        (RewardProfile.BRANDAO_REWARD, False),
        (RewardProfile.BRANDAO_REWARD, True),
    ],
)
def test_fouled_team_keeps_its_ordinary_reward(
    reward_profile: RewardProfile, individual_agent_rewards: bool
):
    # Yellow leaves the ball stuck in its own area: a penalty for blue.
    simulation = _simulation(ball=(0.66, 0.1, 0.0, 0.0))
    rules = MatchRules(
        reward_profile=reward_profile, individual_agent_rewards=individual_agent_rewards
    )

    def step(stopped_seconds):
        referee = initial_referee_state()._replace(
            goal_area_stopped_seconds=jnp.float32(stopped_seconds)
        )
        context = _context(simulation, simulation)._replace(
            referee_state=referee, dt=jnp.float32(0.02)
        )
        return rules.step(context)

    foul, ordinary = step(4.99), step(0.0)

    assert int(foul.event) == RefereeEvent.GOAL_AREA_STOPPED_BALL
    assert int(foul.offending_team) == config.TEAM_YELLOW
    np.testing.assert_allclose(foul.reward, ordinary.reward)


@pytest.mark.parametrize(
    "ball,robots",
    [
        # Rolling faster than the 0.1 m/s stuck-ball threshold.
        ((-0.66, 0.0, 0.15, 0.0), []),
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
    direction = 1 if restart_team == config.TEAM_BLUE else -1
    expected_mark = np.broadcast_to([direction * config.PENALTY_MARK_X, 0.0], ball.shape)
    np.testing.assert_allclose(ball, expected_mark, atol=1e-6)

    ball_distance = np.linalg.norm(positions - ball[:, None, :], axis=-1)
    assert ball_distance.min() >= 0.10 - 1e-5
    pair_distance = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    pair_distance += np.eye(6)[None] * 10.0
    assert pair_distance.min() >= config.ROBOT_SIZE * 2.0**0.5
    # Players start clear of the ball and walls, so a frozen step leaves it in place.
    np.testing.assert_allclose(np.asarray(stepped.ball[:, 0:2]), ball, atol=1e-6)


@pytest.mark.parametrize(
    "attacker_speed, ball, expected_event",
    [
        # Ordinary contact with the goalkeeper plays on.
        (0.5, (0.0, 0.5, 0.0, 0.0), RefereeEvent.NONE),
        # So does a hard contact while both dispute the ball.
        (1.2, (-0.6225, 0.065, 0.0, 0.0), RefereeEvent.NONE),
        # An avoidable hard hit on the goalkeeper inside its area is a block.
        (1.2, (0.0, 0.5, 0.0, 0.0), RefereeEvent.GOALKEEPER_BLOCK),
    ],
    ids=["ordinary", "dispute", "hard-hit"],
)
def test_goalkeeper_block_is_only_a_personal_foul_on_the_goalkeeper(
    attacker_speed: float, ball: tuple, expected_event: RefereeEvent
):
    before = _simulation(ball=ball)
    robots = before.robots.at[config.TEAM_BLUE, 0, :3].set([-0.66, 0.0, 0.0])
    robots = robots.at[config.TEAM_YELLOW, 0, :3].set([-0.495, 0.0, np.pi])
    wheels = before.wheel_speeds.at[config.TEAM_YELLOW, 0].set(attacker_speed)
    before = before._replace(robots=robots, wheel_speeds=wheels)
    after = before._replace(robots=robots.at[config.TEAM_YELLOW, 0, 0].set(-0.585))
    referee = initial_referee_state()
    referee = referee._replace(lane_seconds=referee.lane_seconds.at[1, 0, 0].set(0.3))

    result = _call(before, after, referee)

    assert int(result.event) == expected_event
    if expected_event == RefereeEvent.GOALKEEPER_BLOCK:
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
    np.testing.assert_allclose(result.observation, build_observation(result.simulation), atol=1e-7)


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


def test_off_referee_mode_only_stops_play_for_goals():
    rules = dict(referee_mode="off")
    assert int(_call(*_contact_scene(config.TEAM_BLUE, 1.2), **rules).event) == RefereeEvent.NONE
    foul = _defender_foul_scene()
    assert int(_call(foul, foul, initial_referee_state(), **rules).event) == RefereeEvent.NONE
    stopped = _simulation()
    referee = initial_referee_state()._replace(field_impasse_seconds=jnp.float32(4.99))
    result = _call(stopped, stopped, referee, **rules)
    assert int(result.event) == RefereeEvent.NONE
    assert int(result.referee_state.phase) == MatchPhase.PLAYING

    goal = MatchRules(referee_mode="off").step(_context(stopped, stopped, goal=1))

    assert int(goal.event) == RefereeEvent.GOAL
    assert int(goal.restart) == RestartType.KICKOFF
    assert int(goal.restart_team) == config.TEAM_YELLOW


def test_simplified_referee_mode_keeps_area_fouls_and_free_balls_only():
    rules = dict(referee_mode="simplified")
    assert int(_call(*_contact_scene(config.TEAM_BLUE, 1.2), **rules).event) == RefereeEvent.NONE
    foul = _defender_foul_scene()
    assert int(_call(foul, foul, initial_referee_state(), **rules).referee_state.advantage_team) == (
        config.TEAM_YELLOW
    )
    holding = _simulation(ball=(-0.66, 0.0, 0.2, 0.0))
    holding = holding._replace(robots=holding.robots.at[0, 0, :2].set(jnp.asarray([-0.70, 0.0])))
    referee = initial_referee_state()._replace(goalkeeper_hold_seconds=jnp.asarray([4.99, 0.0]))
    assert int(_call(holding, holding, referee).event) == RefereeEvent.GOALKEEPER_TIMEOUT
    assert int(_call(holding, holding, referee, **rules).event) == RefereeEvent.NONE

    # A stuck ball inside a goal area is freed rather than left to stall.
    stuck = _simulation(ball=(-0.66, 0.1, 0.0, 0.0))
    referee = initial_referee_state()._replace(
        field_impasse_seconds=jnp.float32(4.99), field_impasse_quadrant=jnp.int32(2)
    )
    result = _call(stuck, stuck, referee, **rules)

    assert int(result.event) == RefereeEvent.FIELD_IMPASSE
    assert int(result.restart) == RestartType.FREE_BALL
    np.testing.assert_allclose(result.simulation.ball[:2], [-0.375, 0.40])


@pytest.mark.parametrize("referee_mode", ["simplified", "full"])
@pytest.mark.parametrize("speed, stuck", [(0.08, True), (0.12, False)])
def test_ball_slower_than_vssreferee_threshold_counts_as_stuck(
    referee_mode: str, speed: float, stuck: bool
):
    simulation = _simulation(ball=(0.0, 0.2, speed, 0.0))

    result = _call(simulation, simulation, initial_referee_state(), referee_mode=referee_mode)

    expected = pytest.approx(config.DT) if stuck else 0.0
    assert float(result.referee_state.field_impasse_seconds) == expected
