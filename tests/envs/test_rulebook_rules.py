"""Regression scenarios from RoboCore Mini/VSSS v3.0, revised 27 May 2025."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import config
from vsss_sim.envs.match_geometry import (
    _ball_goal_area_membership,
    _goal_area_membership,
    _robot_ball_contacts,
    _robot_contacts,
    _robot_vertices,
)
from vsss_sim.envs.match_referee import MatchPhase, RefereeEvent, RestartType, initial_referee_state
from vsss_sim.envs.match_restarts import _restart_positions
from vsss_sim.envs.match_rules import MatchRules, MatchStepInput
from vsss_sim.physics import jax_backend as physics


def state(ball=(0.0, 0.0, 0.0, 0.0)):
    s = physics.empty_state()
    xy = jnp.asarray(
        [[[-0.70, 0], [-0.25, -0.4], [-0.25, 0.4]], [[0.70, 0], [0.25, -0.4], [0.25, 0.4]]]
    )
    return s._replace(ball=jnp.asarray(ball), robots=s.robots.at[..., :2].set(xy))


def step(before, after=None, referee=None, goal=0, dt=0.1, rules=None):
    context = MatchStepInput(
        before,
        before if after is None else after,
        jnp.int32(goal),
        jnp.int32(1),
        jax.random.PRNGKey(5),
        initial_referee_state() if referee is None else referee,
        jnp.float32(dt),
    )
    return (MatchRules() if rules is None else rules).step(context)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("offset,expected", [(-0.01, False), (0.01, False), (0.03, True)])
def test_entire_ball_must_cross_goal_line(sign, offset, expected):
    s = state((sign * (0.75 + offset), 0.0, sign * 0.2, 0.0))
    after, goal = jax.jit(physics._ball_wall_collisions)(s)
    assert int(goal) == (sign if expected else 0)
    if not expected:
        assert float(after.ball[2]) * sign > 0  # mouth stays open while straddling
        assert float(after.ball[0]) == pytest.approx(float(s.ball[0]))


def test_ball_can_roll_through_mouth_before_becoming_a_goal():
    s = physics.empty_state()._replace(ball=jnp.asarray([0.70, 0.0, 1.0, 0.0]))
    actions = jnp.zeros((2, 3, 2))
    for _ in range(10):
        s, info = physics.step(s, actions)
        if int(info["goal"]):
            assert float(s.ball[0]) - config.BALL_RADIUS > 0.75
            break
    else:
        pytest.fail("ball bounced at the mouth before crossing the goal line")


@pytest.mark.parametrize("sign", [-1, 1])
def test_ball_overlapping_a_post_does_not_score(sign):
    s = state((sign * 0.78, 0.195, sign * 0.2, 0.0))
    after, goal = physics._ball_wall_collisions(s)
    assert int(goal) == 0
    assert float(after.ball[2]) * sign < 0


@pytest.mark.parametrize("sign,side", [(1, 0), (-1, 1)])
def test_area_requires_whole_ball_but_only_any_part_of_robot(sign, side):
    # At y=.25 the straight edge is .60; the central arc is elsewhere.
    ball = jnp.asarray([-sign * 0.59, 0.25, 0.0, 0.0])
    assert not bool(_ball_goal_area_membership(ball)[side])
    ball = ball.at[0].set(-sign * (0.60 + config.BALL_RADIUS))
    assert bool(_ball_goal_area_membership(ball)[side])
    s = state()
    robots = s.robots.at[0, 0, :3].set(jnp.asarray([-sign * 0.58, 0.25, 0.0]))
    assert bool(_goal_area_membership(robots)[0, 0, side])


@pytest.mark.parametrize("sign,side", [(1, 0), (-1, 1)])
def test_curved_goal_area_includes_ball_and_rotated_robot(sign, side):
    ball = jnp.asarray([-sign * 0.575, 0.0, 0.0, 0.0])
    assert bool(jax.jit(_ball_goal_area_membership)(ball)[side])
    robots = state().robots.at[0, 0, :3].set(jnp.asarray([-sign * 0.52, 0.0, jnp.pi / 4]))
    assert bool(jax.jit(_goal_area_membership)(robots)[0, 0, side])
    assert not bool(_ball_goal_area_membership(ball.at[1].set(0.15))[side])


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize(
    "touches,expected", [(False, RefereeEvent.NONE), (True, RefereeEvent.ATTACKER_GOAL_AREA)]
)
def test_second_attacker_needs_contact_not_proximity_and_no_defender_is_required(
    team, touches, expected
):
    direction = 1 if team == 0 else -1
    s = state((direction * 0.66, 0.0, 0.0, 0.0))
    robots = s.robots.at[team, 0, :2].set(jnp.asarray([direction * 0.6025, 0.0]))
    robots = robots.at[team, 1, :2].set(jnp.asarray([direction * 0.69, 0.05 if touches else 0.105]))
    robots = robots.at[1 - team, 0, :2].set(jnp.asarray([-direction * 0.3, 0.0]))
    s = s._replace(robots=robots)
    previous = initial_referee_state()
    previous = previous._replace(attack_primary=previous.attack_primary.at[1 - team].set(0))
    result = step(s, referee=previous)
    assert int(result.event) == expected
    if touches:
        assert int(result.offending_team) == team
        assert int(result.restart_team) == 1 - team
        assert int(result.restart) == RestartType.GOAL_KICK


def test_push_into_area_is_exempt_until_defender_leaves():
    before = state((-0.52, 0.25, 0.0, 0.0))
    robots = before.robots.at[0, 1, :2].set(jnp.asarray([-0.54, 0.25]))
    robots = robots.at[1, 0, :2].set(jnp.asarray([-0.465, 0.25]))
    before = before._replace(robots=robots)
    after = before._replace(robots=robots.at[0, 1, 0].set(-0.57).at[1, 0, 0].set(-0.495))
    entered = step(before, after)
    assert bool(entered.referee_state.defender_pushed_in[0, 1])
    assert int(entered.event) == RefereeEvent.NONE
    contest = after._replace(
        ball=jnp.asarray([-0.63, 0.25, 0.0, 0.0]),
        robots=after.robots.at[0, 1, 0].set(-0.66).at[1, 0, 0].set(-0.59),
    )
    continued = step(contest, referee=entered.referee_state)
    assert int(continued.event) == RefereeEvent.NONE
    assert bool(continued.referee_state.defender_pushed_in[0, 1])
    # Without a preceding forced entry the same second-player touch is a
    # defender foul; the attackers play the advantage before the penalty.
    assert int(step(contest).referee_state.advantage_team) == 1
    left = contest._replace(robots=contest.robots.at[0, 1, :2].set(jnp.asarray([-0.4, 0.3])))
    assert not bool(
        step(left, referee=continued.referee_state).referee_state.defender_pushed_in[0, 1]
    )


def test_voluntary_area_entry_near_an_opponent_is_not_a_push_exemption():
    before = state((-0.63, 0.25, 0.0, 0.0))
    robots = before.robots.at[0, 1, :2].set(jnp.asarray([-0.54, 0.25]))
    robots = robots.at[1, 0, :2].set(jnp.asarray([-0.59, 0.25]))
    before = before._replace(robots=robots)
    after = before._replace(robots=robots.at[0, 1, 0].set(-0.66))
    result = step(before, after)
    assert int(result.event) == RefereeEvent.NONE
    assert int(result.referee_state.advantage_team) == 1


@pytest.mark.parametrize("kind", ["possession", "impasse"])
def test_goal_area_timers_expire_at_five_seconds(kind):
    s = state((-0.66, 0.0, 0.20 if kind == "possession" else 0.0, 0.0))
    if kind == "impasse":
        s = s._replace(robots=s.robots.at[1, 0, :2].set(jnp.asarray([-0.59, 0.0])))
    referee = initial_referee_state()
    if kind == "possession":
        referee = referee._replace(goalkeeper_hold_seconds=jnp.asarray([4.89, 0.0]))
        event, restart = RefereeEvent.GOALKEEPER_TIMEOUT, RestartType.PENALTY_KICK
    else:
        referee = referee._replace(goal_area_impasse_seconds=jnp.float32(4.89))
        event, restart = RefereeEvent.GOAL_AREA_IMPASSE, RestartType.GOAL_KICK
    almost = step(s, referee=referee)
    assert int(almost.event) == RefereeEvent.NONE
    result = step(s, referee=almost.referee_state)
    assert int(result.event) == event
    assert int(result.restart) == restart


def test_goalkeeper_dispute_resets_possession_timer_and_attacking_goal_has_advantage():
    s = state((-0.66, 0.0, 0.0, 0.0))
    s = s._replace(robots=s.robots.at[1, 0, :2].set(jnp.asarray([-0.59, 0.0])))
    referee = initial_referee_state()._replace(goalkeeper_hold_seconds=jnp.asarray([4.99, 0.0]))
    result = step(s, referee=referee)
    assert int(result.event) == RefereeEvent.NONE
    assert float(result.referee_state.goalkeeper_hold_seconds[0]) == 0
    s = s._replace(robots=s.robots.at[0, 1, :2].set(jnp.asarray([-0.62, 0.0])))
    result = step(s, goal=-1)
    assert int(result.event) == RefereeEvent.GOAL
    assert result.simulation.score.tolist() == [0, 1]


@pytest.mark.parametrize("team", [0, 1])
def test_free_kick_uses_TL_mark_and_defenders_touch_area_line(team):
    s = state((0.1, 0.2, 0.1, 0.1))
    placed = _restart_positions(s, jnp.int32(RestartType.FREE_KICK), jnp.int32(team), jnp.int32(-1))
    direction = 1 if team == 0 else -1
    np.testing.assert_allclose(placed.ball, [direction * config.PENALTY_MARK_X, 0, 0, 0])
    assert np.all(np.asarray(placed.robots[team, 1:, 0]) * direction < 0)
    for robot in np.asarray(placed.robots[1 - team, 1:]):
        vertices = np.asarray(_robot_vertices(robot))
        assert np.max(direction * vertices[:, 0]) == pytest.approx(0.60, abs=1e-6)


def test_repeated_free_balls_follow_quadrant_then_centre_then_reposition():
    s = state((-0.20, -0.20, 0.0, 0.0))
    referee = initial_referee_state()._replace(
        field_impasse_seconds=jnp.float32(4.99), field_impasse_quadrant=jnp.int32(0)
    )
    first = step(s, referee=referee)
    assert int(first.referee_state.free_ball_count) == 1
    np.testing.assert_allclose(first.simulation.ball[:2], [-0.375, -0.40])
    second_ref = first.referee_state._replace(
        phase=jnp.int32(MatchPhase.PLAYING), field_impasse_seconds=jnp.float32(4.99)
    )
    second = step(first.simulation, referee=second_ref)
    assert int(second.referee_state.free_ball_count) == 2
    np.testing.assert_allclose(second.simulation.ball[:2], [0, 0])
    np.testing.assert_array_equal(
        second.simulation.robots[..., :3], first.simulation.robots[..., :3]
    )
    third_ref = second.referee_state._replace(
        phase=jnp.int32(MatchPhase.PLAYING), field_impasse_seconds=jnp.float32(4.99)
    )
    third = step(second.simulation, referee=third_ref)
    assert int(third.referee_state.free_ball_count) == 3
    np.testing.assert_allclose(third.simulation.ball[:2], [0, 0])
    assert np.min(np.linalg.norm(np.asarray(third.simulation.robots[..., :2]), axis=-1)) > 0.25
    moved = third.simulation._replace(ball=third.simulation.ball.at[0].set(0.1))
    assert (
        int(
            step(
                moved, referee=third.referee_state._replace(phase=jnp.int32(0))
            ).referee_state.free_ball_count
        )
        == 0
    )


@pytest.mark.parametrize("team", [0, 1])
def test_kickoff_respects_possession_and_defending_footprint_clearance(team):
    keys = jax.random.split(jax.random.PRNGKey(5), 32)
    states = jax.jit(jax.vmap(lambda key: physics.reset_kickoff(key, jnp.int32(team))))(keys)
    xy = np.asarray(states.robots[:, 1 - team, :, :2])
    assert (
        np.min(np.linalg.norm(xy, axis=-1)) - config.ROBOT_SIZE / np.sqrt(2) > config.CENTER_RADIUS
    )
    assert np.all(
        np.linalg.norm(np.asarray(states.robots[:, team, 1, :2]), axis=-1) < config.CENTER_RADIUS
    )


def test_goal_kickoff_stays_legal_with_random_curriculum_resets():
    s = state()
    result = step(s, goal=1, rules=MatchRules(init_mode="random"))
    np.testing.assert_allclose(result.simulation.ball, 0)
    # Yellow conceded, so yellow receives the only kicker inside the circle.
    assert np.linalg.norm(np.asarray(result.simulation.robots[1, 1, :2])) < 0.20
    assert np.min(np.linalg.norm(np.asarray(result.simulation.robots[0, :, :2]), axis=-1)) > 0.25


def test_contact_helpers_distinguish_nearby_robots_and_ball_from_touching():
    s = state((0.0, 0.0, 0.0, 0.0))
    robots = s.robots.at[0, 0, :3].set(jnp.asarray([0.11, 0, 0.0]))
    robots = robots.at[0, 1, :3].set(jnp.asarray([0.03, 0, 0.0]))
    assert not bool(_robot_ball_contacts(robots, s.ball)[0, 0])
    assert not bool(_robot_contacts(robots)[0, 0, 0, 1])
    assert bool(_robot_ball_contacts(robots, s.ball)[0, 1])


def test_ball_reentry_starts_a_new_attack_with_a_new_primary():
    s = state((0.66, 0.0, 0.0, 0.0))
    s = s._replace(
        robots=s.robots.at[0, 0, :2]
        .set(jnp.asarray([0.6025, 0.0]))
        .at[0, 1, :2]
        .set(jnp.asarray([0.69, 0.105]))
        .at[1, 0, :2]
        .set(jnp.asarray([0.3, -0.3]))
    )
    first = step(s)
    assert int(first.event) == RefereeEvent.NONE
    assert int(first.referee_state.attack_primary[1]) == 0
    outside = s._replace(ball=s.ball.at[0].set(0.3))
    ended = step(s, outside, first.referee_state)
    assert int(ended.referee_state.attack_primary[1]) == -1
    reentry = s._replace(
        robots=s.robots.at[0, 0, :2]
        .set(jnp.asarray([0.69, 0.105]))
        .at[0, 1, :2]
        .set(jnp.asarray([0.6025, 0.0]))
    )
    new_attack = step(outside, reentry, ended.referee_state)
    assert int(new_attack.event) == RefereeEvent.NONE
    assert int(new_attack.referee_state.attack_primary[1]) == 1


def test_free_ball_uses_quadrant_where_stoppage_started():
    s = state((0.0002, 0.30, -0.0002, 0.0))
    first = step(s)
    referee = first.referee_state._replace(field_impasse_seconds=jnp.float32(4.99))
    drifted = s._replace(ball=s.ball.at[0].set(-0.0002))
    result = step(s, drifted, referee)
    assert int(result.quadrant) == 3  # initial positive x, positive y
    np.testing.assert_allclose(result.simulation.ball[:2], [0.375, 0.4])


def test_stopped_frame_preserves_restart_history_and_positions():
    s = state((0.2, 0.2, 0.0, 0.0))
    referee = initial_referee_state()._replace(
        field_impasse_seconds=jnp.float32(4.99), field_impasse_quadrant=jnp.int32(3)
    )
    stopped = step(s, referee=referee)
    assert int(stopped.referee_state.phase) == MatchPhase.STOPPED
    moving = stopped.simulation._replace(ball=stopped.simulation.ball.at[0].add(0.1))
    resumed = step(stopped.simulation, moving, stopped.referee_state)
    assert int(resumed.referee_state.phase) == MatchPhase.PLAYING
    assert int(resumed.referee_state.free_ball_count) == 1
    np.testing.assert_array_equal(resumed.simulation.ball, stopped.simulation.ball)
    np.testing.assert_array_equal(resumed.simulation.robots, stopped.simulation.robots)
