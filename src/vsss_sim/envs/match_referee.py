"""JAX-compatible referee state and event resolution."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import config, physics
from .match_geometry import (
    _PLAY_INVOLVEMENT_DISTANCE,
    _ball_between,
    _ball_goal_area_membership,
    _goal_area_membership,
    _robot_ball_contacts,
    _robot_ball_distance,
    _robot_contacts,
)

_BALL_STUCK_SPEED = 0.01
_FREE_BALL_STOPPED_SPEED = 0.001  # m/s (1 mm/s)
_FREE_BALL_TIMEOUT_SECONDS = 5.0
_RULE_TIMEOUT_SECONDS = 5.0
_BALL_MOVEMENT_TOLERANCE = 0.001


class RefereeEvent(IntEnum):
    """Calls that can be inferred from the simulator's 2D state."""

    NONE = 0
    GOAL = 1
    ATTACKER_GOAL_AREA = 2
    DEFENDER_GOAL_AREA = 3
    GOALKEEPER_TIMEOUT = 4
    GOAL_AREA_IMPASSE = 5
    FIELD_IMPASSE = 6
    PERSONAL_FOUL = 7
    GOALKEEPER_BLOCK = 8
    GOAL_AREA_STOPPED_BALL = 9


class RestartType(IntEnum):
    """Restart awarded after a referee call."""

    NONE = 0
    KICKOFF = 1
    PENALTY_KICK = 2
    GOAL_KICK = 3
    FREE_BALL = 4
    FREE_KICK = 5


class MatchPhase(IntEnum):
    PLAYING = 0
    STOPPED = 1


class RefereeState(NamedTuple):
    """Per-match history for goal-area rules, possession, and impasses."""

    goal_area_seconds: jax.Array
    goalkeeper_hold_seconds: jax.Array
    goal_area_impasse_seconds: jax.Array
    field_impasse_seconds: jax.Array
    goal_area_stopped_seconds: jax.Array
    attack_primary: jax.Array
    defender_pushed_in: jax.Array
    field_impasse_quadrant: jax.Array
    free_ball_count: jax.Array
    free_ball_quadrant: jax.Array
    free_ball_origin: jax.Array
    phase: jax.Array


class _RefereeDecision(NamedTuple):
    event: jax.Array
    offending_team: jax.Array
    restart: jax.Array
    restart_team: jax.Array
    quadrant: jax.Array
    state: RefereeState


def initial_referee_state() -> RefereeState:
    """Create empty state at the beginning of a match or after a goal."""
    return RefereeState(
        goal_area_seconds=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32),
        goalkeeper_hold_seconds=jnp.zeros((config.N_TEAMS,), dtype=jnp.float32),
        goal_area_impasse_seconds=jnp.zeros((), dtype=jnp.float32),
        field_impasse_seconds=jnp.zeros((), dtype=jnp.float32),
        goal_area_stopped_seconds=jnp.zeros((), dtype=jnp.float32),
        attack_primary=jnp.full((config.N_TEAMS,), -1, dtype=jnp.int32),
        defender_pushed_in=jnp.zeros((config.N_TEAMS, config.N_ROBOTS), dtype=jnp.bool_),
        field_impasse_quadrant=jnp.int32(-1),
        free_ball_count=jnp.int32(0),
        free_ball_quadrant=jnp.int32(-1),
        free_ball_origin=jnp.zeros((2,), dtype=jnp.float32),
        phase=jnp.asarray(MatchPhase.PLAYING, dtype=jnp.int32),
    )


def _initiated_contacts(simulation_before, simulation_after, contacts):
    """Observed closing contact initiated by one team, including legal pushes.

    A 2D engine cannot infer intent or toppling. Require actual footprint
    contact and greater advance by the initiator, never proximity alone.
    """
    before, after = simulation_before.robots, simulation_after.robots
    contacts_before = _robot_contacts(before)
    hits = []
    for team in range(config.N_TEAMS):
        delta = before[1 - team, None, :, :2] - before[team, :, None, :2]
        distance = jnp.linalg.norm(delta, axis=-1)
        direction = delta / jnp.maximum(distance[..., None], 1e-8)
        motion = after[team, :, None, :2] - before[team, :, None, :2]
        other_motion = after[1 - team, None, :, :2] - before[1 - team, None, :, :2]
        advance = jnp.sum(motion * direction, axis=-1)
        other_advance = -jnp.sum(other_motion * direction, axis=-1)
        after_distance = jnp.linalg.norm(
            after[1 - team, None, :, :2] - after[team, :, None, :2], axis=-1
        )
        closing = (distance > after_distance + 1e-4) | contacts_before[team, :, 1 - team, :]
        hits.append(
            contacts[team, :, 1 - team, :]
            & closing
            & (advance > 1e-4)
            & (advance > other_advance + 1e-4)
        )
    return jnp.stack(hits)


def _illegal_push(simulation_before, simulation_after, initiated=None):
    """Approximate disruptive personal contact anywhere on the field (9.1)."""
    if initiated is None:
        contacts = _robot_contacts(simulation_after.robots)
        initiated = _initiated_contacts(simulation_before, simulation_after, contacts)
    legal_ball_push = []
    for team in range(config.N_TEAMS):
        legal_ball_push.append(
            _ball_between(
                simulation_after.robots[team, :, None, :],
                simulation_after.robots[1 - team, None, :, :],
                simulation_after.ball,
            )
        )
    return initiated & ~jnp.stack(legal_ball_push)


def _free_ball_quadrant(ball: jax.Array) -> jax.Array:
    return (ball[0] >= 0.0).astype(jnp.int32) + 2 * (ball[1] >= 0.0).astype(jnp.int32)


def _resolve_referee(
    simulation_before: physics.SimState,
    simulation: physics.SimState,
    previous: RefereeState,
    goal: jax.Array,
    dt: jax.Array | None = None,
) -> _RefereeDecision:
    """Resolve fouls, goalkeeper possession, and ball-stoppage restarts."""
    dt = jnp.float32(config.DT) if dt is None else jnp.asarray(dt, dtype=jnp.float32)
    areas_before = _goal_area_membership(simulation_before.robots)
    areas_after = _goal_area_membership(simulation.robots)
    area_seconds = jnp.where(
        areas_after,
        previous.goal_area_seconds + dt,
        jnp.zeros_like(previous.goal_area_seconds),
    )
    ball_areas = _ball_goal_area_membership(simulation.ball)
    ball_in_goal_area = jnp.any(ball_areas)
    contacts = _robot_contacts(simulation.robots)
    ball_contacts = _robot_ball_contacts(simulation.robots, simulation.ball)
    near_ball = (
        _robot_ball_distance(simulation.robots, simulation.ball) <= _PLAY_INVOLVEMENT_DISTANCE
    )
    ball_areas_before = _ball_goal_area_membership(simulation_before.ball)
    near_ball_before = (
        _robot_ball_distance(simulation_before.robots, simulation_before.ball)
        <= _PLAY_INVOLVEMENT_DISTANCE
    )
    initiated = _initiated_contacts(simulation_before, simulation, contacts)

    pushed_in = []
    for team in range(config.N_TEAMS):
        forced_entry = (
            areas_after[team, :, team]
            & ~areas_before[team, :, team]
            & near_ball_before[team]
            & jnp.any(initiated[1 - team].T & near_ball_before[1 - team][None, :], axis=-1)
        )
        pushed_in.append(
            areas_after[team, :, team] & (previous.defender_pushed_in[team] | forced_entry)
        )
    pushed_in = jnp.stack(pushed_in)

    attacking_violations, defending_violations, primary_attackers = [], [], []
    robot_ids = jnp.arange(config.N_ROBOTS)
    for defending_team in range(config.N_TEAMS):
        attacker = 1 - defending_team
        attacking_area = areas_after[attacker, :, defending_team]
        active_attackers = attacking_area & near_ball[attacker]
        attack = ball_areas[defending_team] & jnp.any(active_attackers)
        old_primary = previous.attack_primary[defending_team]
        safe_primary = jnp.maximum(old_primary, 0)
        keep_primary = (
            ball_areas_before[defending_team] & (old_primary >= 0) & attacking_area[safe_primary]
        )
        first_active = jnp.argmax(
            jnp.where(active_attackers, area_seconds[attacker, :, defending_team], -1.0)
        )
        primary = jnp.where(keep_primary, old_primary, jnp.where(attack, first_active, -1)).astype(
            jnp.int32
        )
        primary_attackers.append(jnp.where(ball_areas[defending_team], primary, -1))
        active_players = areas_after[:, :, defending_team] & near_ball
        touches_play = jnp.any(contacts[attacker] & active_players[None, :, :], axis=(1, 2))
        second_attacker = attacking_area & (robot_ids != primary)
        attacking_violations.append(
            attack & jnp.any(second_attacker & (ball_contacts[attacker] | touches_play))
        )

        defending_area = areas_after[defending_team, :, defending_team]
        goalkeeper = jnp.argmax(
            jnp.where(defending_area, area_seconds[defending_team, :, defending_team], -1.0)
        )
        second_defender = defending_area & (robot_ids != goalkeeper) & ~pushed_in[defending_team]
        touches_attacker = jnp.any(
            contacts[defending_team, :, attacker, :] & active_attackers[None, :], axis=-1
        )
        defending_violations.append(
            attack & jnp.any(second_defender & (ball_contacts[defending_team] | touches_attacker))
        )
    attacking_violations = jnp.stack(attacking_violations)
    defending_violations = jnp.stack(defending_violations)
    primary_attackers = jnp.stack(primary_attackers)
    has_attacking_violation = jnp.any(attacking_violations)
    has_defending_violation = jnp.any(defending_violations)
    attacking_team = 1 - jnp.argmax(attacking_violations.astype(jnp.int32))
    defending_team = jnp.argmax(defending_violations.astype(jnp.int32))

    # The goalkeeper is the longest-standing robot in its own goal area.
    goalkeeper_indices = []
    goalkeeper_present = []
    goalkeeper_holds = []
    goalkeeper_blocks = []
    for team in range(config.N_TEAMS):
        own_area = areas_after[team, :, team]
        stay = area_seconds[team, :, team]
        goalkeeper_index = jnp.argmax(jnp.where(own_area, stay, -1.0))
        has_goalkeeper = jnp.any(own_area)
        goalkeeper = simulation.robots[team, goalkeeper_index]
        opponents = simulation.robots[1 - team]
        opponent_near_ball = jnp.any(near_ball[1 - team])
        keeper_touches_ball = ball_contacts[team, goalkeeper_index]
        # Possession must end with the ball outside the area. Dribbling or
        # briefly losing contact inside it does not restart the five seconds.
        holds_ball = (
            ball_areas[team]
            & has_goalkeeper
            & ~opponent_near_ball
            & (keeper_touches_ball | (previous.goalkeeper_hold_seconds[team] > 0))
        )
        attacker_in_area = areas_after[1 - team, :, team]
        blocks = (
            attacker_in_area
            & has_goalkeeper
            & initiated[1 - team, :, goalkeeper_index]
            & ~_ball_between(opponents, goalkeeper, simulation.ball)
        )
        goalkeeper_indices.append(goalkeeper_index)
        goalkeeper_present.append(has_goalkeeper)
        goalkeeper_holds.append(holds_ball)
        goalkeeper_blocks.append(jnp.any(blocks))

    goalkeeper_indices = jnp.stack(goalkeeper_indices)
    goalkeeper_present = jnp.stack(goalkeeper_present)
    goalkeeper_holds = jnp.stack(goalkeeper_holds)
    goalkeeper_blocks = jnp.stack(goalkeeper_blocks)
    hold_seconds = jnp.where(
        goalkeeper_holds,
        previous.goalkeeper_hold_seconds + dt,
        jnp.zeros_like(previous.goalkeeper_hold_seconds),
    )
    goalkeeper_timeouts = hold_seconds >= _RULE_TIMEOUT_SECONDS - 1e-6
    has_goalkeeper_timeout = jnp.any(goalkeeper_timeouts)
    timeout_team = jnp.argmax(goalkeeper_timeouts.astype(jnp.int32))

    ball_side = jnp.argmax(ball_areas.astype(jnp.int32))
    goalkeeper_index = goalkeeper_indices[ball_side]
    attacker_near_impasse_ball = jnp.any(near_ball[1 - ball_side])
    goal_area_contest = (
        ball_in_goal_area
        & goalkeeper_present[ball_side]
        & ball_contacts[ball_side, goalkeeper_index]
        & attacker_near_impasse_ball
    )
    ball_speed = jnp.linalg.norm(simulation.ball[2:4])
    goal_area_ball_is_stuck = ball_speed <= _BALL_STUCK_SPEED
    field_ball_is_stopped = ball_speed < _FREE_BALL_STOPPED_SPEED
    impasse = jnp.where(
        ball_in_goal_area,
        goal_area_ball_is_stuck & goal_area_contest,
        field_ball_is_stopped,
    )
    goal_area_impasse_seconds = jnp.where(
        impasse & ball_in_goal_area,
        previous.goal_area_impasse_seconds + dt,
        jnp.zeros_like(previous.goal_area_impasse_seconds),
    )
    field_impasse_seconds = jnp.where(
        impasse & ~ball_in_goal_area,
        previous.field_impasse_seconds + dt,
        jnp.zeros_like(previous.field_impasse_seconds),
    )
    # A ball left at rest in a goal area without a contest is the defending
    # team's fault: its goalkeeper had the free-ball timeout to clear it.
    goal_area_stopped_seconds = jnp.where(
        ball_in_goal_area & field_ball_is_stopped & ~goal_area_contest,
        previous.goal_area_stopped_seconds + dt,
        jnp.zeros_like(previous.goal_area_stopped_seconds),
    )
    goal_area_impasse = (
        goal_area_impasse_seconds >= _RULE_TIMEOUT_SECONDS - 1e-6
    ) & ball_in_goal_area
    field_impasse = (
        field_impasse_seconds >= _FREE_BALL_TIMEOUT_SECONDS - 1e-6
    ) & ~ball_in_goal_area
    goal_area_stopped_ball = (
        goal_area_stopped_seconds >= _FREE_BALL_TIMEOUT_SECONDS - 1e-6
    ) & ball_in_goal_area

    # Only physical, initiated contact qualifies; proximity is not a foul.
    push_hits = _illegal_push(simulation_before, simulation, initiated)
    push_by_team = jnp.any(push_hits, axis=(1, 2))
    has_personal_foul = jnp.any(push_by_team)
    personal_foul_team = jnp.argmax(push_by_team.astype(jnp.int32))

    goal_event = goal != 0
    has_goalkeeper_block = jnp.any(goalkeeper_blocks)
    goalkeeper_blocking_team = jnp.argmax(goalkeeper_blocks.astype(jnp.int32))
    event = jnp.where(
        goal_event,
        jnp.int32(RefereeEvent.GOAL),
        jnp.where(
            has_defending_violation,
            jnp.int32(RefereeEvent.DEFENDER_GOAL_AREA),
            jnp.where(
                has_attacking_violation,
                jnp.int32(RefereeEvent.ATTACKER_GOAL_AREA),
                jnp.where(
                    has_goalkeeper_block,
                    jnp.int32(RefereeEvent.GOALKEEPER_BLOCK),
                    jnp.where(
                        has_goalkeeper_timeout,
                        jnp.int32(RefereeEvent.GOALKEEPER_TIMEOUT),
                        jnp.where(
                            has_personal_foul,
                            jnp.int32(RefereeEvent.PERSONAL_FOUL),
                            jnp.where(
                                goal_area_impasse,
                                jnp.int32(RefereeEvent.GOAL_AREA_IMPASSE),
                                jnp.where(
                                    goal_area_stopped_ball,
                                    jnp.int32(RefereeEvent.GOAL_AREA_STOPPED_BALL),
                                    jnp.where(
                                        field_impasse,
                                        jnp.int32(RefereeEvent.FIELD_IMPASSE),
                                        jnp.int32(RefereeEvent.NONE),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    offending_team = jnp.where(
        event == RefereeEvent.DEFENDER_GOAL_AREA,
        defending_team,
        jnp.where(
            event == RefereeEvent.ATTACKER_GOAL_AREA,
            attacking_team,
            jnp.where(
                event == RefereeEvent.GOALKEEPER_BLOCK,
                1 - goalkeeper_blocking_team,
                jnp.where(
                    event == RefereeEvent.GOALKEEPER_TIMEOUT,
                    timeout_team,
                    jnp.where(
                        event == RefereeEvent.PERSONAL_FOUL,
                        personal_foul_team,
                        jnp.where(
                            event == RefereeEvent.GOAL_AREA_STOPPED_BALL,
                            ball_side,
                            jnp.int32(-1),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    personal_restart = jnp.int32(RestartType.FREE_KICK)
    conceding_team = jnp.where(goal == 1, config.TEAM_YELLOW, config.TEAM_BLUE).astype(jnp.int32)
    restart_team = jnp.where(
        event == RefereeEvent.GOAL,
        conceding_team,
        jnp.where(
            event == RefereeEvent.GOAL_AREA_IMPASSE,
            ball_side,
            jnp.where(
                event == RefereeEvent.FIELD_IMPASSE,
                jnp.int32(-1),
                1 - offending_team,
            ),
        ),
    ).astype(jnp.int32)
    restart = jnp.where(
        event == RefereeEvent.GOAL,
        jnp.int32(RestartType.KICKOFF),
        jnp.where(
            (event == RefereeEvent.DEFENDER_GOAL_AREA)
            | (event == RefereeEvent.GOALKEEPER_TIMEOUT)
            | (event == RefereeEvent.GOAL_AREA_STOPPED_BALL),
            jnp.int32(RestartType.PENALTY_KICK),
            jnp.where(
                (event == RefereeEvent.ATTACKER_GOAL_AREA)
                | (event == RefereeEvent.GOALKEEPER_BLOCK)
                | (event == RefereeEvent.GOAL_AREA_IMPASSE),
                jnp.int32(RestartType.GOAL_KICK),
                jnp.where(
                    event == RefereeEvent.FIELD_IMPASSE,
                    jnp.int32(RestartType.FREE_BALL),
                    jnp.where(
                        event == RefereeEvent.PERSONAL_FOUL,
                        personal_restart,
                        jnp.int32(RestartType.NONE),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    restart_team = jnp.where(restart == RestartType.NONE, -1, restart_team).astype(jnp.int32)
    moved_since_free_ball = (
        jnp.linalg.norm(simulation.ball[:2] - previous.free_ball_origin) > _BALL_MOVEMENT_TOLERANCE
    ) | (ball_speed >= _FREE_BALL_STOPPED_SPEED)
    previous_count = jnp.where(moved_since_free_ball, 0, previous.free_ball_count)
    stopped_quadrant = jnp.where(
        previous.field_impasse_seconds > 0,
        previous.field_impasse_quadrant,
        _free_ball_quadrant(simulation.ball),
    )
    quadrant = jnp.where(
        event == RefereeEvent.FIELD_IMPASSE,
        jnp.where(previous_count > 0, previous.free_ball_quadrant, stopped_quadrant),
        -1,
    ).astype(jnp.int32)
    free_ball_count = jnp.where(
        event == RefereeEvent.FIELD_IMPASSE, jnp.minimum(previous_count + 1, 3), previous_count
    ).astype(jnp.int32)
    free_ball_mark = jnp.asarray(
        [
            jnp.where((quadrant & 1) != 0, config.FREE_BALL_MARK_X, -config.FREE_BALL_MARK_X),
            jnp.where((quadrant & 2) != 0, config.FREE_BALL_MARK_Y, -config.FREE_BALL_MARK_Y),
        ]
    )
    free_ball_origin = jnp.where(free_ball_count >= 2, jnp.zeros(2), free_ball_mark)
    whistle = (restart != RestartType.NONE) & (event != RefereeEvent.GOAL)
    next_state = RefereeState(
        goal_area_seconds=jnp.where(
            goal_event | whistle, jnp.zeros_like(area_seconds), area_seconds
        ),
        goalkeeper_hold_seconds=jnp.where(
            goal_event | whistle, jnp.zeros_like(hold_seconds), hold_seconds
        ),
        goal_area_impasse_seconds=jnp.where(
            goal_event | whistle,
            jnp.zeros_like(goal_area_impasse_seconds),
            goal_area_impasse_seconds,
        ),
        field_impasse_seconds=jnp.where(
            goal_event | whistle,
            jnp.zeros_like(field_impasse_seconds),
            field_impasse_seconds,
        ),
        goal_area_stopped_seconds=jnp.where(
            goal_event | whistle,
            jnp.zeros_like(goal_area_stopped_seconds),
            goal_area_stopped_seconds,
        ),
        attack_primary=jnp.where(goal_event | whistle, -1, primary_attackers),
        defender_pushed_in=jnp.where(goal_event | whistle, False, pushed_in),
        field_impasse_quadrant=jnp.where(goal_event | whistle, -1, stopped_quadrant),
        free_ball_count=jnp.where(
            goal_event | (whistle & (event != RefereeEvent.FIELD_IMPASSE)), 0, free_ball_count
        ),
        free_ball_quadrant=jnp.where(
            event == RefereeEvent.FIELD_IMPASSE, quadrant, previous.free_ball_quadrant
        ),
        free_ball_origin=jnp.where(
            event == RefereeEvent.FIELD_IMPASSE, free_ball_origin, previous.free_ball_origin
        ),
        phase=jnp.where(
            whistle,
            jnp.int32(MatchPhase.STOPPED),
            jnp.int32(MatchPhase.PLAYING),
        ),
    )
    return _RefereeDecision(
        event=event,
        offending_team=offending_team,
        restart=restart,
        restart_team=restart_team,
        quadrant=quadrant,
        state=next_state,
    )
