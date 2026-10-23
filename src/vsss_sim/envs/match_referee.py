"""JAX-compatible referee state and event resolution."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import config, physics
from .match_geometry import (
    _PLAY_INVOLVEMENT_DISTANCE,
    _ROBOT_CONTACT_DISTANCE,
    _ball_between,
    _ball_goal_area_membership,
    _goal_area_membership,
    _players_near_each_other,
    _robot_ball_distance,
)

_BALL_STUCK_SPEED = 0.01
_FREE_BALL_STOPPED_SPEED = 0.001  # m/s (1 mm/s)
_FREE_BALL_TIMEOUT_SECONDS = 5.0
# Rule 15 and the 3v3 reference setups place each designated player 20 cm away.
_GOALKEEPER_HOLD_SPEED = 0.05
_RULE_TIMEOUT_SECONDS = 10.0


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
    defender_area_violation_pending: jax.Array
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
        defender_area_violation_pending=jnp.zeros((config.N_TEAMS,), dtype=jnp.bool_),
        phase=jnp.asarray(MatchPhase.PLAYING, dtype=jnp.int32),
    )


def _illegal_push(
    simulation_before: physics.SimState,
    simulation_after: physics.SimState,
    areas: jax.Array,
    ball_areas: jax.Array,
):
    """Find a local close pair and identify which robot initiated contact."""
    before = simulation_before.robots
    after = simulation_after.robots
    ball = simulation_after.ball
    ball_in_goal_area = jnp.any(ball_areas)
    near_ball = _robot_ball_distance(after, ball) <= jnp.float32(_PLAY_INVOLVEMENT_DISTANCE)
    robot_in_area = jnp.any(areas, axis=-1)
    hits_by_team = []
    for offender in range(config.N_TEAMS):
        victim = 1 - offender
        before_delta = before[victim, None, :, 0:2] - before[offender, :, None, 0:2]
        after_delta = after[victim, None, :, 0:2] - after[offender, :, None, 0:2]
        before_distance = jnp.linalg.norm(before_delta, axis=-1)
        after_distance = jnp.linalg.norm(after_delta, axis=-1)
        contact = after_distance <= jnp.float32(_ROBOT_CONTACT_DISTANCE)
        closing = before_distance > after_distance + 1e-4
        direction = before_delta / jnp.maximum(before_distance[..., None], 1e-8)
        offender_motion = after[offender, :, None, 0:2] - before[offender, :, None, 0:2]
        victim_motion = after[victim, None, :, 0:2] - before[victim, None, :, 0:2]
        offender_advance = jnp.sum(offender_motion * direction, axis=-1)
        victim_advance = -jnp.sum(victim_motion * direction, axis=-1)
        initiated_contact = offender_advance > victim_advance + 1e-4
        first_robots = jnp.broadcast_to(
            after[offender, :, None, :],
            (config.N_ROBOTS, config.N_ROBOTS, after.shape[-1]),
        )
        second_robots = jnp.broadcast_to(
            after[victim, None, :, :],
            (config.N_ROBOTS, config.N_ROBOTS, after.shape[-1]),
        )
        ball_between = _ball_between(
            first_robots,
            second_robots,
            ball,
        )
        involved_area = robot_in_area[offender, :, None] | robot_in_area[victim, None, :]
        near_ball_pair = near_ball[offender, :, None] | near_ball[victim, None, :]
        goal_play = involved_area | (ball_in_goal_area & near_ball_pair)
        hits_by_team.append(contact & closing & initiated_contact & ~ball_between & goal_play)
    return jnp.stack(hits_by_team)


def _free_ball_quadrant(ball: jax.Array) -> jax.Array:
    return (ball[0] >= 0.0).astype(jnp.int32) + 2 * (ball[1] >= 0.0).astype(jnp.int32)


def _second_robot_participates(
    area_members: jax.Array,
    area_seconds: jax.Array,
    participates: jax.Array,
) -> jax.Array:
    """Require an involved player beyond the longest-standing area occupant."""
    longest_stay = jnp.max(jnp.where(area_members, area_seconds, -1.0))
    longest_count = jnp.sum(area_members & (area_seconds == longest_stay))
    second = area_members & ((area_seconds < longest_stay) | (longest_count > 1)) & participates
    return jnp.any(second)


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
    nearby_players = _players_near_each_other(simulation.robots)
    near_ball = _robot_ball_distance(simulation.robots, simulation.ball) <= jnp.float32(
        _PLAY_INVOLVEMENT_DISTANCE
    )
    involved = near_ball | jnp.any(nearby_players, axis=(-2, -1))

    pushed_in = []
    for team in range(config.N_TEAMS):
        opponent = 1 - team
        near_opponent = jnp.any(nearby_players[team, :, opponent, :], axis=-1)
        opponent_near_ball = jnp.any(near_ball[opponent])
        pushed_in.append(
            areas_after[team]
            & ~areas_before[team]
            & near_opponent[:, None]
            & near_ball[team, :, None]
            & opponent_near_ball
        )
    pushed_in = jnp.stack(pushed_in)
    participation = involved[..., None] & ~pushed_in

    attacking_violations = []
    defending_violations = []
    pending_defender_violations = []
    for team in range(config.N_TEAMS):
        attacking_area = areas_after[team, :, 1 - team]
        attacking_ball_dispute = (
            ball_areas[1 - team]
            & jnp.any(near_ball[team])
            & jnp.any(near_ball[1 - team])
        )
        attacking_violations.append(
            attacking_ball_dispute
            & (jnp.sum(attacking_area) >= 2)
            & _second_robot_participates(
                attacking_area,
                area_seconds[team, :, 1 - team],
                participation[team, :, 1 - team],
            )
        )

        defending_area = areas_after[team, :, team]
        second_defender_participates = (jnp.sum(defending_area) >= 2) & _second_robot_participates(
            defending_area,
            area_seconds[team, :, team],
            participation[team, :, team],
        )
        active_ball_dispute = jnp.any(defending_area & near_ball[team]) & jnp.any(
            near_ball[1 - team]
        )
        violation_in_area = ball_areas[team] & second_defender_participates
        immediate_violation = violation_in_area & active_ball_dispute
        deferred_violation = previous.defender_area_violation_pending[team] & ~ball_areas[team]
        defending_violations.append(immediate_violation | deferred_violation)
        pending_defender_violations.append(
            ball_areas[team]
            & (
                previous.defender_area_violation_pending[team]
                | (violation_in_area & ~active_ball_dispute)
            )
        )
    attacking_violations = jnp.stack(attacking_violations)
    defending_violations = jnp.stack(defending_violations)
    pending_defender_violations = jnp.stack(pending_defender_violations)
    has_attacking_violation = jnp.any(attacking_violations)
    has_defending_violation = jnp.any(defending_violations)
    attacking_team = jnp.argmax(attacking_violations.astype(jnp.int32))
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
        opponents_before = simulation_before.robots[1 - team]
        goalkeeper_before = simulation_before.robots[team, goalkeeper_index]
        before_to_keeper = goalkeeper_before[None, 0:2] - opponents_before[:, 0:2]
        before_distance = jnp.linalg.norm(before_to_keeper, axis=-1)
        opponent_to_keeper = jnp.linalg.norm(opponents[:, 0:2] - goalkeeper[None, 0:2], axis=-1)
        opponent_near_keeper = jnp.any(opponent_to_keeper <= _PLAY_INVOLVEMENT_DISTANCE)
        opponent_near_ball = jnp.any(near_ball[1 - team])
        keeper_ball_distance = jnp.linalg.norm(goalkeeper[0:2] - simulation.ball[0:2])
        keeper_touches_ball = keeper_ball_distance <= (
            config.ROBOT_RADIUS + config.BALL_RADIUS + 0.01
        )
        ball_speed = jnp.linalg.norm(simulation.ball[2:4])
        attacker_contests = opponent_near_keeper | opponent_near_ball
        holds_ball = (
            ball_areas[team]
            & has_goalkeeper
            & keeper_touches_ball
            & (ball_speed <= _GOALKEEPER_HOLD_SPEED)
            & ~attacker_contests
        )

        # Rule 14: an attacker blocking the goalkeeper without a ball contest
        # causes a goal kick. The 50-percent area test applies to both robots.
        attacker_in_area = areas_after[1 - team, :, team]
        attacker_near_keeper = opponent_to_keeper <= _ROBOT_CONTACT_DISTANCE
        before_direction = before_to_keeper / jnp.maximum(before_distance[:, None], 1e-8)
        attacker_motion = opponents[:, 0:2] - opponents_before[:, 0:2]
        keeper_motion = goalkeeper[0:2] - goalkeeper_before[0:2]
        attacker_advance = jnp.sum(attacker_motion * before_direction, axis=-1)
        keeper_advance = -jnp.sum(keeper_motion[None, :] * before_direction, axis=-1)
        attacker_closing = (before_distance > opponent_to_keeper + 1e-4) & (
            attacker_advance > keeper_advance + 1e-4
        )
        no_ball_contest = (keeper_ball_distance > _PLAY_INVOLVEMENT_DISTANCE) & (
            jnp.linalg.norm(opponents[:, 0:2] - simulation.ball[None, 0:2], axis=-1)
            > _PLAY_INVOLVEMENT_DISTANCE
        )
        blocks = (
            attacker_in_area
            & attacker_near_keeper
            & attacker_closing
            & no_ball_contest
            & has_goalkeeper
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
    goalkeeper_timeouts = hold_seconds >= _RULE_TIMEOUT_SECONDS
    has_goalkeeper_timeout = jnp.any(goalkeeper_timeouts)
    timeout_team = jnp.argmax(goalkeeper_timeouts.astype(jnp.int32))

    ball_side = jnp.argmax(ball_areas.astype(jnp.int32))
    goalkeeper_index = goalkeeper_indices[ball_side]
    goalkeeper = simulation.robots[ball_side, goalkeeper_index]
    goalkeeper_distance = jnp.linalg.norm(goalkeeper[0:2] - simulation.ball[0:2])
    attacker_near_impasse_ball = jnp.any(near_ball[1 - ball_side])
    goal_area_contest = (
        ball_in_goal_area
        & goalkeeper_present[ball_side]
        & (goalkeeper_distance <= config.ROBOT_RADIUS + config.BALL_RADIUS + 0.01)
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
    goal_area_impasse = (goal_area_impasse_seconds >= _RULE_TIMEOUT_SECONDS) & ball_in_goal_area
    field_impasse = (field_impasse_seconds >= _FREE_BALL_TIMEOUT_SECONDS) & ~ball_in_goal_area
    goal_area_stopped_ball = (
        goal_area_stopped_seconds >= _FREE_BALL_TIMEOUT_SECONDS
    ) & ball_in_goal_area

    # Approximate personal fouls only for a closing opponent pair in a goal-
    # area play. This avoids turning harmless distant collisions into whistles.
    push_hits = _illegal_push(simulation_before, simulation, areas_after, ball_areas)
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
    personal_offender_in_own_area = (
        jnp.any(areas_after[offending_team, :, offending_team]) & ball_areas[offending_team]
    )
    personal_offender_in_attack_area = jnp.any(areas_after[offending_team, :, 1 - offending_team])
    personal_restart = jnp.where(
        personal_offender_in_own_area,
        jnp.int32(RestartType.PENALTY_KICK),
        jnp.where(
            personal_offender_in_attack_area,
            jnp.int32(RestartType.GOAL_KICK),
            jnp.int32(RestartType.FREE_KICK),
        ),
    )
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
    quadrant = jnp.where(
        event == RefereeEvent.FIELD_IMPASSE,
        _free_ball_quadrant(simulation.ball),
        jnp.int32(-1),
    ).astype(jnp.int32)
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
        defender_area_violation_pending=jnp.where(
            goal_event | whistle,
            jnp.zeros_like(pending_defender_violations),
            pending_defender_violations,
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
