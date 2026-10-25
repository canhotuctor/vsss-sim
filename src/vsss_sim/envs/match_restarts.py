"""Deterministic player formations for referee restarts."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config, physics
from .match_referee import RestartType

_FREE_BALL_PLAYER_DISTANCE = 0.20


def _restart_positions(
    simulation: physics.SimState,
    restart: jax.Array,
    restart_team: jax.Array,
    quadrant: jax.Array,
    free_ball_stage: jax.Array = 1,
) -> physics.SimState:
    """Choose legal fixed formations; repeated free balls follow Rule 15."""
    half_length = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_width = jnp.float32(config.FIELD_WIDTH / 2.0)
    robot_margin = jnp.float32(config.ROBOT_RADIUS)
    formation_team = jnp.where(restart_team < 0, config.TEAM_BLUE, restart_team)
    direction = jnp.where(formation_team == config.TEAM_BLUE, 1.0, -1.0)

    penalty_ball = jnp.asarray([direction * config.PENALTY_MARK_X, 0.0], dtype=jnp.float32)
    # The goal-kick (TM) mark is on the front edge of its goal area.
    goal_kick_ball = jnp.asarray(
        [
            -direction * (half_length - config.GOAL_AREA_LENGTH_X),
            0.0,
        ],
        dtype=jnp.float32,
    )
    quadrant_x = jnp.where((quadrant & 1) != 0, 1.0, -1.0)
    quadrant_y = jnp.where((quadrant & 2) != 0, 1.0, -1.0)
    free_ball_mark = jnp.asarray(
        [
            quadrant_x * config.FREE_BALL_MARK_X,
            quadrant_y * config.FREE_BALL_MARK_Y,
        ],
        dtype=jnp.float32,
    )
    free_kick_ball = penalty_ball  # TL and PK share the marked spot (Fig. 1).

    penalty_own_goalkeeper = jnp.asarray(
        [
            -direction * (half_length - config.GOAL_AREA_LENGTH_X - 0.03),
            0.0,
        ],
        dtype=jnp.float32,
    )
    # Stage the kicker behind the ball; other field players stay beyond halfway.
    penalty_kicker = jnp.asarray(
        [penalty_ball[0] - direction * 0.242, direction * 0.133],
        dtype=jnp.float32,
    )
    penalty_support_x = -direction * config.ROBOT_SIZE * 0.8
    penalty_support = jnp.asarray(
        [penalty_support_x, -direction * 0.40],
        dtype=jnp.float32,
    )
    penalty_attacking = jnp.stack((penalty_own_goalkeeper, penalty_kicker, penalty_support))
    penalty_defending = jnp.stack(
        (
            jnp.asarray([direction * (half_length - robot_margin), 0.0]),
            jnp.asarray([penalty_support_x, -0.20]),
            jnp.asarray([penalty_support_x, 0.20]),
        )
    )
    penalty_robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), jnp.float32)
    penalty_robots = penalty_robots.at[formation_team].set(penalty_attacking)
    penalty_robots = penalty_robots.at[1 - formation_team].set(penalty_defending)

    own_goal_x = -direction * half_length
    goal_kick_kicker = jnp.asarray(
        [
            goal_kick_ball[0] - direction * 0.094,
            -direction * 0.187,
        ],
        dtype=jnp.float32,
    )
    goal_kick_team = jnp.stack(
        (
            goal_kick_kicker,
            jnp.asarray([own_goal_x + direction * 0.30, -0.40]),
            jnp.asarray([own_goal_x + direction * 0.30, 0.40]),
        )
    )
    goal_kick_opponents = jnp.stack(
        (
            jnp.asarray([-direction * (config.ROBOT_SIZE / 2.0), 0.30]),
            jnp.asarray([direction * 0.28, -0.35]),
            jnp.asarray([direction * 0.28, 0.35]),
        )
    )
    goal_kick_robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), jnp.float32)
    goal_kick_robots = goal_kick_robots.at[formation_team].set(goal_kick_team)
    goal_kick_robots = goal_kick_robots.at[1 - formation_team].set(goal_kick_opponents)

    free_kick_kicker = free_kick_ball.at[0].add(-direction * 0.10)
    free_kick_kicker = jnp.clip(
        free_kick_kicker,
        jnp.asarray([-half_length + robot_margin, -half_width + robot_margin]),
        jnp.asarray([half_length - robot_margin, half_width - robot_margin]),
    )
    support_x = -direction * 0.30
    free_kick_team = jnp.stack(
        (
            free_kick_kicker,
            jnp.stack((support_x, jnp.clip(free_kick_ball[1] - 0.30, -0.55, 0.55))),
            jnp.stack((support_x, jnp.clip(free_kick_ball[1] + 0.30, -0.55, 0.55))),
        )
    )
    defending_goalkeeper = jnp.asarray(
        [direction * (half_length - robot_margin), 0.0], dtype=jnp.float32
    )
    # Axis-aligned defenders touch the front goal-area boundary from outside.
    defensive_x = direction * (half_length - config.GOAL_AREA_LENGTH_X - robot_margin)
    free_kick_defending = jnp.stack(
        (
            defending_goalkeeper,
            jnp.asarray([defensive_x, -0.25]),
            jnp.asarray([defensive_x, 0.25]),
        )
    )
    free_kick_robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), jnp.float32)
    free_kick_robots = free_kick_robots.at[formation_team].set(free_kick_team)
    free_kick_robots = free_kick_robots.at[1 - formation_team].set(free_kick_defending)


    free_ball_support_y = -quadrant_y * 0.42
    free_ball_inner_support_y = free_ball_support_y / 2.0
    # Keep exactly one player per side on the 20 cm boundary; move teammates
    # away from the ball's quadrant so no other robot enters its exclusion zone.
    blue_free_ball_kicker = jnp.stack(
        (
            free_ball_mark[0] - _FREE_BALL_PLAYER_DISTANCE,
            free_ball_mark[1],
        )
    )
    yellow_free_ball_kicker = jnp.stack(
        (
            free_ball_mark[0] + _FREE_BALL_PLAYER_DISTANCE,
            free_ball_mark[1],
        )
    )
    blue_free_ball_supports = jnp.stack(
        (
            jnp.stack((jnp.float32(-0.45), free_ball_support_y)),
            jnp.stack((jnp.float32(-0.45), free_ball_inner_support_y)),
        )
    )
    yellow_free_ball_supports = jnp.stack(
        (
            jnp.stack((jnp.float32(0.45), free_ball_support_y)),
            jnp.stack((jnp.float32(0.45), free_ball_inner_support_y)),
        )
    )
    blue_free_ball_robots = jnp.concatenate(
        (
            blue_free_ball_kicker[None, :],
            blue_free_ball_supports,
        ),
        axis=0,
    )
    yellow_free_ball_robots = jnp.concatenate(
        (
            yellow_free_ball_kicker[None, :],
            yellow_free_ball_supports,
        ),
        axis=0,
    )
    free_ball_robots = jnp.stack((blue_free_ball_robots, yellow_free_ball_robots))
    center_formation = jnp.asarray(
        [
            [[-0.28, 0.0], [-0.40, -0.30], [-0.40, 0.30]],
            [[0.28, 0.0], [0.40, -0.30], [0.40, 0.30]],
        ]
    )
    free_ball_mark = jnp.where(free_ball_stage >= 2, jnp.zeros(2), free_ball_mark)
    free_ball_robots = jnp.where(free_ball_stage >= 3, center_formation, free_ball_robots)

    restart_ball = jnp.where(
        restart == RestartType.PENALTY_KICK,
        penalty_ball,
        jnp.where(
            restart == RestartType.GOAL_KICK,
            goal_kick_ball,
            jnp.where(
                restart == RestartType.FREE_BALL,
                free_ball_mark,
                free_kick_ball,
            ),
        ),
    )
    restart_robots = jnp.where(
        restart == RestartType.PENALTY_KICK,
        penalty_robots,
        jnp.where(
            restart == RestartType.GOAL_KICK,
            goal_kick_robots,
            jnp.where(
                restart == RestartType.FREE_BALL,
                free_ball_robots,
                free_kick_robots,
            ),
        ),
    )
    angles = jnp.arctan2(
        restart_ball[None, None, 1] - restart_robots[..., 1],
        restart_ball[None, None, 0] - restart_robots[..., 0],
    )
    # These defenders must keep an entire side on the goal-area/goal line.
    is_free_kick_defender = (restart == RestartType.FREE_KICK) & (
        jnp.arange(config.N_TEAMS) != formation_team
    )
    is_penalty_keeper = (
        (restart == RestartType.PENALTY_KICK)
        & (jnp.arange(config.N_TEAMS)[:, None] != formation_team)
        & (jnp.arange(config.N_ROBOTS)[None, :] == 0)
    )
    angles = jnp.where(
        is_free_kick_defender[:, None] | is_penalty_keeper,
        jnp.where(direction > 0, jnp.pi, 0.0),
        angles,
    )
    robots = jnp.zeros_like(simulation.robots)
    robots = robots.at[..., 0:2].set(restart_robots)
    robots = robots.at[..., 2].set(angles)
    # On the second occurrence the ball alone moves to centre. Keep headings
    # and positions unchanged, but stop motion during the whistle.
    robots = jnp.where(
        (restart == RestartType.FREE_BALL) & (free_ball_stage == 2),
        simulation.robots.at[..., 3:].set(0.0),
        robots,
    )
    stopped_ball = simulation.ball.at[0:2].set(restart_ball).at[2:4].set(0.0)
    return simulation._replace(
        ball=stopped_ball,
        robots=robots,
        wheel_speeds=jnp.zeros_like(simulation.wheel_speeds),
    )
