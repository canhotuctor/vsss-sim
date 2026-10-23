"""Deterministic player formations for referee restarts."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import config, physics
from .match_referee import RestartType

_FREE_BALL_PLAYER_DISTANCE = 0.20
# The free-kick taker stands this far behind the ball; no player may start closer.
_FREE_KICK_CLEARANCE = 0.10
# Squares of any heading cannot touch once their centres are a full diagonal
# apart, and stay off a wall when their centre is half a diagonal from it.
_ROBOT_HALF_DIAGONAL = config.ROBOT_SIZE / 2.0 * 2.0**0.5
_FORMATION_SPACING = 2.0 * _ROBOT_HALF_DIAGONAL + 0.005
_FORMATION_ITERATIONS = 16


def _inside_field(points: jax.Array) -> jax.Array:
    """Clamp robot centres inside the walls and the 45-degree corner chamfers."""
    half_length = jnp.float32(config.FIELD_LENGTH / 2.0 - _ROBOT_HALF_DIAGONAL)
    half_width = jnp.float32(config.FIELD_WIDTH / 2.0 - _ROBOT_HALF_DIAGONAL)
    points = jnp.clip(
        points,
        jnp.stack((-half_length, -half_width)),
        jnp.stack((half_length, half_width)),
    )
    # A square's support along a chamfer normal is at most ROBOT_SIZE in |x| + |y|.
    corner = jnp.float32(
        config.FIELD_LENGTH / 2.0
        + config.FIELD_WIDTH / 2.0
        - config.FIELD_CHAMFER
        - config.ROBOT_SIZE
    )
    excess = jnp.maximum(jnp.sum(jnp.abs(points), axis=-1) - corner, 0.0)
    return points - jnp.sign(points) * excess[..., None] * 0.5


def _clear_ball(points: jax.Array, ball: jax.Array, clearance: float) -> jax.Array:
    """Slide robots closer than ``clearance`` to the ball along y until they clear it.

    A robot moves to the side of the ball it already occupies (towards the field
    centre when level with the ball), or to the other side if that would leave
    the field. Robots already clear of the ball are unchanged.
    """
    y_limit = config.FIELD_WIDTH / 2.0 - _ROBOT_HALF_DIAGONAL
    offset = points - ball
    too_close = jnp.linalg.norm(offset, axis=-1) < clearance - 1e-6
    dy = jnp.sqrt(jnp.maximum(clearance**2 - offset[..., 0] ** 2, 0.0))
    level = jnp.abs(offset[..., 1]) < 1e-6
    side = jnp.where(level, -jnp.sign(ball[1]), jnp.sign(offset[..., 1]))
    side = jnp.where(side == 0.0, 1.0, side)
    preferred_y = ball[1] + side * dy
    cleared_y = jnp.where(jnp.abs(preferred_y) <= y_limit, preferred_y, ball[1] - side * dy)
    return points.at[..., 1].set(jnp.where(too_close, cleared_y, points[..., 1]))


def _separate_formation(robots: jax.Array, ball: jax.Array) -> jax.Array:
    """Spread a ``(team, robot, 2)`` formation so no robot touches another or the ball.

    A fixed number of push-apart passes keeps this branch-free under ``vmap``.
    Formations already spaced out and inside the walls pass through unchanged.
    """
    points = robots.reshape((config.N_TEAMS * config.N_ROBOTS, 2))
    order = jnp.arange(points.shape[0])
    # Coincident robots separate along y, lower index first.
    coincident = jnp.where(order[:, None] > order[None, :], 1.0, -1.0)[..., None] * jnp.asarray(
        [0.0, 1.0], dtype=jnp.float32
    )
    not_self = ~jnp.eye(points.shape[0], dtype=jnp.bool_)
    for _ in range(_FORMATION_ITERATIONS):
        delta = points[:, None, :] - points[None, :, :]
        distance = jnp.linalg.norm(delta, axis=-1)
        direction = jnp.where(
            distance[..., None] > 1e-6,
            delta / jnp.maximum(distance[..., None], 1e-6),
            coincident,
        )
        overlap = jnp.where(not_self, jnp.maximum(_FORMATION_SPACING - distance, 0.0), 0.0)
        points = points + 0.5 * jnp.sum(overlap[..., None] * direction, axis=1)
        points = _clear_ball(_inside_field(points), ball, _FREE_KICK_CLEARANCE)
    return points.reshape(robots.shape)


def _restart_positions(
    simulation: physics.SimState,
    restart: jax.Array,
    restart_team: jax.Array,
    quadrant: jax.Array,
) -> physics.SimState:
    """Place both teams in a deterministic legal-enough restart formation.

    Restart markings and formations are fixed proxies: the current simulator
    has no human referee placement phase or model of the field's curved arcs.
    """
    half_length = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_width = jnp.float32(config.FIELD_WIDTH / 2.0)
    robot_margin = jnp.float32(config.ROBOT_RADIUS)
    formation_team = jnp.where(restart_team < 0, config.TEAM_BLUE, restart_team)
    direction = jnp.where(formation_team == config.TEAM_BLUE, 1.0, -1.0)
    ball = simulation.ball

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
    free_kick_ball = jnp.clip(
        ball[0:2],
        jnp.asarray([-half_length + robot_margin, -half_width + robot_margin]),
        jnp.asarray([half_length - robot_margin, half_width - robot_margin]),
    )

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
    support_x = jnp.clip(
        free_kick_ball[0] - direction * 0.24,
        -half_length + robot_margin,
        half_length - robot_margin,
    )
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
    defensive_x = direction * (half_length - config.GOAL_AREA_LENGTH_X - 0.08)
    free_kick_defending = jnp.stack(
        (
            defending_goalkeeper,
            jnp.asarray([defensive_x, -0.42]),
            jnp.asarray([defensive_x, 0.42]),
        )
    )
    free_kick_robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), jnp.float32)
    free_kick_robots = free_kick_robots.at[formation_team].set(free_kick_team)
    free_kick_robots = free_kick_robots.at[1 - formation_team].set(free_kick_defending)
    # The foul spot can be anywhere, so the ball-relative spots above may clip
    # the kicker onto the ball near an end line or overlap the fixed defensive
    # spots; spread the formation before play resumes.
    free_kick_robots = _separate_formation(free_kick_robots, free_kick_ball)

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
    robots = jnp.zeros_like(simulation.robots)
    robots = robots.at[..., 0:2].set(restart_robots)
    robots = robots.at[..., 2].set(angles)
    stopped_ball = simulation.ball.at[0:2].set(restart_ball).at[2:4].set(0.0)
    return simulation._replace(
        ball=stopped_ball,
        robots=robots,
        wheel_speeds=jnp.zeros_like(simulation.wheel_speeds),
    )
