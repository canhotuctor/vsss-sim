"""JAX physics backend for vsss-sim.

Pure-functional mirror of ``numpy_backend.py``. ``SimState`` is a
:class:`typing.NamedTuple` PyTree — register-free, immutable, ``vmap``-friendly.

Float dtype is ``float32`` (GPU default). Tests assert semantic parity with the
float64 numpy backend within a generous tolerance.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import config
from .numpy_backend import SimState as NumpySimState


# ---------------------------------------------------------------------------
# State container (PyTree)
# ---------------------------------------------------------------------------

class SimState(NamedTuple):
    """Immutable simulation state. Fields are JAX arrays."""

    ball: jnp.ndarray          # (4,)   float32   [x, y, vx, vy]
    robots: jnp.ndarray        # (N_TEAMS, N_ROBOTS, 6) float32 [x, y, theta, vx, vy, omega]
    score: jnp.ndarray         # (2,)   int32
    t: jnp.ndarray             # ()     float32
    wheel_speeds: jnp.ndarray  # (N_TEAMS, N_ROBOTS, 2) float32 — last applied (m/s)


def empty_state() -> SimState:
    """Return an all-zero ``SimState``."""
    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32),
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
        wheel_speeds=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32),
    )


def from_numpy(np_state: NumpySimState) -> SimState:
    """Convert a NumPy ``SimState`` to a JAX ``SimState``."""
    return SimState(
        ball=jnp.asarray(np_state.ball, dtype=jnp.float32),
        robots=jnp.asarray(np_state.robots, dtype=jnp.float32),
        score=jnp.asarray(np_state.score, dtype=jnp.int32),
        t=jnp.asarray(np_state.t, dtype=jnp.float32),
        wheel_speeds=jnp.asarray(np_state.wheel_speeds, dtype=jnp.float32),
    )


def to_numpy(state: SimState) -> NumpySimState:
    """Convert a JAX ``SimState`` to a NumPy ``SimState``."""
    return NumpySimState(
        ball=np.asarray(state.ball, dtype=np.float64),
        robots=np.asarray(state.robots, dtype=np.float64),
        score=np.asarray(state.score, dtype=np.int32),
        t=float(state.t),
        wheel_speeds=np.asarray(state.wheel_speeds, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Differential-drive kinematics (vectorised over robots)
# ---------------------------------------------------------------------------

def _diff_drive(
    v_left: jnp.ndarray,
    v_right: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert wheel speeds to body velocities.

    Returns ``(vx, vy, omega)`` shaped like the inputs.
    """
    v = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / config.ROBOT_WHEELBASE
    vx = v * jnp.cos(theta)
    vy = v * jnp.sin(theta)
    return vx, vy, omega


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

_BLUE_STARTS = jnp.array(
    [[-0.55, 0.0], [-0.30, 0.30], [-0.30, -0.30]], dtype=jnp.float32
)
_YELLOW_STARTS = jnp.array(
    [[0.55, 0.0], [0.30, -0.30], [0.30, 0.30]], dtype=jnp.float32
)


def reset_kickoff(key: jnp.ndarray) -> SimState:
    """Place robots and ball for a standard kickoff (functional)."""
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0 - config.ROBOT_RADIUS)
    clear = jnp.float32(config.KICKOFF_CLEAR_DIST)

    key_b, key_y = jax.random.split(key)
    blue_jitter = jax.random.uniform(key_b, (3, 2), minval=-0.05, maxval=0.05)
    yellow_jitter = jax.random.uniform(key_y, (3, 2), minval=-0.05, maxval=0.05)

    blue = _BLUE_STARTS + blue_jitter
    blue = blue.at[:, 0].set(jnp.clip(blue[:, 0], -half_l, -clear))
    blue = blue.at[:, 1].set(jnp.clip(blue[:, 1], -half_l, half_l))

    yellow = _YELLOW_STARTS + yellow_jitter
    yellow = yellow.at[:, 0].set(jnp.clip(yellow[:, 0], clear, half_l))
    yellow = yellow.at[:, 1].set(jnp.clip(yellow[:, 1], -half_l, half_l))

    robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32)
    robots = robots.at[config.TEAM_BLUE, :, 0:2].set(blue)
    robots = robots.at[config.TEAM_YELLOW, :, 0:2].set(yellow)

    # Face toward the ball at origin.
    theta = jnp.arctan2(-robots[:, :, 1], -robots[:, :, 0])
    robots = robots.at[:, :, 2].set(theta)

    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=robots,
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
        wheel_speeds=jnp.zeros(
            (config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32
        ),
    )


# ---------------------------------------------------------------------------
def reset_random(key: jnp.ndarray) -> SimState:
    """Place robots and ball at uniformly random positions anywhere on the field.

    All robots (both teams) sample from the full field; no half-restriction.
    - Ball: random within the inner 80 % of the field.
    - All robots: random x/y within field bounds (margin from walls), random headings.
    - No overlap-rejection; any initial interpenetration resolves in the first
      few physics sub-steps.
    """
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    margin = jnp.float32(config.ROBOT_SIZE)

    key_ball, key_x, key_y, key_t = jax.random.split(key, 4)

    # Ball: inner 80 % of the field
    ball_x = jax.random.uniform(key_ball, (), minval=-(half_l - margin) * 0.8, maxval=(half_l - margin) * 0.8)
    ball_y = jax.random.uniform(key_ball, (), minval=-(half_w - margin) * 0.8, maxval=(half_w - margin) * 0.8)
    ball = jnp.array([ball_x, ball_y, 0.0, 0.0], dtype=jnp.float32)

    # All robots: full field
    all_x = jax.random.uniform(key_x, (config.N_TEAMS, config.N_ROBOTS), minval=-half_l + margin, maxval=half_l - margin)
    all_y = jax.random.uniform(key_y, (config.N_TEAMS, config.N_ROBOTS), minval=-half_w + margin, maxval=half_w - margin)
    all_t = jax.random.uniform(key_t, (config.N_TEAMS, config.N_ROBOTS), minval=-jnp.pi, maxval=jnp.pi)

    robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32)
    robots = robots.at[:, :, 0].set(all_x)
    robots = robots.at[:, :, 1].set(all_y)
    robots = robots.at[:, :, 2].set(all_t)

    return SimState(
        ball=ball,
        robots=robots,
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
        wheel_speeds=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32),
    )


# Robot–wall collisions (vectorised)
# ---------------------------------------------------------------------------

def _robot_wall_collisions(state: SimState) -> SimState:
    """Clamp robots (OBB) inside field boundaries; zero outward velocity."""
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half = jnp.float32(config.ROBOT_SIZE / 2.0)

    theta = state.robots[:, :, 2]
    extent = half * (jnp.abs(jnp.cos(theta)) + jnp.abs(jnp.sin(theta)))

    lim_x = half_l - extent
    lim_y = half_w - extent

    x = state.robots[:, :, 0]
    y = state.robots[:, :, 1]
    vx = state.robots[:, :, 3]
    vy = state.robots[:, :, 4]

    exceeded_neg_x = x < -lim_x
    exceeded_pos_x = x > lim_x
    exceeded_neg_y = y < -lim_y
    exceeded_pos_y = y > lim_y

    new_x = jnp.clip(x, -lim_x, lim_x)
    new_y = jnp.clip(y, -lim_y, lim_y)

    new_vx = jnp.where(exceeded_neg_x & (vx < 0), 0.0, vx)
    new_vx = jnp.where(exceeded_pos_x & (new_vx > 0), 0.0, new_vx)
    new_vy = jnp.where(exceeded_neg_y & (vy < 0), 0.0, vy)
    new_vy = jnp.where(exceeded_pos_y & (new_vy > 0), 0.0, new_vy)

    robots = state.robots
    robots = robots.at[:, :, 0].set(new_x)
    robots = robots.at[:, :, 1].set(new_y)
    robots = robots.at[:, :, 3].set(new_vx)
    robots = robots.at[:, :, 4].set(new_vy)
    return state._replace(robots=robots)


# ---------------------------------------------------------------------------
# Ball–wall collisions and goal detection
# ---------------------------------------------------------------------------

def _ball_wall_collisions(state: SimState) -> tuple[SimState, jnp.ndarray]:
    """Reflect ball off field walls and detect goals.

    Returns
    -------
    new_state : SimState
    goal : int32 scalar (+1 blue, -1 yellow, 0 none).
    """
    r = jnp.float32(config.BALL_RADIUS)
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half_goal = jnp.float32(config.GOAL_WIDTH / 2.0)
    goal_depth = jnp.float32(config.GOAL_DEPTH)
    e_wall = jnp.float32(config.BALL_WALL_RESTITUTION)

    bx, by, bvx, bvy = state.ball[0], state.ball[1], state.ball[2], state.ball[3]

    # --- y walls (top / bottom of field) ---
    hit_bot = by - r < -half_w
    hit_top = by + r > half_w
    by = jnp.where(hit_bot, -half_w + r, by)
    by = jnp.where(hit_top, half_w - r, by)
    bvy = jnp.where(hit_bot, jnp.abs(bvy) * e_wall, bvy)
    bvy = jnp.where(hit_top, -jnp.abs(bvy) * e_wall, bvy)

    # --- x walls / goals (use post-y-clamp by for goal-y check) ---
    in_goal_y = jnp.abs(by) <= half_goal
    hit_left = bx - r < -half_l
    hit_right = bx + r > half_l

    left_goal = hit_left & in_goal_y    # yellow scores (-1)
    right_goal = hit_right & in_goal_y  # blue scores (+1)

    # Left side: clamp to back-of-net if goal, else to wall.
    bx = jnp.where(
        left_goal,
        -half_l - goal_depth + r,
        jnp.where(hit_left, -half_l + r, bx),
    )
    bvx = jnp.where(hit_left, jnp.abs(bvx) * e_wall, bvx)

    # Right side: same treatment.
    bx = jnp.where(
        right_goal,
        half_l + goal_depth - r,
        jnp.where(hit_right, half_l - r, bx),
    )
    bvx = jnp.where(hit_right, -jnp.abs(bvx) * e_wall, bvx)

    goal = jnp.where(
        right_goal,
        jnp.int32(1),
        jnp.where(left_goal, jnp.int32(-1), jnp.int32(0)),
    )

    new_ball = jnp.stack([bx, by, bvx, bvy])
    return state._replace(ball=new_ball), goal


# ---------------------------------------------------------------------------
# Ball–robot collisions (circle vs OBB) — fori_loop over the 6 robots
# ---------------------------------------------------------------------------

_N_ROBOTS_TOTAL = config.N_TEAMS * config.N_ROBOTS


def _ball_obb_penetration(
    ball_pos: jnp.ndarray, rob_pos: jnp.ndarray, theta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (world-space normal, penetration) for a ball vs square OBB."""
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    r_ball = jnp.float32(config.BALL_RADIUS)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    dx = ball_pos[0] - rob_pos[0]
    dy = ball_pos[1] - rob_pos[1]
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    clamp_x = jnp.clip(local_x, -half, half)
    clamp_y = jnp.clip(local_y, -half, half)
    diff_x = local_x - clamp_x
    diff_y = local_y - clamp_y
    dist = jnp.sqrt(diff_x * diff_x + diff_y * diff_y)

    # Outside-face case
    safe_dist = jnp.where(dist < 1e-9, 1.0, dist)
    lnx_out = diff_x / safe_dist
    lny_out = diff_y / safe_dist
    pen_out = r_ball - dist

    # Ball-centre-inside case: push out along shortest face.
    pen_x_in = half - jnp.abs(local_x)
    pen_y_in = half - jnp.abs(local_y)
    use_x = pen_x_in <= pen_y_in
    lnx_in = jnp.where(use_x, jnp.sign(local_x), 0.0)
    lny_in = jnp.where(use_x, 0.0, jnp.sign(local_y))
    pen_in = jnp.where(use_x, pen_x_in, pen_y_in) + r_ball

    inside = dist < 1e-9
    lnx = jnp.where(inside, lnx_in, lnx_out)
    lny = jnp.where(inside, lny_in, lny_out)
    penetration = jnp.where(inside, pen_in, pen_out)

    # Rotate normal back to world frame.
    nx = cos_t * lnx - sin_t * lny
    ny = sin_t * lnx + cos_t * lny
    return jnp.stack([nx, ny]), penetration


def _resolve_ball_robot_pair(
    ball: jnp.ndarray,   # (4,)
    robot: jnp.ndarray,  # (6,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve one ball-robot collision; returns updated (ball, robot)."""
    ball_pos = ball[0:2]
    ball_vel = ball[2:4]
    rob_pos = robot[0:2]
    rob_vel = robot[3:5]
    theta = robot[2]

    normal, penetration = _ball_obb_penetration(ball_pos, rob_pos, theta)
    collide = penetration > 0.0

    m_b = jnp.float32(config.BALL_MASS)
    m_r = jnp.float32(config.ROBOT_MASS)
    total_m = m_b + m_r
    e = jnp.float32(config.BALL_ROBOT_RESTITUTION)

    # Positional correction (only when colliding).
    bp_corr = ball_pos + normal * penetration * (m_r / total_m)
    rp_corr = rob_pos - normal * penetration * (m_b / total_m)

    rel_vel = ball_vel - rob_vel
    vel_along = jnp.dot(rel_vel, normal)
    do_impulse = collide & (vel_along < 0)

    j = -(1.0 + e) * vel_along / (1.0 / m_b + 1.0 / m_r)
    impulse = j * normal

    bv_new = jnp.where(do_impulse, ball_vel + impulse / m_b, ball_vel)
    rv_new = jnp.where(do_impulse, rob_vel - impulse / m_r, rob_vel)

    bp_new = jnp.where(collide, bp_corr, ball_pos)
    rp_new = jnp.where(collide, rp_corr, rob_pos)

    new_ball = jnp.concatenate([bp_new, bv_new])
    new_robot = robot.at[0:2].set(rp_new).at[3:5].set(rv_new)
    return new_ball, new_robot


def _ball_robot_collisions(state: SimState) -> SimState:
    """Resolve elastic collisions between the ball and the 6 robots (sequential)."""
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)

    def body(i, carry):
        ball, robots_flat = carry
        new_ball, new_robot = _resolve_ball_robot_pair(ball, robots_flat[i])
        robots_flat = robots_flat.at[i].set(new_robot)
        return new_ball, robots_flat

    new_ball, new_robots_flat = jax.lax.fori_loop(
        0, _N_ROBOTS_TOTAL, body, (state.ball, robots_flat)
    )
    return state._replace(
        ball=new_ball,
        robots=new_robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6),
    )


# ---------------------------------------------------------------------------
# Robot–robot collisions (OBB vs OBB via SAT) — fori_loop over 15 pairs
# ---------------------------------------------------------------------------

# Pre-compute the 15 unique (i, j) pairs for 6 robots.
_PAIR_I, _PAIR_J = jnp.triu_indices(_N_ROBOTS_TOTAL, k=1)
_N_PAIRS = int(_PAIR_I.shape[0])


def _sat_square_overlap(
    pos_a: jnp.ndarray, theta_a: jnp.ndarray,
    pos_b: jnp.ndarray, theta_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SAT overlap test for two square OBBs of side ROBOT_SIZE.

    Returns ``(overlapping_bool, normal_from_B_to_A, min_overlap)``.
    """
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    delta = pos_a - pos_b
    ca, sa = jnp.cos(theta_a), jnp.sin(theta_a)
    cb, sb = jnp.cos(theta_b), jnp.sin(theta_b)

    axes = jnp.stack([
        jnp.stack([ca, sa]),
        jnp.stack([-sa, ca]),
        jnp.stack([cb, sb]),
        jnp.stack([-sb, cb]),
    ])  # (4, 2)

    # Support of each square along each axis: half * (|u1·axis| + |u2·axis|).
    def support(axis, c, s):
        ax, ay = axis[0], axis[1]
        return half * (jnp.abs(c * ax + s * ay) + jnp.abs(-s * ax + c * ay))

    sup_a = jax.vmap(support, in_axes=(0, None, None))(axes, ca, sa)  # (4,)
    sup_b = jax.vmap(support, in_axes=(0, None, None))(axes, cb, sb)
    proj = axes @ delta  # (4,)
    dist = jnp.abs(proj)
    overlaps = sup_a + sup_b - dist  # (4,)

    overlapping = jnp.all(overlaps > 0)

    # Pick the axis with the minimum overlap (only meaningful when overlapping).
    min_idx = jnp.argmin(overlaps)
    min_overlap = overlaps[min_idx]
    axis = axes[min_idx]
    sign = jnp.where(proj[min_idx] >= 0, 1.0, -1.0)
    normal = axis * sign

    # When not overlapping, return safe zeros.
    normal = jnp.where(overlapping, normal, jnp.zeros(2, dtype=jnp.float32))
    min_overlap = jnp.where(overlapping, min_overlap, jnp.float32(0.0))
    return overlapping, normal, min_overlap


def _resolve_robot_pair(
    robots_flat: jnp.ndarray, i: jnp.ndarray, j: jnp.ndarray
) -> jnp.ndarray:
    e = jnp.float32(config.ROBOT_WALL_RESTITUTION)
    a = robots_flat[i]
    b = robots_flat[j]
    pos_a, theta_a, vel_a = a[0:2], a[2], a[3:5]
    pos_b, theta_b, vel_b = b[0:2], b[2], b[3:5]

    overlapping, normal, overlap = _sat_square_overlap(pos_a, theta_a, pos_b, theta_b)

    pa_new = pos_a + normal * overlap * 0.5
    pb_new = pos_b - normal * overlap * 0.5

    rel_vel = vel_a - vel_b
    vel_along = jnp.dot(rel_vel, normal)
    do_impulse = overlapping & (vel_along < 0)
    j_imp = -(1.0 + e) * vel_along * 0.5
    va_new = jnp.where(do_impulse, vel_a + j_imp * normal, vel_a)
    vb_new = jnp.where(do_impulse, vel_b - j_imp * normal, vel_b)

    pa_final = jnp.where(overlapping, pa_new, pos_a)
    pb_final = jnp.where(overlapping, pb_new, pos_b)

    new_a = a.at[0:2].set(pa_final).at[3:5].set(va_new)
    new_b = b.at[0:2].set(pb_final).at[3:5].set(vb_new)
    robots_flat = robots_flat.at[i].set(new_a).at[j].set(new_b)
    return robots_flat


def _robot_robot_collisions(state: SimState) -> SimState:
    """Resolve inelastic collisions between robots (OBB vs OBB)."""
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)

    def body(k, robots_flat):
        i = _PAIR_I[k]
        j = _PAIR_J[k]
        return _resolve_robot_pair(robots_flat, i, j)

    new_robots_flat = jax.lax.fori_loop(0, _N_PAIRS, body, robots_flat)
    return state._replace(
        robots=new_robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6)
    )


# ---------------------------------------------------------------------------
# Public step (jit-compiled)
# ---------------------------------------------------------------------------

def _substep(
    state: SimState,
    target_speeds: jnp.ndarray,
    sub_dt: jnp.ndarray,
    max_delta: jnp.ndarray,
) -> tuple[SimState, jnp.ndarray]:
    """One physics sub-step. Returns ``(new_state, goal_event)``."""
    # Slew current wheel speeds toward the commanded target (torque limit).
    delta = jnp.clip(target_speeds - state.wheel_speeds, -max_delta, max_delta)
    wheel_speeds = state.wheel_speeds + delta
    state = state._replace(wheel_speeds=wheel_speeds)

    v_l = wheel_speeds[:, :, 0]
    v_r = wheel_speeds[:, :, 1]
    theta = state.robots[:, :, 2]
    vx, vy, omega = _diff_drive(v_l, v_r, theta)

    robots = state.robots
    robots = robots.at[:, :, 3].set(vx)
    robots = robots.at[:, :, 4].set(vy)
    robots = robots.at[:, :, 5].set(omega)
    robots = robots.at[:, :, 0].add(vx * sub_dt)
    robots = robots.at[:, :, 1].add(vy * sub_dt)
    new_theta = robots[:, :, 2] + omega * sub_dt
    new_theta = (new_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    robots = robots.at[:, :, 2].set(new_theta)
    state = state._replace(robots=robots)

    # Ball friction (rolling).
    ball_vel = state.ball[2:4]
    speed = jnp.linalg.norm(ball_vel)
    safe_speed = jnp.where(speed > 1e-6, speed, 1.0)
    decel = jnp.minimum(speed, config.BALL_FRICTION * 9.81 * sub_dt)
    new_vel = ball_vel - (ball_vel / safe_speed) * decel
    new_vel = jnp.where(speed > 1e-6, new_vel, ball_vel)
    ball = state.ball.at[2:4].set(new_vel)
    ball = ball.at[0:2].add(new_vel * sub_dt)
    state = state._replace(ball=ball)

    # Collisions.
    state = _robot_wall_collisions(state)
    state, goal = _ball_wall_collisions(state)
    state = _ball_robot_collisions(state)
    state = _robot_robot_collisions(state)
    return state, goal


@partial(jax.jit, static_argnames=("sub_steps",))
def step(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = 4,
) -> tuple[SimState, dict]:
    """Advance the simulation by one control timestep (functional + jitted).

    Parameters
    ----------
    state : SimState
    actions : (N_TEAMS, N_ROBOTS, 2) jnp.float32 — normalised wheel speeds in [-1, 1].
    dt : float — control timestep (default ``config.DT``).
    sub_steps : int — physics sub-steps per control step (static, default 4).

    Returns
    -------
    new_state : SimState
    info : dict with ``"goal"`` (int32 scalar).
    """
    actions = jnp.clip(actions, -1.0, 1.0).astype(jnp.float32)
    target_speeds = actions * jnp.float32(config.ROBOT_MAX_WHEEL_SPEED)
    sub_dt = jnp.float32(dt / sub_steps)
    max_delta = jnp.float32(config.ROBOT_WHEEL_ACCEL_LIMIT) * sub_dt

    def body(_, carry):
        state, goal_acc = carry
        state, g = _substep(state, target_speeds, sub_dt, max_delta)
        goal_acc = jnp.where(goal_acc == 0, g, goal_acc)
        return state, goal_acc

    state, goal = jax.lax.fori_loop(
        0, sub_steps, body, (state, jnp.int32(0))
    )
    state = state._replace(t=state.t + jnp.float32(dt))
    return state, {"goal": goal}
