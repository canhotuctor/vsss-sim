"""
2-D physics engine for the VSSS simulator (NumPy backend).

Coordinate system
-----------------
- Origin at field centre.
- +x points toward TEAM_YELLOW's goal (right end-line).
- +y points upward (in the rendered top-down view).
- Angles measured from +x, counter-clockwise positive.

Team convention
---------------
- TEAM_BLUE  (index 0) defends the goal at x = -FIELD_LENGTH/2
  and attacks toward +x.
- TEAM_YELLOW (index 1) defends the goal at x = +FIELD_LENGTH/2
  and attacks toward -x.

State arrays
-----------
ball  : ndarray shape (4,)   [x, y, vx, vy]
robots: ndarray shape (N_TEAMS, N_ROBOTS, 6)
        axis-2 layout: [x, y, theta, vx, vy, omega]

Design note – GPU readiness
---------------------------
All physics operations use NumPy broadcastable array expressions.
Replacing ``np`` with ``torch`` (and adjusting ``np.linalg.norm``
calls) is sufficient to run batch environments on CUDA via PyTorch.
The Isaac Gym / PhysX path would wrap this module with a C++/CUDA
backend while preserving the same state-array layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import config

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """Mutable simulation state for a single environment instance."""

    ball: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    robots: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (config.N_TEAMS, config.N_ROBOTS, 6), dtype=np.float64
        )
    )
    score: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.int32)
    )
    t: float = 0.0

    def copy(self) -> "SimState":
        """Return a deep copy of this state."""
        return SimState(
            ball=self.ball.copy(),
            robots=self.robots.copy(),
            score=self.score.copy(),
            t=self.t,
        )


# ---------------------------------------------------------------------------
# Differential-drive kinematics  (vectorised over robots)
# ---------------------------------------------------------------------------

def _diff_drive(
    v_left: np.ndarray,
    v_right: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert wheel speeds to body velocities.

    Parameters
    ----------
    v_left, v_right : (...,) wheel speeds in m/s
    theta : (...,)  heading angles in radians

    Returns
    -------
    vx, vy, omega  (same shape as inputs)
    """
    v = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / config.ROBOT_WHEELBASE
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    return vx, vy, omega


# ---------------------------------------------------------------------------
# Wall collisions
# ---------------------------------------------------------------------------

def _ball_wall_collisions(state: SimState) -> int:
    """
    Reflect ball off field walls and detect goals.

    Returns
    -------
    goal : int
        +1 if TEAM_BLUE scored (ball entered yellow goal at +x),
        -1 if TEAM_YELLOW scored (ball entered blue goal at -x),
         0 otherwise.
    """
    bx, by, bvx, bvy = state.ball
    r = config.BALL_RADIUS
    half_l = config.FIELD_LENGTH / 2.0
    half_w = config.FIELD_WIDTH / 2.0
    half_goal = config.GOAL_WIDTH / 2.0

    # --- y walls (top / bottom of field) ---
    if by - r < -half_w:
        state.ball[1] = -half_w + r
        state.ball[3] = abs(bvy) * config.BALL_WALL_RESTITUTION
    elif by + r > half_w:
        state.ball[1] = half_w - r
        state.ball[3] = -abs(bvy) * config.BALL_WALL_RESTITUTION

    # --- x walls (end-lines) ---
    bx = state.ball[0]
    by = state.ball[1]

    if bx - r < -half_l:
        if abs(by) <= half_goal:
            # Ball in blue's goal → yellow scores; clamp to back of net
            state.ball[0] = -half_l - config.GOAL_DEPTH + r
            state.ball[2] = abs(state.ball[2]) * config.BALL_WALL_RESTITUTION
            return -1
        # Bounce off back wall
        state.ball[0] = -half_l + r
        state.ball[2] = abs(state.ball[2]) * config.BALL_WALL_RESTITUTION

    elif bx + r > half_l:
        if abs(by) <= half_goal:
            # Ball in yellow's goal → blue scores; clamp to back of net
            state.ball[0] = half_l + config.GOAL_DEPTH - r
            state.ball[2] = -abs(state.ball[2]) * config.BALL_WALL_RESTITUTION
            return 1
        # Bounce off back wall
        state.ball[0] = half_l - r
        state.ball[2] = -abs(state.ball[2]) * config.BALL_WALL_RESTITUTION

    return 0


def _robot_wall_collisions(state: SimState) -> None:
    """Clamp robots (OBB) inside field boundaries."""
    half_l = config.FIELD_LENGTH / 2.0
    half_w = config.FIELD_WIDTH / 2.0
    half = config.ROBOT_SIZE / 2.0

    # Per-robot OBB extent: max projection of a square onto world x / y axes
    theta = state.robots[:, :, 2]
    extent = half * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))  # (N_TEAMS, N_ROBOTS)

    lim_x = half_l - extent
    lim_y = half_w - extent

    x = state.robots[:, :, 0]
    y = state.robots[:, :, 1]

    exceeded_neg_x = x < -lim_x
    exceeded_pos_x = x > lim_x
    exceeded_neg_y = y < -lim_y
    exceeded_pos_y = y > lim_y

    state.robots[:, :, 0] = np.clip(x, -lim_x, lim_x)
    state.robots[:, :, 1] = np.clip(y, -lim_y, lim_y)

    state.robots[:, :, 3] = np.where(
        exceeded_neg_x & (state.robots[:, :, 3] < 0), 0.0, state.robots[:, :, 3]
    )
    state.robots[:, :, 3] = np.where(
        exceeded_pos_x & (state.robots[:, :, 3] > 0), 0.0, state.robots[:, :, 3]
    )
    state.robots[:, :, 4] = np.where(
        exceeded_neg_y & (state.robots[:, :, 4] < 0), 0.0, state.robots[:, :, 4]
    )
    state.robots[:, :, 4] = np.where(
        exceeded_pos_y & (state.robots[:, :, 4] > 0), 0.0, state.robots[:, :, 4]
    )


# ---------------------------------------------------------------------------
# Ball-robot collision (circle vs OBB)
# ---------------------------------------------------------------------------

def _ball_obb_penetration(
    ball_pos: np.ndarray,
    rob_pos: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, float]:
    """
    Return (normal, penetration) for a ball (circle) vs robot (square OBB).

    normal  – world-space unit vector pointing from robot toward ball.
    penetration > 0 means the shapes overlap by that depth.
    """
    half = config.ROBOT_SIZE / 2.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Ball centre in robot-local frame
    dx = ball_pos[0] - rob_pos[0]
    dy = ball_pos[1] - rob_pos[1]
    local_x =  cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    # Closest point on the AABB (local frame)
    clamp_x = max(-half, min(half, local_x))
    clamp_y = max(-half, min(half, local_y))

    diff_x = local_x - clamp_x
    diff_y = local_y - clamp_y
    dist = math.hypot(diff_x, diff_y)

    if dist < 1e-9:
        # Ball centre inside OBB: push out along shortest face
        pen_x = half - abs(local_x)
        pen_y = half - abs(local_y)
        if pen_x <= pen_y:
            lnx, lny = math.copysign(1.0, local_x), 0.0
            penetration = pen_x + config.BALL_RADIUS
        else:
            lnx, lny = 0.0, math.copysign(1.0, local_y)
            penetration = pen_y + config.BALL_RADIUS
    else:
        lnx, lny = diff_x / dist, diff_y / dist
        penetration = config.BALL_RADIUS - dist

    # Rotate normal back to world frame
    nx = cos_t * lnx - sin_t * lny
    ny = sin_t * lnx + cos_t * lny
    return np.array([nx, ny]), penetration


def _ball_robot_collisions(state: SimState) -> None:
    """Resolve elastic collisions between the ball and all robots (OBB)."""
    ball_pos = state.ball[0:2].copy()
    ball_vel = state.ball[2:4].copy()

    for team in range(config.N_TEAMS):
        for r_idx in range(config.N_ROBOTS):
            rob_pos = state.robots[team, r_idx, 0:2].copy()
            rob_vel = state.robots[team, r_idx, 3:5].copy()
            theta   = float(state.robots[team, r_idx, 2])

            normal, penetration = _ball_obb_penetration(ball_pos, rob_pos, theta)
            if penetration <= 0:
                continue

            m_b = config.BALL_MASS
            m_r = config.ROBOT_MASS
            total_m = m_b + m_r

            # Separate proportionally to mass
            ball_pos = ball_pos + normal * penetration * (m_r / total_m)
            rob_pos  = rob_pos  - normal * penetration * (m_b / total_m)

            # Impulse along collision normal
            rel_vel   = ball_vel - rob_vel
            vel_along = float(np.dot(rel_vel, normal))
            if vel_along >= 0:
                state.ball[0:2] = ball_pos
                state.robots[team, r_idx, 0:2] = rob_pos
                continue

            e = config.BALL_ROBOT_RESTITUTION
            j = -(1.0 + e) * vel_along / (1.0 / m_b + 1.0 / m_r)
            impulse  = j * normal
            ball_vel = ball_vel + impulse / m_b
            rob_vel  = rob_vel  - impulse / m_r

            state.ball[0:2] = ball_pos
            state.ball[2:4] = ball_vel
            state.robots[team, r_idx, 0:2] = rob_pos
            state.robots[team, r_idx, 3:5] = rob_vel

            ball_pos = state.ball[0:2]
            ball_vel = state.ball[2:4]


# ---------------------------------------------------------------------------
# Robot-robot collisions (OBB vs OBB via SAT)
# ---------------------------------------------------------------------------

def _sat_square_overlap(
    pos_a: np.ndarray, theta_a: float,
    pos_b: np.ndarray, theta_b: float,
) -> tuple[bool, np.ndarray, float]:
    """
    SAT overlap test for two square OBBs of side ROBOT_SIZE.

    Returns (overlapping, normal, min_overlap).
    normal points from B toward A (push-out direction for A).
    """
    half = config.ROBOT_SIZE / 2.0
    delta = pos_a - pos_b

    # Four candidate separating axes (local axes of each box)
    ca, sa = math.cos(theta_a), math.sin(theta_a)
    cb, sb = math.cos(theta_b), math.sin(theta_b)
    axes = (
        np.array([ ca,  sa]),
        np.array([-sa,  ca]),
        np.array([ cb,  sb]),
        np.array([-sb,  cb]),
    )

    min_overlap = float("inf")
    best_axis: np.ndarray = axes[0]

    for axis in axes:
        ax, ay = float(axis[0]), float(axis[1])
        # Support of each square along axis = half*(|u1·axis|+|u2·axis|)
        sup_a = half * (abs(ca * ax + sa * ay) + abs(-sa * ax + ca * ay))
        sup_b = half * (abs(cb * ax + sb * ay) + abs(-sb * ax + cb * ay))
        dist = abs(float(np.dot(delta, axis)))
        overlap = sup_a + sup_b - dist
        if overlap <= 0:
            return False, np.zeros(2), 0.0
        if overlap < min_overlap:
            min_overlap = overlap
            sign = 1.0 if float(np.dot(delta, axis)) >= 0 else -1.0
            best_axis = axis * sign

    return True, best_axis, min_overlap


def _robot_robot_collisions(state: SimState) -> None:
    """Resolve inelastic collisions between robots (OBB vs OBB)."""
    e = config.ROBOT_WALL_RESTITUTION
    n_total = config.N_TEAMS * config.N_ROBOTS
    robots_flat = state.robots.reshape(n_total, 6)

    for i in range(n_total - 1):
        for j in range(i + 1, n_total):
            pos_i   = robots_flat[i, 0:2].copy()
            pos_j   = robots_flat[j, 0:2].copy()
            theta_i = float(robots_flat[i, 2])
            theta_j = float(robots_flat[j, 2])
            vel_i   = robots_flat[i, 3:5].copy()
            vel_j   = robots_flat[j, 3:5].copy()

            overlapping, normal, overlap = _sat_square_overlap(
                pos_i, theta_i, pos_j, theta_j
            )
            if not overlapping:
                continue

            # Equal-mass separation
            robots_flat[i, 0:2] = pos_i + normal * overlap * 0.5
            robots_flat[j, 0:2] = pos_j - normal * overlap * 0.5

            rel_vel   = vel_i - vel_j
            vel_along = float(np.dot(rel_vel, normal))
            if vel_along >= 0:
                continue

            j_imp = -(1.0 + e) * vel_along * 0.5
            robots_flat[i, 3:5] = vel_i + j_imp * normal
            robots_flat[j, 3:5] = vel_j - j_imp * normal

    state.robots = robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6)


# ---------------------------------------------------------------------------
# Public step function
# ---------------------------------------------------------------------------

def step(
    state: SimState,
    actions: np.ndarray,
    dt: float = config.DT,
    sub_steps: int = 4,
) -> dict:
    """
    Advance the simulation by one control timestep.

    Parameters
    ----------
    state : SimState
        Mutable simulation state (modified in-place).
    actions : ndarray of shape (N_TEAMS, N_ROBOTS, 2)
        Normalised wheel speeds in [-1, 1] for each robot,
        axis-2 layout: [v_left, v_right].
        Pass zeros for an uncontrolled team.
    dt : float
        Control timestep (default: config.DT = 1/60 s).
    sub_steps : int
        Number of physics sub-steps per control step (increases
        collision accuracy at high speeds).

    Returns
    -------
    info : dict
        ``'goal'`` : +1 (blue scored), -1 (yellow scored), 0 (no goal).
    """
    wheel_speeds = np.clip(actions, -1.0, 1.0) * config.ROBOT_MAX_WHEEL_SPEED
    sub_dt = dt / sub_steps
    goal_result = 0

    for _ in range(sub_steps):
        # --- robot velocities via diff-drive kinematics ---
        v_l = wheel_speeds[:, :, 0]   # (N_TEAMS, N_ROBOTS)
        v_r = wheel_speeds[:, :, 1]
        theta = state.robots[:, :, 2]

        vx, vy, omega = _diff_drive(v_l, v_r, theta)
        state.robots[:, :, 3] = vx
        state.robots[:, :, 4] = vy
        state.robots[:, :, 5] = omega

        # --- integrate robots ---
        state.robots[:, :, 0] += vx * sub_dt
        state.robots[:, :, 1] += vy * sub_dt
        state.robots[:, :, 2] += omega * sub_dt

        # Wrap heading to [-π, π]
        state.robots[:, :, 2] = (
            (state.robots[:, :, 2] + math.pi) % (2.0 * math.pi) - math.pi
        )

        # --- ball friction (rolling) ---
        ball_vel = state.ball[2:4]
        speed = float(np.linalg.norm(ball_vel))
        if speed > 1e-6:
            decel = min(speed, config.BALL_FRICTION * 9.81 * sub_dt)
            state.ball[2:4] -= (ball_vel / speed) * decel

        # --- integrate ball ---
        state.ball[0:2] += state.ball[2:4] * sub_dt

        # --- collisions ---
        _robot_wall_collisions(state)
        g = _ball_wall_collisions(state)
        if g != 0 and goal_result == 0:
            goal_result = g
        _ball_robot_collisions(state)
        _robot_robot_collisions(state)

    state.t += dt
    return {"goal": goal_result}


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _default_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def reset_kickoff(
    state: SimState,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Place robots and ball for a standard kickoff.

    - Ball at field centre (0, 0).
    - Blue robots on the left half (x < 0), Yellow on the right half (x > 0).
    - All robots are at least KICKOFF_CLEAR_DIST from the centre.
    - Angles point roughly toward the ball.
    - All velocities zeroed.
    """
    rng = _default_rng(rng)
    state.ball[:] = 0.0
    state.robots[:] = 0.0

    half_l = config.FIELD_LENGTH / 2.0 - config.ROBOT_RADIUS
    clear = config.KICKOFF_CLEAR_DIST

    # Fixed starting positions with small random jitter
    blue_starts = np.array([
        [-0.55, 0.0],
        [-0.30, 0.30],
        [-0.30, -0.30],
    ], dtype=np.float64)

    yellow_starts = np.array([
        [0.55, 0.0],
        [0.30, -0.30],
        [0.30, 0.30],
    ], dtype=np.float64)

    jitter = rng.uniform(-0.05, 0.05, size=blue_starts.shape)
    blue_starts = np.clip(blue_starts + jitter, -half_l, half_l)
    blue_starts[:, 0] = np.clip(blue_starts[:, 0], -half_l, -clear)

    jitter = rng.uniform(-0.05, 0.05, size=yellow_starts.shape)
    yellow_starts = np.clip(yellow_starts + jitter, -half_l, half_l)
    yellow_starts[:, 0] = np.clip(yellow_starts[:, 0], clear, half_l)

    state.robots[config.TEAM_BLUE, :, 0:2] = blue_starts
    state.robots[config.TEAM_YELLOW, :, 0:2] = yellow_starts

    # Face toward the ball
    for team in range(config.N_TEAMS):
        for r in range(config.N_ROBOTS):
            dx = -state.robots[team, r, 0]
            dy = -state.robots[team, r, 1]
            state.robots[team, r, 2] = math.atan2(dy, dx)
