"""Pure-functional JAX physics engine for vsss-sim.

``SimState`` is a :class:`typing.NamedTuple` PyTree: register-free, immutable,
JIT-compatible, and ``vmap``-friendly. Physics arrays use ``float32`` so the
same implementation runs efficiently on CPU and accelerators.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import config

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
    ball_spin: jnp.ndarray     # (3,)   float32   [wx, wy, wz] angular velocity (rad/s)
    ball_vertical: jnp.ndarray # (2,)   float32   [z, vz] height above floor (m), vertical velocity


def _ball_at_rest() -> dict:
    """Zero spin and height for a ball lying on the floor."""
    return {
        "ball_spin": jnp.zeros(3, dtype=jnp.float32),
        "ball_vertical": jnp.zeros(2, dtype=jnp.float32),
    }


def empty_state() -> SimState:
    """Return an all-zero ``SimState``."""
    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32),
        robots=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32),
        score=jnp.zeros(2, dtype=jnp.int32),
        t=jnp.zeros((), dtype=jnp.float32),
        wheel_speeds=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32),
        **_ball_at_rest(),
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

def reset_kickoff(key: jnp.ndarray, possession_team=None) -> SimState:
    """Legal kickoff: a random coin toss initially, explicit possession after goals.

    Only the possessing team places a kicker inside the centre circle. Robot
    footprints of the defending team stay outside it, including placement jitter.
    """
    if possession_team is None:
        possession_team = jax.random.randint(jax.random.fold_in(key, 2), (), 0, config.N_TEAMS)
    key_a, key_d = jax.random.split(key)
    attacking = jnp.asarray([[-0.65, 0.0], [-0.12, 0.0], [-0.35, 0.35]])
    defending = jnp.asarray([[-0.65, 0.0], [-0.30, -0.30], [-0.30, 0.30]])
    attacking += jax.random.uniform(key_a, (3, 2), minval=-0.02, maxval=0.02)
    defending += jax.random.uniform(key_d, (3, 2), minval=-0.02, maxval=0.02)
    direction = jnp.where(possession_team == config.TEAM_BLUE, 1.0, -1.0)
    robots = jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 6), dtype=jnp.float32)
    robots = robots.at[possession_team, :, :2].set(direction * attacking)
    robots = robots.at[1 - possession_team, :, :2].set(-direction * defending)
    theta = jnp.arctan2(-robots[:, :, 1], -robots[:, :, 0])
    robots = robots.at[:, :, 2].set(theta)
    return SimState(
        ball=jnp.zeros(4, dtype=jnp.float32), robots=robots,
        score=jnp.zeros(2, dtype=jnp.int32), t=jnp.zeros((), dtype=jnp.float32),
        wheel_speeds=jnp.zeros((config.N_TEAMS, config.N_ROBOTS, 2), dtype=jnp.float32),
        **_ball_at_rest(),
    )


# ---------------------------------------------------------------------------
def reset_random(key: jnp.ndarray) -> SimState:
    """Place robots and ball at uniformly random positions anywhere on the field.

    All robots (both teams) sample from the full field; no half-restriction.
    - Ball: random anywhere within field bounds (margin from walls).
    - All robots: random x/y within field bounds (margin from walls), random headings.
    - No overlap-rejection; any initial interpenetration resolves in the first
      few physics sub-steps.
    """
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    margin = jnp.float32(config.ROBOT_SIZE)

    key_bx, key_by, key_x, key_y, key_t = jax.random.split(key, 5)

    # Ball: full field (same margin from walls as robots)
    ball_x = jax.random.uniform(key_bx, (), minval=-half_l + margin, maxval=half_l - margin)
    ball_y = jax.random.uniform(key_by, (), minval=-half_w + margin, maxval=half_w - margin)
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
        **_ball_at_rest(),
    )


# Robot–wall collisions (vectorised)
# ---------------------------------------------------------------------------

def _robot_wall_collisions(state: SimState) -> SimState:
    """Keep robot footprints inside the field and the two goal pockets."""
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half = jnp.float32(config.ROBOT_SIZE / 2.0)

    theta = state.robots[:, :, 2]
    extent = half * (jnp.abs(jnp.cos(theta)) + jnp.abs(jnp.sin(theta)))

    lim_y = half_w - extent

    x = state.robots[:, :, 0]
    y = state.robots[:, :, 1]
    vx = state.robots[:, :, 3]
    vy = state.robots[:, :, 4]

    # Once the centre is behind the goal line, the pocket side walls confine
    # the robot. At the entrance its entire footprint must fit between posts.
    half_goal = jnp.float32(config.GOAL_WIDTH / 2.0)
    in_pocket = (jnp.abs(x) > half_l) & (jnp.abs(y) < half_goal)
    pocket_y = half_goal - extent
    new_y = jnp.where(in_pocket, jnp.clip(y, -pocket_y, pocket_y),
                      jnp.clip(y, -lim_y, lim_y))
    fits_opening = jnp.abs(new_y) + extent <= half_goal
    lim_x = half_l - extent + jnp.where(in_pocket | fits_opening, config.GOAL_DEPTH, 0.0)
    # Side-wall contact applies only behind the goal line.
    effective_lim_y = jnp.where(in_pocket, pocket_y, lim_y)

    exceeded_neg_x = x < -lim_x
    exceeded_pos_x = x > lim_x
    exceeded_neg_y = y < -effective_lim_y
    exceeded_pos_y = y > effective_lim_y

    new_x = jnp.clip(x, -lim_x, lim_x)

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
# Ball spin — contact friction, floor contact, and flight
# ---------------------------------------------------------------------------

_BALL_K = config.BALL_INERTIA_FACTOR  # I / (m r²)
_UP = (0.0, 0.0, 1.0)
_NO_VELOCITY = (0.0, 0.0, 0.0)


def _contact_friction(
    velocity: jnp.ndarray,
    spin: jnp.ndarray,
    normal: jnp.ndarray,
    surface_velocity: jnp.ndarray,
    normal_impulse: jnp.ndarray,
    friction: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Coulomb friction where a surface touches the ball, as velocity changes.

    ``normal`` is the unit 3D contact normal pointing into the ball and
    ``normal_impulse`` the normal impulse per unit ball mass (m/s). Friction
    opposes the slip of the touching points and at most stops it, so it never
    adds energy. Returns ``(velocity change, spin change, slip stopped)``.
    """
    r = jnp.float32(config.BALL_RADIUS)
    normal = jnp.asarray(normal, dtype=jnp.float32)
    offset = -r * normal  # contact point relative to the ball centre
    slip = velocity + jnp.cross(spin, offset) - jnp.asarray(surface_velocity, jnp.float32)
    slip = slip - jnp.dot(slip, normal) * normal
    slip_speed = jnp.linalg.norm(slip)
    # A tangential impulse p per unit mass changes the slip by p·(1 + 1/k).
    stopping = slip_speed / (1.0 + 1.0 / _BALL_K)
    available = friction * normal_impulse
    impulse = jnp.minimum(available, stopping)
    dv = -impulse * slip / jnp.where(slip_speed > 1e-9, slip_speed, 1.0)
    dspin = jnp.cross(offset, dv) / (_BALL_K * r * r)
    return dv, dspin, stopping <= available


def _ball_face_impulse(
    velocity: jnp.ndarray,
    spin: jnp.ndarray,
    normal: jnp.ndarray,
    surface_velocity: jnp.ndarray,
    restitution: float,
    friction: float,
    mass_share: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Velocity and spin changes when a vertical face strikes the ball.

    ``normal`` points from the face into the ball. ``mass_share`` scales the
    normal impulse for a movable face: m_face / (m_ball + m_face). Friction
    treats the face as immovable. Both changes are zero unless approaching.
    """
    normal = jnp.asarray(normal, dtype=jnp.float32)
    approach = jnp.dot(velocity - jnp.asarray(surface_velocity, jnp.float32), normal)
    normal_impulse = jnp.where(
        approach < 0.0, -(1.0 + restitution) * approach * mass_share, 0.0
    )
    dv, dspin, _ = _contact_friction(
        velocity, spin, normal, surface_velocity, normal_impulse, friction
    )
    return dv + normal_impulse * normal, dspin


def _wall_bounce(
    velocity: jnp.ndarray, spin: jnp.ndarray, hit: jnp.ndarray, normal: tuple
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a static wall's restitution and friction where ``hit`` is set."""
    dv, dspin = _ball_face_impulse(
        velocity, spin, normal, _NO_VELOCITY,
        config.BALL_WALL_RESTITUTION, config.BALL_WALL_FRICTION,
    )
    return velocity + jnp.where(hit, dv, 0.0), spin + jnp.where(hit, dspin, 0.0)


def _ball_floor(state: SimState, sub_dt: jnp.ndarray) -> SimState:
    """Advance the ball's floor contact, or its flight, by one sub-step.

    On the floor, sliding friction acts on the contact point until the ball
    rolls; a rolling ball then slows by rolling resistance. A ball knocked
    upward flies under gravity and bounces when it lands.
    """
    g = jnp.float32(config.GRAVITY)
    r = jnp.float32(config.BALL_RADIUS)
    z, vz = state.ball_vertical[0], state.ball_vertical[1]

    grounded = (z <= 0.0) & (vz <= 0.0)
    # Flights last only a few sub-steps, so follow the exact ballistic arc.
    z_air = z + vz * sub_dt - 0.5 * g * sub_dt * sub_dt
    vz_air = vz - g * sub_dt
    lands = ~grounded & (z_air <= 0.0)
    impact = -jnp.sqrt(jnp.maximum(vz * vz + 2.0 * g * z, 0.0))
    bounce = -jnp.float32(config.BALL_FLOOR_RESTITUTION) * impact
    # A bounce whose flight would not outlast one sub-step settles instead.
    bounce = jnp.where(bounce > 0.5 * g * sub_dt, bounce, 0.0)
    on_floor = grounded | lands
    new_vz = jnp.where(grounded, 0.0, jnp.where(lands, bounce, vz_air))
    new_z = jnp.where(on_floor, 0.0, z_air)
    # Floor normal impulse per unit mass: supporting the weight, or a landing.
    normal_impulse = jnp.where(
        grounded, g * sub_dt, jnp.where(lands, bounce - impact, 0.0)
    )

    planar = jnp.stack([state.ball[2], state.ball[3], jnp.float32(0.0)])
    dv, dspin, slip_stopped = _contact_friction(
        planar, state.ball_spin, _UP, _NO_VELOCITY, normal_impulse,
        config.BALL_SLIDING_FRICTION,
    )
    velocity = planar[:2] + dv[:2]
    spin = state.ball_spin + dspin

    # A rolling ball slows by rolling resistance, and its spin follows.
    rolling = on_floor & slip_stopped
    speed = jnp.linalg.norm(velocity)
    decel = jnp.minimum(speed, config.BALL_FRICTION * g * sub_dt)
    rolled = velocity - velocity / jnp.where(speed > 1e-6, speed, 1.0) * decel
    velocity = jnp.where(rolling, rolled, velocity)
    rolling_spin = spin.at[0].set(-velocity[1] / r).at[1].set(velocity[0] / r)
    spin = jnp.where(rolling, rolling_spin, spin)
    # Floor contact also slows spin about the vertical axis.
    wz = spin[2]
    slowed = wz - jnp.sign(wz) * jnp.minimum(jnp.abs(wz), config.BALL_SPIN_DECEL * sub_dt)
    spin = spin.at[2].set(jnp.where(on_floor, slowed, wz))

    return state._replace(
        ball=state.ball.at[2:4].set(velocity),
        ball_spin=spin,
        ball_vertical=jnp.stack([new_z, new_vz]),
    )


# ---------------------------------------------------------------------------
# Chamfer collisions — shared geometry helpers
# ---------------------------------------------------------------------------

# 45° chamfer sign combinations for the 4 corners: (sx, sy).
# The chamfer half-plane for corner (sx, sy) is: sx*x + sy*y ≤ chamfer_diag.
_CHAMFER_SX = (1., -1.,  1., -1.)
_CHAMFER_SY = (1.,  1., -1., -1.)
_INV_SQRT2 = float(0.5 ** 0.5)


def _ball_chamfer_collisions(state: SimState) -> SimState:
    """Bounce the ball off the four 45° corner chamfers, with wall friction.

    Each chamfer is a half-plane  sx·x + sy·y ≤ D − r·√2  where
    D = half_l + half_w − chamfer and (sx,sy) ∈ {±1}².
    """
    r = jnp.float32(config.BALL_RADIUS)
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    c = jnp.float32(config.FIELD_CHAMFER)

    chamfer_diag = half_l + half_w - c
    # Limit in the "sum space" (sx·bx + sy·by must stay below this).
    limit = chamfer_diag - r * jnp.float32(2.0 ** 0.5)

    bx, by = state.ball[0], state.ball[1]
    velocity = jnp.stack([state.ball[2], state.ball[3], state.ball_vertical[1]])
    spin = state.ball_spin

    for sx, sy in zip(_CHAMFER_SX, _CHAMFER_SY):
        sx_f = jnp.float32(sx)
        sy_f = jnp.float32(sy)

        pen = sx_f * bx + sy_f * by - limit
        hit = pen > 0.0

        # Push ball back along the chamfer normal n = (-sx, -sy)/√2.
        # Displacement = (pen/√2) · n = pen · (-sx, -sy) / 2.
        bx = jnp.where(hit, bx - sx_f * pen * 0.5, bx)
        by = jnp.where(hit, by - sy_f * pen * 0.5, by)
        velocity, spin = _wall_bounce(
            velocity, spin, hit, (-sx * _INV_SQRT2, -sy * _INV_SQRT2, 0.0)
        )

    return state._replace(
        ball=jnp.stack([bx, by, velocity[0], velocity[1]]),
        ball_spin=spin,
        ball_vertical=state.ball_vertical.at[1].set(velocity[2]),
    )


def _robot_chamfer_collisions(state: SimState) -> SimState:
    """Clamp robots (OBB) inside the four 45° corner chamfers; zero outward velocity.

    For each chamfer half-plane sx·x + sy·y ≤ D, the effective limit for the
    robot centre is D − support(OBB, (sx,sy)/√2), where
      support = half · (|sx·cos θ + sy·sin θ| + |sy·cos θ − sx·sin θ|).
    """
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_w = jnp.float32(config.FIELD_WIDTH / 2.0)
    half = jnp.float32(config.ROBOT_SIZE / 2.0)
    c = jnp.float32(config.FIELD_CHAMFER)
    inv_sqrt2 = jnp.float32(_INV_SQRT2)

    chamfer_diag = half_l + half_w - c

    rx = state.robots[:, :, 0]
    ry = state.robots[:, :, 1]
    theta = state.robots[:, :, 2]
    vx = state.robots[:, :, 3]
    vy = state.robots[:, :, 4]

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    for sx, sy in zip(_CHAMFER_SX, _CHAMFER_SY):
        sx_f = jnp.float32(sx)
        sy_f = jnp.float32(sy)

        # OBB support in direction (sx, sy)/√2 — expressed in "sum space" (×√2).
        support_sum = half * (
            jnp.abs(sx_f * cos_t + sy_f * sin_t)
            + jnp.abs(sy_f * cos_t - sx_f * sin_t)
        )
        limit = chamfer_diag - support_sum

        pen = sx_f * rx + sy_f * ry - limit
        hit = pen > 0.0

        rx = jnp.where(hit, rx - sx_f * pen * 0.5, rx)
        ry = jnp.where(hit, ry - sy_f * pen * 0.5, ry)

        # Zero the velocity component pointing into the corner.
        v_out = (sx_f * vx + sy_f * vy) * inv_sqrt2
        do_zero = hit & (v_out > 0.0)
        vx = jnp.where(do_zero, vx - v_out * sx_f * inv_sqrt2, vx)
        vy = jnp.where(do_zero, vy - v_out * sy_f * inv_sqrt2, vy)

    robots = state.robots.at[:, :, 0].set(rx).at[:, :, 1].set(ry)
    robots = robots.at[:, :, 3].set(vx).at[:, :, 4].set(vy)
    return state._replace(robots=robots)


# ---------------------------------------------------------------------------
# Ball–wall collisions and goal detection
# ---------------------------------------------------------------------------

def _ball_wall_collisions(state: SimState) -> tuple[SimState, jnp.ndarray]:
    """Bounce the ball off field walls, with wall friction, and detect goals.

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

    bx, by = state.ball[0], state.ball[1]
    velocity = jnp.stack([state.ball[2], state.ball[3], state.ball_vertical[1]])
    spin = state.ball_spin

    # --- y walls (top / bottom of field) ---
    hit_bot = by - r < -half_w
    hit_top = by + r > half_w
    by = jnp.where(hit_bot, -half_w + r, by)
    by = jnp.where(hit_top, half_w - r, by)
    velocity, spin = _wall_bounce(velocity, spin, hit_bot, (0.0, 1.0, 0.0))
    velocity, spin = _wall_bounce(velocity, spin, hit_top, (0.0, -1.0, 0.0))

    # The mouth is open while a ball is straddling the goal line. Do not
    # bounce it off that line before its trailing edge can cross (Rule 8.1).
    in_goal_y = jnp.abs(by) + r <= half_goal + 1e-7
    left_goal = (bx + r < -half_l) & in_goal_y
    right_goal = (bx - r > half_l) & in_goal_y

    # At the goal line, the portions outside the 40 cm mouth act as posts.
    # Only resolve this front-wall contact while the ball centre is still on
    # the field side; once inside, the pocket's side rails handle contact.
    hit_left_front = (
        (bx >= -half_l) & (bx - r < -half_l) & ~in_goal_y
    )
    hit_right_front = (
        (bx <= half_l) & (bx + r > half_l) & ~in_goal_y
    )
    bx = jnp.where(hit_left_front, -half_l + r, bx)
    bx = jnp.where(hit_right_front, half_l - r, bx)
    velocity, spin = _wall_bounce(velocity, spin, hit_left_front, (1.0, 0.0, 0.0))
    velocity, spin = _wall_bounce(velocity, spin, hit_right_front, (-1.0, 0.0, 0.0))

    # The 10 cm pocket has side rails from the goal line to the back wall.
    # Constrain the ball by its radius and reflect it away from either rail.
    overlaps_pocket_depth = (
        ((bx + r > half_l) & (bx - r < half_l + goal_depth))
        | ((bx - r < -half_l) & (bx + r > -half_l - goal_depth))
    )
    hit_upper_rail = overlaps_pocket_depth & (by + r > half_goal)
    hit_lower_rail = overlaps_pocket_depth & (by - r < -half_goal)
    by = jnp.where(hit_upper_rail, half_goal - r, by)
    by = jnp.where(hit_lower_rail, -half_goal + r, by)
    velocity, spin = _wall_bounce(velocity, spin, hit_upper_rail, (0.0, -1.0, 0.0))
    velocity, spin = _wall_bounce(velocity, spin, hit_lower_rail, (0.0, 1.0, 0.0))

    # Rear goal walls sit exactly GOAL_DEPTH beyond each end line.
    in_goal_y = jnp.abs(by) + r <= half_goal + 1e-7
    left_back = (bx - r < -half_l - goal_depth) & in_goal_y
    right_back = (bx + r > half_l + goal_depth) & in_goal_y
    bx = jnp.where(left_back, -half_l - goal_depth + r, bx)
    bx = jnp.where(right_back, half_l + goal_depth - r, bx)
    velocity, spin = _wall_bounce(velocity, spin, left_back, (1.0, 0.0, 0.0))
    velocity, spin = _wall_bounce(velocity, spin, right_back, (-1.0, 0.0, 0.0))

    goal = jnp.where(
        right_goal,
        jnp.int32(1),
        jnp.where(left_goal, jnp.int32(-1), jnp.int32(0)),
    )

    new_state = state._replace(
        ball=jnp.stack([bx, by, velocity[0], velocity[1]]),
        ball_spin=spin,
        ball_vertical=state.ball_vertical.at[1].set(velocity[2]),
    )
    return new_state, goal


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
    """Resolve ball collisions, with face friction, against all 6 robots in parallel.

    All penetrations are computed simultaneously via vmap, then corrections are
    summed. Slightly less accurate than sequential resolution but equivalent
    throughput-wise on GPU and correct enough for RL over config.SUB_STEPS sub-steps.
    (config.SUB_STEPS is 6 by default, but may be greater for higher accuracy.)
    """
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)
    m_b = jnp.float32(config.BALL_MASS)
    m_r = jnp.float32(config.ROBOT_MASS)
    total_m = m_b + m_r

    normals, penetrations = jax.vmap(_ball_obb_penetration, in_axes=(None, 0, 0))(
        state.ball[0:2], robots_flat[:, 0:2], robots_flat[:, 2]
    )  # normals: (6, 2), penetrations: (6,)

    collide = penetrations > 0.0  # (6,)

    # Ball position: sum push-out from all colliding robots
    ball_pos_delta = (normals * penetrations[:, None] * (m_r / total_m))
    ball_pos_delta = jnp.where(collide[:, None], ball_pos_delta, 0.0)
    ball_pos_new = state.ball[0:2] + ball_pos_delta.sum(axis=0)

    # Robot positions: each gets its own push-back
    rob_pos_delta = (-normals * penetrations[:, None] * (m_b / total_m))
    rob_pos_new = robots_flat[:, 0:2] + jnp.where(collide[:, None], rob_pos_delta, 0.0)

    # Velocity impulses, with friction. Each face moves with its robot,
    # including the robot's spin, at the point touching the ball.
    ball_vel = jnp.stack([state.ball[2], state.ball[3], state.ball_vertical[1]])
    arm = state.ball[0:2] - normals * jnp.float32(config.BALL_RADIUS) - robots_flat[:, 0:2]
    surface = robots_flat[:, 3:5] + robots_flat[:, 5:6] * jnp.stack(
        [-arm[:, 1], arm[:, 0]], axis=-1
    )
    flat = jnp.zeros((_N_ROBOTS_TOTAL, 1), dtype=jnp.float32)
    normals_3d = jnp.concatenate([normals, flat], axis=-1)
    dv, dspin = jax.vmap(_ball_face_impulse, in_axes=(None, None, 0, 0, None, None, None))(
        ball_vel, state.ball_spin, normals_3d, jnp.concatenate([surface, flat], axis=-1),
        config.BALL_ROBOT_RESTITUTION, config.BALL_ROBOT_FRICTION, m_r / total_m,
    )
    dv = jnp.where(collide[:, None], dv, 0.0)
    dspin = jnp.where(collide[:, None], dspin, 0.0)
    ball_vel_new = ball_vel + dv.sum(axis=0)

    # Each robot takes the opposite normal impulse.
    normal_dv = jnp.sum(dv * normals_3d, axis=-1)
    rob_vel_new = robots_flat[:, 3:5] - normal_dv[:, None] * normals * (m_b / m_r)

    new_robots_flat = robots_flat.at[:, 0:2].set(rob_pos_new).at[:, 3:5].set(rob_vel_new)
    return state._replace(
        ball=jnp.concatenate([ball_pos_new, ball_vel_new[:2]]),
        robots=new_robots_flat.reshape(config.N_TEAMS, config.N_ROBOTS, 6),
        ball_spin=state.ball_spin + dspin.sum(axis=0),
        ball_vertical=state.ball_vertical.at[1].set(ball_vel_new[2]),
    )


# ---------------------------------------------------------------------------
# Robot–robot collisions (OBB vs OBB via SAT) — dense masked contacts
# ---------------------------------------------------------------------------

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
    selected = jnp.arange(4) == min_idx
    min_overlap = jnp.sum(jnp.where(selected, overlaps, 0.0))
    axis = jnp.sum(jnp.where(selected[:, None], axes, 0.0), axis=0)
    selected_proj = jnp.sum(jnp.where(selected, proj, 0.0))
    sign = jnp.where(selected_proj >= 0, 1.0, -1.0)
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
    """Resolve inelastic collisions between all 15 robot pairs in parallel.

    A dense contact tensor and row/column reductions accumulate each robot's
    simultaneous corrections without gathering pairs or scatter-adding them.
    """
    robots_flat = state.robots.reshape(_N_ROBOTS_TOTAL, 6)
    e = jnp.float32(config.ROBOT_WALL_RESTITUTION)
    positions = robots_flat[:, 0:2]
    velocities = robots_flat[:, 3:5]

    # Pack only the 15 i < j contacts using static slices. Evaluating the full
    # 6x6 grid would duplicate SAT work and reverse tied/coincident contact axes.
    first = jnp.concatenate(
        [
            jnp.broadcast_to(robots_flat[index:index + 1], (_N_ROBOTS_TOTAL - index - 1, 6))
            for index in range(_N_ROBOTS_TOTAL - 1)
        ],
        axis=0,
    )
    second = jnp.concatenate(
        [robots_flat[index + 1:] for index in range(_N_ROBOTS_TOTAL - 1)], axis=0
    )
    overlapping, normals, overlaps = jax.vmap(_sat_square_overlap)(
        first[:, :2], first[:, 2], second[:, :2], second[:, 2]
    )

    # Position corrections
    sep = jnp.where(overlapping[:, None], normals * overlaps[:, None] * 0.5, 0.0)

    # Velocity impulses
    rel_vel = first[:, 3:5] - second[:, 3:5]
    vel_along = jnp.sum(rel_vel * normals, axis=-1)
    do_impulse = overlapping & (vel_along < 0)
    j_imp = jnp.where(do_impulse, -(1.0 + e) * vel_along * 0.5, 0.0)
    impulse = j_imp[:, None] * normals

    def contact_matrix(deltas: jnp.ndarray) -> jnp.ndarray:
        """Expand packed contacts into upper-triangular rows without scatters."""
        rows = []
        offset = 0
        for index in range(_N_ROBOTS_TOTAL):
            width = _N_ROBOTS_TOTAL - index - 1
            rows.append(
                jnp.pad(deltas[offset:offset + width], ((index + 1, 0), (0, 0)))
            )
            offset += width
        return jnp.stack(rows)

    sep = contact_matrix(sep)
    impulse = contact_matrix(impulse)

    # Each upper-triangular contact contributes +delta to its row robot and
    # -delta to its column robot, with no diagonal or duplicate contacts.
    new_pos = positions + jnp.sum(sep, axis=1) - jnp.sum(sep, axis=0)
    new_vel = velocities + jnp.sum(impulse, axis=1) - jnp.sum(impulse, axis=0)

    new_robots_flat = jnp.concatenate(
        (new_pos, robots_flat[:, 2:3], new_vel, robots_flat[:, 5:6]), axis=1
    )
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

    # Ball floor contact (skidding, rolling) or flight, then motion.
    state = _ball_floor(state, sub_dt)
    state = state._replace(ball=state.ball.at[0:2].add(state.ball[2:4] * sub_dt))

    # Collisions.
    state = _robot_wall_collisions(state)
    state = _robot_chamfer_collisions(state)
    state, goal = _ball_wall_collisions(state)
    state = _ball_chamfer_collisions(state)
    state = _ball_robot_collisions(state)
    state = _robot_robot_collisions(state)
    # The floor absorbs any downward push from wall or robot friction.
    z, vz = state.ball_vertical[0], state.ball_vertical[1]
    vz = jnp.where(z <= 0.0, jnp.maximum(vz, 0.0), vz)
    return state._replace(ball_vertical=jnp.stack([z, vz])), goal


def _control_inputs(
    actions: jnp.ndarray, dt: float, sub_steps: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Wheel targets, sub-step duration, and slew limit for one control step."""
    actions = jnp.clip(actions, -1.0, 1.0).astype(jnp.float32)
    target_speeds = actions * jnp.float32(config.ROBOT_MAX_WHEEL_SPEED)
    sub_dt = jnp.float32(dt / sub_steps)
    max_delta = jnp.float32(config.ROBOT_WHEEL_ACCEL_LIMIT) * sub_dt
    return target_speeds, sub_dt, max_delta


def _control_substep(
    carry: tuple[SimState, jnp.ndarray],
    target_speeds: jnp.ndarray,
    sub_dt: jnp.ndarray,
    max_delta: jnp.ndarray,
) -> tuple[tuple[SimState, jnp.ndarray], jnp.ndarray]:
    """One sub-step of a control interval; the state freezes after a goal."""
    state, goal_acc = carry
    state, g = jax.lax.cond(
        goal_acc == 0,
        lambda current: _substep(current, target_speeds, sub_dt, max_delta),
        lambda current: (current, jnp.int32(0)),
        state,
    )
    goal_acc = jnp.where(goal_acc == 0, g, goal_acc)
    return (state, goal_acc), g


@partial(jax.jit, static_argnames=("sub_steps",))
def step(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = config.SUB_STEPS,
) -> tuple[SimState, dict]:
    """Advance the simulation by one control timestep (functional + jitted).

    Parameters
    ----------
    state : SimState
    actions : (N_TEAMS, N_ROBOTS, 2) jnp.float32 — normalised wheel speeds in [-1, 1].
    dt : float — control timestep (default ``config.DT``).
    sub_steps : int — physics sub-steps per control step (static, default 6).

    Returns
    -------
    new_state : SimState
    info : dict with ``"goal"`` (int32 scalar).
    """
    inputs = _control_inputs(actions, dt, sub_steps)

    def body(_, carry):
        carry, _ = _control_substep(carry, *inputs)
        return carry

    state, goal = jax.lax.fori_loop(
        0, sub_steps, body, (state, jnp.int32(0))
    )
    state = state._replace(t=state.t + jnp.float32(dt))
    return state, {"goal": goal}


@partial(jax.jit, static_argnames=("sub_steps",))
def step_with_frames(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = config.SUB_STEPS,
) -> tuple[SimState, dict, SimState]:
    """Advance one control timestep exactly like :func:`step`, keeping sub-steps.

    Rendering uses this to draw every intermediate state from one engine call.

    Returns
    -------
    new_state : SimState — identical to ``step``'s result.
    info : dict with ``"goal"`` (first goal, int32 scalar) and ``"goals"``
        (``(sub_steps,)`` per-sub-step goal events).
    frames : SimState whose arrays gain a leading ``sub_steps`` axis: the state
        after each sub-step. Sub-steps after a goal repeat the frozen state, and
        ``t`` advances only on ``new_state``.
    """
    inputs = _control_inputs(actions, dt, sub_steps)

    def body(carry, _):
        carry, g = _control_substep(carry, *inputs)
        return carry, (carry[0], g)

    (state, goal), (frames, goals) = jax.lax.scan(
        body, (state, jnp.int32(0)), None, length=sub_steps
    )
    state = state._replace(t=state.t + jnp.float32(dt))
    return state, {"goal": goal, "goals": goals}, frames
