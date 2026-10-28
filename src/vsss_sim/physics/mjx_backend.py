"""MuJoCo MJX physics engine for vsss-sim.

An alternative to :mod:`jax_backend` with the same functional API: ``step``
and ``step_with_frames`` take and return :class:`SimState`, so match rules,
referee restarts, and observations work unchanged. Resets and the state
container are shared with the JAX engine.

The MuJoCo model is planar. Each robot is a box on slide-x, slide-y, and
yaw-hinge joints, driven by two wheel velocity actuators whose force limits
model wheel traction, plus a lateral actuator that resists sideways slip. The
ball is a sphere on two slide joints with rolling friction applied as a force.
Walls are planes, plus four boxes beside the goal mouths whose faces form the
end lines and goal-pocket rails. Contacts are frictionless, like the JAX
engine, and damped to reproduce the configured restitution.

Requires the optional ``mujoco-mjx`` dependency (``vsss-sim[mjx]``).
"""
from __future__ import annotations

import contextlib
import functools
import io
import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import mujoco

from .. import config
from .jax_backend import (
    SimState,
    _control_inputs,
    empty_state,
    reset_kickoff,
    reset_random,
)

# MJX 3.17 prints a notice when the optional NVIDIA Warp backend is missing.
# This engine always uses MJX's JAX implementation, so the notice is noise.
with contextlib.redirect_stdout(io.StringIO()):
    from mujoco import mjx

__all__ = [
    "MJXOptions",
    "DEFAULT_OPTIONS",
    "SimState",
    "build_mjcf",
    "empty_state",
    "reset_kickoff",
    "reset_random",
    "step",
    "step_with_frames",
]

# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MJXOptions:
    """Static MuJoCo solver settings. Each distinct value compiles separately.

    steps_per_substep : MuJoCo steps per physics sub-step. The MuJoCo timestep
        is ``dt / (sub_steps * steps_per_substep)``; 4 gives 240 Hz at the
        default 10 Hz control rate and 6 sub-steps.
    solver_iterations / ls_iterations : Newton solver and line-search limits.
    max_contact_points : keep only the deepest contacts each MuJoCo step;
        ``-1`` keeps all 366 candidate contacts.
    """

    steps_per_substep: int = 4
    solver_iterations: int = 4
    ls_iterations: int = 8
    max_contact_points: int = -1

    def __post_init__(self) -> None:
        for name in ("steps_per_substep", "solver_iterations", "ls_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        value = self.max_contact_points
        if isinstance(value, bool) or not isinstance(value, int) or value == 0 or value < -1:
            raise ValueError("max_contact_points must be a positive integer or -1")


DEFAULT_OPTIONS = MJXOptions()

# ---------------------------------------------------------------------------
# Physical parameters that the JAX engine does not need
# ---------------------------------------------------------------------------

_GRAVITY = 9.81
_N_ROBOTS_TOTAL = config.N_TEAMS * config.N_ROBOTS
_BALL_QPOS = 3 * _N_ROBOTS_TOTAL  # ball x, y follow the robots' x, y, yaw

# config.ROBOT_INERTIA does not match its documented solid-square formula for
# the current ROBOT_MASS, so the yaw inertia is derived from mass and size.
_ROBOT_YAW_INERTIA = config.ROBOT_MASS * 2.0 * config.ROBOT_SIZE**2 / 12.0

# Wheel traction: each wheel carries half the robot's weight. A coefficient of
# one matches ROBOT_WHEEL_ACCEL_LIMIT (1 g) and caps how hard robots can push.
_WHEEL_GRIP = config.ROBOT_WHEEL_ACCEL_LIMIT / _GRAVITY
_WHEEL_FORCE_LIMIT = _WHEEL_GRIP * config.ROBOT_MASS * _GRAVITY / 2.0
_LATERAL_FORCE_LIMIT = _WHEEL_GRIP * config.ROBOT_MASS * _GRAVITY
# Velocity-tracking time constant of the wheel and lateral grip actuators.
# Euler integration keeps their force limits exact (implicitfast would damp
# saturated actuators) and stays non-oscillatory while timestep < this.
_GRIP_TIME_CONSTANT = 0.01
_WHEEL_KV = config.ROBOT_MASS / (2.0 * _GRIP_TIME_CONSTANT)
_LATERAL_KV = config.ROBOT_MASS / _GRIP_TIME_CONSTANT

# Contacts are damped springs (MuJoCo direct solref). Each type is a
# dimensionless (phase, damping) pair for MuJoCo timestep h: stiffness
# (phase / h)^2 and damping coefficient damping / h. Behaviour then holds at
# any timestep, and contacts stiffen as it shrinks. MuJoCo applies contact
# damping explicitly, so damping above one would reverse the approach
# velocity within a single step.
_BALL_CONTACT_PHASE = 0.6  # about 6 mm of penetration per m/s at 240 Hz
# Box contacts last only a few steps, so these values were tuned in MJX at
# 0.4-1.3 m/s: robot-robot e = 0.15-0.27 for ROBOT_WALL_RESTITUTION (0.2),
# and robot-wall e < 0.1 to match the JAX engine's inelastic walls.
_ROBOT_CONTACT = (0.4, 0.5)
_WALL_CONTACT = (0.3, 0.9)
_CONTACT_SOLIMP = "0.9 0.95 0.001 0.5 2"

_WALL_THICKNESS = 0.15
_WALL_HEIGHT = 0.2

# 45° chamfer sign combinations for the 4 corners, as in the JAX engine.
_CHAMFER_SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _ball_contact(restitution: float) -> tuple[float, float]:
    """(phase, damping) of a linear spring contact with this restitution."""
    log_e = math.log(restitution)
    damping_ratio = -log_e / math.sqrt(math.pi**2 + log_e**2)
    return _BALL_CONTACT_PHASE, 2.0 * damping_ratio * _BALL_CONTACT_PHASE


def _solref(contact: tuple[float, float], timestep: float) -> str:
    """Direct (-stiffness, -damping) MuJoCo solref for one contact type."""
    phase, damping = contact
    return f"{-(phase / timestep) ** 2:.6g} {-damping / timestep:.6g}"


# ---------------------------------------------------------------------------
# MJCF model
# ---------------------------------------------------------------------------


def _robot_name(index: int) -> str:
    team = "blue" if index // config.N_ROBOTS == config.TEAM_BLUE else "yellow"
    return f"{team}{index % config.N_ROBOTS}"


def _wall_geoms() -> list[str]:
    """Static field geometry: side, goal-back, and chamfer planes; post boxes."""
    half_l = config.FIELD_LENGTH / 2.0
    half_w = config.FIELD_WIDTH / 2.0
    half_goal = config.GOAL_WIDTH / 2.0
    back = half_l + config.GOAL_DEPTH
    chamfer_diag = half_l + half_w - config.FIELD_CHAMFER

    # Planes are infinite half-spaces facing the field. None of them cuts the
    # field or the pockets: |y| <= half_w, |x| <= back, and |x| + |y| reaches
    # the chamfer limit only at the corners.
    planes = [
        ("wall_top", (0.0, half_w), (0, -1)),
        ("wall_bottom", (0.0, -half_w), (0, 1)),
        ("goal_back_right", (back, 0.0), (-1, 0)),
        ("goal_back_left", (-back, 0.0), (1, 0)),
    ]
    planes += [
        (f"chamfer{index}", (sx * chamfer_diag / 2, sy * chamfer_diag / 2), (-sx, -sy))
        for index, (sx, sy) in enumerate(_CHAMFER_SIGNS)
    ]
    geoms = [
        f'<geom name="{name}" type="plane" size="0 0 1" pos="{x} {y} 0" '
        f'zaxis="{nx} {ny} 0"/>'
        for name, (x, y), (nx, ny) in planes
    ]
    # Each box's inner face is the end line beside the mouth; its face at
    # |y| = half_goal is the pocket rail out to the goal-back plane.
    size_x = _WALL_THICKNESS / 2.0
    size_y = (half_w + _WALL_THICKNESS - half_goal) / 2.0
    center_x, center_y = half_l + size_x, half_goal + size_y
    for sx, side in ((1, "right"), (-1, "left")):
        for sy, end in ((1, "top"), (-1, "bottom")):
            geoms.append(
                f'<geom name="post_{side}_{end}" type="box" '
                f'pos="{sx * center_x} {sy * center_y} 0" '
                f'size="{size_x} {size_y} {_WALL_HEIGHT / 2}"/>'
            )
    return geoms


def build_mjcf(options: MJXOptions = DEFAULT_OPTIONS, timestep: float | None = None) -> str:
    """Return the MJCF document for the field, six robots, and the ball.

    ``timestep`` defaults to the MuJoCo timestep at the default control rate
    and sub-step count. Contact stiffness is derived from it.
    """
    if timestep is None:
        timestep = config.DT / (config.SUB_STEPS * options.steps_per_substep)
    half = config.ROBOT_SIZE / 2.0
    wheel_offset = config.ROBOT_WHEELBASE / 2.0
    r = config.BALL_RADIUS
    walls = _wall_geoms()
    wall_names = [geom.split('"')[1] for geom in walls]
    robots = [_robot_name(index) for index in range(_N_ROBOTS_TOTAL)]

    bodies, actuators = [], []
    for name in robots:
        bodies.append(
            f'<body name="{name}" pos="0 0 {half}">'
            f'<joint name="{name}_x" type="slide" axis="1 0 0"/>'
            f'<joint name="{name}_y" type="slide" axis="0 1 0"/>'
            f'<joint name="{name}_yaw" type="hinge" axis="0 0 1"/>'
            f'<inertial pos="0 0 0" mass="{config.ROBOT_MASS}" '
            f'diaginertia="{_ROBOT_YAW_INERTIA:.6g} {_ROBOT_YAW_INERTIA:.6g} '
            f'{_ROBOT_YAW_INERTIA:.6g}"/>'
            f'<geom name="{name}" type="box" size="{half} {half} {half}"/>'
            # The robot drives along its local +x; the left wheel is at +y.
            f'<site name="{name}_left" pos="0 {wheel_offset} 0"/>'
            f'<site name="{name}_right" pos="0 {-wheel_offset} 0"/>'
            f'<site name="{name}_center" pos="0 0 0"/>'
            "</body>"
        )
        for wheel in ("left", "right"):
            actuators.append(
                f'<velocity name="{name}_{wheel}" site="{name}_{wheel}" '
                f'gear="1 0 0 0 0 0" kv="{_WHEEL_KV:.6g}" forcelimited="true" '
                f'forcerange="{-_WHEEL_FORCE_LIMIT:.6g} {_WHEEL_FORCE_LIMIT:.6g}"/>'
            )
        actuators.append(
            f'<velocity name="{name}_lateral" site="{name}_center" '
            f'gear="0 1 0 0 0 0" kv="{_LATERAL_KV:.6g}" forcelimited="true" '
            f'forcerange="{-_LATERAL_FORCE_LIMIT:.6g} {_LATERAL_FORCE_LIMIT:.6g}"/>'
        )
    bodies.append(
        f'<body name="ball" pos="0 0 {r}">'
        '<joint name="ball_x" type="slide" axis="1 0 0"/>'
        '<joint name="ball_y" type="slide" axis="0 1 0"/>'
        f'<geom name="ball" type="sphere" size="{r}" mass="{config.BALL_MASS}"/>'
        "</body>"
    )

    def pair(geom1: str, geom2: str, contact: tuple[float, float]) -> str:
        return (
            f'<pair geom1="{geom1}" geom2="{geom2}" condim="1" '
            f'solref="{_solref(contact, timestep)}" solimp="{_CONTACT_SOLIMP}"/>'
        )

    # Explicit pairs set per-type restitution and skip impossible contacts.
    ball_robot = _ball_contact(config.BALL_ROBOT_RESTITUTION)
    pairs = []
    for index, name in enumerate(robots):
        pairs += [pair(name, wall, _WALL_CONTACT) for wall in wall_names]
        pairs += [pair(name, other, _ROBOT_CONTACT) for other in robots[index + 1:]]
        pairs.append(pair("ball", name, ball_robot))
    ball_wall = _ball_contact(config.BALL_WALL_RESTITUTION)
    pairs += [pair("ball", wall, ball_wall) for wall in wall_names]

    numeric = (
        f'<numeric name="max_contact_points" data="{options.max_contact_points}"/>'
        if options.max_contact_points > 0
        else ""
    )
    return (
        '<mujoco model="vsss_3v3">'
        '<compiler angle="radian"/>'
        f'<option timestep="{timestep:.9g}" gravity="0 0 0" integrator="Euler" '
        f'solver="Newton" iterations="{options.solver_iterations}" '
        f'ls_iterations="{options.ls_iterations}">'
        '<flag eulerdamp="disable"/>'
        "</option>"
        f"<custom>{numeric}</custom>"
        # Automatic contacts are disabled; only the explicit pairs collide.
        '<default><geom contype="0" conaffinity="0"/></default>'
        f"<worldbody>{''.join(walls)}{''.join(bodies)}</worldbody>"
        f"<contact>{''.join(pairs)}</contact>"
        f"<actuator>{''.join(actuators)}</actuator>"
        "</mujoco>"
    )


@functools.lru_cache(maxsize=None)
def _compiled_model(options: MJXOptions, timestep: float) -> tuple[mjx.Model, mjx.Data]:
    """Build the MJX model and a template ``Data`` once per configuration."""
    if timestep >= _GRIP_TIME_CONSTANT:
        raise ValueError(
            f"MuJoCo timestep {timestep:.4g} s must stay below the "
            f"{_GRIP_TIME_CONSTANT} s grip time constant; "
            "increase physics_substeps or MJXOptions.steps_per_substep"
        )
    mj_model = mujoco.MjModel.from_xml_string(build_mjcf(options, timestep))
    # The first call usually happens while tracing ``step``; the cache must
    # hold concrete arrays, not tracers from that trace.
    with jax.ensure_compile_time_eval():
        model = mjx.put_model(mj_model, impl="jax")
        return model, mjx.make_data(model)


# ---------------------------------------------------------------------------
# SimState <-> MuJoCo state
# ---------------------------------------------------------------------------


def _to_data(template: mjx.Data, state: SimState) -> mjx.Data:
    """Load positions and velocities from ``state`` into a fresh ``Data``."""
    robots = state.robots.reshape(_N_ROBOTS_TOTAL, 6)
    qpos = jnp.concatenate([robots[:, 0:3].reshape(-1), state.ball[0:2]])
    qvel = jnp.concatenate([robots[:, 3:6].reshape(-1), state.ball[2:4]])
    return template.replace(qpos=qpos, qvel=qvel)


def _to_state(state: SimState, data: mjx.Data, wheel_speeds: jnp.ndarray) -> SimState:
    """Read robot and ball state back from MuJoCo; wrap headings to [-pi, pi)."""
    qpos = data.qpos.astype(jnp.float32)
    qvel = data.qvel.astype(jnp.float32)
    pose = qpos[:_BALL_QPOS].reshape(_N_ROBOTS_TOTAL, 3)
    theta = (pose[:, 2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    pose = pose.at[:, 2].set(theta)
    robots = jnp.concatenate(
        [pose, qvel[:_BALL_QPOS].reshape(_N_ROBOTS_TOTAL, 3)], axis=1
    )
    ball = jnp.concatenate([qpos[_BALL_QPOS:], qvel[_BALL_QPOS:]])
    return state._replace(
        ball=ball,
        robots=robots.reshape(config.N_TEAMS, config.N_ROBOTS, 6),
        wheel_speeds=wheel_speeds,
    )


def _wheel_controls(wheel_speeds: jnp.ndarray) -> jnp.ndarray:
    """Actuator controls per robot: left, right, and zero lateral velocity."""
    wheels = wheel_speeds.reshape(_N_ROBOTS_TOTAL, 2)
    lateral = jnp.zeros((_N_ROBOTS_TOTAL, 1), dtype=wheels.dtype)
    return jnp.concatenate([wheels, lateral], axis=1).reshape(-1)


def _ball_rolling_friction(qvel: jnp.ndarray, timestep: float) -> jnp.ndarray:
    """Generalized force decelerating the ball at mu_r * g without reversing it."""
    ball_vel = qvel[_BALL_QPOS:]
    speed = jnp.linalg.norm(ball_vel)
    safe_speed = jnp.where(speed > 1e-6, speed, 1.0)
    decel = jnp.minimum(config.BALL_FRICTION * _GRAVITY, speed / timestep)
    force = -config.BALL_MASS * decel * ball_vel / safe_speed
    force = jnp.where(speed > 1e-6, force, 0.0)
    return jnp.zeros_like(qvel).at[_BALL_QPOS:].set(force)


def _goal_event(ball_pos: jnp.ndarray) -> jnp.ndarray:
    """+1 blue scores (+x goal), -1 yellow scores, 0 none (Rule 8.1).

    The whole ball must be past the goal line. Soft pocket rails let the ball
    sink slightly into them, so the mouth test uses the ball centre.
    """
    half_l = jnp.float32(config.FIELD_LENGTH / 2.0)
    half_goal = jnp.float32(config.GOAL_WIDTH / 2.0)
    r = jnp.float32(config.BALL_RADIUS)
    bx, by = ball_pos[0], ball_pos[1]
    in_mouth = jnp.abs(by) < half_goal
    right_goal = (bx - r > half_l) & in_mouth
    left_goal = (bx + r < -half_l) & in_mouth
    return jnp.where(
        right_goal, jnp.int32(1), jnp.where(left_goal, jnp.int32(-1), jnp.int32(0))
    )


# ---------------------------------------------------------------------------
# Public step (jit-compiled)
# ---------------------------------------------------------------------------


def _substep(
    model: mjx.Model,
    data: mjx.Data,
    wheel_speeds: jnp.ndarray,
    target_speeds: jnp.ndarray,
    max_delta: jnp.ndarray,
    steps: int,
    timestep: float,
) -> tuple[mjx.Data, jnp.ndarray, jnp.ndarray]:
    """One physics sub-step: slew wheels, then run ``steps`` MuJoCo steps."""
    # Same torque-limit slew as the JAX engine, so wheel_speeds keeps its meaning.
    delta = jnp.clip(target_speeds - wheel_speeds, -max_delta, max_delta)
    wheel_speeds = wheel_speeds + delta
    data = data.replace(ctrl=_wheel_controls(wheel_speeds))

    def mujoco_step(_, current):
        friction = _ball_rolling_friction(current.qvel, timestep)
        return mjx.step(model, current.replace(qfrc_applied=friction))

    data = jax.lax.fori_loop(0, steps, mujoco_step, data)
    return data, wheel_speeds, _goal_event(data.qpos[_BALL_QPOS:])


def _control_substep(
    model: mjx.Model,
    carry: tuple[mjx.Data, jnp.ndarray, jnp.ndarray],
    target_speeds: jnp.ndarray,
    max_delta: jnp.ndarray,
    steps: int,
    timestep: float,
) -> tuple[tuple[mjx.Data, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """One sub-step of a control interval; the state freezes after a goal."""
    data, wheel_speeds, goal_acc = carry
    data, wheel_speeds, g = jax.lax.cond(
        goal_acc == 0,
        lambda d, w: _substep(model, d, w, target_speeds, max_delta, steps, timestep),
        lambda d, w: (d, w, jnp.int32(0)),
        data,
        wheel_speeds,
    )
    goal_acc = jnp.where(goal_acc == 0, g, goal_acc)
    return (data, wheel_speeds, goal_acc), g


def _control_loop(
    state: SimState, actions: jnp.ndarray, dt: float, sub_steps: int, options: MJXOptions
):
    """Initial carry and a sub-step function for one control interval."""
    target_speeds, _, max_delta = _control_inputs(actions, dt, sub_steps)
    timestep = dt / (sub_steps * options.steps_per_substep)
    model, template = _compiled_model(options, timestep)

    def substep(carry):
        return _control_substep(
            model, carry, target_speeds, max_delta, options.steps_per_substep, timestep
        )

    carry = (_to_data(template, state), state.wheel_speeds, jnp.int32(0))
    return carry, substep


@partial(jax.jit, static_argnames=("dt", "sub_steps", "options"))
def step(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = config.SUB_STEPS,
    options: MJXOptions = DEFAULT_OPTIONS,
) -> tuple[SimState, dict]:
    """Advance the simulation by one control timestep (functional + jitted).

    Parameters
    ----------
    state : SimState
    actions : (N_TEAMS, N_ROBOTS, 2) jnp.float32 — normalised wheel speeds in [-1, 1].
    dt : float — control timestep (static, default ``config.DT``); it sets the
        MuJoCo timestep and contact stiffness, so it must be a Python number.
    sub_steps : int — physics sub-steps per control step (static, default 6).
    options : MJXOptions — static MuJoCo solver settings.

    Returns
    -------
    new_state : SimState
    info : dict with ``"goal"`` (int32 scalar).
    """
    carry, substep = _control_loop(state, actions, dt, sub_steps, options)
    data, wheel_speeds, goal = jax.lax.fori_loop(
        0, sub_steps, lambda _, current: substep(current)[0], carry
    )
    new_state = _to_state(state, data, wheel_speeds)
    return new_state._replace(t=state.t + jnp.float32(dt)), {"goal": goal}


@partial(jax.jit, static_argnames=("dt", "sub_steps", "options"))
def step_with_frames(
    state: SimState,
    actions: jnp.ndarray,
    dt: float = config.DT,
    sub_steps: int = config.SUB_STEPS,
    options: MJXOptions = DEFAULT_OPTIONS,
) -> tuple[SimState, dict, SimState]:
    """Advance one control timestep exactly like :func:`step`, keeping sub-steps.

    Returns
    -------
    new_state : SimState — identical to ``step``'s result.
    info : dict with ``"goal"`` (first goal, int32 scalar) and ``"goals"``
        (``(sub_steps,)`` per-sub-step goal events).
    frames : SimState whose arrays gain a leading ``sub_steps`` axis: the state
        after each sub-step. Sub-steps after a goal repeat the frozen state, and
        ``t`` advances only on ``new_state``.
    """
    carry, substep = _control_loop(state, actions, dt, sub_steps, options)

    def body(current, _):
        current, g = substep(current)
        return current, (_to_state(state, current[0], current[1]), g)

    (data, wheel_speeds, goal), (frames, goals) = jax.lax.scan(
        body, carry, None, length=sub_steps
    )
    new_state = _to_state(state, data, wheel_speeds)
    new_state = new_state._replace(t=state.t + jnp.float32(dt))
    return new_state, {"goal": goal, "goals": goals}, frames
