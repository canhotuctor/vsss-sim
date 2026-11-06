"""Benchmark batched collision physics with ordinary or contact-heavy starts."""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from vsss_sim import config as c
from vsss_sim.physics import jax_backend as p

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=200)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--substeps", type=int, default=6)
parser.add_argument("--repeats", type=int, default=7)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--scenario", choices=["kickoff", "contact"], default="kickoff")
parser.add_argument("--drive", choices=c.ROBOT_DRIVE_MODELS, default=c.ROBOT_DRIVE_MODEL)
parser.add_argument(
    "--restart-each-chunk",
    action="store_true",
    help="Replay the exact same 20-step starting batch each repetition",
)
args = parser.parse_args()
B, STEPS = args.envs, args.steps
print(
    f"jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()} batch={B} rollout_chunk={STEPS} substeps={args.substeps} drive={args.drive} scenario={args.scenario} seed={args.seed}",
    flush=True,
)
initial = jax.vmap(p.reset_kickoff)(jax.random.split(jax.random.PRNGKey(args.seed), B))
rng = np.random.default_rng(args.seed)
actions = jnp.asarray(rng.uniform(-1, 1, (STEPS, B, c.N_TEAMS, c.N_ROBOTS, 2)).astype(np.float32))
if args.scenario == "contact":
    positions = jnp.array(
        [[-0.5, -0.4], [-0.25, -0.4], [0.0, -0.4], [0.25, -0.4], [0.5, -0.4], [0.0, 0.4]],
        dtype=jnp.float32,
    )
    robots = jnp.broadcast_to(initial.robots[:1], initial.robots.shape).reshape(B, 6, 6)
    robots = robots.at[:, :, :2].set(positions)
    robots = robots.at[:, :, 3:].set(0.0)
    # Equal mix of a ball under sustained wall pressure and an opposing-robot pinch.
    wall_y = c.FIELD_WIDTH / 2 - 2 * c.BALL_RADIUS - c.ROBOT_SIZE / 2 + 0.002
    robots = robots.at[: B // 2, 0, :3].set(jnp.array([0.0, wall_y, jnp.pi / 2]))
    robots = robots.at[B // 2 :, 0, :3].set(jnp.array([-0.06, 0.0, 0.0]))
    robots = robots.at[B // 2 :, 1, :3].set(jnp.array([0.06, 0.0, jnp.pi]))
    ball = jnp.zeros_like(initial.ball)
    ball = ball.at[: B // 2, 1].set(c.FIELD_WIDTH / 2 - c.BALL_RADIUS)
    wheels = jnp.zeros_like(initial.wheel_speeds)
    wheels = wheels.at[:, 0, 0, :].set(c.ROBOT_MAX_WHEEL_SPEED)
    wheels = wheels.at[B // 2 :, 0, 1, :].set(c.ROBOT_MAX_WHEEL_SPEED)
    initial = initial._replace(
        robots=robots.reshape(B, c.N_TEAMS, c.N_ROBOTS, 6), ball=ball, wheel_speeds=wheels
    )
    actions = jnp.zeros_like(actions)
    actions = actions.at[:, :, 0, 0, :].set(1.0)
    actions = actions.at[:, B // 2 :, 0, 1, :].set(1.0)
vstep = jax.vmap(lambda state, action: p.step(state, action, sub_steps=args.substeps, drive_model=args.drive)[0])


def rollout(states, seq):
    return jax.lax.scan(lambda carry, action: (vstep(carry, action), None), states, seq)[0]


compiled = jax.jit(rollout)
started = time.perf_counter()
states = compiled(initial, actions)
jax.block_until_ready(states)
compile_and_first = time.perf_counter() - started
states = compiled(states, actions)
jax.block_until_ready(states)
samples = []
for _ in range(args.repeats):
    started = time.perf_counter()
    states = compiled(initial if args.restart_each_chunk else states, actions)
    jax.block_until_ready(states)
    samples.append(time.perf_counter() - started)
median = statistics.median(samples)
print(
    f"physics compile_and_first_s={compile_and_first:.3f} median_s={median:.5f} env_steps_s={B * STEPS / median:.0f} samples_s={samples}",
    flush=True,
)
ball_xy = np.asarray(states.ball[:, :2])
robots = np.asarray(states.robots).reshape(B, 6, 6)
separation = np.linalg.norm(robots[:, :, :2] - ball_xy[:, None, :], axis=-1)
print(
    f"final_near_robot_fraction={float(np.mean(np.min(separation, axis=1) < c.ROBOT_SIZE / 2 + c.BALL_RADIUS + 0.01)):.3f} final_past_goal_line_fraction={float(np.mean(np.abs(ball_xy[:, 0]) > c.FIELD_LENGTH / 2)):.3f} final_near_side_wall_fraction={float(np.mean(np.abs(ball_xy[:, 1]) > c.FIELD_WIDTH / 2 - c.BALL_RADIUS - 0.01)):.3f}",
    flush=True,
)
