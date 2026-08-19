"""Jumanji-compatible, fully JAX-resident IEEE VSSS environment.

This module is an optional integration layer. Install ``vsss-sim[stoix]`` to
use it. Unlike :class:`VSSVecEnv`, neither observations nor bookkeeping cross
the device boundary, so a trainer can ``jit``/``vmap`` the complete rollout.
"""
from __future__ import annotations

from functools import cached_property
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, transition, truncation

from .. import config
from ..config import InitMode
from ..physics import jax_backend as physics

_OBS_DIM = 4 + config.N_TEAMS * config.N_ROBOTS * 7
_ACTION_DIM = config.N_ROBOTS * 2
_NORM_POS_X = config.FIELD_LENGTH / 2.0
_NORM_POS_Y = config.FIELD_WIDTH / 2.0
_NORM_VEL = config.ROBOT_MAX_WHEEL_SPEED * config.VELOCITY_NORM_HEADROOM
_NORM_OMEGA = _NORM_VEL / (config.ROBOT_WHEELBASE / 2.0)


class State(NamedTuple):
    """Complete JAX environment state; safe to pass through ``jit`` and ``vmap``."""

    simulation: physics.SimState
    key: chex.PRNGKey
    step_count: jax.Array


def build_observation(simulation: physics.SimState) -> jax.Array:
    """Build the normalized 46-element policy observation on device."""
    ball = simulation.ball
    robots = simulation.robots.reshape(config.N_TEAMS * config.N_ROBOTS, 6)

    ball_obs = jnp.asarray(
        [
            ball[0] / _NORM_POS_X,
            ball[1] / _NORM_POS_Y,
            ball[2] / _NORM_VEL,
            ball[3] / _NORM_VEL,
        ],
        dtype=jnp.float32,
    )
    robot_obs = jnp.stack(
        [
            robots[:, 0] / _NORM_POS_X,
            robots[:, 1] / _NORM_POS_Y,
            jnp.sin(robots[:, 2]),
            jnp.cos(robots[:, 2]),
            robots[:, 3] / _NORM_VEL,
            robots[:, 4] / _NORM_VEL,
            robots[:, 5] / _NORM_OMEGA,
        ],
        axis=-1,
    ).reshape(-1)
    return jnp.concatenate([ball_obs, robot_obs]).astype(jnp.float32)


def _crowding_penalty(simulation: physics.SimState) -> jax.Array:
    """JAX equivalent of ``VSSEnv._crowding_penalty``."""
    blue_pos = simulation.robots[config.TEAM_BLUE, :, 0:2]
    half_l = config.FIELD_LENGTH / 2.0
    half_goal_y = config.GOAL_AREA_LENGTH_Y / 2.0
    goal_x = config.GOAL_AREA_LENGTH_X

    def crowded(x_min: float, x_max: float) -> jax.Array:
        in_area = (
            (blue_pos[:, 0] >= x_min)
            & (blue_pos[:, 0] <= x_max)
            & (jnp.abs(blue_pos[:, 1]) <= half_goal_y)
        )
        return jnp.sum(in_area) >= 2

    count = crowded(-half_l, -half_l + goal_x).astype(jnp.float32)
    count += crowded(half_l - goal_x, half_l).astype(jnp.float32)
    return count * jnp.float32(config.GOAL_AREA_CROWDING_PENALTY)


class VSSJumanjiEnv(Environment[State, specs.BoundedArray, jax.Array]):
    """Single-agent VSSS match exposed through Jumanji's pure JAX API.

    Blue is controlled by the six-dimensional continuous action. Yellow uses
    either a stationary or random wheel-speed policy. Goals cause an in-match
    kickoff; reaching ``max_episode_steps`` truncates the episode.
    """

    def __init__(
        self,
        opponent_policy: str = "stationary",
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        init_mode: InitMode | str = InitMode.KICKOFF,
    ) -> None:
        if opponent_policy not in ("stationary", "random"):
            raise ValueError(
                "VSSJumanjiEnv requires a JAX-native opponent_policy: "
                "choose 'stationary' or 'random'."
            )
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.opponent_policy = opponent_policy
        self.max_episode_steps = int(max_episode_steps)
        self.init_mode = InitMode(init_mode)
        self._reset_simulation = (
            physics.reset_kickoff
            if self.init_mode == InitMode.KICKOFF
            else physics.reset_random
        )
        super().__init__()

    @cached_property
    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(_OBS_DIM,),
            dtype=jnp.float32,
            minimum=-5.0,
            maximum=5.0,
            name="observation",
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(_ACTION_DIM,),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[jax.Array]]:
        state_key, reset_key = jax.random.split(key)
        simulation = self._reset_simulation(reset_key)
        state = State(
            simulation=simulation,
            key=state_key,
            step_count=jnp.zeros((), dtype=jnp.int32),
        )
        return state, restart(build_observation(simulation), dtype=jnp.float32)

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[jax.Array]]:
        next_key, opponent_key, kickoff_key = jax.random.split(state.key, 3)
        blue = jnp.asarray(action, dtype=jnp.float32).reshape(config.N_ROBOTS, 2)
        if self.opponent_policy == "stationary":
            yellow = jnp.zeros_like(blue)
        else:
            yellow = jax.random.uniform(
                opponent_key, blue.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32
            )

        ball_x_before = state.simulation.ball[0]
        actions = jnp.stack([blue, yellow], axis=0)
        simulation, physics_info = physics.step(state.simulation, actions)
        goal = physics_info["goal"]
        ball_x_after = simulation.ball[0]

        score_delta = jnp.asarray([goal == 1, goal == -1], dtype=jnp.int32)
        simulation = simulation._replace(score=simulation.score + score_delta)

        # A goal starts a new kickoff inside the same match. Preserve match-level
        # score and time while replacing positions and velocities.
        def kickoff(scored_simulation: physics.SimState) -> physics.SimState:
            fresh = physics.reset_kickoff(kickoff_key)
            return fresh._replace(score=scored_simulation.score, t=scored_simulation.t)

        simulation = jax.lax.cond(goal != 0, kickoff, lambda sim: sim, simulation)
        step_count = state.step_count + jnp.int32(1)
        next_state = State(simulation=simulation, key=next_key, step_count=step_count)

        reward = (
            goal.astype(jnp.float32)
            + jnp.float32(config.BALL_FORWARD_REWARD_COEF)
            * (ball_x_after - ball_x_before)
            + _crowding_penalty(simulation)
        )
        observation = build_observation(simulation)
        extras = {
            "goal": goal,
            "score_blue": simulation.score[config.TEAM_BLUE],
            "score_yellow": simulation.score[config.TEAM_YELLOW],
            "sim_time": simulation.t,
        }
        timestep = jax.lax.cond(
            step_count >= self.max_episode_steps,
            lambda: truncation(
                reward, observation, discount=jnp.float32(1.0), extras=extras
            ),
            lambda: transition(reward, observation, extras=extras, dtype=jnp.float32),
        )
        return next_state, timestep
