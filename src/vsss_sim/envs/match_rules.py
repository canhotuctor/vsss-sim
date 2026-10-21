"""Shared, JAX-compatible match transitions for referee rules and rewards.

Framework adapters keep their own reset, rendering, and timestep contracts.
This module coordinates scoring, referee decisions, observations, and rewards
so scalar, batched, and Jumanji matches share match semantics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import config, physics
from ..config import InitMode, RewardProfile
from .match_referee import (
    MatchPhase,
    RefereeEvent,
    RefereeState,
    RestartType,
    _resolve_referee,
    initial_referee_state,
)
from .match_restarts import _restart_positions

_NORM_POS_X = config.FIELD_LENGTH / 2.0
_NORM_POS_Y = config.FIELD_WIDTH / 2.0
_NORM_VEL = config.ROBOT_MAX_WHEEL_SPEED * config.VELOCITY_NORM_HEADROOM
_NORM_OMEGA = _NORM_VEL / (config.ROBOT_WHEELBASE / 2.0)


class MatchStepInput(NamedTuple):
    """Facts needed to resolve one completed control step.

    Pass ``referee_state`` from the previous result to retain referee timers,
    pending goal-area violations, and restart phase. ``dt`` is the control
    interval used by those timers. Omitting either supports one-off transitions.
    """

    simulation_before: physics.SimState
    simulation_after_physics: physics.SimState
    goal: jax.Array
    step_count: jax.Array
    kickoff_key: jax.Array
    referee_state: RefereeState | None = None
    dt: jax.Array | None = None


class MatchStepResult(NamedTuple):
    """Canonical match result before framework-specific packaging."""

    simulation: physics.SimState
    observation: jax.Array
    reward: jax.Array
    referee_state: RefereeState
    event: jax.Array
    offending_team: jax.Array
    restart: jax.Array
    restart_team: jax.Array
    quadrant: jax.Array


def build_observation(simulation: physics.SimState) -> jax.Array:
    """Build the normalized policy observation for scalar or batched state."""
    leading_shape = simulation.robots.shape[:-3]
    ball = simulation.ball
    robots = simulation.robots.reshape(
        leading_shape + (config.N_TEAMS * config.N_ROBOTS, 6)
    )

    ball_observation = jnp.stack(
        [
            ball[..., 0] / _NORM_POS_X,
            ball[..., 1] / _NORM_POS_Y,
            ball[..., 2] / _NORM_VEL,
            ball[..., 3] / _NORM_VEL,
        ],
        axis=-1,
    )
    robot_observation = jnp.stack(
        [
            robots[..., 0] / _NORM_POS_X,
            robots[..., 1] / _NORM_POS_Y,
            jnp.sin(robots[..., 2]),
            jnp.cos(robots[..., 2]),
            robots[..., 3] / _NORM_VEL,
            robots[..., 4] / _NORM_VEL,
            robots[..., 5] / _NORM_OMEGA,
        ],
        axis=-1,
    ).reshape(leading_shape + (config.N_TEAMS * config.N_ROBOTS * 7,))
    return jnp.concatenate([ball_observation, robot_observation], axis=-1).astype(
        jnp.float32
    )


def _crowding_penalty(simulation: physics.SimState) -> jax.Array:
    """Penalize two or more blue robots in either goal area."""
    blue_positions = simulation.robots[..., config.TEAM_BLUE, :, 0:2]
    half_length = config.FIELD_LENGTH / 2.0
    half_goal_width = config.GOAL_AREA_LENGTH_Y / 2.0
    goal_depth = config.GOAL_AREA_LENGTH_X

    crowding = jnp.zeros(blue_positions.shape[:-2], dtype=jnp.float32)
    for x_min, x_max in (
        (-half_length, -half_length + goal_depth),
        (half_length - goal_depth, half_length),
    ):
        in_area = (
            (blue_positions[..., 0] >= x_min)
            & (blue_positions[..., 0] <= x_max)
            & (jnp.abs(blue_positions[..., 1]) <= half_goal_width)
        )
        crowding += (jnp.sum(in_area, axis=-1) >= 2).astype(jnp.float32)
    return crowding * jnp.float32(config.GOAL_AREA_CROWDING_PENALTY)


def _angle_reward(
    point_a: jax.Array, vertex: jax.Array, point_c: jax.Array
) -> jax.Array:
    """Return the Brandão normalized angle term in [-1, 0]."""
    vector_a = point_a - vertex
    vector_c = point_c - vertex
    norm_product = jnp.linalg.norm(vector_a) * jnp.linalg.norm(vector_c)
    cosine = jnp.sum(vector_a * vector_c) / jnp.maximum(norm_product, 1e-8)
    angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
    return angle / jnp.pi - 1.0


def _default_reward(context: MatchStepInput) -> jax.Array:
    before = context.simulation_before.ball[0]
    after = context.simulation_after_physics.ball[0]
    return (
        context.goal.astype(jnp.float32)
        + jnp.float32(config.BALL_FORWARD_REWARD_COEF) * (after - before)
        + _crowding_penalty(context.simulation_after_physics)
    ).astype(jnp.float32)


def _brandao_reward(
    context: MatchStepInput,
    max_episode_steps: int,
    individual_agent_rewards: bool = False,
) -> jax.Array:
    """Brandão et al. (2022), Eq. (6)-(16), from blue's team perspective.

    The shared decentralized policy receives one reward per robot: universal
    ball terms and goal outcome are shared, while positioning rewards are
    assigned to the individual robot. The centralized policy retains its
    historical scalar team reward.
    """
    simulation = context.simulation_after_physics
    ball_before = context.simulation_before.ball[:2]
    ball_after = simulation.ball[:2]
    blue_positions = simulation.robots[config.TEAM_BLUE, :, :2]

    half_length = jnp.float32(config.FIELD_LENGTH / 2.0)
    opponent_goal = jnp.asarray([half_length, 0.0], dtype=jnp.float32)
    own_goal = jnp.asarray([-half_length, 0.0], dtype=jnp.float32)

    distance_before = jnp.linalg.norm(ball_before - opponent_goal)
    distance_after = jnp.linalg.norm(ball_after - opponent_goal)
    speed_reward = jnp.clip((distance_before - distance_after - 0.05) / 0.14, -1.0, 1.0)

    nearest_distance = jnp.min(jnp.linalg.norm(blue_positions - ball_after, axis=-1))
    proximity_reward = -jnp.minimum(nearest_distance, 1.0)
    offensive_reward = jax.vmap(
        lambda robot: _angle_reward(robot, ball_after, opponent_goal)
    )(blue_positions)
    defensive_reward = jax.vmap(
        lambda robot: _angle_reward(ball_after, robot, own_goal)
    )(blue_positions)

    universal_reward = 0.7 * speed_reward + 0.1 * proximity_reward
    if individual_agent_rewards:
        continuous_reward = (
            universal_reward
            + 0.1 * offensive_reward
            + 0.1 * defensive_reward
        )
    else:
        continuous_reward = (
            universal_reward
            + 0.1 * jnp.mean(offensive_reward)
            + 0.1 * jnp.mean(defensive_reward)
        )
    horizon = jnp.float32(max_episode_steps)
    goal_reward = 10.0 * context.goal.astype(jnp.float32) * (
        1.0 + (horizon - context.step_count.astype(jnp.float32)) / horizon
    )
    return jnp.where(context.goal != 0, goal_reward, continuous_reward).astype(
        jnp.float32
    )


class MatchRules:
    """Configured match rules with one JAX-compatible step interface."""

    def __init__(
        self,
        reward_profile: RewardProfile | str = RewardProfile.DEFAULT,
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        init_mode: InitMode | str = InitMode.KICKOFF,
        kickoff_on_goal: bool = True,
        individual_agent_rewards: bool = False,
    ) -> None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.reward_profile = RewardProfile(reward_profile)
        self.max_episode_steps = int(max_episode_steps)
        self.init_mode = InitMode(init_mode)
        self.kickoff_on_goal = bool(kickoff_on_goal)
        self.individual_agent_rewards = bool(individual_agent_rewards)

    def observation(self, simulation: physics.SimState) -> jax.Array:
        return build_observation(simulation)

    def step(self, context: MatchStepInput) -> MatchStepResult:
        """Apply reward, scoring, referee events, restarts, and observations."""
        referee_state = (
            initial_referee_state() if context.referee_state is None else context.referee_state
        )
        resuming = referee_state.phase == MatchPhase.STOPPED
        effective_goal = jnp.where(resuming, jnp.int32(0), context.goal)
        reward_context = context._replace(goal=effective_goal)
        if self.reward_profile is RewardProfile.BRANDAO_REWARD:
            reward = _brandao_reward(
                reward_context,
                self.max_episode_steps,
                individual_agent_rewards=self.individual_agent_rewards,
            )
        else:
            reward = _default_reward(reward_context)

        decision = _resolve_referee(
            reward_context.simulation_before,
            reward_context.simulation_after_physics,
            referee_state,
            effective_goal,
            reward_context.dt,
        )
        event = jnp.where(resuming, jnp.int32(RefereeEvent.NONE), decision.event).astype(jnp.int32)
        offending_team = jnp.where(resuming, jnp.int32(-1), decision.offending_team)
        restart = jnp.where(resuming, jnp.int32(RestartType.NONE), decision.restart)
        if not self.kickoff_on_goal:
            restart = jnp.where(
                event == RefereeEvent.GOAL,
                jnp.int32(RestartType.NONE),
                restart,
            )
        restart_team = jnp.where(resuming, jnp.int32(-1), decision.restart_team)
        quadrant = jnp.where(resuming, jnp.int32(-1), decision.quadrant)

        if self.reward_profile is RewardProfile.BRANDAO_REWARD:
            foul_event = (
                (event == RefereeEvent.ATTACKER_GOAL_AREA)
                | (event == RefereeEvent.DEFENDER_GOAL_AREA)
                | (event == RefereeEvent.GOALKEEPER_TIMEOUT)
                | (event == RefereeEvent.PERSONAL_FOUL)
                | (event == RefereeEvent.GOALKEEPER_BLOCK)
            )
            foul_reward = jnp.where(offending_team == config.TEAM_BLUE, -1.0, 1.0).astype(
                jnp.float32
            )
            reward = jnp.where(foul_event, foul_reward, reward)
        else:
            penalty_reward = jnp.where(
                restart == RestartType.PENALTY_KICK,
                jnp.where(offending_team == config.TEAM_BLUE, -1.0, 1.0),
                0.0,
            ).astype(jnp.float32)
            reward = reward + penalty_reward
        reward = jnp.where(resuming, jnp.zeros_like(reward), reward)

        score_delta = jnp.asarray(
            [effective_goal == 1, effective_goal == -1], dtype=jnp.int32
        )
        scored_simulation = context.simulation_after_physics._replace(
            score=context.simulation_after_physics.score + score_delta
        )

        if self.kickoff_on_goal:

            def kickoff(simulation: physics.SimState) -> physics.SimState:
                if self.init_mode is InitMode.RANDOM:
                    fresh = physics.reset_random(context.kickoff_key)
                else:
                    fresh = physics.reset_kickoff(context.kickoff_key)
                return fresh._replace(score=simulation.score, t=simulation.t)

            simulation = jax.lax.cond(
                effective_goal != 0,
                kickoff,
                lambda state: state,
                scored_simulation,
            )
        else:
            simulation = scored_simulation

        whistle = ~resuming & (restart != RestartType.NONE) & (event != RefereeEvent.GOAL)
        simulation = jax.lax.cond(
            whistle,
            lambda state: _restart_positions(state, restart, restart_team, quadrant),
            lambda state: state,
            simulation,
        )

        def resume_frame(_: None) -> physics.SimState:
            return context.simulation_before._replace(t=context.simulation_after_physics.t)

        simulation = jax.lax.cond(resuming, resume_frame, lambda _: simulation, operand=None)
        next_referee_state = jax.lax.cond(
            (effective_goal != 0) & self.kickoff_on_goal,
            lambda _: initial_referee_state(),
            lambda _: decision.state,
            operand=None,
        )
        next_referee_state = jax.lax.cond(
            resuming,
            lambda _: referee_state._replace(phase=jnp.int32(MatchPhase.PLAYING)),
            lambda _: next_referee_state,
            operand=None,
        )

        return MatchStepResult(
            simulation=simulation,
            observation=build_observation(simulation),
            reward=reward,
            referee_state=next_referee_state,
            event=event,
            offending_team=offending_team.astype(jnp.int32),
            restart=restart.astype(jnp.int32),
            restart_team=restart_team.astype(jnp.int32),
            quadrant=quadrant.astype(jnp.int32),
        )
