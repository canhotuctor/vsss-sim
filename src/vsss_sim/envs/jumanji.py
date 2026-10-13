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
from jumanji.types import TimeStep, restart, termination, transition, truncation

from .. import config
from ..config import InitMode, RewardProfile
from ..perspective import switch_team_perspective
from ..physics import jax_backend as physics
from .match_rules import MatchRules, MatchStepInput, build_observation

_OBS_DIM = 4 + config.N_TEAMS * config.N_ROBOTS * 7
_ACTION_DIM = config.N_ROBOTS * 2
class State(NamedTuple):
    """Complete JAX environment state; safe to pass through ``jit`` and ``vmap``."""

    simulation: physics.SimState
    key: chex.PRNGKey
    step_count: jax.Array


class VSSJumanjiEnv(Environment[State, specs.BoundedArray, jax.Array]):
    """Single-agent VSSS match exposed through Jumanji's pure JAX API.

    Blue is controlled by the six-dimensional continuous action. Yellow uses
    either a stationary or random wheel-speed policy. Goals cause an in-match
    kickoff by default; optionally, a goal can terminate the episode. Reaching
    ``max_episode_steps`` truncates the episode.
    """

    def __init__(
        self,
        opponent_policy: str = "stationary",
        max_episode_steps: int = config.MAX_EPISODE_STEPS,
        init_mode: InitMode | str = InitMode.KICKOFF,
        terminate_on_goal: bool = False,
        reward_profile: RewardProfile | str = RewardProfile.DEFAULT,
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
        self.terminate_on_goal = bool(terminate_on_goal)
        self.reward_profile = RewardProfile(reward_profile)
        self._match_rules = MatchRules(
            reward_profile=self.reward_profile,
            max_episode_steps=self.max_episode_steps,
            init_mode=self.init_mode,
            kickoff_on_goal=not self.terminate_on_goal,
        )
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
        return state, restart(self._match_rules.observation(simulation), dtype=jnp.float32)

    def step(
        self,
        state: State,
        action: chex.Array,
        opponent_action: chex.Array | None = None,
        opponent_mode: chex.Array | int | None = None,
    ) -> tuple[State, TimeStep[jax.Array]]:
        next_key, opponent_key, kickoff_key = jax.random.split(state.key, 3)
        blue = jnp.asarray(action, dtype=jnp.float32).reshape(config.N_ROBOTS, 2)
        if opponent_mode is not None:
            # Runtime modes are used by historical self-play. 0 = learned,
            # 1 = random, 2 = stationary. The learned action is already in
            # yellow's robot-local wheel coordinates.
            random_yellow = jax.random.uniform(
                opponent_key, blue.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32
            )
            learned_yellow = jnp.asarray(opponent_action, dtype=jnp.float32).reshape(
                config.N_ROBOTS, 2
            )
            yellow = jnp.where(
                jnp.asarray(opponent_mode) == 0,
                learned_yellow,
                jnp.where(jnp.asarray(opponent_mode) == 1, random_yellow, jnp.zeros_like(blue)),
            )
        elif opponent_action is not None:
            yellow = jnp.asarray(opponent_action, dtype=jnp.float32).reshape(
                config.N_ROBOTS, 2
            )
        elif self.opponent_policy == "stationary":
            yellow = jnp.zeros_like(blue)
        else:
            yellow = jax.random.uniform(
                opponent_key, blue.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32
            )

        actions = jnp.stack([blue, yellow], axis=0)
        simulation_after_physics, physics_info = physics.step(state.simulation, actions)
        goal = physics_info["goal"]
        step_count = state.step_count + jnp.int32(1)
        match_step = self._match_rules.step(
            MatchStepInput(
                simulation_before=state.simulation,
                simulation_after_physics=simulation_after_physics,
                goal=goal,
                step_count=step_count,
                kickoff_key=kickoff_key,
            )
        )
        simulation = match_step.simulation
        reward = match_step.reward
        observation = match_step.observation
        next_state = State(simulation=simulation, key=next_key, step_count=step_count)
        extras = {
            "goal": goal,
            "score_blue": simulation.score[config.TEAM_BLUE],
            "score_yellow": simulation.score[config.TEAM_YELLOW],
            "sim_time": simulation.t,
        }
        if self.terminate_on_goal:
            timestep = jax.lax.cond(
                goal != 0,
                lambda: termination(reward, observation, extras=extras),
                lambda: jax.lax.cond(
                    step_count >= self.max_episode_steps,
                    lambda: truncation(
                        reward, observation, discount=jnp.float32(1.0), extras=extras
                    ),
                    lambda: transition(
                        reward, observation, extras=extras, dtype=jnp.float32
                    ),
                ),
            )
        else:
            timestep = jax.lax.cond(
                step_count >= self.max_episode_steps,
                lambda: truncation(
                    reward, observation, discount=jnp.float32(1.0), extras=extras
                ),
                lambda: transition(reward, observation, extras=extras, dtype=jnp.float32),
            )
        return next_state, timestep

    def opponent_observation(self, state: State) -> jax.Array:
        """Return the same match observation from yellow's canonical view."""
        return switch_team_perspective(build_observation(state.simulation))
