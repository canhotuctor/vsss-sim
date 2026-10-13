"""Environment-agnostic symmetry augmentation for pure-JAX RL environments.

The wrapper exposes a symmetry-transformed view of an environment while keeping
the underlying simulation in its original coordinates.  A group element is
sampled once per episode: observations are transformed into the policy frame,
and policy actions are mapped back into the simulation frame before stepping.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import chex
import jax
from jumanji.types import TimeStep


class Symmetry(Protocol):
    """A group action on an environment's observation and action spaces."""

    def sample(self, key: chex.PRNGKey) -> Any:
        """Sample and return one symmetry group element."""

    def transform_observation(self, observation: chex.Array, element: Any) -> chex.Array:
        """Map an observation from simulation coordinates to policy coordinates."""

    def transform_action(self, action: chex.Array, element: Any) -> chex.Array:
        """Map an action from simulation coordinates to policy coordinates."""

    def inverse_action(self, action: chex.Array, element: Any) -> chex.Array:
        """Map an action from policy coordinates to simulation coordinates."""


class SymmetryEnvState(NamedTuple):
    """Underlying environment state plus the episode's fixed group element."""

    env_state: Any
    element: Any


class SymmetryEnv:
    """Apply random episode-level symmetry augmentation to a pure-JAX environment.

    The wrapped object only needs ``reset``, ``step``, ``observation_spec`` and
    ``action_spec`` attributes, matching the small interface used by Jumanji and
    this project's PPO trainer.  It deliberately does not depend on VSSS state
    layouts, so the same wrapper can be paired with an SSL symmetry definition.
    """

    def __init__(self, env: Any, symmetry: Symmetry) -> None:
        self.env = env
        self.symmetry = symmetry

    @property
    def observation_spec(self):
        return self.env.observation_spec

    @property
    def action_spec(self):
        return self.env.action_spec

    def reset(self, key: chex.PRNGKey) -> tuple[SymmetryEnvState, TimeStep]:
        env_key, symmetry_key = jax.random.split(key)
        env_state, timestep = self.env.reset(env_key)
        element = self.symmetry.sample(symmetry_key)
        observation = self.symmetry.transform_observation(timestep.observation, element)
        return SymmetryEnvState(env_state, element), timestep.replace(observation=observation)

    def step(
        self,
        state: SymmetryEnvState,
        action: chex.Array,
        opponent_action: chex.Array | None = None,
        opponent_mode: chex.Array | int | None = None,
    ) -> tuple[SymmetryEnvState, TimeStep]:
        simulation_action = self.symmetry.inverse_action(action, state.element)
        env_state, timestep = self.env.step(
            state.env_state,
            simulation_action,
            opponent_action=opponent_action,
            opponent_mode=opponent_mode,
        )
        observation = self.symmetry.transform_observation(
            timestep.observation, state.element
        )
        return (
            SymmetryEnvState(env_state, state.element),
            timestep.replace(observation=observation),
        )

    def opponent_observation(self, state: SymmetryEnvState) -> chex.Array:
        """Keep the opponent in physical canonical coordinates under augmentation."""
        return self.env.opponent_observation(state.env_state)
