"""End-to-end JAX PPO for the VSSS Jumanji environment.

The hot path is entirely device-resident: Flax policy inference, VMAP'd
environment stepping, ``lax.scan`` rollout collection and GAE, and Optax
minibatch updates. Python is only needed between PPO updates for logging.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from ..envs.brandao_observation import (
    FRAME_FEATURES,
    HISTORY_LENGTH,
    mirror_x_axis_observation,
)
from ..envs.jumanji import State as EnvState
from ..envs.jumanji import VSSJumanjiEnv
from ..envs.match_referee import RefereeEvent

_LOG_2PI = jnp.log(jnp.float32(2.0 * jnp.pi))


def _curriculum_scores(returns: jax.Array, valid: jax.Array):
    """Batch min-max return normalization and bounded learnability heuristic."""
    any_valid = jnp.any(valid)
    low = jnp.min(jnp.where(valid, returns, jnp.inf))
    high = jnp.max(jnp.where(valid, returns, -jnp.inf))
    span = high - low
    normalized = jnp.where(span > 1e-8, (returns - low) / jnp.maximum(span, 1e-8), 0.5)
    scores = jnp.where(valid, normalized * (1.0 - normalized), 0.0)
    return scores, jnp.where(any_valid, low, 0.0), jnp.where(any_valid, high, 0.0)


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters that determine one compiled PPO update."""

    num_envs: int = 256
    rollout_length: int = 128
    learning_rate: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    mirror_augmentation: bool = False
    curriculum_replay_fraction: float = 0.0
    curriculum_buffer_size: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.curriculum_replay_fraction <= 1.0:
            raise ValueError("curriculum_replay_fraction must be between 0 and 1")
        if self.curriculum_buffer_size < 1:
            raise ValueError("curriculum_buffer_size must be positive")
        batch_size = self.num_envs * self.rollout_length
        if self.num_envs < 1 or self.rollout_length < 1:
            raise ValueError("num_envs and rollout_length must be at least 1")
        if self.update_epochs < 1 or self.num_minibatches < 1:
            raise ValueError("update_epochs and num_minibatches must be at least 1")
        if batch_size % self.num_minibatches:
            raise ValueError(
                "num_envs * rollout_length must be divisible by num_minibatches"
            )
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_length

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with a state-independent action scale."""

    action_size: int
    hidden_sizes: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor = observation
        critic = observation
        hidden_init = nn.initializers.orthogonal(jnp.sqrt(2.0))

        for size in self.hidden_sizes:
            actor = nn.tanh(nn.Dense(size, kernel_init=hidden_init)(actor))
            critic = nn.tanh(nn.Dense(size, kernel_init=hidden_init)(critic))

        mean = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(actor)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_size,))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return mean, jnp.broadcast_to(log_std, mean.shape), jnp.squeeze(value, axis=-1)


class BrandaoSharedActorCritic(nn.Module):
    """Paper-shaped per-robot actor and value networks with a shared policy."""

    @nn.compact
    def __call__(
        self, observation: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor = observation
        for size in (300, 200, 100):
            actor = nn.tanh(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(
                    actor
                )
            )
        alpha = nn.softplus(
            nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01), name="alpha")(
                actor
            )
        ) + 1.0
        beta = nn.softplus(
            nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01), name="beta")(
                actor
            )
        ) + 1.0

        critic = observation
        for size in (256, 128, 64):
            critic = nn.tanh(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(
                    critic
                )
            )
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return alpha, beta, jnp.squeeze(value, axis=-1)


class BrandaoSetActorCritic(nn.Module):
    """Shared per-robot policy with separate invariant ally and opponent sets."""

    hidden_sizes: tuple[int, ...] = (300, 200, 100)

    @nn.compact
    def __call__(
        self, observation: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Every frame has fixed semantic slots: self, two allies, three
        # opponents, ball, heading, bearings, distances, previous actions, time.
        frames = observation.reshape(observation.shape[:-1] + (8, 53))
        positions = frames[..., :14].reshape(frames.shape[:-1] + (7, 2))
        self_position = positions[..., 0, :]
        ally_positions = positions[..., 1:3, :]
        opponent_positions = positions[..., 3:6, :]
        ball_position = positions[..., 6, :]

        heading = frames[..., 14:17]
        bearings = frames[..., 17:38].reshape(frames.shape[:-1] + (7, 3))
        ally_bearings = bearings[..., :2, :]
        opponent_bearings = bearings[..., 2:5, :]
        goal_bearings = bearings[..., 5:7, :].reshape(bearings.shape[:-2] + (6,))
        distances = frames[..., 38:46]
        ally_distances = distances[..., :2, None]
        opponent_distances = distances[..., 2:5, None]
        ball_and_goal_distances = distances[..., 5:8]
        previous_actions = frames[..., 46:52].reshape(frames.shape[:-1] + (3, 2))

        ally_features = jnp.concatenate(
            [
                ally_positions,
                ally_bearings,
                ally_distances,
                previous_actions[..., 1:3, :],
            ],
            axis=-1,
        )
        opponent_features = jnp.concatenate(
            [opponent_positions, opponent_bearings, opponent_distances], axis=-1
        )

        # Each Dense module is applied with shared weights across all members
        # and all eight frames. Sum pooling is invariant to entity order.
        ally_hidden = nn.tanh(nn.Dense(32, name="ally_encoder_hidden")(ally_features))
        ally_embeddings = nn.Dense(16, name="ally_encoder_embedding")(ally_hidden)
        ally_summary = jnp.sum(ally_embeddings, axis=-2)
        opponent_hidden = nn.tanh(
            nn.Dense(32, name="opponent_encoder_hidden")(opponent_features)
        )
        opponent_embeddings = nn.Dense(16, name="opponent_encoder_embedding")(
            opponent_hidden
        )
        opponent_summary = jnp.sum(opponent_embeddings, axis=-2)

        self_features = jnp.concatenate(
            [
                self_position,
                heading,
                goal_bearings,
                ball_position,
                ball_and_goal_distances,
                previous_actions[..., 0, :],
                frames[..., 52:53],
            ],
            axis=-1,
        )
        encoded_frames = jnp.concatenate(
            [self_features, ally_summary, opponent_summary], axis=-1
        )
        encoded = encoded_frames.reshape(encoded_frames.shape[:-2] + (8 * 51,))

        actor = encoded
        for size in self.hidden_sizes:
            actor = nn.tanh(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(
                    actor
                )
            )
        alpha = nn.softplus(
            nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01), name="alpha")(
                actor
            )
        ) + 1.0
        beta = nn.softplus(
            nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01), name="beta")(
                actor
            )
        ) + 1.0

        critic = encoded
        for size in self.hidden_sizes:
            critic = nn.tanh(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(
                    critic
                )
            )
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return alpha, beta, jnp.squeeze(value, axis=-1)


def create_policy_model(
    policy_architecture: str, hidden_sizes: tuple[int, ...] = (64, 64)
) -> nn.Module:
    """Construct the selected checkpoint-compatible actor-critic model."""
    if policy_architecture == "joint":
        return ActorCritic(action_size=6, hidden_sizes=hidden_sizes)
    if policy_architecture == "brandao_shared":
        return BrandaoSharedActorCritic()
    if policy_architecture == "brandao_sets":
        return BrandaoSetActorCritic(hidden_sizes=hidden_sizes)
    raise ValueError(
        "policy_architecture must be 'joint', 'brandao_shared', or 'brandao_sets'"
    )


def deterministic_policy_action(
    model: nn.Module,
    params: dict,
    observation: jax.Array,
    policy_architecture: str,
) -> jax.Array:
    """Map observations to the selected policy's deterministic wheel speeds."""
    first, second, _ = model.apply(params, observation)
    if policy_architecture in ("brandao_shared", "brandao_sets"):
        return 2.0 * first / (first + second) - 1.0
    return jnp.tanh(first)


class RunnerState(NamedTuple):
    """All mutable training state carried from one PPO update to the next."""

    train_state: TrainState
    env_state: EnvState
    observation: jax.Array
    key: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    env_steps: jax.Array
    opponent_indices: jax.Array
    opponent_modes: jax.Array
    opponent_categories: jax.Array
    opponent_generations: jax.Array
    initial_level_keys: jax.Array
    replay_level_keys: jax.Array
    replay_weights: jax.Array
    replay_valid: jax.Array


class _Transition(NamedTuple):
    observation: jax.Array
    raw_action: jax.Array
    old_log_prob: jax.Array
    old_value: jax.Array
    reward: jax.Array
    bootstrap_value: jax.Array
    discount: jax.Array
    episode_end: jax.Array
    completed_return: jax.Array
    completed_length: jax.Array
    goal: jax.Array
    referee_event: jax.Array
    opponent_index: jax.Array
    opponent_mode: jax.Array
    opponent_category: jax.Array
    opponent_generation: jax.Array
    level_key: jax.Array
    replay_selected: jax.Array
    mirrored_old_log_prob: jax.Array | None = None
    mirrored_old_value: jax.Array | None = None
    mirrored_bootstrap_value: jax.Array | None = None


class _TrainingBatch(NamedTuple):
    observation: jax.Array
    raw_action: jax.Array
    old_log_prob: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    target: jax.Array
    mirrored_old_log_prob: jax.Array | None = None
    mirrored_old_value: jax.Array | None = None
    mirrored_advantage: jax.Array | None = None
    mirrored_target: jax.Array | None = None


def _reconstruct_brandao_observations(
    initial_observation: jax.Array,
    rollout_frames: jax.Array,
    episode_ends: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """Rebuild selected eight-frame Brandao observations from one stored frame.

    ``rollout_frames[t]`` is the newest frame of the pre-action observation at
    step ``t``.  The initial history supplies step zero, and an episode ending
    at step ``t - 1`` makes every frame of observation ``t`` the reset frame.
    """
    rollout_length, num_envs = episode_ends.shape
    initial_frames = jnp.moveaxis(
        initial_observation.reshape(
            num_envs, 3, HISTORY_LENGTH, FRAME_FEATURES
        ),
        2,
        0,
    )
    frames = rollout_frames.reshape(
        rollout_length, num_envs, 3, FRAME_FEATURES
    )
    all_frames = jnp.concatenate((initial_frames, frames[1:]), axis=0)

    previous_episode_ends = jnp.concatenate(
        (jnp.zeros_like(episode_ends[:1]), episode_ends[:-1]), axis=0
    )
    step_numbers = jnp.arange(rollout_length, dtype=jnp.int32)

    def track_episode_start(
        episode_start: jax.Array, data: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        step, previous_episode_end = data
        episode_start = jnp.where(previous_episode_end, step, episode_start)
        return episode_start, episode_start

    _, episode_starts = jax.lax.scan(
        track_episode_start,
        jnp.full((num_envs,), -(HISTORY_LENGTH - 1), dtype=jnp.int32),
        (step_numbers, previous_episode_ends),
    )
    steps, envs = indices // num_envs, indices % num_envs
    history_offsets = jnp.arange(HISTORY_LENGTH, dtype=jnp.int32)
    source_steps = jnp.maximum(
        steps[:, None] - (HISTORY_LENGTH - 1) + history_offsets[None, :],
        episode_starts[steps, envs][:, None],
    ) + (HISTORY_LENGTH - 1)
    frames = all_frames[source_steps, envs[:, None]]
    return jnp.swapaxes(frames, 1, 2).reshape(
        (indices.shape[0], 3, HISTORY_LENGTH * FRAME_FEATURES)
    )


def _normal_log_prob(
    raw_action: jax.Array, mean: jax.Array, log_std: jax.Array
) -> jax.Array:
    variance_term = jnp.square((raw_action - mean) / jnp.exp(log_std))
    return jnp.sum(-0.5 * (variance_term + 2.0 * log_std + _LOG_2PI), axis=-1)


def _normal_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (1.0 + _LOG_2PI), axis=-1)


def _sample_beta(
    key: jax.Array, alpha: jax.Array, beta: jax.Array
) -> jax.Array:
    """Sample independent Beta variates with one batched gamma sampler.

    The leading parameter axis combines both gamma factors in one draw. Keep
    the ratio in log space, as in JAX's Beta sampler, to avoid underflow for
    small concentrations. Grouped learner/opponent arrays share this call
    while retaining independent draws for every element.
    """
    alpha, beta = jnp.broadcast_arrays(alpha, beta)
    log_alpha, log_beta = jax.random.loggamma(key, jnp.stack((alpha, beta)))
    log_max = jnp.maximum(log_alpha, log_beta)
    alpha_scaled = jnp.exp(log_alpha - log_max)
    beta_scaled = jnp.exp(log_beta - log_max)
    return alpha_scaled / (alpha_scaled + beta_scaled)


def _beta_log_prob(
    sample: jax.Array, alpha: jax.Array, beta: jax.Array
) -> jax.Array:
    sample = jnp.clip(sample, 1e-6, 1.0 - 1e-6)
    log_normalizer = (
        jsp.special.gammaln(alpha)
        + jsp.special.gammaln(beta)
        - jsp.special.gammaln(alpha + beta)
    )
    log_density = (
        (alpha - 1.0) * jnp.log(sample)
        + (beta - 1.0) * jnp.log1p(-sample)
        - log_normalizer
    )
    # The policy's wheel space is [-1, 1], an affine transform of Beta [0, 1].
    return jnp.sum(log_density, axis=-1) - 2.0 * jnp.log(jnp.float32(2.0))


def _beta_entropy(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    total = alpha + beta
    entropy = (
        jsp.special.betaln(alpha, beta)
        - (alpha - 1.0) * jsp.special.digamma(alpha)
        - (beta - 1.0) * jsp.special.digamma(beta)
        + (total - 2.0) * jsp.special.digamma(total)
    )
    return jnp.sum(entropy + jnp.log(jnp.float32(2.0)), axis=-1)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    discounts: jax.Array,
    episode_ends: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute truncation-aware generalized advantages and value targets.

    ``discounts`` controls value bootstrapping. ``episode_ends`` separately
    prevents GAE from leaking from a reset episode into the preceding one.
    This distinction correctly bootstraps time-limit truncations.
    """

    def backward(gae: jax.Array, data: tuple[jax.Array, ...]):
        reward, value, bootstrap, discount, episode_end = data
        delta = reward + gamma * discount * bootstrap - value
        continuation = 1.0 - episode_end.astype(jnp.float32)
        gae = delta + gamma * gae_lambda * continuation * gae
        return gae, gae

    initial = jnp.zeros_like(rewards[-1])
    _, advantages = jax.lax.scan(
        backward,
        initial,
        (rewards, values, bootstrap_values, discounts, episode_ends),
        reverse=True,
    )
    return advantages, advantages + values


class PPO:
    """Compiled PPO trainer specialized to a pure-JAX VSSS environment."""

    def __init__(self, env: VSSJumanjiEnv, config: PPOConfig = PPOConfig()) -> None:
        self.env = env
        self.config = config
        self.policy_architecture = env.policy_architecture
        if config.mirror_augmentation and self.policy_architecture not in (
            "brandao_shared",
            "brandao_sets",
        ):
            raise ValueError("x-axis mirror augmentation requires a Brandao observation")
        self.model = create_policy_model(self.policy_architecture, config.hidden_sizes)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
        self._batched_reset = jax.vmap(env.reset)
        self._batched_step = jax.vmap(env.step)
        self._batched_step_with_opponent = jax.vmap(
            lambda state, action, opponent_action, opponent_mode: env.step(
                state,
                action,
                opponent_action=opponent_action,
                opponent_mode=opponent_mode,
            )
        )
        self._batched_opponent_observation = jax.vmap(env.opponent_observation)
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_initialize_self_play = jax.jit(self._initialize_self_play)
        self._compiled_update = jax.jit(self._update)
        self._compiled_update_many = jax.jit(
            jax.vmap(self._update, in_axes=(0, 0, 0, 0))
        )
        self._compiled_train = jax.jit(self._train, static_argnames=("num_updates",))
        self._compiled_train_legacy = jax.jit(
            self._train_legacy, static_argnames=("num_updates",)
        )
        self._compiled_train_many = jax.jit(
            self._train_many, static_argnames=("num_updates",)
        )

    @property
    def is_shared_agent_policy(self) -> bool:
        return self.policy_architecture in ("brandao_shared", "brandao_sets")

    def initialize(
        self,
        key: jax.Array,
        opponent_pool: dict | None = None,
        opponent_probabilities: jax.Array | None = None,
        opponent_generations: jax.Array | None = None,
    ) -> RunnerState:
        """Initialize one lineage, optionally assigning historical opponents."""
        if opponent_pool is None:
            return self._compiled_initialize(key)
        if opponent_probabilities is None:
            raise ValueError("opponent_probabilities are required with opponent_pool")
        if opponent_generations is None:
            opponent_generations = jnp.arange(
                jax.tree_util.tree_leaves(opponent_pool)[0].shape[0], dtype=jnp.int32
            )
        return self._compiled_initialize_self_play(
            key, opponent_pool, opponent_probabilities, opponent_generations
        )

    def initialize_lineages(
        self,
        keys: jax.Array,
        opponent_pools: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array | None = None,
    ) -> RunnerState:
        """Initialize independent learner states along a leading lineage axis."""
        if opponent_generations is None:
            pool_size = jax.tree_util.tree_leaves(opponent_pools)[0].shape[1]
            opponent_generations = jnp.broadcast_to(
                jnp.arange(pool_size, dtype=jnp.int32),
                (keys.shape[0], pool_size),
            )
        return jax.jit(
            jax.vmap(self._initialize_self_play, in_axes=(0, 0, 0, 0))
        )(keys, opponent_pools, opponent_probabilities, opponent_generations)

    def update(
        self,
        runner: RunnerState,
        opponent_pool: dict | None = None,
        opponent_probabilities: jax.Array | None = None,
        opponent_generations: jax.Array | None = None,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Collect one rollout and perform PPO, optionally against frozen history."""
        if opponent_pool is None:
            return self._compiled_update(runner, None, None)
        if opponent_probabilities is None:
            raise ValueError("opponent_probabilities are required with opponent_pool")
        if opponent_generations is None:
            opponent_generations = jnp.arange(
                jax.tree_util.tree_leaves(opponent_pool)[0].shape[0], dtype=jnp.int32
            )
        return self._compiled_update(
            runner, opponent_pool, opponent_probabilities, opponent_generations
        )

    def update_lineages(
        self,
        runners: RunnerState,
        opponent_pools: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array | None = None,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Update independent learner states batched along a lineage axis."""
        if opponent_generations is None:
            pool_size = jax.tree_util.tree_leaves(opponent_pools)[0].shape[1]
            opponent_generations = jnp.broadcast_to(
                jnp.arange(pool_size, dtype=jnp.int32),
                (runners.env_steps.shape[0], pool_size),
            )
        return self._compiled_update_many(
            runners, opponent_pools, opponent_probabilities, opponent_generations
        )

    def train(
        self,
        runner: RunnerState,
        num_updates: int,
        opponent_pool: dict | None = None,
        opponent_probabilities: jax.Array | None = None,
        opponent_generations: jax.Array | None = None,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Run PPO generations inside one compiled scan."""
        if num_updates < 1:
            raise ValueError("num_updates must be at least 1")
        if opponent_pool is None:
            return self._compiled_train(runner, None, None, None, num_updates=num_updates)
        if opponent_probabilities is None:
            raise ValueError("opponent_probabilities are required with opponent_pool")
        if opponent_generations is None:
            opponent_generations = jnp.arange(
                jax.tree_util.tree_leaves(opponent_pool)[0].shape[0], dtype=jnp.int32
            )
        return self._compiled_train(
            runner,
            opponent_pool,
            opponent_probabilities,
            opponent_generations,
            num_updates=num_updates,
        )

    def train_lineages(
        self,
        runners: RunnerState,
        opponent_pools: dict,
        opponent_probabilities: jax.Array,
        num_updates: int,
        opponent_generations: jax.Array | None = None,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Run a fixed number of updates for independent batched lineages."""
        if num_updates < 1:
            raise ValueError("num_updates must be at least 1")
        if opponent_generations is None:
            pool_size = jax.tree_util.tree_leaves(opponent_pools)[0].shape[1]
            opponent_generations = jnp.broadcast_to(
                jnp.arange(pool_size, dtype=jnp.int32),
                (runners.env_steps.shape[0], pool_size),
            )
        return self._compiled_train_many(
            runners,
            opponent_pools,
            opponent_probabilities,
            opponent_generations,
            num_updates=num_updates,
        )

    def compile_train_lineages(
        self,
        runners: RunnerState,
        opponent_pools: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array,
        num_updates: int,
    ):
        """Compile a fixed-size batched-lineage run without executing it."""
        if num_updates < 1:
            raise ValueError("num_updates must be at least 1")
        return self._compiled_train_many.lower(
            runners,
            opponent_pools,
            opponent_probabilities,
            opponent_generations,
            num_updates=num_updates,
        ).compile()

    def compile_train(self, runner: RunnerState, num_updates: int):
        """Compile, but do not execute, a fixed-length multi-generation train run."""
        if num_updates < 1:
            raise ValueError("num_updates must be at least 1")
        return self._compiled_train_legacy.lower(runner, num_updates=num_updates).compile()

    def act(
        self,
        params: dict,
        observation: jax.Array,
        key: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Return bounded wheel actions for evaluation or inference."""
        first, second, _ = self.model.apply(params, observation)
        if self.policy_architecture in ("brandao_shared", "brandao_sets"):
            if deterministic:
                sample = first / (first + second)
            else:
                if key is None:
                    raise ValueError("key is required for stochastic actions")
                sample = _sample_beta(key, first, second)
            return 2.0 * sample - 1.0
        if deterministic:
            raw_action = first
        else:
            if key is None:
                raise ValueError("key is required for stochastic actions")
            raw_action = first + jnp.exp(second) * jax.random.normal(key, first.shape)
        return jnp.tanh(raw_action)

    def _initialize(self, key: jax.Array) -> RunnerState:
        model_key, reset_key, runner_key = jax.random.split(key, 3)
        dummy_observation = jnp.zeros(self.env.observation_spec.shape, dtype=jnp.float32)
        params = self.model.init(model_key, dummy_observation)
        train_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        env_state, timestep = self._batched_reset(reset_keys)
        return_shape = (
            (self.config.num_envs, 3)
            if self.is_shared_agent_policy
            else (self.config.num_envs,)
        )
        zeros_float = jnp.zeros(return_shape, dtype=jnp.float32)
        zeros_int = jnp.zeros(self.config.num_envs, dtype=jnp.int32)
        return RunnerState(
            train_state=train_state,
            env_state=env_state,
            observation=timestep.observation,
            key=runner_key,
            episode_return=zeros_float,
            episode_length=zeros_int,
            env_steps=jnp.zeros((), dtype=jnp.int32),
            opponent_indices=jnp.zeros(self.config.num_envs, dtype=jnp.int32),
            opponent_modes=jnp.full(self.config.num_envs, 2, dtype=jnp.int32),
            opponent_categories=jnp.full(self.config.num_envs, 4, dtype=jnp.int32),
            opponent_generations=jnp.full(self.config.num_envs, -1, dtype=jnp.int32),
            initial_level_keys=jax.vmap(lambda k: jax.random.split(k, 3)[1])(reset_keys),
            replay_level_keys=jnp.zeros((self.config.curriculum_buffer_size, 2), dtype=jnp.uint32),
            replay_weights=jnp.zeros((self.config.curriculum_buffer_size,), dtype=jnp.float32),
            replay_valid=jnp.zeros((self.config.curriculum_buffer_size,), dtype=jnp.bool_),
        )

    def _sample_opponent(
        self,
        key: jax.Array,
        opponent_pool: dict,
        category_probabilities: jax.Array,
        generation_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pool_size = jax.tree_util.tree_leaves(opponent_pool)[0].shape[0]
        category_key, all_key, recent_key = jax.random.split(key, 3)
        category = jax.random.categorical(
            category_key, jnp.log(category_probabilities)
        ).astype(jnp.int32)
        recent_count = max(1, (pool_size + 1) // 2)
        all_index = jax.random.randint(all_key, (), 0, pool_size, dtype=jnp.int32)
        recent_index = pool_size - recent_count + jax.random.randint(
            recent_key, (), 0, recent_count, dtype=jnp.int32
        )
        opponent_index = jnp.where(
            category == 0,
            all_index,
            jnp.where(category == 1, recent_index, jnp.maximum(pool_size - 1, 0)),
        )
        mode = jnp.where(category < 3, 0, category - 2).astype(jnp.int32)
        generation = jnp.where(mode == 0, generation_ids[opponent_index], -1)
        return opponent_index, mode, category, generation

    def _initialize_self_play(
        self,
        key: jax.Array,
        opponent_pool: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array,
    ) -> RunnerState:
        runner = self._initialize(key)
        sample_key, next_key = jax.random.split(runner.key)
        sample_keys = jax.random.split(sample_key, self.config.num_envs)
        indices, modes, categories, generations = jax.vmap(
            self._sample_opponent, in_axes=(0, None, None, None)
        )(sample_keys, opponent_pool, opponent_probabilities, opponent_generations)
        return runner._replace(
            key=next_key,
            opponent_indices=indices,
            opponent_modes=modes,
            opponent_categories=categories,
            opponent_generations=generations,
        )

    def _apply_opponent_policy(
        self,
        opponent_pool: dict,
        opponent_indices: jax.Array,
        observations: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Apply sampled opponents without storing a full parameter tree per env.

        A single-policy pool is shared across the whole environment batch. For
        larger pools, gather only the selected trees needed for this inference
        call; the runner state retains indices rather than those gathered trees.
        """
        pool_size = jax.tree_util.tree_leaves(opponent_pool)[0].shape[0]
        if pool_size == 1:
            shared_params = jax.tree_util.tree_map(lambda value: value[0], opponent_pool)
            return self.model.apply(shared_params, observations)

        selected_params = jax.tree_util.tree_map(
            lambda value: value[opponent_indices], opponent_pool
        )
        return jax.vmap(self.model.apply)(selected_params, observations)

    def _sample_policy(
        self, params: dict, observation: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        first, second, value = self.model.apply(params, observation)
        if self.policy_architecture in ("brandao_shared", "brandao_sets"):
            sample = _sample_beta(key, first, second)
            log_prob = _beta_log_prob(sample, first, second)
            return 2.0 * sample - 1.0, sample, log_prob, value
        raw_action = first + jnp.exp(second) * jax.random.normal(key, first.shape)
        log_prob = _normal_log_prob(raw_action, first, second)
        return jnp.tanh(raw_action), raw_action, log_prob, value

    def _maybe_reset(
        self,
        done: jax.Array,
        state: EnvState,
        observation: jax.Array,
        key: jax.Array,
        old_opponent_index: jax.Array,
        old_opponent_mode: jax.Array,
        old_opponent_category: jax.Array,
        old_opponent_generation: jax.Array,
        opponent_pool: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array,
        level_key: jax.Array,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def reset(args) -> tuple[
            EnvState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ]:
            reset_key, sample_key = args
            if self.config.curriculum_replay_fraction:
                state_key, _, obs_key = jax.random.split(reset_key, 3)
                reset_state, reset_timestep = self.env.reset_level(level_key, state_key, obs_key)
            else:
                reset_state, reset_timestep = self.env.reset(reset_key)
            index, mode, category, generation = self._sample_opponent(
                sample_key, opponent_pool, opponent_probabilities, opponent_generations
            )
            return (
                reset_state,
                reset_timestep.observation,
                index,
                mode,
                category,
                generation,
            )

        def keep(_: None) -> tuple[
            EnvState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ]:
            return (
                state,
                observation,
                old_opponent_index,
                old_opponent_mode,
                old_opponent_category,
                old_opponent_generation,
            )

        reset_key, sample_key = jax.random.split(key)
        return jax.lax.cond(done, reset, keep, (reset_key, sample_key))

    def _collect_rollout(
        self,
        runner: RunnerState,
        opponent_pool: dict | None,
        opponent_probabilities: jax.Array | None,
        opponent_generations: jax.Array | None,
    ) -> tuple[RunnerState, _Transition]:
        def rollout_step(carry: RunnerState, _: None):
            policy_key, opponent_key, reset_key, next_key = jax.random.split(carry.key, 4)
            if opponent_pool is not None:
                opponent_observation = self._batched_opponent_observation(carry.env_state)
                opponent_first, opponent_second, _ = self._apply_opponent_policy(
                    opponent_pool,
                    carry.opponent_indices,
                    opponent_observation,
                )
            if self.is_shared_agent_policy and opponent_pool is not None:
                first, second, value = self.model.apply(
                    carry.train_state.params, carry.observation
                )
                raw_action, opponent_sample = _sample_beta(
                    policy_key,
                    jnp.stack((first, opponent_first)),
                    jnp.stack((second, opponent_second)),
                )
                action = 2.0 * raw_action - 1.0
                log_prob = _beta_log_prob(raw_action, first, second)
                opponent_action = 2.0 * opponent_sample - 1.0
            else:
                action, raw_action, log_prob, value = self._sample_policy(
                    carry.train_state.params, carry.observation, policy_key
                )
                if opponent_pool is not None:
                    opponent_raw_action = (
                        opponent_first
                        + jnp.exp(opponent_second)
                        * jax.random.normal(opponent_key, opponent_first.shape)
                    )
                    opponent_action = jnp.tanh(opponent_raw_action)
            if self.config.mirror_augmentation:
                mirrored_observation = mirror_x_axis_observation(carry.observation)
                mirrored_first, mirrored_second, mirrored_value = self.model.apply(
                    carry.train_state.params, mirrored_observation
                )
                mirrored_log_prob = _beta_log_prob(
                    raw_action[..., ::-1], mirrored_first, mirrored_second
                )
            else:
                mirrored_log_prob = None
                mirrored_value = None
            if opponent_pool is None:
                stepped_state, timestep = self._batched_step(carry.env_state, action)
            else:
                stepped_state, timestep = self._batched_step_with_opponent(
                    carry.env_state,
                    action,
                    opponent_action,
                    carry.opponent_modes,
                )

            # Compute this before autoreset: a time-limit truncation bootstraps
            # from its final observation, not from the next episode's reset obs.
            _, _, bootstrap_value = self.model.apply(
                carry.train_state.params, timestep.observation
            )
            if self.config.mirror_augmentation:
                _, _, mirrored_bootstrap_value = self.model.apply(
                    carry.train_state.params,
                    mirror_x_axis_observation(timestep.observation),
                )
            else:
                mirrored_bootstrap_value = None
            episode_end = timestep.last()
            running_return = carry.episode_return + timestep.reward
            running_length = carry.episode_length + jnp.int32(1)
            return_episode_end = (
                episode_end[..., None]
                if self.is_shared_agent_policy
                else episode_end
            )
            completed_return = jnp.where(return_episode_end, running_return, 0.0)
            completed_length = jnp.where(episode_end, running_length, 0)

            reset_keys = jax.random.split(reset_key, self.config.num_envs)
            if self.config.curriculum_replay_fraction:
                weights = jnp.where(carry.replay_valid, carry.replay_weights + 1e-3, 0.0)
                weights = jnp.where(jnp.any(carry.replay_valid), weights, jnp.ones_like(weights))
                cumulative = jnp.cumsum(weights) / jnp.sum(weights)
                def select_level(k):
                    fresh_key, choice_key, coin_key = jax.random.split(jax.random.fold_in(k, 99991), 3)
                    selected = jnp.minimum(jnp.searchsorted(cumulative, jax.random.uniform(choice_key)), self.config.curriculum_buffer_size - 1)
                    use_replay = (jax.random.uniform(coin_key) < self.config.curriculum_replay_fraction) & jnp.any(carry.replay_valid)
                    return jnp.where(use_replay, carry.replay_level_keys[selected], fresh_key), use_replay
                level_keys, replay_selected = jax.vmap(select_level)(reset_keys)
            else:
                replay_selected = jnp.zeros_like(episode_end)
                level_keys = jax.vmap(lambda k: jax.random.split(k, 3)[1])(reset_keys)
            if opponent_pool is None:
                next_env_state, next_observation = jax.vmap(self._maybe_reset_legacy)(
                    episode_end, stepped_state, timestep.observation, reset_keys, level_keys
                )
                next_opponent_indices = carry.opponent_indices
                next_opponent_modes = carry.opponent_modes
                next_opponent_categories = carry.opponent_categories
                next_opponent_generations = carry.opponent_generations
            else:
                reset_args = (
                    episode_end,
                    stepped_state,
                    timestep.observation,
                    reset_keys,
                    carry.opponent_indices,
                    carry.opponent_modes,
                    carry.opponent_categories,
                    carry.opponent_generations,
                    level_keys,
                )
                (
                    next_env_state,
                    next_observation,
                    next_opponent_indices,
                    next_opponent_modes,
                    next_opponent_categories,
                    next_opponent_generations,
                ) = jax.vmap(
                    lambda done, state, obs, key, index, mode, category, generation, level: self._maybe_reset(
                        done,
                        state,
                        obs,
                        key,
                        index,
                        mode,
                        category,
                        generation,
                        opponent_pool,
                        opponent_probabilities,
                        opponent_generations,
                        level,
                    )
                )(*reset_args)
            next_runner = RunnerState(
                train_state=carry.train_state,
                env_state=next_env_state,
                observation=next_observation,
                key=next_key,
                episode_return=jnp.where(return_episode_end, 0.0, running_return),
                episode_length=jnp.where(episode_end, 0, running_length),
                env_steps=carry.env_steps + jnp.int32(self.config.num_envs),
                opponent_indices=next_opponent_indices,
                opponent_modes=next_opponent_modes,
                opponent_categories=next_opponent_categories,
                opponent_generations=next_opponent_generations,
                initial_level_keys=jnp.where(episode_end[:, None], level_keys, carry.initial_level_keys),
                replay_level_keys=carry.replay_level_keys,
                replay_weights=carry.replay_weights,
                replay_valid=carry.replay_valid,
            )
            transition_data = _Transition(
                # Consecutive Brandao histories overlap in seven of eight
                # frames. Keep only the newest frame and rebuild histories
                # for the selected optimization minibatch.
                observation=(
                    carry.observation[..., -FRAME_FEATURES:]
                    if self.is_shared_agent_policy
                    else carry.observation
                ),
                raw_action=raw_action,
                old_log_prob=log_prob,
                old_value=value,
                reward=timestep.reward,
                bootstrap_value=bootstrap_value,
                discount=timestep.discount,
                episode_end=episode_end,
                completed_return=completed_return,
                completed_length=completed_length,
                goal=timestep.extras["goal"],
                referee_event=timestep.extras["event"],
                opponent_index=carry.opponent_indices,
                opponent_mode=carry.opponent_modes,
                opponent_category=carry.opponent_categories,
                opponent_generation=carry.opponent_generations,
                level_key=carry.initial_level_keys,
                replay_selected=episode_end & replay_selected,
                mirrored_old_log_prob=mirrored_log_prob,
                mirrored_old_value=mirrored_value,
                mirrored_bootstrap_value=mirrored_bootstrap_value,
            )
            return next_runner, transition_data

        return jax.lax.scan(
            rollout_step,
            runner,
            xs=None,
            length=self.config.rollout_length,
        )

    def _maybe_reset_legacy(
        self,
        done: jax.Array,
        state: EnvState,
        observation: jax.Array,
        key: jax.Array,
        level_key: jax.Array,
    ) -> tuple[EnvState, jax.Array]:
        def reset(reset_key: jax.Array) -> tuple[EnvState, jax.Array]:
            if self.config.curriculum_replay_fraction:
                state_key, _, obs_key = jax.random.split(reset_key, 3)
                reset_state, reset_timestep = self.env.reset_level(level_key, state_key, obs_key)
            else:
                reset_state, reset_timestep = self.env.reset(reset_key)
            return reset_state, reset_timestep.observation

        def keep(_: None) -> tuple[EnvState, jax.Array]:
            return state, observation

        return jax.lax.cond(done, reset, keep, key)

    def _loss(
        self, params: dict, batch: _TrainingBatch
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        first, second, value = self.model.apply(params, batch.observation)
        if self.policy_architecture in ("brandao_shared", "brandao_sets"):
            log_prob = _beta_log_prob(batch.raw_action, first, second)
            entropy = jnp.mean(_beta_entropy(first, second))
        else:
            log_prob = _normal_log_prob(batch.raw_action, first, second)
            entropy = jnp.mean(_normal_entropy(second))
        raw_log_ratio = log_prob - batch.old_log_prob
        # Very large density ratios can overflow float32 before PPO clipping
        # applies. Bound only the numerical tail; ordinary ratios are unchanged.
        log_ratio = jnp.clip(raw_log_ratio, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)

        unclipped_policy = -batch.advantage * ratio
        clipped_policy = -batch.advantage * jnp.clip(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        policy_loss = jnp.mean(jnp.maximum(unclipped_policy, clipped_policy))

        clipped_value = batch.old_value + jnp.clip(
            value - batch.old_value,
            -self.config.clip_epsilon,
            self.config.clip_epsilon,
        )
        value_loss = 0.5 * jnp.mean(
            jnp.maximum(jnp.square(value - batch.target), jnp.square(clipped_value - batch.target))
        )
        total_loss = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
            "clip_fraction": jnp.mean(
                (jnp.abs(ratio - 1.0) > self.config.clip_epsilon).astype(jnp.float32)
            ),
            "ratio_log_clipped_fraction": jnp.mean(
                (jnp.abs(raw_log_ratio) > 20.0).astype(jnp.float32)
            ),
        }
        return total_loss, metrics

    def _optimize(
        self,
        train_state: TrainState,
        batch: _TrainingBatch,
        key: jax.Array,
        reconstruct_observation: Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[TrainState, dict[str, jax.Array], jax.Array]:
        def minibatch_step(state: TrainState, indices: jax.Array):
            if reconstruct_observation is None:
                minibatch = jax.tree_util.tree_map(lambda value: value[indices], batch)
            else:
                minibatch = _TrainingBatch(
                    observation=reconstruct_observation(indices),
                    raw_action=batch.raw_action[indices],
                    old_log_prob=batch.old_log_prob[indices],
                    old_value=batch.old_value[indices],
                    advantage=batch.advantage[indices],
                    target=batch.target[indices],
                    mirrored_old_log_prob=(
                        batch.mirrored_old_log_prob[indices]
                        if batch.mirrored_old_log_prob is not None
                        else None
                    ),
                    mirrored_old_value=(
                        batch.mirrored_old_value[indices]
                        if batch.mirrored_old_value is not None
                        else None
                    ),
                    mirrored_advantage=(
                        batch.mirrored_advantage[indices]
                        if batch.mirrored_advantage is not None
                        else None
                    ),
                    mirrored_target=(
                        batch.mirrored_target[indices]
                        if batch.mirrored_target is not None
                        else None
                    ),
                )
            (_, metrics), grads = jax.value_and_grad(self._loss, has_aux=True)(
                state.params, minibatch
            )
            if self.config.mirror_augmentation:
                mirrored_observation = mirror_x_axis_observation(minibatch.observation)
                mirrored_action = jnp.flip(minibatch.raw_action, axis=-1)
                mirrored_batch = _TrainingBatch(
                    observation=mirrored_observation,
                    raw_action=mirrored_action,
                    old_log_prob=minibatch.mirrored_old_log_prob,
                    old_value=minibatch.mirrored_old_value,
                    advantage=minibatch.mirrored_advantage,
                    target=minibatch.mirrored_target,
                )
                (_, mirrored_metrics), mirrored_grads = jax.value_and_grad(
                    self._loss, has_aux=True
                )(state.params, mirrored_batch)
                grads = jax.tree_util.tree_map(
                    lambda original, reflected: (original + reflected) * 0.5,
                    grads,
                    mirrored_grads,
                )
                metrics = jax.tree_util.tree_map(
                    lambda original, reflected: (original + reflected) * 0.5,
                    metrics,
                    mirrored_metrics,
                )
            return state.apply_gradients(grads=grads), metrics

        def epoch_step(carry: tuple[TrainState, jax.Array], _: None):
            state, epoch_key = carry
            epoch_key, permutation_key = jax.random.split(epoch_key)
            batch_size = batch.observation.shape[0]
            minibatch_size = batch_size // self.config.num_minibatches
            minibatch_indices = jax.random.permutation(
                permutation_key, batch_size
            ).reshape(
                self.config.num_minibatches, minibatch_size
            )
            state, metrics = jax.lax.scan(minibatch_step, state, minibatch_indices)
            return (state, epoch_key), metrics

        (train_state, key), metrics = jax.lax.scan(
            epoch_step,
            (train_state, key),
            xs=None,
            length=self.config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return train_state, metrics, key

    def _update(
        self,
        runner: RunnerState,
        opponent_pool: dict | None = None,
        opponent_probabilities: jax.Array | None = None,
        opponent_generations: jax.Array | None = None,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        initial_observation = runner.observation
        rollout_runner, trajectory = self._collect_rollout(
            runner, opponent_pool, opponent_probabilities, opponent_generations
        )
        curriculum_return_min = jnp.float32(0.0)
        curriculum_return_max = jnp.float32(0.0)
        if self.config.curriculum_replay_fraction:
            valid = trajectory.episode_end.reshape(-1)
            returns = trajectory.completed_return
            if self.is_shared_agent_policy:
                returns = jnp.mean(returns, axis=-1)
            returns = returns.reshape(-1)
            score, curriculum_return_min, curriculum_return_max = _curriculum_scores(returns, valid)
            keys = trajectory.level_key.reshape((-1, 2))
            capacity = self.config.curriculum_buffer_size
            # Pad so top_k also works for short rollout batches.
            valid = jnp.concatenate((valid, jnp.zeros(capacity, dtype=jnp.bool_)))
            score = jnp.concatenate((score, jnp.zeros(capacity)))
            keys = jnp.concatenate((keys, jnp.zeros((capacity, 2), dtype=jnp.uint32)))
            sample_key, next_key = jax.random.split(rollout_runner.key)
            random_order = jax.random.uniform(sample_key, valid.shape)
            _, chosen = jax.lax.top_k(jnp.where(valid, random_order, -1.0), capacity)
            any_valid = jnp.any(valid)
            rollout_runner = rollout_runner._replace(
                key=next_key,
                replay_level_keys=jnp.where(any_valid, keys[chosen], rollout_runner.replay_level_keys),
                replay_weights=jnp.where(any_valid, score[chosen], rollout_runner.replay_weights),
                replay_valid=jnp.where(any_valid, valid[chosen], rollout_runner.replay_valid),
            )
        episode_ends = trajectory.episode_end
        discounts = trajectory.discount
        if trajectory.reward.ndim > episode_ends.ndim:
            episode_ends = episode_ends[..., None]
            discounts = discounts[..., None]
        advantages, targets = compute_gae(
            rewards=trajectory.reward,
            values=trajectory.old_value,
            bootstrap_values=trajectory.bootstrap_value,
            discounts=discounts,
            episode_ends=episode_ends,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if self.config.mirror_augmentation:
            mirrored_advantages, mirrored_targets = compute_gae(
                rewards=trajectory.reward,
                values=trajectory.mirrored_old_value,
                bootstrap_values=trajectory.mirrored_bootstrap_value,
                discounts=discounts,
                episode_ends=episode_ends,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            combined_mean = 0.5 * (
                jnp.mean(advantages) + jnp.mean(mirrored_advantages)
            )
            combined_variance = 0.5 * (
                jnp.mean(jnp.square(advantages - combined_mean))
                + jnp.mean(jnp.square(mirrored_advantages - combined_mean))
            )
            combined_std = jnp.sqrt(combined_variance + 1e-8)
            advantages = (advantages - combined_mean) / combined_std
            mirrored_advantages = (
                mirrored_advantages - combined_mean
            ) / combined_std
        else:
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + 1e-8
            )
            mirrored_advantages = None
            mirrored_targets = None
        batch = _TrainingBatch(
            observation=trajectory.observation,
            raw_action=trajectory.raw_action,
            old_log_prob=trajectory.old_log_prob,
            old_value=trajectory.old_value,
            advantage=advantages,
            target=targets,
            mirrored_old_log_prob=trajectory.mirrored_old_log_prob,
            mirrored_old_value=trajectory.mirrored_old_value,
            mirrored_advantage=mirrored_advantages,
            mirrored_target=mirrored_targets,
        )
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((self.config.batch_size,) + x.shape[2:]), batch
        )
        if self.is_shared_agent_policy:
            rollout_frames = trajectory.observation
            episode_ends_for_observation = trajectory.episode_end

            def reconstruct_observation(indices: jax.Array) -> jax.Array:
                return _reconstruct_brandao_observations(
                    initial_observation,
                    rollout_frames,
                    episode_ends_for_observation,
                    indices,
                )

        else:
            reconstruct_observation = None
        train_state, metrics, key = self._optimize(
            rollout_runner.train_state,
            batch,
            rollout_runner.key,
            reconstruct_observation,
        )
        runner = rollout_runner._replace(train_state=train_state, key=key)

        episode_count = jnp.sum(trajectory.episode_end)
        safe_episode_count = jnp.maximum(episode_count, 1)
        blue_goals_by_env = jnp.sum(trajectory.goal == 1, axis=0)
        yellow_goals_by_env = jnp.sum(trajectory.goal == -1, axis=0)
        metrics.update(
            {
                "curriculum_replay_resets": jnp.sum(trajectory.replay_selected),
                "curriculum_total_resets": jnp.sum(trajectory.episode_end),
                "curriculum_buffer_count": jnp.sum(runner.replay_valid),
                "curriculum_return_min": curriculum_return_min,
                "curriculum_return_max": curriculum_return_max,
                "mean_reward": jnp.mean(trajectory.reward),
                "agent_transitions": runner.env_steps
                * (3 if self.is_shared_agent_policy else 1),
                "episodes": episode_count,
                "mean_episode_return": jnp.sum(trajectory.completed_return)
                / (
                    safe_episode_count * 3
                    if self.is_shared_agent_policy
                    else safe_episode_count
                ),
                "mean_episode_length": jnp.sum(trajectory.completed_length)
                / safe_episode_count,
                "blue_goals": jnp.sum(blue_goals_by_env),
                "yellow_goals": jnp.sum(yellow_goals_by_env),
                "blue_goals_by_env": blue_goals_by_env,
                "yellow_goals_by_env": yellow_goals_by_env,
                "mean_blue_goals_per_env": jnp.mean(blue_goals_by_env),
                "mean_yellow_goals_per_env": jnp.mean(yellow_goals_by_env),
                "mean_goal_delta_per_env": jnp.mean(
                    blue_goals_by_env - yellow_goals_by_env
                ),
                "referee_event_counts": jnp.bincount(
                    trajectory.referee_event.reshape(-1), length=len(RefereeEvent)
                ),
                "env_steps": runner.env_steps,
            }
        )
        if opponent_pool is not None:
            metrics.update(
                {
                    "opponent_selector_counts": jnp.bincount(
                        trajectory.opponent_category.reshape(-1), length=5
                    ),
                    "opponent_generations": trajectory.opponent_generation,
                }
            )
        return runner, metrics

    def _train(
        self,
        runner: RunnerState,
        opponent_pool: dict | None,
        opponent_probabilities: jax.Array | None,
        opponent_generations: jax.Array | None,
        num_updates: int,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Scan complete PPO updates without returning control to Python."""
        return jax.lax.scan(
            lambda carry, _: self._update(
                carry, opponent_pool, opponent_probabilities, opponent_generations
            ),
            runner,
            xs=None,
            length=num_updates,
        )

    def _train_legacy(
        self, runner: RunnerState, num_updates: int
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Keep the original one-argument compiled callable for legacy callers."""
        return jax.lax.scan(
            lambda carry, _: self._update(carry, None, None, None),
            runner,
            xs=None,
            length=num_updates,
        )

    def _train_many(
        self,
        runners: RunnerState,
        opponent_pools: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array,
        num_updates: int,
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Scan PPO updates while keeping each lineage on its own vmap lane."""
        return jax.lax.scan(
            lambda carry, _: self._compiled_update_many(
                carry, opponent_pools, opponent_probabilities, opponent_generations
            ),
            runners,
            xs=None,
            length=num_updates,
        )
