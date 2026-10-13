"""End-to-end JAX PPO for the VSSS Jumanji environment.

The hot path is entirely device-resident: Flax policy inference, VMAP'd
environment stepping, ``lax.scan`` rollout collection and GAE, and Optax
minibatch updates. Python is only needed between PPO updates for logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from ..envs.jumanji import State as EnvState
from ..envs.jumanji import VSSJumanjiEnv

_LOG_2PI = jnp.log(jnp.float32(2.0 * jnp.pi))


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

    def __post_init__(self) -> None:
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


def create_policy_model(
    policy_architecture: str, hidden_sizes: tuple[int, ...] = (64, 64)
) -> nn.Module:
    """Construct the selected checkpoint-compatible actor-critic model."""
    if policy_architecture == "joint":
        return ActorCritic(action_size=6, hidden_sizes=hidden_sizes)
    if policy_architecture == "brandao_shared":
        return BrandaoSharedActorCritic()
    raise ValueError(
        "policy_architecture must be 'joint' or 'brandao_shared'"
    )


def deterministic_policy_action(
    model: nn.Module,
    params: dict,
    observation: jax.Array,
    policy_architecture: str,
) -> jax.Array:
    """Map observations to the selected policy's deterministic wheel speeds."""
    first, second, _ = model.apply(params, observation)
    if policy_architecture == "brandao_shared":
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
    opponent_params: dict | None
    opponent_indices: jax.Array
    opponent_modes: jax.Array
    opponent_categories: jax.Array
    opponent_generations: jax.Array


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
    opponent_index: jax.Array
    opponent_mode: jax.Array
    opponent_category: jax.Array
    opponent_generation: jax.Array


class _TrainingBatch(NamedTuple):
    observation: jax.Array
    raw_action: jax.Array
    old_log_prob: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    target: jax.Array


def _normal_log_prob(
    raw_action: jax.Array, mean: jax.Array, log_std: jax.Array
) -> jax.Array:
    variance_term = jnp.square((raw_action - mean) / jnp.exp(log_std))
    return jnp.sum(-0.5 * (variance_term + 2.0 * log_std + _LOG_2PI), axis=-1)


def _normal_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (1.0 + _LOG_2PI), axis=-1)


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
        return self.policy_architecture == "brandao_shared"

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
        if self.policy_architecture == "brandao_shared":
            if deterministic:
                sample = first / (first + second)
            else:
                if key is None:
                    raise ValueError("key is required for stochastic actions")
                sample = jax.random.beta(key, first, second)
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
            opponent_params=None,
            opponent_indices=jnp.zeros(self.config.num_envs, dtype=jnp.int32),
            opponent_modes=jnp.full(self.config.num_envs, 2, dtype=jnp.int32),
            opponent_categories=jnp.full(self.config.num_envs, 4, dtype=jnp.int32),
            opponent_generations=jnp.full(self.config.num_envs, -1, dtype=jnp.int32),
        )

    def _sample_opponent(
        self,
        key: jax.Array,
        opponent_pool: dict,
        category_probabilities: jax.Array,
        generation_ids: jax.Array,
    ) -> tuple[dict, jax.Array, jax.Array, jax.Array, jax.Array]:
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
        params = jax.tree_util.tree_map(lambda value: value[opponent_index], opponent_pool)
        generation = jnp.where(mode == 0, generation_ids[opponent_index], -1)
        return params, opponent_index, mode, category, generation

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
        params, indices, modes, categories, generations = jax.vmap(
            self._sample_opponent, in_axes=(0, None, None, None)
        )(sample_keys, opponent_pool, opponent_probabilities, opponent_generations)
        return runner._replace(
            key=next_key,
            opponent_params=params,
            opponent_indices=indices,
            opponent_modes=modes,
            opponent_categories=categories,
            opponent_generations=generations,
        )

    def _sample_policy(
        self, params: dict, observation: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        first, second, value = self.model.apply(params, observation)
        if self.policy_architecture == "brandao_shared":
            sample = jax.random.beta(key, first, second)
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
        old_opponent_params: dict,
        old_opponent_index: jax.Array,
        old_opponent_mode: jax.Array,
        old_opponent_category: jax.Array,
        old_opponent_generation: jax.Array,
        opponent_pool: dict,
        opponent_probabilities: jax.Array,
        opponent_generations: jax.Array,
    ) -> tuple[EnvState, jax.Array, dict, jax.Array, jax.Array, jax.Array, jax.Array]:
        def reset(args) -> tuple[
            EnvState, jax.Array, dict, jax.Array, jax.Array, jax.Array, jax.Array
        ]:
            reset_key, sample_key = args
            reset_state, reset_timestep = self.env.reset(reset_key)
            params, index, mode, category, generation = self._sample_opponent(
                sample_key, opponent_pool, opponent_probabilities, opponent_generations
            )
            return (
                reset_state,
                reset_timestep.observation,
                params,
                index,
                mode,
                category,
                generation,
            )

        def keep(_: None) -> tuple[
            EnvState, jax.Array, dict, jax.Array, jax.Array, jax.Array, jax.Array
        ]:
            return (
                state,
                observation,
                old_opponent_params,
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
            action, raw_action, log_prob, value = self._sample_policy(
                carry.train_state.params, carry.observation, policy_key
            )
            if opponent_pool is None:
                stepped_state, timestep = self._batched_step(carry.env_state, action)
            else:
                opponent_observation = self._batched_opponent_observation(carry.env_state)
                first, second, _ = jax.vmap(self.model.apply)(
                    carry.opponent_params, opponent_observation
                )
                if self.policy_architecture == "brandao_shared":
                    opponent_sample = jax.random.beta(opponent_key, first, second)
                    opponent_action = 2.0 * opponent_sample - 1.0
                else:
                    opponent_raw_action = first + jnp.exp(second) * jax.random.normal(
                        opponent_key, first.shape
                    )
                    opponent_action = jnp.tanh(opponent_raw_action)
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
            if opponent_pool is None:
                next_env_state, next_observation = jax.vmap(self._maybe_reset_legacy)(
                    episode_end, stepped_state, timestep.observation, reset_keys
                )
                next_opponent_params = None
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
                    carry.opponent_params,
                    carry.opponent_indices,
                    carry.opponent_modes,
                    carry.opponent_categories,
                    carry.opponent_generations,
                )
                (
                    next_env_state,
                    next_observation,
                    next_opponent_params,
                    next_opponent_indices,
                    next_opponent_modes,
                    next_opponent_categories,
                    next_opponent_generations,
                ) = jax.vmap(
                    lambda done, state, obs, key, params, index, mode, category, generation: self._maybe_reset(
                        done,
                        state,
                        obs,
                        key,
                        params,
                        index,
                        mode,
                        category,
                        generation,
                        opponent_pool,
                        opponent_probabilities,
                        opponent_generations,
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
                opponent_params=next_opponent_params,
                opponent_indices=next_opponent_indices,
                opponent_modes=next_opponent_modes,
                opponent_categories=next_opponent_categories,
                opponent_generations=next_opponent_generations,
            )
            transition_data = _Transition(
                observation=carry.observation,
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
                opponent_index=carry.opponent_indices,
                opponent_mode=carry.opponent_modes,
                opponent_category=carry.opponent_categories,
                opponent_generation=carry.opponent_generations,
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
    ) -> tuple[EnvState, jax.Array]:
        def reset(reset_key: jax.Array) -> tuple[EnvState, jax.Array]:
            reset_state, reset_timestep = self.env.reset(reset_key)
            return reset_state, reset_timestep.observation

        def keep(_: None) -> tuple[EnvState, jax.Array]:
            return state, observation

        return jax.lax.cond(done, reset, keep, key)

    def _loss(
        self, params: dict, batch: _TrainingBatch
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        first, second, value = self.model.apply(params, batch.observation)
        if self.policy_architecture == "brandao_shared":
            log_prob = _beta_log_prob(batch.raw_action, first, second)
            entropy = jnp.mean(_beta_entropy(first, second))
        else:
            log_prob = _normal_log_prob(batch.raw_action, first, second)
            entropy = jnp.mean(_normal_entropy(second))
        log_ratio = log_prob - batch.old_log_prob
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
        }
        return total_loss, metrics

    def _optimize(
        self,
        train_state: TrainState,
        batch: _TrainingBatch,
        key: jax.Array,
    ) -> tuple[TrainState, dict[str, jax.Array], jax.Array]:
        def minibatch_step(state: TrainState, minibatch: _TrainingBatch):
            (_, metrics), grads = jax.value_and_grad(self._loss, has_aux=True)(
                state.params, minibatch
            )
            return state.apply_gradients(grads=grads), metrics

        def epoch_step(carry: tuple[TrainState, jax.Array], _: None):
            state, epoch_key = carry
            epoch_key, permutation_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(permutation_key, self.config.batch_size)
            shuffled = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (self.config.num_minibatches, self.config.minibatch_size) + x.shape[1:]
                ),
                shuffled,
            )
            state, metrics = jax.lax.scan(minibatch_step, state, minibatches)
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
        rollout_runner, trajectory = self._collect_rollout(
            runner, opponent_pool, opponent_probabilities, opponent_generations
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
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        batch = _TrainingBatch(
            observation=trajectory.observation,
            raw_action=trajectory.raw_action,
            old_log_prob=trajectory.old_log_prob,
            old_value=trajectory.old_value,
            advantage=advantages,
            target=targets,
        )
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((self.config.batch_size,) + x.shape[2:]), batch
        )
        train_state, metrics, key = self._optimize(
            rollout_runner.train_state, batch, rollout_runner.key
        )
        runner = rollout_runner._replace(train_state=train_state, key=key)

        episode_count = jnp.sum(trajectory.episode_end)
        safe_episode_count = jnp.maximum(episode_count, 1)
        metrics.update(
            {
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
                "blue_goals": jnp.sum(trajectory.goal == 1),
                "yellow_goals": jnp.sum(trajectory.goal == -1),
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
