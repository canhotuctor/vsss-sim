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


class RunnerState(NamedTuple):
    """All mutable training state carried from one PPO update to the next."""

    train_state: TrainState
    env_state: EnvState
    observation: jax.Array
    key: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    env_steps: jax.Array


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
        self.model = ActorCritic(
            action_size=env.action_spec.shape[0],
            hidden_sizes=config.hidden_sizes,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
        self._batched_reset = jax.vmap(env.reset)
        self._batched_step = jax.vmap(env.step)
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)

    def initialize(self, key: jax.Array) -> RunnerState:
        """Initialize model, optimizer, and VMAP'd environments."""
        return self._compiled_initialize(key)

    def update(self, runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        """Collect one rollout and perform all PPO epochs in one JIT call."""
        return self._compiled_update(runner)

    def act(
        self,
        params: dict,
        observation: jax.Array,
        key: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Return bounded wheel actions for evaluation or inference."""
        mean, log_std, _ = self.model.apply(params, observation)
        if deterministic:
            raw_action = mean
        else:
            if key is None:
                raise ValueError("key is required for stochastic actions")
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
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
        zeros_float = jnp.zeros(self.config.num_envs, dtype=jnp.float32)
        zeros_int = jnp.zeros(self.config.num_envs, dtype=jnp.int32)
        return RunnerState(
            train_state=train_state,
            env_state=env_state,
            observation=timestep.observation,
            key=runner_key,
            episode_return=zeros_float,
            episode_length=zeros_int,
            env_steps=jnp.zeros((), dtype=jnp.int32),
        )

    def _sample_policy(
        self, params: dict, observation: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        mean, log_std, value = self.model.apply(params, observation)
        raw_action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        log_prob = _normal_log_prob(raw_action, mean, log_std)
        return jnp.tanh(raw_action), raw_action, log_prob, value

    def _maybe_reset(
        self,
        done: jax.Array,
        state: EnvState,
        observation: jax.Array,
        key: jax.Array,
    ) -> tuple[EnvState, jax.Array]:
        def reset(reset_key: jax.Array) -> tuple[EnvState, jax.Array]:
            reset_state, reset_timestep = self.env.reset(reset_key)
            return reset_state, reset_timestep.observation

        def keep(_: jax.Array) -> tuple[EnvState, jax.Array]:
            return state, observation

        return jax.lax.cond(done, reset, keep, key)

    def _collect_rollout(self, runner: RunnerState) -> tuple[RunnerState, _Transition]:
        def rollout_step(carry: RunnerState, _: None):
            policy_key, reset_key, next_key = jax.random.split(carry.key, 3)
            action, raw_action, log_prob, value = self._sample_policy(
                carry.train_state.params, carry.observation, policy_key
            )
            stepped_state, timestep = self._batched_step(carry.env_state, action)

            # Compute this before autoreset: a time-limit truncation bootstraps
            # from its final observation, not from the next episode's reset obs.
            _, _, bootstrap_value = self.model.apply(
                carry.train_state.params, timestep.observation
            )
            episode_end = timestep.last()
            running_return = carry.episode_return + timestep.reward
            running_length = carry.episode_length + jnp.int32(1)
            completed_return = jnp.where(episode_end, running_return, 0.0)
            completed_length = jnp.where(episode_end, running_length, 0)

            reset_keys = jax.random.split(reset_key, self.config.num_envs)
            next_env_state, next_observation = jax.vmap(self._maybe_reset)(
                episode_end, stepped_state, timestep.observation, reset_keys
            )
            next_runner = RunnerState(
                train_state=carry.train_state,
                env_state=next_env_state,
                observation=next_observation,
                key=next_key,
                episode_return=jnp.where(episode_end, 0.0, running_return),
                episode_length=jnp.where(episode_end, 0, running_length),
                env_steps=carry.env_steps + jnp.int32(self.config.num_envs),
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
            )
            return next_runner, transition_data

        return jax.lax.scan(
            rollout_step,
            runner,
            xs=None,
            length=self.config.rollout_length,
        )

    def _loss(
        self, params: dict, batch: _TrainingBatch
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        mean, log_std, value = self.model.apply(params, batch.observation)
        log_prob = _normal_log_prob(batch.raw_action, mean, log_std)
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
        entropy = jnp.mean(_normal_entropy(log_std))
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

    def _update(self, runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        rollout_runner, trajectory = self._collect_rollout(runner)
        advantages, targets = compute_gae(
            rewards=trajectory.reward,
            values=trajectory.old_value,
            bootstrap_values=trajectory.bootstrap_value,
            discounts=trajectory.discount,
            episode_ends=trajectory.episode_end,
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
                "episodes": episode_count,
                "mean_episode_return": jnp.sum(trajectory.completed_return)
                / safe_episode_count,
                "mean_episode_length": jnp.sum(trajectory.completed_length)
                / safe_episode_count,
                "blue_goals": jnp.sum(trajectory.goal == 1),
                "yellow_goals": jnp.sum(trajectory.goal == -1),
                "env_steps": runner.env_steps,
            }
        )
        return runner, metrics
