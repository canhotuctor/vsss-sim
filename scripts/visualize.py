#!/usr/bin/env python3
"""Render episodes using a policy snapshot produced by native JAX PPO."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from vsss_sim import config
from vsss_sim.config import InitMode, RewardProfile
from vsss_sim.envs import VSSEnv
from vsss_sim.envs.brandao_observation import (
    advance_history,
    build_agent_features,
    flatten_history,
    initialize_history,
)
from vsss_sim.envs.match_referee import MatchPhase
from vsss_sim.rl.ppo import create_policy_model


def load_snapshot(snapshot_path: Path):
    """Load parameters and their experiment settings from an archived snapshot."""
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"policy snapshot not found: {snapshot_path}")

    metadata_path = snapshot_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"snapshot metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported snapshot schema in {metadata_path}")
    if metadata.get("artifact") != snapshot_path.name:
        raise ValueError(f"snapshot metadata artifact does not match {snapshot_path.name}")

    experiment_path = snapshot_path.parents[2] / "config.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {experiment_path}")
    experiment_config = json.loads(experiment_path.read_text(encoding="utf-8"))

    policy_architecture = metadata.get("policy_architecture", "joint")
    model = create_policy_model(
        policy_architecture,
        tuple(int(size) for size in metadata["hidden_sizes"]),
    )
    dummy_observation = jnp.zeros(
        (int(metadata["observation_size"]),), dtype=jnp.float32
    )
    template = model.init(jax.random.PRNGKey(0), dummy_observation)
    params = serialization.from_bytes(template, snapshot_path.read_bytes())
    return model, params, metadata, experiment_config


def resolve_simulation_timing(
    experiment_config: dict, policy_architecture: str
) -> tuple[float, int]:
    """Resolve the archived control cadence, including legacy config defaults."""
    is_brandao = policy_architecture in ("brandao_shared", "brandao_sets")
    default_rate_hz = 10.0 if is_brandao else config.FPS
    default_substeps = 6 if is_brandao else config.SUB_STEPS

    if experiment_config.get("control_rate_hz") is not None:
        control_rate_hz = experiment_config["control_rate_hz"]
    elif experiment_config.get("control_fps") is not None:
        control_rate_hz = experiment_config["control_fps"]
    elif experiment_config.get("control_dt") is not None:
        control_dt = experiment_config["control_dt"]
        if (
            isinstance(control_dt, bool)
            or not isinstance(control_dt, (int, float))
            or not math.isfinite(control_dt)
            or control_dt <= 0
        ):
            raise ValueError("experiment control_dt must be a positive number")
        control_rate_hz = 1.0 / control_dt
    else:
        control_rate_hz = default_rate_hz

    physics_substeps = experiment_config.get("physics_substeps")
    if physics_substeps is None:
        physics_substeps = experiment_config.get("sub_steps", default_substeps)
    if (
        isinstance(control_rate_hz, bool)
        or not isinstance(control_rate_hz, (int, float))
        or not math.isfinite(control_rate_hz)
        or control_rate_hz <= 0
    ):
        raise ValueError("experiment control_rate_hz must be a positive finite number")
    if (
        isinstance(physics_substeps, bool)
        or not isinstance(physics_substeps, (int, float))
        or not math.isfinite(physics_substeps)
        or int(physics_substeps) != physics_substeps
        or physics_substeps < 1
    ):
        raise ValueError("experiment physics_substeps must be a positive integer")
    return float(control_rate_hz), int(physics_substeps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        type=Path,
        help="JAX snapshot .msgpack file, e.g. output/experiments/<name>/lineage_000/snapshots/generation_00000010.msgpack",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--speedup-rate",
        type=float,
        default=1.0,
        help="Maximum playback speed relative to real time (default: 1.0x).",
    )
    parser.add_argument(
        "--opponent", choices=("stationary", "random"), default="stationary"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--init-mode", choices=tuple(mode.value for mode in InitMode))
    parser.add_argument(
        "--reward-profile", choices=tuple(profile.value for profile in RewardProfile)
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("max-episode-steps must be positive")
    if not math.isfinite(args.speedup_rate) or args.speedup_rate <= 0.0:
        parser.error("speedup-rate must be a positive finite number")

    model, params, metadata, experiment_config = load_snapshot(args.snapshot)
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else experiment_config.get("max_episode_steps", config.MAX_EPISODE_STEPS)
    )
    policy_architecture = metadata.get("policy_architecture", "joint")
    try:
        control_rate_hz, physics_substeps = resolve_simulation_timing(
            experiment_config, policy_architecture
        )
    except ValueError as error:
        parser.error(str(error))
    policy_episode_steps = max_episode_steps
    init_mode = args.init_mode or experiment_config.get("init_mode", InitMode.KICKOFF.value)
    reward_profile = args.reward_profile or experiment_config.get(
        "reward_profile", RewardProfile.DEFAULT.value
    )
    env = VSSEnv(
        opponent_policy=args.opponent,
        render_mode="human",
        control_rate_hz=control_rate_hz,
        physics_substeps=physics_substeps,
        speedup_rate=args.speedup_rate,
        max_episode_steps=max_episode_steps,
        init_mode=init_mode,
        reward_profile=reward_profile,
    )
    key = jax.random.PRNGKey(args.seed)

    def action_for(observation: np.ndarray) -> np.ndarray:
        nonlocal key
        if policy_architecture in ("brandao_shared", "brandao_sets"):
            first, second, _ = model.apply(
                params, jnp.asarray(observation, dtype=jnp.float32)
            )
            if args.stochastic:
                key, sample_key = jax.random.split(key)
                sample = jax.random.beta(sample_key, first, second)
                action = 2.0 * sample - 1.0
            else:
                action = 2.0 * first / (first + second) - 1.0
            return np.asarray(jax.device_get(action), dtype=np.float32)
        mean, log_std, _ = model.apply(params, jnp.asarray(observation, dtype=jnp.float32))
        raw_action = mean
        if args.stochastic:
            key, sample_key = jax.random.split(key)
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(sample_key, mean.shape)
        return np.asarray(jax.device_get(jnp.tanh(raw_action)), dtype=np.float32)

    print(
        f"snapshot generation={metadata['generation']} lineage={metadata['lineage']} "
        f"— architecture={policy_architecture}, opponent={args.opponent}, "
        f"control={control_rate_hz:g} Hz × {physics_substeps} substeps, "
        f"speedup cap={args.speedup_rate:g}x, init={init_mode}, "
        f"stochastic={args.stochastic}"
    )
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    goals_blue = goals_yellow = 0
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            reward_total = 0.0
            score_blue = score_yellow = 0
            if policy_architecture in ("brandao_shared", "brandao_sets"):
                previous_actions = jnp.zeros((2, 3, 2), dtype=jnp.float32)
                observation_key = jax.random.PRNGKey(args.seed + episode)
                first_features = build_agent_features(
                    env._state,
                    previous_actions,
                    jnp.float32(1.0),
                    observation_key,
                    entity_ordered_actions=policy_architecture == "brandao_sets",
                )
                history = initialize_history(first_features)
                policy_steps = 0
                done = False
                while policy_steps < policy_episode_steps and not done:
                    robot_observation = flatten_history(history, config.TEAM_BLUE)
                    blue_action = action_for(robot_observation)
                    # VSSEnv zeroes every command on a STOPPED step; record what
                    # was applied, as VSSJumanjiEnv does during training.
                    stopped = int(np.asarray(env._referee_state.phase)) == MatchPhase.STOPPED
                    observation, reward, terminated, truncated, info = env.step(
                        blue_action.reshape(-1)
                    )
                    reward_total += float(reward)
                    goal = int(info.get("goal", 0))
                    if goal:
                        goals_blue += goal == 1
                        goals_yellow += goal == -1
                    score_blue = int(info["score_blue"])
                    score_yellow = int(info["score_yellow"])
                    done = bool(goal or terminated or truncated)
                    previous_actions = previous_actions.at[config.TEAM_BLUE].set(
                        np.zeros_like(blue_action) if stopped else blue_action
                    )
                    policy_steps += 1
                    if not done and policy_steps < policy_episode_steps:
                        remaining = jnp.float32(
                            1.0 - policy_steps / policy_episode_steps
                        )
                        observation_key, sample_key = jax.random.split(observation_key)
                        features = build_agent_features(
                            env._state,
                            previous_actions,
                            remaining,
                            sample_key,
                            entity_ordered_actions=policy_architecture == "brandao_sets",
                        )
                        history = advance_history(
                            history, features, jnp.asarray(True)
                        )
                steps = policy_steps
            else:
                steps = 0
                done = False
                while not done:
                    observation, reward, terminated, truncated, info = env.step(
                        action_for(observation)
                    )
                    reward_total += float(reward)
                    steps += 1
                    goal = int(info.get("goal", 0))
                    goals_blue += goal == 1
                    goals_yellow += goal == -1
                    score_blue = int(info["score_blue"])
                    score_yellow = int(info["score_yellow"])
                    done = bool(terminated or truncated)

            episode_rewards.append(reward_total)
            episode_lengths.append(steps)
            print(
                f"  episode {episode:>2}: reward={reward_total:+.3f} "
                f"length={steps} score(blue:yellow)="
                f"{score_blue}:{score_yellow}"
            )
    finally:
        env.close()

    print(
        f"\nsummary over {args.episodes} episode(s): "
        f"mean reward={np.mean(episode_rewards):+.3f} "
        f"mean length={np.mean(episode_lengths):.0f} "
        f"goals blue:yellow={goals_blue}:{goals_yellow}"
    )


if __name__ == "__main__":
    main()
