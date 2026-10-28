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
from vsss_sim.config import InitMode, PhysicsBackend, RefereeMode, RewardProfile
from vsss_sim.envs import VSSEnv
from vsss_sim.envs.brandao_observation import (
    advance_history,
    build_agent_features,
    flatten_history,
    initialize_history,
)
from vsss_sim.envs.match_referee import MatchPhase
from vsss_sim.perspective import switch_team_perspective
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


def resolve_simulation_timing(experiment_config: dict) -> tuple[float, int]:
    """Resolve the archived control cadence, including legacy config defaults."""
    default_rate_hz = config.FPS
    default_substeps = config.SUB_STEPS

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
    opponent_group = parser.add_mutually_exclusive_group()
    opponent_group.add_argument(
        "--opponent", choices=("stationary", "random"), default="stationary"
    )
    opponent_group.add_argument(
        "--opponent-snapshot",
        type=Path,
        help="Saved .msgpack policy to play as yellow instead of a fixed opponent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions for both saved policies (default: deterministic).",
    )
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--init-mode", choices=tuple(mode.value for mode in InitMode))
    parser.add_argument(
        "--reward-profile", choices=tuple(profile.value for profile in RewardProfile)
    )
    parser.add_argument(
        "--referee-mode",
        choices=tuple(mode.value for mode in RefereeMode),
        help="Referee calls; defaults to the experiment's mode, or full.",
    )
    parser.add_argument(
        "--physics-backend",
        choices=tuple(backend.value for backend in PhysicsBackend),
        help="Physics engine; defaults to the experiment's engine, or jax.",
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
    opponent_model = opponent_params = opponent_metadata = None
    if args.opponent_snapshot is not None:
        opponent_model, opponent_params, opponent_metadata, opponent_config = load_snapshot(
            args.opponent_snapshot
        )
    try:
        control_rate_hz, physics_substeps = resolve_simulation_timing(experiment_config)
        if opponent_metadata is not None:
            opponent_timing = resolve_simulation_timing(opponent_config)
            if opponent_timing != (control_rate_hz, physics_substeps):
                print(
                    "Note: opponent was trained with control timing "
                    f"{opponent_timing[0]:g} Hz × {opponent_timing[1]} substeps; "
                    "the blue policy's timing is used for this match."
                )
    except ValueError as error:
        parser.error(str(error))
    policy_episode_steps = max_episode_steps
    init_mode = args.init_mode or experiment_config.get("init_mode", InitMode.KICKOFF.value)
    reward_profile = args.reward_profile or experiment_config.get(
        "reward_profile", RewardProfile.DEFAULT.value
    )
    referee_mode = args.referee_mode or experiment_config.get(
        "referee_mode", RefereeMode.FULL.value
    )
    physics_backend = args.physics_backend or experiment_config.get(
        "physics_backend", PhysicsBackend.JAX.value
    )
    # The environment invokes the opponent once per control step. For a
    # saved policy, compute its action from the same pre-step state as blue.
    pending_yellow_action = np.zeros((config.N_ROBOTS, 2), dtype=np.float32)
    def saved_opponent(_observation):
        return pending_yellow_action

    env = VSSEnv(
        opponent_policy=saved_opponent if opponent_metadata is not None else args.opponent,
        render_mode="human",
        control_rate_hz=control_rate_hz,
        physics_substeps=physics_substeps,
        speedup_rate=args.speedup_rate,
        max_episode_steps=max_episode_steps,
        init_mode=init_mode,
        reward_profile=reward_profile,
        referee_mode=referee_mode,
        physics_backend=physics_backend,
    )
    # Capture baseline actions too: each team's Brandão history includes its
    # own previously applied commands, including STOPPED-frame zeroing.
    opponent_callback = env._opponent_policy
    actual_yellow_action = np.zeros_like(pending_yellow_action)
    def record_opponent(observation):
        nonlocal actual_yellow_action
        actual_yellow_action = np.asarray(
            opponent_callback(observation), dtype=np.float32
        ).reshape(config.N_ROBOTS, 2)
        return actual_yellow_action
    env._opponent_policy = record_opponent

    key = jax.random.PRNGKey(args.seed)
    blue_apply = jax.jit(model.apply)
    yellow_apply = jax.jit(opponent_model.apply) if opponent_model is not None else None
    opponent_architecture = (
        opponent_metadata.get("policy_architecture", "joint")
        if opponent_metadata is not None else None
    )
    # Separate histories are necessary when one policy uses entity ordering
    # and the other uses the original shared-policy action ordering.
    history_orders = {
        architecture == "brandao_sets"
        for architecture in (policy_architecture, opponent_architecture)
        if architecture in ("brandao_shared", "brandao_sets")
    }
    history_advance = {}
    for entity_ordered in history_orders:
        def next_history(history, simulation, previous_actions, remaining, sample_key,
                         entity_ordered=entity_ordered):
            features = build_agent_features(
                simulation, previous_actions, remaining, sample_key,
                entity_ordered_actions=entity_ordered,
            )
            return advance_history(history, features, jnp.asarray(True))
        history_advance[entity_ordered] = jax.jit(next_history)

    def action_for(apply, policy_params, architecture, observation):
        nonlocal key
        first, second, _ = apply(
            policy_params, jnp.asarray(observation, dtype=jnp.float32)
        )
        if architecture in ("brandao_shared", "brandao_sets"):
            if args.stochastic:
                key, sample_key = jax.random.split(key)
                action = 2.0 * jax.random.beta(sample_key, first, second) - 1.0
            else:
                action = 2.0 * first / (first + second) - 1.0
        else:
            raw_action = first
            if args.stochastic:
                key, sample_key = jax.random.split(key)
                raw_action = first + jnp.exp(second) * jax.random.normal(
                    sample_key, first.shape
                )
            action = jnp.tanh(raw_action)
        return np.asarray(jax.device_get(action), dtype=np.float32).reshape(
            config.N_ROBOTS, 2
        )

    opponent_label = args.opponent
    if opponent_metadata is not None:
        opponent_label = (
            f"saved generation={opponent_metadata['generation']} "
            f"lineage={opponent_metadata['lineage']} architecture={opponent_architecture}"
        )
    print(
        f"snapshot generation={metadata['generation']} lineage={metadata['lineage']} "
        f"— architecture={policy_architecture}, opponent={opponent_label}, "
        f"control={control_rate_hz:g} Hz × {physics_substeps} substeps, "
        f"speedup cap={args.speedup_rate:g}x, init={init_mode}, "
        f"referee={referee_mode}, physics={physics_backend}, "
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
            previous_actions = jnp.zeros((2, 3, 2), dtype=jnp.float32)
            observation_key = jax.random.PRNGKey(args.seed + episode)
            histories = {
                entity_ordered: initialize_history(build_agent_features(
                    env._state, previous_actions, jnp.float32(1.0), observation_key,
                    entity_ordered_actions=entity_ordered,
                ))
                for entity_ordered in history_orders
            }
            steps = 0
            done = False
            while steps < policy_episode_steps and not done:
                if policy_architecture in ("brandao_shared", "brandao_sets"):
                    blue_observation = flatten_history(
                        histories[policy_architecture == "brandao_sets"], config.TEAM_BLUE
                    )
                else:
                    blue_observation = observation
                blue_action = action_for(
                    blue_apply, params, policy_architecture, blue_observation
                )
                if opponent_metadata is not None:
                    if opponent_architecture in ("brandao_shared", "brandao_sets"):
                        yellow_observation = flatten_history(
                            histories[opponent_architecture == "brandao_sets"],
                            config.TEAM_YELLOW,
                        )
                    else:
                        yellow_observation = switch_team_perspective(observation)
                    pending_yellow_action = action_for(
                        yellow_apply, opponent_params, opponent_architecture,
                        yellow_observation,
                    )

                # VSSEnv applies zero commands on a STOPPED step, just as the
                # training environment does when it constructs its history.
                stopped = int(np.asarray(env._referee_state.phase)) == MatchPhase.STOPPED
                observation, reward, terminated, truncated, info = env.step(
                    blue_action.reshape(-1)
                )
                reward_total += float(reward)
                goal = int(info.get("goal", 0))
                goals_blue += goal == 1
                goals_yellow += goal == -1
                score_blue = int(info["score_blue"])
                score_yellow = int(info["score_yellow"])
                steps += 1
                done = bool(terminated or truncated or (
                    goal and policy_architecture in ("brandao_shared", "brandao_sets")
                ))
                if stopped:
                    previous_actions = jnp.zeros_like(previous_actions)
                else:
                    previous_actions = jnp.asarray(
                        np.stack([blue_action, actual_yellow_action]), dtype=jnp.float32
                    )
                if not done and steps < policy_episode_steps and histories:
                    remaining = jnp.float32(1.0 - steps / policy_episode_steps)
                    observation_key, sample_key = jax.random.split(observation_key)
                    histories = {
                        order: history_advance[order](
                            history, env._state, previous_actions, remaining, sample_key
                        )
                        for order, history in histories.items()
                    }

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
