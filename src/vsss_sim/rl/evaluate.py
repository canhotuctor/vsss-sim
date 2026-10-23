"""Independent, long-match evaluation for historical self-play experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import config as sim_config
from ..envs.brandao_observation import OBSERVATION_SIZE as BRANDAO_OBSERVATION_SIZE
from ..envs.jumanji import VSSJumanjiEnv
from .ppo import create_policy_model, deterministic_policy_action
from .self_play import ExperimentConfig, SnapshotArchive


def _evaluate_batch(
    env: VSSJumanjiEnv,
    model,
    current_params: dict,
    opponent_params: dict,
    opponent_modes: jax.Array,
    seeds: jax.Array,
) -> jax.Array:
    """Run a fixed batch of matches and return blue/yellow final scores."""
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    states, timesteps = jax.vmap(env.reset)(keys)

    def match_step(carry, _):
        states, observation = carry
        blue_action = deterministic_policy_action(
            model, current_params, observation, env.policy_architecture
        )
        opponent_observations = jax.vmap(env.opponent_observation)(states)
        yellow_action = jax.vmap(
            lambda params, obs: deterministic_policy_action(
                model, params, obs, env.policy_architecture
            )
        )(opponent_params, opponent_observations)
        next_states, timestep = jax.vmap(
            lambda state, action, opponent_action, mode: env.step(
                state,
                action,
                opponent_action=opponent_action,
                opponent_mode=mode,
            )
        )(states, blue_action, yellow_action, opponent_modes)
        return (next_states, timestep.observation), None

    (states, _), _ = jax.lax.scan(
        match_step,
        (states, timesteps.observation),
        xs=None,
        length=env.max_episode_steps,
    )
    return states.simulation.score


def _summary(scores: np.ndarray) -> dict[str, Any]:
    blue = scores[:, sim_config.TEAM_BLUE]
    yellow = scores[:, sim_config.TEAM_YELLOW]
    wins = int(np.sum(blue > yellow))
    draws = int(np.sum(blue == yellow))
    losses = int(np.sum(blue < yellow))
    return {
        "matches": int(scores.shape[0]),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(scores.shape[0], 1),
        "goals_for": int(blue.sum()),
        "goals_against": int(yellow.sum()),
        "goal_difference": int((blue - yellow).sum()),
        "mean_goal_difference": float(np.mean(blue - yellow)) if scores.size else 0.0,
    }


def _evaluate_group(
    env: VSSJumanjiEnv,
    model,
    current_params: dict,
    opponent_params: list[dict],
    opponent_modes: np.ndarray,
    opponent_generations: list[int | None],
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for start in range(0, len(opponent_params), batch_size):
        end = min(start + batch_size, len(opponent_params))
        params_batch = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values), *opponent_params[start:end]
        )
        modes = jnp.asarray(opponent_modes[start:end], dtype=jnp.int32)
        match_seeds = jnp.arange(seed + start, seed + end, dtype=jnp.uint32)
        match_scores = _evaluate_batch(
            env, model, current_params, params_batch, modes, match_seeds
        )
        jax.block_until_ready(match_scores)
        host_scores = np.asarray(match_scores, dtype=np.int32)
        scores.append(host_scores)
        for offset, score in enumerate(host_scores):
            records.append(
                {
                    "seed": seed + start + offset,
                    "opponent_generation": opponent_generations[start + offset],
                    "blue_goals": int(score[sim_config.TEAM_BLUE]),
                    "yellow_goals": int(score[sim_config.TEAM_YELLOW]),
                }
            )
    all_scores = np.concatenate(scores, axis=0) if scores else np.zeros((0, 2), dtype=np.int32)
    return {"summary": _summary(all_scores), "matches": records}


def evaluate_experiment(
    config: ExperimentConfig,
    lineage: int | None = None,
    generation: int | None = None,
    matches: int | None = None,
    seed: int | None = None,
    opponents: str = "all",
    batch_size: int = 32,
) -> Path:
    """Evaluate current policies against own history and optional fixed baselines."""
    if opponents not in ("all", "history", "random", "stationary"):
        raise ValueError("opponents must be all, history, random, or stationary")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    match_count = config.evaluation_matches if matches is None else matches
    if match_count < 1:
        raise ValueError("matches must be at least 1")
    base_seed = config.seed + 10_000_000 if seed is None else seed
    lineages = range(config.num_lineages) if lineage is None else (lineage,)
    if any(value < 0 or value >= config.num_lineages for value in lineages):
        raise ValueError("lineage index is outside the configured experiment")

    policy_hz = config.control_rate_hz
    horizon = int(round(config.evaluation_seconds * policy_hz))
    # Evaluation is a match-level rollout.  Goals reset the field position
    # while the match score and wall-clock horizon continue, so use the
    # environment's in-match kickoff path instead of episode termination.
    env = VSSJumanjiEnv(
        opponent_policy="stationary",
        max_episode_steps=horizon,
        init_mode=config.init_mode,
        terminate_on_goal=False,
        reward_profile=config.reward_profile,
        policy_architecture=config.policy_architecture,
        control_rate_hz=config.control_rate_hz,
        physics_substeps=config.physics_substeps,
    )
    model = create_policy_model(config.policy_architecture, config.hidden_sizes)
    observation_size = (
        BRANDAO_OBSERVATION_SIZE
        if config.policy_architecture in ("brandao_shared", "brandao_sets")
        else 46
    )
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((observation_size,), dtype=jnp.float32)
    )
    output_records = []

    for lineage_id in lineages:
        archive = SnapshotArchive(config.output_root, config.experiment_name, lineage_id)
        snapshots = archive.list_snapshots()
        if not snapshots:
            raise FileNotFoundError(f"no snapshots for lineage {lineage_id}")
        current_metadata = (
            next((item for item in snapshots if item["generation"] == generation), None)
            if generation is not None
            else snapshots[-1]
        )
        if current_metadata is None:
            raise FileNotFoundError(
                f"no generation {generation} for lineage {lineage_id}"
            )
        current_generation = int(current_metadata["generation"])
        current_params = archive.load(
            template,
            current_generation,
            expected_hidden_sizes=config.hidden_sizes,
            expected_policy_architecture=config.policy_architecture,
        )
        historical = [
            item for item in snapshots if item["generation"] < current_generation
        ]
        if not historical:
            historical = [current_metadata]
        if config.active_pool_limit is not None:
            historical = historical[-config.active_pool_limit :]
        historical_generations = [int(item["generation"]) for item in historical]
        historical_params = [
            archive.load(
                template,
                item,
                expected_hidden_sizes=config.hidden_sizes,
                expected_policy_architecture=config.policy_architecture,
            )
            for item in historical_generations
        ]

        groups = []
        if opponents in ("all", "history"):
            history_probabilities = config.opponent_weights.probabilities(
                historical_generations
            )[: len(historical_generations)]
            if history_probabilities.sum() <= 0.0:
                raise ValueError("historical selector weights are all zero")
            history_probabilities = history_probabilities / history_probabilities.sum()
            rng = np.random.default_rng(base_seed + lineage_id)
            choices = rng.choice(
                len(historical_generations),
                size=match_count,
                p=history_probabilities,
            )
            groups.append(
                (
                    "history",
                    [historical_params[int(index)] for index in choices],
                    np.zeros(match_count, dtype=np.int32),
                    [historical_generations[int(index)] for index in choices],
                )
            )
        if opponents in ("all", "random"):
            groups.append(
                (
                    "random",
                    [current_params] * match_count,
                    np.ones(match_count, dtype=np.int32),
                    [None] * match_count,
                )
            )
        if opponents in ("all", "stationary"):
            groups.append(
                (
                    "stationary",
                    [current_params] * match_count,
                    np.full(match_count, 2, dtype=np.int32),
                    [None] * match_count,
                )
            )

        for group_index, (name, params, modes, opponent_gens) in enumerate(groups):
            result = _evaluate_group(
                env,
                model,
                current_params,
                params,
                modes,
                opponent_gens,
                seed=base_seed + lineage_id * 1_000_000 + group_index * match_count,
                batch_size=batch_size,
            )
            output_records.append(
                {
                    "lineage": lineage_id,
                    "generation": current_generation,
                    "opponent_type": name,
                    "summary": result["summary"],
                    "matches": result["matches"],
                }
            )

    result = {
        "experiment_name": config.experiment_name,
        "reward_profile": config.reward_profile,
        "policy_architecture": config.policy_architecture,
        "evaluation_seconds": config.evaluation_seconds,
        "control_steps": horizon,
        "seed": base_seed,
        "results": output_records,
    }
    evaluation_dir = Path(config.output_root) / config.experiment_name / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    target = evaluation_dir / f"evaluation_seed_{base_seed}.json"
    if target.exists():
        raise FileExistsError(f"evaluation output already exists: {target}")
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return target


def evaluate_experiment_exhaustive_history(
    config: ExperimentConfig,
    lineage: int | None = None,
    generation: int | None = None,
    matches_per_snapshot: int | None = None,
    seed: int | None = None,
    batch_size: int = 32,
) -> Path:
    """Evaluate each final policy against every prior snapshot of its lineage."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    match_count = (
        config.evaluation_matches
        if matches_per_snapshot is None
        else matches_per_snapshot
    )
    if match_count < 1:
        raise ValueError("matches_per_snapshot must be at least 1")
    base_seed = config.seed + 10_000_000 if seed is None else seed
    lineages = range(config.num_lineages) if lineage is None else (lineage,)
    if any(value < 0 or value >= config.num_lineages for value in lineages):
        raise ValueError("lineage index is outside the configured experiment")

    policy_hz = config.control_rate_hz
    horizon = int(round(config.evaluation_seconds * policy_hz))
    # Keep the same match-level semantics as the regular evaluator: a goal
    # triggers a kickoff, but does not end the 120-second match rollout.
    env = VSSJumanjiEnv(
        opponent_policy="stationary",
        max_episode_steps=horizon,
        init_mode=config.init_mode,
        terminate_on_goal=False,
        reward_profile=config.reward_profile,
        policy_architecture=config.policy_architecture,
        control_rate_hz=config.control_rate_hz,
        physics_substeps=config.physics_substeps,
    )
    model = create_policy_model(config.policy_architecture, config.hidden_sizes)
    observation_size = (
        BRANDAO_OBSERVATION_SIZE
        if config.policy_architecture in ("brandao_shared", "brandao_sets")
        else 46
    )
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((observation_size,), dtype=jnp.float32)
    )
    output_records = []

    for lineage_id in lineages:
        archive = SnapshotArchive(config.output_root, config.experiment_name, lineage_id)
        snapshots = archive.list_snapshots()
        if not snapshots:
            raise FileNotFoundError(f"no snapshots for lineage {lineage_id}")
        current_metadata = (
            next((item for item in snapshots if item["generation"] == generation), None)
            if generation is not None
            else snapshots[-1]
        )
        if current_metadata is None:
            raise FileNotFoundError(
                f"no generation {generation} for lineage {lineage_id}"
            )
        current_generation = int(current_metadata["generation"])
        current_params = archive.load(
            template,
            current_generation,
            expected_hidden_sizes=config.hidden_sizes,
            expected_policy_architecture=config.policy_architecture,
        )
        historical = [
            item for item in snapshots if item["generation"] < current_generation
        ]
        if not historical:
            historical = [current_metadata]
        if config.active_pool_limit is not None:
            historical = historical[-config.active_pool_limit :]
        historical_generations = [int(item["generation"]) for item in historical]
        historical_params = [
            archive.load(
                template,
                item,
                expected_hidden_sizes=config.hidden_sizes,
                expected_policy_architecture=config.policy_architecture,
            )
            for item in historical_generations
        ]

        opponent_params = []
        opponent_generations = []
        for snapshot_generation, snapshot_params in zip(
            historical_generations, historical_params
        ):
            opponent_params.extend([snapshot_params] * match_count)
            opponent_generations.extend([snapshot_generation] * match_count)
        result = _evaluate_group(
            env,
            model,
            current_params,
            opponent_params,
            np.zeros(len(opponent_params), dtype=np.int32),
            opponent_generations,
            seed=base_seed + lineage_id * 1_000_000,
            batch_size=batch_size,
        )
        output_records.append(
            {
                "lineage": lineage_id,
                "generation": current_generation,
                "opponent_type": "history_exhaustive",
                "matches_per_snapshot": match_count,
                "opponent_generations": historical_generations,
                "summary": result["summary"],
                "matches": result["matches"],
            }
        )

    result = {
        "experiment_name": config.experiment_name,
        "reward_profile": config.reward_profile,
        "evaluation_mode": "exhaustive_history",
        "policy_architecture": config.policy_architecture,
        "evaluation_seconds": config.evaluation_seconds,
        "control_steps": horizon,
        "matches_per_snapshot": match_count,
        "seed": base_seed,
        "results": output_records,
    }
    evaluation_dir = Path(config.output_root) / config.experiment_name / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    target = evaluation_dir / f"evaluation_exhaustive_history_seed_{base_seed}.json"
    if target.exists():
        raise FileExistsError(f"evaluation output already exists: {target}")
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return target
