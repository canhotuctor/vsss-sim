"""Historical self-play experiment configuration, archives, and orchestration."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .. import config as sim_config
from ..config import InitMode, RefereeMode, RewardProfile
from ..envs.brandao_observation import OBSERVATION_SIZE as BRANDAO_OBSERVATION_SIZE
from ..envs.jumanji import VSSJumanjiEnv
from ..envs.match_referee import RefereeEvent
from .ppo import PPO, PPOConfig, deterministic_policy_action

_SNAPSHOT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_POLICY_ARCHITECTURES = ("joint", "brandao_shared", "brandao_sets")
_BRANDAO_POLICY_ARCHITECTURES = ("brandao_shared", "brandao_sets")


@dataclass(frozen=True)
class OpponentSamplingWeights:
    """Percent mixture over historical selectors and optional baseline modes."""

    all_history: float = 200.0 / 3.0
    recent_half: float = 200.0 / 9.0
    latest: float = 100.0 / 9.0
    random: float = 0.0
    stationary: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0.0 for value in values.values()):
            raise ValueError("opponent sampling weights must be non-negative")
        if not np.isclose(sum(values.values()), 100.0, atol=1e-5):
            raise ValueError("opponent sampling weights must sum to 100")

    def probabilities(self, generations: list[int] | tuple[int, ...]) -> np.ndarray:
        """Return probabilities for snapshots in order, then random/stationary."""
        ordered = tuple(sorted(generations))
        if not ordered:
            raise ValueError("historical opponent pool must contain a snapshot")
        if len(set(ordered)) != len(ordered):
            raise ValueError("snapshot generations must be unique")

        count = len(ordered)
        probabilities = np.zeros(count + 2, dtype=np.float32)
        if self.all_history:
            probabilities[:count] += self.all_history / 100.0 / count
        if self.recent_half:
            recent_count = max(1, (count + 1) // 2)
            probabilities[count - recent_count : count] += (
                self.recent_half / 100.0 / recent_count
            )
        if self.latest:
            probabilities[count - 1] += self.latest / 100.0
        probabilities[count] = self.random / 100.0
        probabilities[count + 1] = self.stationary / 100.0
        probabilities /= probabilities.sum()
        return probabilities

    def category_probabilities(self) -> np.ndarray:
        """Return selector weights in JAX category order, normalized to one."""
        values = np.asarray(
            [self.all_history, self.recent_half, self.latest, self.random, self.stationary],
            dtype=np.float32,
        )
        return values / 100.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable parameters for one independent historical self-play run."""

    experiment_name: str
    num_generations: int
    num_lineages: int = 1
    num_envs_per_lineage: int = 256
    rollout_length: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 8
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    snapshot_interval: int = 10
    generation_offset: int = 0
    initial_policy_snapshot: str | None = None
    initial_opponent_snapshot: str | None = None
    active_pool_limit: int | None = None
    max_episode_steps: int = sim_config.MAX_EPISODE_STEPS
    terminate_on_goal: bool = False
    init_mode: str = "kickoff"
    reward_profile: str = RewardProfile.DEFAULT.value
    referee_mode: str = RefereeMode.FULL.value
    policy_architecture: str = "joint"
    mirror_augmentation: bool | None = None
    opponent_update_threshold: float = 0.6
    opponent_evaluation_interval: int = 1
    opponent_update_score: str = "evaluation"
    opponent_evaluation_matches: int | None = None
    output_root: str = "output/experiments"
    evaluation_matches: int = 100
    evaluation_seconds: int = 120
    opponent_weights: OpponentSamplingWeights = field(
        default_factory=OpponentSamplingWeights
    )
    control_rate_hz: float | None = None
    physics_substeps: int | None = None
    require_gpu: bool = False

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.experiment_name) or self.experiment_name in (".", ".."):
            raise ValueError("experiment_name must be a simple filesystem-safe name")
        for name in (
            "num_generations",
            "num_lineages",
            "num_envs_per_lineage",
            "rollout_length",
            "update_epochs",
            "num_minibatches",
            "snapshot_interval",
            "opponent_evaluation_interval",
            "max_episode_steps",
            "evaluation_matches",
            "evaluation_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.active_pool_limit is not None and self.active_pool_limit < 1:
            raise ValueError("active_pool_limit must be at least 1 when provided")
        if (
            isinstance(self.generation_offset, bool)
            or not isinstance(self.generation_offset, int)
            or self.generation_offset < 0
        ):
            raise ValueError("generation_offset must be a non-negative integer")
        if self.generation_offset > 0 and not self.initial_policy_snapshot:
            raise ValueError(
                "generation_offset requires an initial_policy_snapshot"
            )
        if self.opponent_evaluation_matches is not None and self.opponent_evaluation_matches < 1:
            raise ValueError("opponent_evaluation_matches must be at least 1 when provided")
        if self.opponent_update_score not in ("evaluation", "training_rollout"):
            raise ValueError("opponent_update_score must be evaluation or training_rollout")
        if self.opponent_update_threshold < 0.0:
            raise ValueError("opponent_update_threshold must be non-negative")
        if self.init_mode not in ("kickoff", "random"):
            raise ValueError("init_mode must be 'kickoff' or 'random'")
        if self.policy_architecture not in _POLICY_ARCHITECTURES:
            raise ValueError(
                "policy_architecture must be 'joint', 'brandao_shared', or 'brandao_sets'"
            )
        is_brandao_policy = self.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
        if self.initial_opponent_snapshot and not is_brandao_policy:
            raise ValueError(
                "initial_opponent_snapshot is only supported for Brandao policies"
            )
        if self.control_rate_hz is None:
            object.__setattr__(self, "control_rate_hz", sim_config.FPS)
        if self.physics_substeps is None:
            object.__setattr__(self, "physics_substeps", sim_config.SUB_STEPS)
        if (
            self.control_rate_hz is None
            or isinstance(self.control_rate_hz, bool)
            or not isinstance(self.control_rate_hz, (int, float))
            or not np.isfinite(self.control_rate_hz)
            or self.control_rate_hz <= 0.0
        ):
            raise ValueError("control_rate_hz must be a positive finite number")
        if (
            self.physics_substeps is None
            or isinstance(self.physics_substeps, bool)
            or not isinstance(self.physics_substeps, (int, float))
            or not np.isfinite(self.physics_substeps)
            or int(self.physics_substeps) != self.physics_substeps
            or self.physics_substeps < 1
        ):
            raise ValueError("physics_substeps must be a positive integer")
        object.__setattr__(self, "physics_substeps", int(self.physics_substeps))
        if self.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES:
            if self.reward_profile == RewardProfile.DEFAULT.value:
                object.__setattr__(
                    self, "reward_profile", RewardProfile.BRANDAO_REWARD.value
                )
            if self.reward_profile != RewardProfile.BRANDAO_REWARD.value:
                raise ValueError(
                    "Brandao per-robot policies require the brandao_reward profile"
                )
            if self.hidden_sizes == (64, 64):
                object.__setattr__(self, "hidden_sizes", (300, 200, 100))
            elif self.hidden_sizes != (300, 200, 100):
                raise ValueError(
                    "Brandao per-robot policies use the paper's 300/200/100 actor layers"
                )
            # Brandão's training episodes end at a goal or after 40 seconds.
            if self.max_episode_steps == sim_config.MAX_EPISODE_STEPS:
                object.__setattr__(self, "max_episode_steps", 400)
            if not self.terminate_on_goal:
                object.__setattr__(self, "terminate_on_goal", True)
        if self.mirror_augmentation is None:
            object.__setattr__(
                self,
                "mirror_augmentation",
                self.policy_architecture == "brandao_sets",
            )
        if self.mirror_augmentation and self.policy_architecture not in (
            "brandao_shared",
            "brandao_sets",
        ):
            raise ValueError("x-axis mirror augmentation requires a Brandao policy")
        try:
            profile = RewardProfile(self.reward_profile)
        except ValueError as error:
            choices = ", ".join(profile.value for profile in RewardProfile)
            raise ValueError(f"reward_profile must be one of: {choices}") from error
        object.__setattr__(self, "reward_profile", profile.value)
        try:
            referee_mode = RefereeMode(self.referee_mode)
        except ValueError as error:
            choices = ", ".join(mode.value for mode in RefereeMode)
            raise ValueError(f"referee_mode must be one of: {choices}") from error
        object.__setattr__(self, "referee_mode", referee_mode.value)
        # Reuse the trainer's validation before initializing a device or allocating state.
        self.ppo_config()

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            num_envs=self.num_envs_per_lineage,
            rollout_length=self.rollout_length,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            update_epochs=self.update_epochs,
            num_minibatches=self.num_minibatches,
            hidden_sizes=self.hidden_sizes,
            mirror_augmentation=bool(self.mirror_augmentation),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        values = dict(raw)
        weight_data = values.pop("opponent_weights", {})
        values["opponent_weights"] = OpponentSamplingWeights(**weight_data)
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(values["hidden_sizes"])
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown experiment config fields: {', '.join(sorted(unknown))}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError("experiment config must be a JSON object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotArchive:
    """Append-only policy snapshot archive scoped to one experiment lineage."""

    def __init__(self, root: str | Path, experiment_name: str, lineage: int) -> None:
        if not _SAFE_NAME.fullmatch(experiment_name) or experiment_name in (".", ".."):
            raise ValueError("experiment_name must be a simple filesystem-safe name")
        if lineage < 0:
            raise ValueError("lineage must be non-negative")
        self.experiment_name = experiment_name
        self.lineage = int(lineage)
        self.directory = Path(root) / experiment_name / f"lineage_{lineage:03d}" / "snapshots"

    def save(
        self,
        params: dict,
        generation: int,
        seed: int,
        hidden_sizes: tuple[int, ...],
        observation_size: int | None = None,
        action_size: int | None = None,
        policy_architecture: str = "joint",
    ) -> Path:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"generation_{generation:08d}"
        payload_path = self.directory / f"{stem}.msgpack"
        metadata_path = self.directory / f"{stem}.json"
        if payload_path.exists() or metadata_path.exists():
            raise FileExistsError(f"snapshot for generation {generation} already exists")
        if policy_architecture not in _POLICY_ARCHITECTURES:
            raise ValueError(f"unknown policy architecture: {policy_architecture}")
        if policy_architecture in _BRANDAO_POLICY_ARCHITECTURES:
            observation_size = 424 if observation_size is None else observation_size
            action_size = 2 if action_size is None else action_size
        else:
            observation_size = 46 if observation_size is None else observation_size
            action_size = 6 if action_size is None else action_size

        metadata = {
            "schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "experiment_name": self.experiment_name,
            "lineage": self.lineage,
            "generation": int(generation),
            "seed": int(seed),
            "hidden_sizes": list(hidden_sizes),
            "observation_size": int(observation_size),
            "action_size": int(action_size),
            "policy_architecture": policy_architecture,
            "artifact": payload_path.name,
        }
        payload = serialization.to_bytes(jax.device_get(params))
        self._atomic_write(payload_path, payload)
        self._atomic_write(
            metadata_path, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
        )
        return payload_path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def list_snapshots(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        snapshots = []
        for path in sorted(self.directory.glob("generation_*.json")):
            with path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            if metadata.get("schema_version") != _SNAPSHOT_SCHEMA_VERSION:
                raise ValueError(f"unsupported snapshot schema in {path}")
            if metadata.get("experiment_name") != self.experiment_name:
                raise ValueError(f"snapshot experiment mismatch in {path}")
            if metadata.get("lineage") != self.lineage:
                raise ValueError(f"snapshot lineage mismatch in {path}")
            payload_path = self.directory / metadata["artifact"]
            if not payload_path.is_file():
                raise ValueError(f"snapshot payload missing for {path}")
            snapshots.append(metadata)
        return sorted(snapshots, key=lambda item: item["generation"])

    def load(
        self,
        template: dict,
        generation: int,
        expected_hidden_sizes: tuple[int, ...] | None = None,
        expected_policy_architecture: str | None = None,
    ) -> dict:
        matches = [item for item in self.list_snapshots() if item["generation"] == generation]
        if not matches:
            raise FileNotFoundError(f"no snapshot for generation {generation} in {self.directory}")
        metadata = matches[0]
        stored_architecture = metadata.get("policy_architecture", "joint")
        if (
            expected_policy_architecture is not None
            and stored_architecture != expected_policy_architecture
        ):
            raise ValueError("snapshot policy architecture does not match experiment")
        if expected_hidden_sizes is not None and tuple(metadata["hidden_sizes"]) != tuple(
            expected_hidden_sizes
        ):
            raise ValueError("snapshot policy architecture does not match current experiment")
        payload_path = self.directory / metadata["artifact"]
        params = serialization.from_bytes(template, payload_path.read_bytes())
        return jax.tree_util.tree_map(jnp.asarray, params)

    def load_pool(
        self,
        template: dict,
        active_limit: int | None = None,
        expected_hidden_sizes: tuple[int, ...] | None = None,
        expected_policy_architecture: str | None = None,
    ) -> tuple[dict, tuple[int, ...]]:
        snapshots = self.list_snapshots()
        if active_limit is not None:
            snapshots = snapshots[-active_limit:]
        if not snapshots:
            raise ValueError(f"historical pool is empty: {self.directory}")
        generations = tuple(item["generation"] for item in snapshots)
        params = [
            self.load(
                template,
                generation,
                expected_hidden_sizes,
                expected_policy_architecture,
            )
            for generation in generations
        ]
        stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *params)
        return stacked, generations


def _stack_lineages(values: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: jnp.stack(items), *values)


def _extract_lineage(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _make_env(config: ExperimentConfig) -> Any:
    return VSSJumanjiEnv(
        opponent_policy="stationary",
        max_episode_steps=config.max_episode_steps,
        init_mode=InitMode(config.init_mode),
        terminate_on_goal=config.terminate_on_goal,
        reward_profile=config.reward_profile,
        policy_architecture=config.policy_architecture,
        control_rate_hz=config.control_rate_hz,
        physics_substeps=config.physics_substeps,
        referee_mode=config.referee_mode,
    )


def _active_pools(
    archives: list[SnapshotArchive],
    template_params: dict,
    config: ExperimentConfig,
) -> tuple[dict, jax.Array, jax.Array, tuple[tuple[int, ...], ...]]:
    pools, probabilities, generation_ids, pool_generations = [], [], [], []
    for archive in archives:
        pool, generations = archive.load_pool(
            template_params,
            active_limit=config.active_pool_limit,
            expected_hidden_sizes=config.hidden_sizes,
            expected_policy_architecture=config.policy_architecture,
        )
        pools.append(pool)
        probabilities.append(config.opponent_weights.category_probabilities())
        generation_ids.append(jnp.asarray(generations, dtype=jnp.int32))
        pool_generations.append(generations)
    return (
        _stack_lineages(pools),
        jnp.asarray(np.stack(probabilities)),
        jnp.stack(generation_ids),
        tuple(pool_generations),
    )


def _brandao_opponent_pools(
    archives: list[SnapshotArchive],
    template_params: dict,
    config: ExperimentConfig,
    selected_generations: list[int],
) -> tuple[dict, jax.Array, jax.Array, tuple[tuple[int, ...], ...]]:
    """Build one fixed historical opponent per lineage for Brandão self-play."""
    pools = []
    for archive, generation in zip(archives, selected_generations, strict=True):
        params = archive.load(
            template_params,
            generation,
            expected_hidden_sizes=config.hidden_sizes,
            expected_policy_architecture=config.policy_architecture,
        )
        pools.append(jax.tree_util.tree_map(lambda value: value[None, ...], params))
    probabilities = jnp.tile(
        jnp.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32),
        (config.num_lineages, 1),
    )
    generation_ids = jnp.asarray(selected_generations, dtype=jnp.int32)[:, None]
    return (
        _stack_lineages(pools),
        probabilities,
        generation_ids,
        tuple((generation,) for generation in selected_generations),
    )


def _snapshot_generations(config: ExperimentConfig) -> set[int]:
    """Return the generation IDs ``run_experiment`` saves after training starts."""
    generations = set(
        range(
            config.generation_offset + config.snapshot_interval,
            config.generation_offset + config.num_generations + 1,
            config.snapshot_interval,
        )
    )
    if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES:
        # A promotion archives its candidate even between regular checkpoints.
        generations.update(range(
            config.generation_offset + config.opponent_evaluation_interval,
            config.generation_offset + config.num_generations + 1,
            config.opponent_evaluation_interval,
        ))
    generations.add(config.generation_offset + config.num_generations)
    return generations


def _lineage_seeds(seed: int, count: int) -> jax.Array:
    root_key = jax.random.PRNGKey(seed)
    return jax.vmap(lambda lineage: jax.random.fold_in(root_key, lineage))(
        jnp.arange(count, dtype=jnp.uint32)
    )


def _write_json_line(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def _opponent_evaluation_summary(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Summarize candidate-vs-opponent matches and apply the goal-delta rule."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != sim_config.N_TEAMS:
        raise ValueError("scores must have shape (matches, teams)")
    if scores.shape[0] < 1:
        raise ValueError("opponent evaluation must contain at least one match")

    goals_for = scores[:, sim_config.TEAM_BLUE]
    goals_against = scores[:, sim_config.TEAM_YELLOW]
    mean_goal_delta = float(np.mean(goals_for - goals_against))
    return {
        "evaluation_sample_count": int(scores.shape[0]),
        "mean_goals_for": float(np.mean(goals_for)),
        "mean_goals_against": float(np.mean(goals_against)),
        "mean_goal_delta": mean_goal_delta,
        "opponent_replaced": mean_goal_delta >= threshold,
    }


def _make_candidate_evaluator(env: VSSJumanjiEnv, model, policy_architecture: str):
    """Build a compiled batch evaluator for single-goal, 40-second matches."""

    def evaluate(candidate_params: dict, opponent_params: dict, seeds: jax.Array):
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        states, timesteps = jax.vmap(env.reset)(keys)
        observations = timesteps.observation
        completed = jnp.zeros((seeds.shape[0],), dtype=jnp.bool_)

        def match_step(carry, _):
            states, observations, completed = carry

            def advance_one(state, observation, done):
                def advance(_):
                    candidate_action = deterministic_policy_action(
                        model, candidate_params, observation, policy_architecture
                    )
                    opponent_observation = env.opponent_observation(state)
                    opponent_action = deterministic_policy_action(
                        model,
                        opponent_params,
                        opponent_observation,
                        policy_architecture,
                    )
                    next_state, timestep = env.step(
                        state,
                        candidate_action,
                        opponent_action=opponent_action,
                        opponent_mode=0,
                    )
                    return next_state, timestep.observation, timestep.last()

                return jax.lax.cond(
                    done,
                    lambda _: (state, observation, done),
                    advance,
                    operand=None,
                )

            states, observations, completed = jax.vmap(advance_one)(
                states, observations, completed
            )
            return (states, observations, completed), None

        (states, _, _), _ = jax.lax.scan(
            match_step,
            (states, observations, completed),
            xs=None,
            length=env.max_episode_steps,
        )
        return states.simulation.score

    return jax.jit(evaluate)


def _require_gpu_backend() -> None:
    """Fail before creating outputs if JAX cannot execute on a GPU."""
    message = (
        "this experiment requires a visible JAX GPU; refusing CPU fallback. "
        "vsss_sim defaults JAX_PLATFORMS to 'cpu' (currently "
        f"{os.environ.get('JAX_PLATFORMS', '<unset>')!r}); "
        "export JAX_PLATFORMS=cuda before starting Python to train on a GPU"
    )
    try:
        gpu_devices = jax.devices("gpu")
        default_backend = jax.default_backend()
    except RuntimeError as error:
        raise RuntimeError(message) from error
    if not gpu_devices or default_backend != "gpu":
        raise RuntimeError(message)


def _load_external_snapshot(
    snapshot_path: str | Path,
    template: dict,
    config: ExperimentConfig,
) -> tuple[dict, dict[str, Any], Path]:
    """Load and validate a policy snapshot used to warm-start an experiment."""
    path = Path(snapshot_path).expanduser().resolve()
    metadata_path = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"initial policy snapshot not found: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"snapshot metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("schema_version") != _SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema in {metadata_path}")
    if metadata.get("artifact") != path.name:
        raise ValueError(f"snapshot metadata artifact does not match {path.name}")
    if metadata.get("policy_architecture", "joint") != config.policy_architecture:
        raise ValueError("initial snapshot policy architecture does not match experiment")
    if tuple(metadata.get("hidden_sizes", ())) != tuple(config.hidden_sizes):
        raise ValueError("initial snapshot hidden sizes do not match experiment")
    expected_observation_size = (
        BRANDAO_OBSERVATION_SIZE
        if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
        else 46
    )
    if int(metadata.get("observation_size", -1)) != expected_observation_size:
        raise ValueError("initial snapshot observation size does not match experiment")
    if int(metadata.get("generation", -1)) < 0:
        raise ValueError("initial snapshot generation metadata is invalid")
    params = serialization.from_bytes(template, path.read_bytes())
    return jax.tree_util.tree_map(jnp.asarray, params), metadata, path


def run_experiment(config: ExperimentConfig, config_source: str | Path | None = None) -> Path:
    """Run one experiment with independent PPO learner states on a lineage axis."""
    if config.require_gpu:
        _require_gpu_backend()
    output_root = Path(config.output_root)
    experiment_dir = output_root / config.experiment_name
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"experiment output already exists: {experiment_dir}; use a new experiment_name"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    resolved_config = config.to_dict()
    (experiment_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    if config_source is not None:
        (experiment_dir / "config_source.txt").write_text(
            str(Path(config_source).resolve()), encoding="utf-8"
        )

    env = _make_env(config)
    trainer = PPO(env, config.ppo_config())
    opponent_evaluator = (
        _make_candidate_evaluator(env, trainer.model, config.policy_architecture)
        if (
            config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
            and config.opponent_update_score == "evaluation"
        )
        else None
    )
    archives = [
        SnapshotArchive(output_root, config.experiment_name, lineage)
        for lineage in range(config.num_lineages)
    ]
    keys = _lineage_seeds(config.seed, config.num_lineages)
    initial_runners = jax.jit(jax.vmap(trainer._initialize))(keys)
    jax.block_until_ready(initial_runners)

    template_params = _extract_lineage(initial_runners.train_state.params, 0)
    initial_policy = (
        _load_external_snapshot(
            config.initial_policy_snapshot, template_params, config
        )
        if config.initial_policy_snapshot is not None
        else None
    )
    if (
        initial_policy is not None
        and int(initial_policy[1]["generation"]) != config.generation_offset
    ):
        raise ValueError(
            "initial policy snapshot generation must equal generation_offset"
        )
    initial_opponent = (
        _load_external_snapshot(
            config.initial_opponent_snapshot, template_params, config
        )
        if config.initial_opponent_snapshot is not None
        else None
    )
    if initial_opponent is not None:
        initial_opponent_generation = int(initial_opponent[1]["generation"])
        if initial_opponent_generation == config.generation_offset:
            raise ValueError(
                "initial opponent and initial policy must use distinct generation IDs"
            )
        if initial_opponent_generation in _snapshot_generations(config):
            raise ValueError(
                f"initial opponent generation {initial_opponent_generation} collides "
                "with a snapshot this run will save; use an opponent generation outside "
                "the run's snapshot schedule"
            )

    if initial_policy is not None:
        stacked_initial_params = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values),
            *([initial_policy[0]] * config.num_lineages),
        )
        initial_runners = initial_runners._replace(
            train_state=initial_runners.train_state.replace(
                params=stacked_initial_params
            )
        )

    for lineage, archive in enumerate(archives):
        initial_params = _extract_lineage(
            initial_runners.train_state.params, lineage
        )
        archive.save(
            initial_params,
            generation=config.generation_offset,
            seed=config.seed,
            hidden_sizes=config.hidden_sizes,
            observation_size=(
                BRANDAO_OBSERVATION_SIZE
                if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                else 46
            ),
            action_size=2 if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES else 6,
            policy_architecture=config.policy_architecture,
        )

    if initial_opponent is not None:
        for archive in archives:
            archive.save(
                initial_opponent[0],
                generation=initial_opponent_generation,
                seed=config.seed,
                hidden_sizes=config.hidden_sizes,
                observation_size=(
                    BRANDAO_OBSERVATION_SIZE
                    if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                    else 46
                ),
                action_size=(
                    2
                    if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                    else 6
                ),
                policy_architecture=config.policy_architecture,
            )

    if initial_policy is not None or initial_opponent is not None:
        warm_start_metadata = {
            "generation_offset": config.generation_offset,
            "initial_policy_snapshot": (
                {
                    "path": str(initial_policy[2]),
                    "source_experiment": initial_policy[1].get("experiment_name"),
                    "source_lineage": initial_policy[1].get("lineage"),
                    "source_generation": initial_policy[1]["generation"],
                }
                if initial_policy is not None
                else None
            ),
            "initial_opponent_snapshot": (
                {
                    "path": str(initial_opponent[2]),
                    "source_experiment": initial_opponent[1].get("experiment_name"),
                    "source_lineage": initial_opponent[1].get("lineage"),
                    "source_generation": initial_opponent[1]["generation"],
                }
                if initial_opponent is not None
                else None
            ),
            "optimizer_state_restored": False,
            "trainer_state": (
                "fresh PPO optimizer and environment state; policy weights loaded from snapshots"
            ),
        }
        (experiment_dir / "warm_start.json").write_text(
            json.dumps(warm_start_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    pool_load_started = time.perf_counter()
    if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES:
        selected_generation = (
            initial_opponent_generation
            if initial_opponent is not None
            else config.generation_offset
        )
        selected_opponent_generations = [selected_generation] * config.num_lineages
        pools, probabilities, generation_ids, pool_generations = _brandao_opponent_pools(
            archives,
            template_params,
            config,
            selected_opponent_generations,
        )
    else:
        selected_opponent_generations = []
        pools, probabilities, generation_ids, pool_generations = _active_pools(
            archives, _extract_lineage(initial_runners.train_state.params, 0), config
        )
    initial_pool_load_seconds = time.perf_counter() - pool_load_started
    runners = trainer.initialize_lineages(keys, pools, probabilities, generation_ids)
    if initial_policy is not None:
        # initialize_lineages creates a fresh optimizer and environment state;
        # replace only its parameters so the optimizer state remains fresh.
        runners = runners._replace(
            train_state=runners.train_state.replace(params=stacked_initial_params)
        )
    jax.block_until_ready(runners)

    metrics_path = experiment_dir / "metrics.jsonl"
    performance_path = experiment_dir / "performance.jsonl"
    _write_json_line(
        performance_path,
        {
            "event": "initial_pool_load",
            "generation": config.generation_offset,
            "seconds": initial_pool_load_seconds,
            "active_snapshots_per_lineage": len(pool_generations[0]),
            "selected_opponent_generation": (
                selected_opponent_generations[0]
                if selected_opponent_generations
                else None
            ),
            "opponent_update_threshold": (
                config.opponent_update_threshold
                if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                else None
            ),
        },
    )
    completed = 0
    steps_per_generation = config.ppo_config().batch_size
    transitions_per_step = (
        3 if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES else 1
    )
    print(f"JAX devices          : {jax.devices()}")
    print(f"Lineages             : {config.num_lineages}")
    print(f"Environments/lineage : {config.num_envs_per_lineage}")
    print(f"Generations          : {config.num_generations}")

    while completed < config.num_generations:
        until_snapshot = config.snapshot_interval - completed % config.snapshot_interval
        chunk_size = min(until_snapshot, config.num_generations - completed)
        if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES:
            until_evaluation = (
                config.opponent_evaluation_interval
                - completed % config.opponent_evaluation_interval
            )
            chunk_size = min(chunk_size, until_evaluation)
        started = time.perf_counter()
        runners, history = trainer.train_lineages(
            runners,
            pools,
            probabilities,
            num_updates=chunk_size,
            opponent_generations=generation_ids,
        )
        jax.block_until_ready((runners, history))
        elapsed = time.perf_counter() - started
        _write_json_line(
            performance_path,
            {
                "event": "train_chunk",
                "generation_start": config.generation_offset + completed + 1,
                "generation_end": config.generation_offset + completed + chunk_size,
                "seconds": elapsed,
                "env_steps": chunk_size * steps_per_generation * config.num_lineages,
                "agent_transitions": (
                    chunk_size
                    * steps_per_generation
                    * config.num_lineages
                    * transitions_per_step
                ),
                "env_steps_per_second": (
                    chunk_size * steps_per_generation * config.num_lineages
                )
                / max(elapsed, 1e-9),
                "agent_transitions_per_second": (
                    chunk_size
                    * steps_per_generation
                    * config.num_lineages
                    * transitions_per_step
                )
                / max(elapsed, 1e-9),
            },
        )
        host_history = jax.device_get(history)
        metric_records = {}
        for update_in_chunk in range(chunk_size):
            generation = (
                config.generation_offset + completed + update_in_chunk + 1
            )
            for lineage in range(config.num_lineages):
                blue_goals_by_env = np.asarray(
                    host_history["blue_goals_by_env"][update_in_chunk, lineage],
                    dtype=np.int64,
                )
                yellow_goals_by_env = np.asarray(
                    host_history["yellow_goals_by_env"][update_in_chunk, lineage],
                    dtype=np.int64,
                )
                referee_event_counts = np.asarray(
                    host_history["referee_event_counts"][update_in_chunk, lineage],
                    dtype=np.int64,
                )
                record = {
                    "generation": generation,
                    "lineage": lineage,
                    "training_chunk_seconds": elapsed,
                    "chunk_generations": chunk_size,
                    "env_steps": int(host_history["env_steps"][update_in_chunk, lineage]),
                    "agent_transitions": int(
                        host_history["env_steps"][update_in_chunk, lineage]
                    )
                    * transitions_per_step,
                    "loss": float(host_history["loss"][update_in_chunk, lineage]),
                    "policy_loss": float(
                        host_history["policy_loss"][update_in_chunk, lineage]
                    ),
                    "value_loss": float(
                        host_history["value_loss"][update_in_chunk, lineage]
                    ),
                    "entropy": float(host_history["entropy"][update_in_chunk, lineage]),
                    "approx_kl": float(
                        host_history["approx_kl"][update_in_chunk, lineage]
                    ),
                    "clip_fraction": float(
                        host_history["clip_fraction"][update_in_chunk, lineage]
                    ),
                    "ratio_log_clipped_fraction": float(
                        host_history["ratio_log_clipped_fraction"][
                            update_in_chunk, lineage
                        ]
                    ),
                    "mean_episode_return": float(
                        host_history["mean_episode_return"][update_in_chunk, lineage]
                    ),
                    "mean_episode_length": float(
                        host_history["mean_episode_length"][update_in_chunk, lineage]
                    ),
                    "episodes": int(host_history["episodes"][update_in_chunk, lineage]),
                    "episode_terminations": int(
                        host_history["episodes"][update_in_chunk, lineage]
                    ),
                    "blue_goals": int(host_history["blue_goals"][update_in_chunk, lineage]),
                    "yellow_goals": int(
                        host_history["yellow_goals"][update_in_chunk, lineage]
                    ),
                    "blue_goals_by_env": blue_goals_by_env.tolist(),
                    "yellow_goals_by_env": yellow_goals_by_env.tolist(),
                    "mean_blue_goals_per_env": float(np.mean(blue_goals_by_env)),
                    "mean_yellow_goals_per_env": float(np.mean(yellow_goals_by_env)),
                    "mean_goal_delta_per_env": float(
                        np.mean(blue_goals_by_env - yellow_goals_by_env)
                    ),
                    "referee_event_counts": {
                        event.name.lower(): int(referee_event_counts[int(event)])
                        for event in RefereeEvent
                    },
                    "opponent_generation": (
                        selected_opponent_generations[lineage]
                        if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                        else None
                    ),
                    "opponent_evaluated": False,
                    "opponent_update_score": config.opponent_update_score,
                    "opponent_evaluation_sample_count": 0,
                    "opponent_evaluation_mean_goals_for": None,
                    "opponent_evaluation_mean_goals_against": None,
                    "opponent_evaluation_mean_goal_delta": None,
                    "opponent_replaced": False,
                    "selected_opponent_generation": (
                        selected_opponent_generations[lineage]
                        if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                        else None
                    ),
                    "opponent_selector_step_counts": {
                        name: int(count)
                        for name, count in zip(
                            ("all_history", "recent_half", "latest", "random", "stationary"),
                            host_history["opponent_selector_counts"][update_in_chunk, lineage],
                            strict=True,
                        )
                    },
                    "opponent_snapshot_step_counts": {
                        str(generation): int(count)
                        for generation, count in zip(
                            *np.unique(
                                np.asarray(
                                    host_history["opponent_generations"][
                                        update_in_chunk, lineage
                                    ]
                                ).reshape(-1),
                                return_counts=True,
                            ),
                            strict=True,
                        )
                        if generation >= 0
                    },
                }
                metric_records[(generation, lineage)] = record
        completed += chunk_size

        should_snapshot = completed % config.snapshot_interval == 0
        is_final = completed == config.num_generations
        snapshot_generation = config.generation_offset + completed

        def save_snapshots(lineages):
            snapshot_started = time.perf_counter()
            for lineage in lineages:
                archive = archives[lineage]
                params = _extract_lineage(runners.train_state.params, lineage)
                archive.save(
                    params,
                    generation=snapshot_generation,
                    seed=config.seed + lineage,
                    hidden_sizes=config.hidden_sizes,
                    observation_size=(
                        BRANDAO_OBSERVATION_SIZE
                        if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
                        else 46
                    ),
                    action_size=2 if config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES else 6,
                    policy_architecture=config.policy_architecture,
                )
            snapshot_seconds = time.perf_counter() - snapshot_started
            _write_json_line(
                performance_path,
                {
                    "event": "snapshot_write",
                    "generation": snapshot_generation,
                    "seconds": snapshot_seconds,
                },
            )

        if should_snapshot or is_final:
            save_snapshots(range(config.num_lineages))

        should_evaluate = (
            config.policy_architecture in _BRANDAO_POLICY_ARCHITECTURES
            and (completed % config.opponent_evaluation_interval == 0 or is_final)
        )
        if should_evaluate:
            replaced_any = False
            for lineage, archive in enumerate(archives):
                candidate_params = _extract_lineage(
                    runners.train_state.params, lineage
                )
                opponent_generation = selected_opponent_generations[lineage]
                evaluation_started = time.perf_counter()
                record = metric_records[(snapshot_generation, lineage)]
                evaluation_seed = None
                if config.opponent_update_score == "training_rollout":
                    scores = np.stack(
                        (record["blue_goals_by_env"], record["yellow_goals_by_env"]),
                        axis=1,
                    )
                else:
                    opponent_params = archive.load(
                        candidate_params,
                        opponent_generation,
                        expected_hidden_sizes=config.hidden_sizes,
                        expected_policy_architecture=config.policy_architecture,
                    )
                    evaluation_matches = (
                        config.opponent_evaluation_matches or config.evaluation_matches
                    )
                    evaluation_seed = (
                        config.seed + snapshot_generation * evaluation_matches
                        + lineage * 10_000_000
                    )
                    evaluation_seeds = jnp.arange(
                        evaluation_seed, evaluation_seed + evaluation_matches,
                        dtype=jnp.uint32,
                    )
                    scores = opponent_evaluator(
                        candidate_params, opponent_params, evaluation_seeds
                    )
                    jax.block_until_ready(scores)
                evaluation = _opponent_evaluation_summary(
                    np.asarray(scores), config.opponent_update_threshold
                )
                replaced = bool(evaluation["opponent_replaced"])
                if replaced:
                    if not (should_snapshot or is_final):
                        save_snapshots([lineage])
                    selected_opponent_generations[lineage] = snapshot_generation
                    replaced_any = True
                record = metric_records[(snapshot_generation, lineage)]
                record.update(
                    {
                        "opponent_generation": opponent_generation,
                        "opponent_evaluated": True,
                        "opponent_evaluation_sample_count": evaluation[
                            "evaluation_sample_count"
                        ],
                        "opponent_evaluation_mean_goals_for": evaluation[
                            "mean_goals_for"
                        ],
                        "opponent_evaluation_mean_goals_against": evaluation[
                            "mean_goals_against"
                        ],
                        "opponent_evaluation_mean_goal_delta": evaluation[
                            "mean_goal_delta"
                        ],
                        "opponent_replaced": replaced,
                        "selected_opponent_generation": (
                            selected_opponent_generations[lineage]
                        ),
                    }
                )
                _write_json_line(
                    performance_path,
                    {
                        "event": "opponent_evaluation",
                        "seconds": time.perf_counter() - evaluation_started,
                        "generation": snapshot_generation,
                        "lineage": lineage,
                        "opponent_generation": opponent_generation,
                        **evaluation,
                        "update_threshold": config.opponent_update_threshold,
                        "selected_opponent_generation": (
                            selected_opponent_generations[lineage]
                        ),
                        "evaluation_seed": evaluation_seed,
                        "score_source": config.opponent_update_score,
                        "policy_action_mode": (
                            "training_stochastic" if config.opponent_update_score == "training_rollout"
                            else "deterministic_mean"
                        ),
                    },
                )

            if replaced_any and not is_final:
                pool_load_started = time.perf_counter()
                pools, probabilities, generation_ids, pool_generations = (
                    _brandao_opponent_pools(
                        archives,
                        _extract_lineage(runners.train_state.params, 0),
                        config,
                        selected_opponent_generations,
                    )
                )
                _write_json_line(
                    performance_path,
                    {
                        "event": "active_pool_reload",
                        "generation": snapshot_generation,
                        "seconds": time.perf_counter() - pool_load_started,
                        "active_snapshots_per_lineage": len(pool_generations[0]),
                    },
                )
        elif (
            should_snapshot and not is_final
            and config.policy_architecture not in _BRANDAO_POLICY_ARCHITECTURES
        ):
            pool_load_started = time.perf_counter()
            pools, probabilities, generation_ids, pool_generations = _active_pools(
                archives,
                _extract_lineage(runners.train_state.params, 0),
                config,
            )
            _write_json_line(
                performance_path,
                {
                    "event": "active_pool_reload",
                    "generation": config.generation_offset + completed,
                    "seconds": time.perf_counter() - pool_load_started,
                    "active_snapshots_per_lineage": len(pool_generations[0]),
                },
            )

        for record in metric_records.values():
            record["run_elapsed_seconds"] = time.perf_counter() - run_started
            _write_json_line(metrics_path, record)

        steps = chunk_size * steps_per_generation * config.num_lineages
        print(
            f"Generations {config.generation_offset + completed - chunk_size + 1}-"
            f"{config.generation_offset + completed}: "
            f"{steps:,} env steps in {elapsed:.2f}s "
            f"({steps / max(elapsed, 1e-9):,.0f} env-steps/s)"
        )

    total_elapsed = time.perf_counter() - run_started
    total_env_steps = (
        config.num_generations
        * steps_per_generation
        * config.num_lineages
    )
    _write_json_line(
        performance_path,
        {
            "event": "experiment_complete",
            "generation_start": config.generation_offset + 1,
            "generation_end": config.generation_offset + config.num_generations,
            "elapsed_seconds": total_elapsed,
            "env_steps": total_env_steps,
            "agent_transitions": total_env_steps * transitions_per_step,
            "env_steps_per_second": total_env_steps / max(total_elapsed, 1e-9),
            "agent_transitions_per_second": (
                total_env_steps * transitions_per_step / max(total_elapsed, 1e-9)
            ),
        },
    )
    return experiment_dir
