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

from ..config import InitMode
from ..envs.jumanji import VSSJumanjiEnv
from .ppo import PPO, PPOConfig

_SNAPSHOT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    update_epochs: int = 4
    num_minibatches: int = 8
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    snapshot_interval: int = 10
    active_pool_limit: int | None = None
    max_episode_steps: int = 300
    terminate_on_goal: bool = False
    init_mode: str = "kickoff"
    output_root: str = "output/experiments"
    evaluation_matches: int = 100
    evaluation_seconds: int = 120
    opponent_weights: OpponentSamplingWeights = field(
        default_factory=OpponentSamplingWeights
    )

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
            "max_episode_steps",
            "evaluation_matches",
            "evaluation_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.active_pool_limit is not None and self.active_pool_limit < 1:
            raise ValueError("active_pool_limit must be at least 1 when provided")
        if self.init_mode not in ("kickoff", "random"):
            raise ValueError("init_mode must be 'kickoff' or 'random'")
        # Reuse the trainer's validation before initializing a device or allocating state.
        self.ppo_config()

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            num_envs=self.num_envs_per_lineage,
            rollout_length=self.rollout_length,
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            num_minibatches=self.num_minibatches,
            hidden_sizes=self.hidden_sizes,
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
        observation_size: int = 46,
        action_size: int = 6,
    ) -> Path:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"generation_{generation:08d}"
        payload_path = self.directory / f"{stem}.msgpack"
        metadata_path = self.directory / f"{stem}.json"
        if payload_path.exists() or metadata_path.exists():
            raise FileExistsError(f"snapshot for generation {generation} already exists")

        metadata = {
            "schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "experiment_name": self.experiment_name,
            "lineage": self.lineage,
            "generation": int(generation),
            "seed": int(seed),
            "hidden_sizes": list(hidden_sizes),
            "observation_size": int(observation_size),
            "action_size": int(action_size),
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
    ) -> dict:
        matches = [item for item in self.list_snapshots() if item["generation"] == generation]
        if not matches:
            raise FileNotFoundError(f"no snapshot for generation {generation} in {self.directory}")
        metadata = matches[0]
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
    ) -> tuple[dict, tuple[int, ...]]:
        snapshots = self.list_snapshots()
        if active_limit is not None:
            snapshots = snapshots[-active_limit:]
        if not snapshots:
            raise ValueError(f"historical pool is empty: {self.directory}")
        generations = tuple(item["generation"] for item in snapshots)
        params = [
            self.load(template, generation, expected_hidden_sizes)
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


def _lineage_seeds(seed: int, count: int) -> jax.Array:
    root_key = jax.random.PRNGKey(seed)
    return jax.vmap(lambda lineage: jax.random.fold_in(root_key, lineage))(
        jnp.arange(count, dtype=jnp.uint32)
    )


def _write_json_line(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def run_experiment(config: ExperimentConfig, config_source: str | Path | None = None) -> Path:
    """Run one experiment with independent PPO learner states on a lineage axis."""
    output_root = Path(config.output_root)
    experiment_dir = output_root / config.experiment_name
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"experiment output already exists: {experiment_dir}; use a new experiment_name"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
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
    archives = [
        SnapshotArchive(output_root, config.experiment_name, lineage)
        for lineage in range(config.num_lineages)
    ]
    keys = _lineage_seeds(config.seed, config.num_lineages)
    initial_runners = jax.jit(jax.vmap(trainer._initialize))(keys)
    jax.block_until_ready(initial_runners)

    for lineage, archive in enumerate(archives):
        initial_params = _extract_lineage(
            initial_runners.train_state.params, lineage
        )
        archive.save(
            initial_params,
            generation=0,
            seed=config.seed,
            hidden_sizes=config.hidden_sizes,
        )

    pool_load_started = time.perf_counter()
    pools, probabilities, generation_ids, pool_generations = _active_pools(
        archives, _extract_lineage(initial_runners.train_state.params, 0), config
    )
    initial_pool_load_seconds = time.perf_counter() - pool_load_started
    runners = trainer.initialize_lineages(keys, pools, probabilities, generation_ids)
    jax.block_until_ready(runners)

    metrics_path = experiment_dir / "metrics.jsonl"
    performance_path = experiment_dir / "performance.jsonl"
    _write_json_line(
        performance_path,
        {
            "event": "initial_pool_load",
            "generation": 0,
            "seconds": initial_pool_load_seconds,
            "active_snapshots_per_lineage": len(pool_generations[0]),
        },
    )
    completed = 0
    steps_per_generation = config.ppo_config().batch_size
    print(f"JAX devices          : {jax.devices()}")
    print(f"Lineages             : {config.num_lineages}")
    print(f"Environments/lineage : {config.num_envs_per_lineage}")
    print(f"Generations          : {config.num_generations}")

    while completed < config.num_generations:
        until_snapshot = config.snapshot_interval - completed % config.snapshot_interval
        chunk_size = min(until_snapshot, config.num_generations - completed)
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
                "generation_start": completed + 1,
                "generation_end": completed + chunk_size,
                "seconds": elapsed,
                "env_steps": chunk_size * steps_per_generation * config.num_lineages,
                "env_steps_per_second": (
                    chunk_size * steps_per_generation * config.num_lineages
                )
                / max(elapsed, 1e-9),
            },
        )
        host_history = jax.device_get(history)
        for update_in_chunk in range(chunk_size):
            generation = completed + update_in_chunk + 1
            for lineage in range(config.num_lineages):
                record = {
                    "generation": generation,
                    "lineage": lineage,
                    "training_chunk_seconds": elapsed,
                    "chunk_generations": chunk_size,
                    "env_steps": int(host_history["env_steps"][update_in_chunk, lineage]),
                    "loss": float(host_history["loss"][update_in_chunk, lineage]),
                    "mean_episode_return": float(
                        host_history["mean_episode_return"][update_in_chunk, lineage]
                    ),
                    "mean_episode_length": float(
                        host_history["mean_episode_length"][update_in_chunk, lineage]
                    ),
                    "episodes": int(host_history["episodes"][update_in_chunk, lineage]),
                    "blue_goals": int(host_history["blue_goals"][update_in_chunk, lineage]),
                    "yellow_goals": int(
                        host_history["yellow_goals"][update_in_chunk, lineage]
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
                _write_json_line(metrics_path, record)
        completed += chunk_size

        should_snapshot = completed % config.snapshot_interval == 0
        is_final = completed == config.num_generations
        if should_snapshot or is_final:
            snapshot_started = time.perf_counter()
            for lineage, archive in enumerate(archives):
                params = _extract_lineage(runners.train_state.params, lineage)
                archive.save(
                    params,
                    generation=completed,
                    seed=config.seed + lineage,
                    hidden_sizes=config.hidden_sizes,
                )
            snapshot_seconds = time.perf_counter() - snapshot_started
            _write_json_line(
                performance_path,
                {
                    "event": "snapshot_write",
                    "generation": completed,
                    "seconds": snapshot_seconds,
                },
            )
            if not is_final:
                pool_load_started = time.perf_counter()
                pools, probabilities, generation_ids, pool_generations = _active_pools(
                    archives,
                    _extract_lineage(runners.train_state.params, 0),
                    config,
                )
                pool_load_seconds = time.perf_counter() - pool_load_started
                _write_json_line(
                    performance_path,
                    {
                        "event": "active_pool_reload",
                        "generation": completed,
                        "seconds": pool_load_seconds,
                        "active_snapshots_per_lineage": len(pool_generations[0]),
                    },
                )

        steps = chunk_size * steps_per_generation * config.num_lineages
        print(
            f"Generations {completed - chunk_size + 1}-{completed}: "
            f"{steps:,} env steps in {elapsed:.2f}s "
            f"({steps / max(elapsed, 1e-9):,.0f} env-steps/s)"
        )

    return experiment_dir
