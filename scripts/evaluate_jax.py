#!/usr/bin/env python3
"""Evaluate a saved JAX historical self-play experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from vsss_sim.rl.evaluate import evaluate_experiment
from vsss_sim.rl.self_play import ExperimentConfig


def _resolve_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    candidate = Path(__file__).resolve().parents[1] / "configs" / "experiments" / candidate.name
    if not candidate.is_file():
        raise FileNotFoundError(f"experiment config not found: {value}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xp", required=True, help="Experiment JSON path or name.")
    parser.add_argument("--lineage", type=int, help="Evaluate one lineage; default evaluates all.")
    parser.add_argument("--generation", type=int, help="Policy snapshot generation; default latest.")
    parser.add_argument("--matches", type=int, help="Matches per opponent group.")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--opponents",
        choices=("all", "history", "random", "stationary"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config_path = _resolve_config(args.xp)
    config = ExperimentConfig.from_json(config_path)
    output = evaluate_experiment(
        config,
        lineage=args.lineage,
        generation=args.generation,
        matches=args.matches,
        seed=args.seed,
        opponents=args.opponents,
        batch_size=args.batch_size,
    )
    print(f"Evaluation written to {output}")


if __name__ == "__main__":
    main()
