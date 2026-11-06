#!/usr/bin/env python3
"""Train many lineages on one GPU: probe how many fit, then run them in waves.

1. Probe: binary-search the largest lineage count that completes a short
   training run (a fresh subprocess per probe, so GPU memory is released).
2. Waves: split the experiment's lineages into the fewest balanced waves of at
   most that size and run them one after another (a semaphore of width one).

Each wave is its own experiment, ``<name>_w00``, ``<name>_w01``, ... with its own
seed (``seed + 1000 * wave``) so lineages never repeat across waves. A wave whose
final snapshot already exists is skipped, so a crashed run resumes where it stopped.

    JAX_PLATFORMS=cuda python scripts/train_in_waves.py \
        --xp experiment27_replay0 experiment27_replay25
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train_jax.py"


def load_config(value: str) -> tuple[dict, Path]:
    path = Path(value)
    if not path.is_file():
        path = REPO / "configs" / "experiments" / path.with_suffix(".json").name
    return json.loads(path.read_text(encoding="utf-8")), path


def run_training(config: dict, directory: Path, log: Path | None) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config['experiment_name']}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    out = open(log, "a") if log else subprocess.DEVNULL
    try:
        return subprocess.run(
            [sys.executable, str(TRAIN), "--xp", str(path)],
            env=env, stdout=out, stderr=subprocess.STDOUT, cwd=REPO,
        ).returncode
    finally:
        if log:
            out.close()


def fits(config: dict, lineages: int, probe_generations: int) -> bool:
    """True if ``lineages`` lineages complete a short run on this GPU."""
    with tempfile.TemporaryDirectory(prefix="vsss-probe-") as tmp:
        probe = {
            **config,
            "experiment_name": f"probe_{lineages}",
            "num_lineages": lineages,
            "num_generations": probe_generations,
            # Hit the snapshot + promotion-evaluation paths too: they allocate more.
            "snapshot_interval": 1,
            "output_root": str(Path(tmp) / "out"),
            "require_gpu": True,
        }
        log = Path(tmp) / "probe.log"
        code = run_training(probe, Path(tmp) / "cfg", log)
        tail = log.read_text(errors="replace")[-400:].strip().splitlines()[-1:] if log.exists() else []
    print(f"  probe {lineages:>3} lineages: {'fits' if code == 0 else 'FAILED'}"
          + ("" if code == 0 else f"  ({tail[0][:90]})" if tail else ""), flush=True)
    return code == 0


def max_lineages(config: dict, probe_generations: int) -> int:
    total = int(config["num_lineages"])
    if not fits(config, 1, probe_generations):
        raise SystemExit("even one lineage does not fit (or training is broken); see probe output")
    low, high = 1, total  # low always fits
    if fits(config, total, probe_generations):
        return total
    high -= 1
    while low < high:
        mid = (low + high + 1) // 2
        if fits(config, mid, probe_generations):
            low = mid
        else:
            high = mid - 1
    return low


def split_waves(total: int, width: int) -> list[int]:
    count = math.ceil(total / width)
    base, extra = divmod(total, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


def wave_done(config: dict) -> bool:
    final = (Path(config["output_root"]) / config["experiment_name"] / "lineage_000" / "snapshots"
             / f"generation_{config['num_generations']:08d}.json")
    return final.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xp", nargs="+", required=True, help="Experiment names/paths, run in order.")
    parser.add_argument("--lineages-per-wave", type=int, help="Skip probing and use this width.")
    parser.add_argument("--probe-generations", type=int, default=2)
    parser.add_argument("--margin", type=int, default=0,
                        help="Subtract this many lineages from the probed maximum for headroom.")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    for name in args.xp:
        config, source = load_config(name)
        total = int(config["num_lineages"])
        print(f"== {config['experiment_name']}: {total} lineages x {config['num_envs_per_lineage']} envs ==")
        width = args.lineages_per_wave or max_lineages(config, args.probe_generations)
        width = max(1, min(total, width - args.margin))
        sizes = split_waves(total, width)
        print(f"  max fitting width (after margin): {width} -> waves {sizes}", flush=True)
        if args.probe_only:
            continue
        out_dir = Path(config["output_root"]) / config["experiment_name"]
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        for index, size in enumerate(sizes):
            wave = {
                **config,
                "experiment_name": f"{config['experiment_name']}_w{index:02d}",
                "num_lineages": size,
                "seed": int(config["seed"]) + 1000 * index,
            }
            if wave_done(wave):
                print(f"  wave {index}: already complete, skipping")
                continue
            partial = Path(wave["output_root"]) / wave["experiment_name"]
            if partial.exists():
                shutil.move(str(partial), str(partial) + ".incomplete")  # run_experiment needs a fresh dir
                print(f"  wave {index}: moved unfinished output to {partial}.incomplete")
            log = Path(wave["output_root"]) / f"{wave['experiment_name']}.log"
            print(f"  wave {index}: {size} lineages, log {log}", flush=True)
            code = run_training(wave, Path(wave["output_root"]) / "_wave_configs", log)
            if code != 0:
                raise SystemExit(f"wave {index} failed (exit {code}); see {log}")
        print(f"== {config['experiment_name']} done ==")


if __name__ == "__main__":
    main()
