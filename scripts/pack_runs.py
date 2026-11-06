#!/usr/bin/env python3
"""Package finished training runs so they can be moved off a training machine.

For every complete run (final-generation snapshot present) matching --pattern:
  <out>/small/<run>/   config.json, performance.jsonl, metrics.jsonl.gz, <run>.log
                       (a few MB: fine to commit to a data branch)
  <out>/snapshots/<run>.tgz   config.json + lineage_*/ snapshots, rooted at <run>/
                       (~90 MB: upload as a release asset, not into git history)
Extract a tgz inside outputs/experiments/ to restore a run for scripts/evaluate_jax.py.
Existing outputs are never overwritten; unfinished runs are skipped.

    python scripts/pack_runs.py --pattern 'experiment27_replay0_l*' --out packed
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path


def is_complete(run: Path) -> bool:
    config = run / "config.json"
    if not config.is_file():
        return False
    final = int(json.loads(config.read_text())["num_generations"])
    offset = int(json.loads(config.read_text()).get("generation_offset", 0))
    return any((run / lineage / "snapshots" / f"generation_{offset + final:08d}.json").is_file()
               for lineage in (p.name for p in run.glob("lineage_*")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--pattern", default="experiment27_*")
    parser.add_argument("--out", default="packed")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    packed = skipped = 0
    for run in sorted(p for p in root.glob(args.pattern) if p.is_dir()):
        if not is_complete(run):
            print(f"skip (unfinished): {run.name}")
            skipped += 1
            continue
        small = out / "small" / run.name
        archive = out / "snapshots" / f"{run.name}.tgz"
        if small.exists() or archive.exists():
            print(f"skip (already packed): {run.name}")
            continue
        small.mkdir(parents=True)
        for name in ("config.json", "performance.jsonl"):
            shutil.copy2(run / name, small / name)
        log = root / f"{run.name}.log"
        if log.is_file():
            shutil.copy2(log, small / log.name)
        with open(run / "metrics.jsonl", "rb") as src, gzip.open(small / "metrics.jsonl.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(run / "config.json", arcname=f"{run.name}/config.json")
            for lineage in sorted(run.glob("lineage_*")):
                tar.add(lineage, arcname=f"{run.name}/{lineage.name}")
        print(f"packed {run.name}: {archive.stat().st_size / 1e6:.0f} MB archive")
        packed += 1
    print(f"done: {packed} packed, {skipped} unfinished")


if __name__ == "__main__":
    main()
