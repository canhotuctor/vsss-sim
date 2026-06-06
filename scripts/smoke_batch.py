#!/usr/bin/env python3
"""
Run smoke.py N times with shared hyperparameters; each run gets its own seed and policy file.

Usage
-----
    python scripts/smoke_batch.py --runs 5 --generations 100 --num-envs 64 \\
        --output-dir runs/batch1

    python scripts/smoke_batch.py --runs 8 --timesteps 50000 --num-envs 32 \\
        --output-dir runs/sweep --seed-start 100 --jobs 2

Policies are written to ``<output-dir>/policy_<NNN>_seed<SEED>.zip``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke.py"


def _run_smoke(
    run_index: int,
    seed: int,
    save_path: Path,
    smoke_extra: list[str],
) -> tuple[int, int, int]:
    """Launch one smoke.py process. Returns (run_index, seed, returncode)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SMOKE_SCRIPT),
        "--seed",
        str(seed),
        "--save-path",
        str(save_path),
        *smoke_extra,
    ]
    print(f"[run {run_index:03d}] seed={seed}  save={save_path}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[run {run_index:03d}] FAILED (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"[run {run_index:03d}] done")
    return run_index, seed, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch N independent smoke.py runs; save one policy per run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help="Number of smoke.py processes to launch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where policy zip files are saved.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Seed for run 0; run i uses seed_start + i.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="How many smoke.py processes to run at once (1 = sequential).",
    )

    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--timesteps", type=int, default=None)
    duration.add_argument("--generations", type=int, default=None)
    duration.add_argument("--forever", action="store_true")

    parser.add_argument("--backend", type=str, default=None, choices=["numpy", "jax"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument(
        "--init-mode",
        type=str,
        default=None,
        choices=["kickoff", "random"],
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    if args.forever:
        parser.error("--forever is not supported in batch mode (runs would never finish)")

    if args.timesteps is None and args.generations is None:
        # smoke.py defaults to 10_000 timesteps when neither is set; mirror that.
        args.timesteps = 10_000

    smoke_extra: list[str] = []
    if args.timesteps is not None:
        smoke_extra.extend(["--timesteps", str(args.timesteps)])
    if args.generations is not None:
        smoke_extra.extend(["--generations", str(args.generations)])
    if args.backend is not None:
        smoke_extra.extend(["--backend", args.backend])
    if args.num_envs != 1:
        smoke_extra.extend(["--num-envs", str(args.num_envs)])
    if args.max_episode_steps is not None:
        smoke_extra.extend(["--max-episode-steps", str(args.max_episode_steps)])
    if args.n_steps is not None:
        smoke_extra.extend(["--n-steps", str(args.n_steps)])
    if args.init_mode is not None:
        smoke_extra.extend(["--init-mode", args.init_mode])

    jobs = [
        (
            i,
            args.seed_start + i,
            args.output_dir / f"policy_{i:03d}_seed{args.seed_start + i}.zip",
            smoke_extra,
        )
        for i in range(args.runs)
    ]

    print(
        f"Launching {args.runs} run(s), jobs={args.jobs}, "
        f"seeds {args.seed_start}..{args.seed_start + args.runs - 1}"
    )
    print(f"Output dir: {args.output_dir.resolve()}")

    failures: list[tuple[int, int, int]] = []

    if args.jobs == 1:
        for job in jobs:
            rc = _run_smoke(*job)
            if rc[2] != 0:
                failures.append(rc)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_smoke, *job) for job in jobs]
            for fut in as_completed(futures):
                rc = fut.result()
                if rc[2] != 0:
                    failures.append(rc)

    if failures:
        failed_ids = ", ".join(f"{idx:03d}" for idx, _, _ in sorted(failures))
        print(f"\n{len(failures)}/{args.runs} run(s) failed: {failed_ids}", file=sys.stderr)
        return 1

    print(f"\nAll {args.runs} runs finished. Policies in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
