#!/usr/bin/env python3
"""
Plot MLflow sweep CSV: mean reward curve + 95% bootstrap CI across runs.

Usage
-----
    # Single sweep
    python scripts/plot_sweep.py runs/sweep/out_sweep_250g_4env.csv

    # Compare multiple sweeps on one figure
    python scripts/plot_sweep.py runs/sweep/out_sweep_250g_{4,16,64}env.csv \\
        -o runs/sweep/out_sweep_250g_combined
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRIC = "ep_reward/mean"
SERIES_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def _clean_value(raw: str) -> float:
    return float(str(raw).strip().strip("'"))


def label_from_path(path: Path) -> str:
    m = re.search(r"_(\d+)env", path.stem)
    if m:
        return rf"{m.group(1)} envs"
    return path.stem


def load_sweep_csv(
    path: Path,
    metric: str = DEFAULT_METRIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (generations, values) with shape (T,) and (T, R) for R runs."""
    by_run: dict[str, dict[int, float]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            if row.get("metric", "").strip() != metric:
                continue
            run = row["Run"].strip()
            generation = int(row["step"])
            value = _clean_value(row["value"])
            by_run.setdefault(run, {})[generation] = value

    if not by_run:
        raise ValueError(f"No rows for metric {metric!r} in {path}")

    runs = sorted(by_run)
    generations = sorted({g for series in by_run.values() for g in series})
    gen_arr = np.asarray(generations, dtype=np.int64)

    values = np.full((len(generations), len(runs)), np.nan, dtype=np.float64)
    for j, run in enumerate(runs):
        for i, generation in enumerate(generations):
            if generation in by_run[run]:
                values[i, j] = by_run[run][generation]

    if np.isnan(values).any():
        raise ValueError(f"Runs have mismatched generations in {path}.")

    return gen_arr, values


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean along axis 0 (time), per run."""
    if window <= 1:
        return values.copy()
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        col = values[:, j]
        cs = np.cumsum(col)
        for i in range(len(col)):
            lo = max(0, i - window + 1)
            out[i, j] = (cs[i] - (cs[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-generation mean and bootstrap CI across runs (axis 1)."""
    rng = np.random.default_rng(seed)
    n_gens, n_runs = values.shape
    mean = values.mean(axis=1)
    if n_runs == 1:
        return mean, mean, mean

    alpha = (100.0 - ci) / 2.0
    lo = np.empty(n_gens, dtype=np.float64)
    hi = np.empty(n_gens, dtype=np.float64)

    for i in range(n_gens):
        sample_vals = values[i]
        boots = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            pick = rng.choice(sample_vals, size=n_runs, replace=True)
            boots[b] = pick.mean()
        lo[i] = np.percentile(boots, alpha)
        hi[i] = np.percentile(boots, 100.0 - alpha)

    return mean, lo, hi


def apply_plot_style(use_tex: bool = False) -> None:
    plt.rc("text", usetex=use_tex)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "mathtext.fontset": "cm",
    })


def _plot_series(
    ax: plt.Axes,
    generations: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    smooth_window: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if smooth_window > 1:
        values = rolling_mean(values, smooth_window)
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    ax.fill_between(generations, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.plot(generations, mean, color=color, linewidth=2, label=label)
    return min(lo.min(), mean.min()), max(hi.max(), mean.max())


def plot_sweeps(
    csv_paths: list[Path],
    output: Path,
    *,
    labels: list[str] | None,
    metric: str = DEFAULT_METRIC,
    smooth_window: int,
    n_boot: int,
    seed: int,
    use_tex: bool,
) -> None:
    apply_plot_style(use_tex=use_tex)
    fig, ax = plt.subplots(figsize=(13.0, 8.0))

    y_min = np.inf
    y_max = -np.inf
    x_min = np.inf
    x_max = -np.inf

    for i, csv_path in enumerate(csv_paths):
        generations, values = load_sweep_csv(csv_path, metric=metric)
        label = labels[i] if labels else label_from_path(csv_path)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        lo, hi = _plot_series(
            ax,
            generations,
            values,
            label=label,
            color=color,
            smooth_window=smooth_window,
            n_boot=n_boot,
            seed=seed + i,
        )
        y_min = min(y_min, lo)
        y_max = max(y_max, hi)
        x_min = min(x_min, generations[0])
        x_max = max(x_max, generations[-1])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward", fontstyle="italic")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(x_min, x_max)

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
    ax.set_ylim(y_min - pad, y_max + pad)

    if len(csv_paths) > 1:
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), format="png", dpi=400)
    try:
        fig.savefig(output.with_suffix(".eps"), format="eps")
        print(f"saved {output.with_suffix('.eps')}")
    except Exception as exc:
        print(f"skipped EPS ({exc})")
    plt.close(fig)
    print(f"saved {output.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep CSV with bootstrap CI.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="One or more MLflow-exported sweep CSV files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path without extension. Default: <csv_stem>_plot (single) "
             "or <first_csv_stem>_combined (multiple).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Legend label per CSV (repeat once per file; default: parsed from filename).",
    )
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Rolling-mean window (generations) applied per run before aggregation.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Render labels with LaTeX (requires a local TeX install).",
    )
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.csv):
        parser.error(f"Expected {len(args.csv)} --label values, got {len(args.label)}")

    if args.output is not None:
        output = args.output
    elif len(args.csv) == 1:
        output = args.csv[0].with_name(f"{args.csv[0].stem}_plot")
    else:
        output = args.csv[0].with_name(f"{args.csv[0].stem}_combined")

    plot_sweeps(
        args.csv,
        output,
        labels=args.label,
        metric=args.metric,
        smooth_window=args.smooth_window,
        n_boot=args.n_boot,
        seed=args.seed,
        use_tex=args.usetex,
    )


if __name__ == "__main__":
    main()
