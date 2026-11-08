#!/usr/bin/env python3
"""Plot performance against ITAndroids (60 Hz) versus training generation, all lineages.

Reads per-snapshot results (``eval_vs_itandroids.py``) from --snapshots-dir plus the
final-policy results, and draws two panels sharing the generation axis:
  A. mean goal difference per match (thin line per lineage, bold bin mean +/- s.e.m.)
  B. goals scored and conceded per match (bin means)

    python scripts/plot_itandroids_evolution.py --out partial-results/figures/itandroids_evolution.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE = "#2563c9", "#c2570c"


def load(paths: list[Path]) -> dict[str, dict[int, dict]]:
    """lineage -> generation -> summary (later files win)."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("agent_hz", 60.0) != 60.0 and "agent60hz" not in path.name and "hz60" not in path.name:
            continue
        out[data["experiment_name"]][int(data["generation"])] = data["summary"]
    return out


def binned(points: list[tuple[int, float]], width: int):
    """Bin (generation, value) pairs; generation 0 gets its own bin. Returns centers, mean, sem, n."""
    bins: dict[int, list[float]] = defaultdict(list)
    for generation, value in points:
        bins[0 if generation == 0 else 1 + (generation - 1) // width].append(value)
    keys = sorted(bins)
    centers = [0 if k == 0 else (k - 1) * width + width / 2 for k in keys]
    mean = [float(np.mean(bins[k])) for k in keys]
    sem = [float(np.std(bins[k], ddof=1) / np.sqrt(len(bins[k]))) if len(bins[k]) > 1 else 0.0 for k in keys]
    return np.array(centers), np.array(mean), np.array(sem), [len(bins[k]) for k in keys]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", default="partial-results/itandroids-snapshots")
    parser.add_argument("--final-glob", nargs="*", default=[
        "partial-results/itandroids/*agent60hz*.json",
        "partial-results/itandroids-diagnostic/*_hz60_fix0.json",
    ])
    parser.add_argument("--bin-width", type=int, default=500)
    parser.add_argument("--out", default="partial-results/figures/itandroids_evolution.png")
    args = parser.parse_args()

    paths = sorted(Path(args.snapshots_dir).glob("*agent60hz*.json"))
    for pattern in args.final_glob:
        paths += sorted(Path().glob(pattern))
    results = load(paths)
    # Diagnostic files are named l00/l01..., not per-experiment; they carry experiment_name already.
    if not results:
        raise SystemExit("no results found")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    ax = axes[0]
    all_gd: list[tuple[int, float]] = []
    all_gf: list[tuple[int, float]] = []
    all_ga: list[tuple[int, float]] = []
    for lineage, by_gen in sorted(results.items()):
        gens = sorted(by_gen)
        gd = [by_gen[g]["mean_goal_difference"] for g in gens]
        ax.plot(gens, gd, color=MUTED, alpha=0.35, linewidth=1)
        for g in gens:
            s = by_gen[g]
            all_gd.append((g, s["mean_goal_difference"]))
            all_gf.append((g, s["goals_for"] / s["matches"]))
            all_ga.append((g, s["goals_against"] / s["matches"]))
    x, m, e, n = binned(all_gd, args.bin_width)
    ax.fill_between(x, m - e, m + e, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, m, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("mean goal difference per match")
    ax.set_title(f"Policy vs ITAndroids (60 Hz): {len(results)} lineages, "
                 f"{sum(len(v) for v in results.values())} evaluated snapshots", loc="left", color=INK)
    ax.text(0.99, 0.04, "grey: one lineage   blue: bin mean ± s.e.m. (bin 500 gens; gen 0 alone)",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)

    ax = axes[1]
    for points, color, label in ((all_gf, BLUE, "goals scored"), (all_ga, ORANGE, "goals conceded")):
        x, m, e, _ = binned(points, args.bin_width)
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                markeredgewidth=1.5, label=label)
        ax.annotate(label, (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                    fontsize=9, va="center")
    ax.set_ylabel("goals per match")
    ax.set_xlabel("training generation of the evaluated snapshot")
    ax.legend(frameon=False, loc="upper left")

    for a in axes:
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.spines["left"].set_color(GRID)
        a.spines["bottom"].set_color(MUTED)
        a.tick_params(colors=MUTED)
        a.set_xlim(-100, max(g for v in results.values() for g in v) + 500)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    for c, mm, ee, nn in zip(*binned(all_gd, args.bin_width)[0:2], binned(all_gd, args.bin_width)[2], binned(all_gd, args.bin_width)[3]):
        print(f"gen~{c:>6.0f}: GD {mm:6.2f} ± {ee:4.2f} (n={nn})")


if __name__ == "__main__":
    main()
