#!/usr/bin/env python3
"""Evaluate a saved policy against the ITAndroids strategy (``vss_fast``).

Runs ``--num-envs`` matches in parallel: physics and the blue policy are
batched in JAX, the yellow ITAndroids agents run in one process each.
Results (plus the simulator version and physics settings) are written as JSON
under ``results/itandroids/`` so they can be committed from the evaluation box.

    JAX_PLATFORMS=cuda python scripts/eval_vs_itandroids.py \
        --xp experiment26_brandao_reward --vss-root ~/itandroids-vss \
        --num-envs 16 --matches 64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xp", required=True, help="Experiment JSON path or name.")
    parser.add_argument("--lineage", type=int, default=0)
    parser.add_argument("--generation", type=int, help="Snapshot generation; default latest.")
    parser.add_argument("--matches", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel matches / agent processes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, help="Override match length (default: experiment's).")
    parser.add_argument("--vss-root", required=True, help="Path to the itandroids-vss checkout.")
    parser.add_argument("--vss-build", help="Directory holding vss_fast*.so (default: <vss-root>/fast/build).")
    parser.add_argument("--strategy", default="EXTREME_OFFENSIVE")
    parser.add_argument("--out-dir", default=str(REPO / "results" / "itandroids"))
    args = parser.parse_args()

    # Heavy imports after parsing so --help works without JAX.
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vsss_sim import config as sim_config
    from vsss_sim.agents.itandroids import ITAndroidsPool, vss_fast_factory
    from vsss_sim.envs.brandao_observation import OBSERVATION_SIZE as BRANDAO_OBS
    from vsss_sim.envs.jumanji import VSSJumanjiEnv
    from vsss_sim.rl.evaluate import _summary
    from vsss_sim.rl.ppo import create_policy_model, deterministic_policy_action
    from vsss_sim.rl.self_play import ExperimentConfig, SnapshotArchive
    xp = Path(args.xp)
    if not xp.is_file():
        xp = REPO / "configs" / "experiments" / xp.with_suffix(".json").name
    cfg = ExperimentConfig.from_json(xp)

    seconds = cfg.evaluation_seconds if args.seconds is None else args.seconds
    horizon = int(round(seconds * cfg.control_rate_hz))
    env = VSSJumanjiEnv(
        opponent_policy="stationary",
        max_episode_steps=horizon,
        init_mode=cfg.init_mode,
        terminate_on_goal=False,
        reward_profile=cfg.reward_profile,
        policy_architecture=cfg.policy_architecture,
        control_rate_hz=cfg.control_rate_hz,
        physics_substeps=cfg.physics_substeps,
        referee_mode=cfg.referee_mode,
    )
    model = create_policy_model(cfg.policy_architecture, cfg.hidden_sizes)
    obs_size = BRANDAO_OBS if cfg.policy_architecture in ("brandao_shared", "brandao_sets") else 46
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((obs_size,), dtype=jnp.float32))
    archive = SnapshotArchive(cfg.output_root, cfg.experiment_name, args.lineage)
    snapshots = archive.list_snapshots()
    if not snapshots:
        raise FileNotFoundError(f"no snapshots for lineage {args.lineage} under {cfg.output_root}")
    generation = int(snapshots[-1]["generation"]) if args.generation is None else args.generation
    params = archive.load(
        template, generation,
        expected_hidden_sizes=cfg.hidden_sizes,
        expected_policy_architecture=cfg.policy_architecture,
    )

    batch = args.num_envs
    reset = jax.jit(jax.vmap(env.reset))
    blue_policy = jax.jit(
        lambda obs: jax.vmap(
            lambda o: deterministic_policy_action(model, params, o, env.policy_architecture)
        )(obs)
    )
    step = jax.jit(jax.vmap(lambda s, a, y: env.step(s, a, opponent_action=y)))

    build = args.vss_build or str(Path(args.vss_root) / "fast" / "build")
    factory = vss_fast_factory(
        str(Path(args.vss_root).resolve()), str(Path(build).resolve()),
        strategy=args.strategy, dt=1.0 / cfg.control_rate_hz,
    )
    records: list[dict] = []
    started = time.time()
    rounds = -(-args.matches // batch)
    with ITAndroidsPool(batch, factory) as pool:
        for round_index in range(rounds):
            seeds = args.seed + round_index * batch + np.arange(batch, dtype=np.uint32)
            states, timesteps = reset(jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds)))
            observation = timesteps.observation
            pool.reset()
            previous_score = np.asarray(states.simulation.score)
            for _ in range(horizon):
                sim = states.simulation
                yellow = pool.step(np.asarray(sim.robots), np.asarray(sim.ball))
                states, timesteps = step(states, blue_policy(observation), jnp.asarray(yellow))
                observation = timesteps.observation
                score = np.asarray(states.simulation.score)
                scored = (score != previous_score).any(axis=-1)
                if scored.any():
                    pool.reset(scored)  # coaches are rebuilt for the new kickoff
                previous_score = score
            final = np.asarray(states.simulation.score, dtype=np.int32)
            for offset, row in enumerate(final):
                if len(records) < args.matches:
                    records.append({
                        "seed": int(seeds[offset]),
                        "blue_goals": int(row[sim_config.TEAM_BLUE]),
                        "yellow_goals": int(row[sim_config.TEAM_YELLOW]),
                    })
            print(f"round {round_index + 1}/{rounds} done ({time.time() - started:.0f}s)", flush=True)
        robot_params = pool.robot_params

    scores = np.array([[r["blue_goals"], r["yellow_goals"]] for r in records], dtype=np.int32)
    result = {
        "opponent": f"itandroids:{args.strategy}",
        "experiment_name": cfg.experiment_name,
        "lineage": args.lineage,
        "generation": generation,
        "match_seconds": seconds,
        "summary": _summary(scores),
        "matches": records,
        "provenance": {
            "vsss_sim_commit": _git("rev-parse", "HEAD"),
            "vsss_sim_dirty": bool(_git("status", "--porcelain")),
            "itandroids_commit": subprocess.run(
                ["git", "-C", args.vss_root, "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "robot_drive_model": sim_config.ROBOT_DRIVE_MODEL,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "agent_robot_params": robot_params,
            "num_envs": batch,
            "wall_seconds": time.time() - started,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{cfg.experiment_name}_g{generation}_seed{args.seed}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
