"""Benchmark a compiled PPO generation, including policy inference and updates."""

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from vsss_sim.rl.ppo import PPO
from vsss_sim.rl.self_play import ExperimentConfig, _make_env

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=200)
parser.add_argument("--rollout", type=int, default=80)
parser.add_argument("--substeps", type=int, default=6)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
base = ExperimentConfig.from_json(
    Path("configs/experiments/experiment26_brandao_shared_online_curriculum_gpu.json")
)
cfg = replace(
    base,
    num_envs_per_lineage=args.envs,
    rollout_length=args.rollout,
    physics_substeps=args.substeps,
    require_gpu=False,
    seed=base.seed if args.seed is None else args.seed,
)
print(
    f"jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()} envs={cfg.num_envs_per_lineage} rollout={cfg.rollout_length} epochs={cfg.update_epochs} minibatches={cfg.num_minibatches} hidden={cfg.hidden_sizes} replay={cfg.curriculum_replay_fraction} substeps={args.substeps} seed={cfg.seed}",
    flush=True,
)
trainer = PPO(_make_env(cfg), cfg.ppo_config())
keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.num_lineages)
initial = jax.jit(jax.vmap(trainer._initialize))(keys)
jax.block_until_ready(initial)
template = jax.tree_util.tree_map(lambda value: value[0], initial.train_state.params)
pools = jax.tree_util.tree_map(lambda value: value[None, None, ...], template)
probabilities = jnp.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
generations = jnp.zeros((1, 1), dtype=jnp.int32)
runners = trainer.initialize_lineages(keys, pools, probabilities, generations)
jax.block_until_ready(runners)
started = time.perf_counter()
executable = trainer.compile_train_lineages(
    runners, pools, probabilities, generations, num_updates=1
)
compile_s = time.perf_counter() - started
runners, metrics = executable(runners, pools, probabilities, generations)
jax.block_until_ready((runners, metrics))
samples = []
for _ in range(args.repeats):
    started = time.perf_counter()
    runners, metrics = executable(runners, pools, probabilities, generations)
    jax.block_until_ready((runners, metrics))
    samples.append(time.perf_counter() - started)
median = statistics.median(samples)
print(
    f"full_generation compile_s={compile_s:.3f} median_s={median:.5f} env_steps_s={cfg.num_envs_per_lineage * cfg.rollout_length / median:.0f} samples_s={samples}",
    flush=True,
)
