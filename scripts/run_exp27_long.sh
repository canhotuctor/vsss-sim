#!/usr/bin/env bash
# Launch the single-lineage, 10000-generation exp27 variant (simplified referee,
# full sampled symmetry augmentation) detached from the shell, so a dropped
# connection does not kill it.
#
#   scripts/run_exp27_long.sh                 # experiment27_long_symfull (promoted opponent)
#   scripts/run_exp27_long.sh experiment27_long_randhist   # random previous-snapshot opponent
#   tail -f outputs/experiments/<name>.log
#
# Needs the project venv (activate it first) and a CUDA GPU (config has require_gpu).
# Output goes to outputs/experiments/<name>/; the run refuses to start
# if that folder already exists. Rename it (or change experiment_name) to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-experiment27_long_symfull}"
out=outputs/experiments
[ -d "$out/$name" ] && { echo "$out/$name already exists; move it first." >&2; exit 1; }
mkdir -p "$out"
JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  nohup python scripts/train_jax.py --xp "$name" > "$out/$name.log" 2>&1 &
echo "started pid $! ; log: $out/$name.log"
echo "check: ps aux | grep train_jax | grep -v grep ; wc -l $out/$name/performance.jsonl"
