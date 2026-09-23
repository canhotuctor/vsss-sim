# Historical Self-Play — Implementation Plan

> Status: implementation completed locally; GPU-scale throughput and memory validation remains for the user's target environment.

## Goal

Add historical self-play to the existing JAX PPO path for IEEE VSSS 3×3. Each current policy trains against frozen historical policies from its own independent lineage. The implementation must support one lineage by default and scale to a 16-lineage experiment without sharing learner state or historical pools.

## Decisions already made

- `num_lineages` is configurable; default 1, initial experiment 16. Environments per lineage are configurable and should be balanced against lineage count using measured device memory and throughput.
- Each lineage has its own PPO learner, seed, and historical pool. No cross-lineage opponent sampling.
- The historical pool begins with the initialized policy snapshot at generation 0, then receives a new snapshot every 10 generations. A generation is one PPO update.
- Snapshots are archived by experiment and lineage. All snapshots are eligible for the initial active pool; a configurable active-pool limit may be tested later. Measure its training-time overhead before choosing a restrictive default.
- Training opponent sources total 100% historical. Sampling-category weights are configurable. Provisional defaults: 66.7% uniform over all history, 22.2% uniform over the recent half, 11.1% latest snapshot, 0% random/stationary. These are starting values for later ablation, not a research conclusion.
- The same centralized 3×3 policy controls either side. Yellow observations/actions use the established team-perspective transformation; this is a policy-interface concern kept independent of the training design.
- `scripts/train_jax.py` remains the generic thin entry point and loads one JSON experiment config (e.g. `--xp experimento26`). Evaluation is a separate script.
- Final evaluation uses 120-second matches (1,800 control steps at 15 Hz), compares against the lineage’s historical pool and compatible fixed references when available, and reports W/D/L plus goal difference. Exact evaluation sample count is open.
- Do not save optimizer/RNG/environment state to resume training in this phase. A policy snapshot contains inference parameters only. Do not incorporate the sampler from the user’s separate partial-work design.
- Do not depend on unavailable VSSS-INF or VSSS-EMC policies.

## Current code and likely seams

- `src/vsss_sim/rl/ppo.py`: `PPO`, `PPOConfig`, `RunnerState`; a runner state already holds optimizer, environment, observations, and PRNG key. `PPO.update` performs rollout plus update in JIT; `PPO.train` compiles a fixed number of updates with `lax.scan`.
- `src/vsss_sim/envs/jumanji.py`: `VSSJumanjiEnv` currently embeds stationary/random opponent action logic and accepts only those two JAX-native opponent choices. Blue is controlled; Yellow is computed internally.
- `scripts/train_jax.py`: currently parses direct CLI hyperparameters, builds one environment/trainer, then trains a single learner. It has no per-experiment JSON orchestration.
- Existing relevant tests: `tests/rl/test_ppo.py`, `tests/envs/test_jumanji_env.py`; add tests at these existing seams where possible.
- Preserve pre-existing user changes in `scripts/train_jax.py`, symmetry modules/tests, `AGENTS.md`, README/docs, and `output/`.

## Proposed architecture

Keep one deep training interface around a lineage update: the learner receives a sampled frozen opponent policy for each environment episode, while the experiment orchestrator owns lineage creation, snapshot lifecycle, weighted selection, generation cadence, and logging. The trainer remains responsible for PPO math and JAX compilation; the environment remains responsible for applying actions and perspective conversion.

The selected opponent should be fixed for one match/episode and vary across parallel environments and subsequent resets. This is the recommended starting contract because it gives each match a coherent opponent and spreads a rollout across the configured historical distribution. Confirm this behavior before implementing the opponent-pool seam; if the user prefers a different sampling unit, adjust the plan first.

For lineage parallelism, first prove the single-lineage path, then benchmark independent learner states represented on a JAX lineage axis against independently launched workers. Choose based on measured throughput, compilation, memory, and Mac/GPU compatibility. Whichever representation is selected, parameters, PRNG streams, environment state, and pools must remain independent per lineage.

## Implementation phases

### Phase 1 — Pin down contracts and single-lineage opponent seam

- Define the opponent inference contract: parameters are frozen during a match; the controlled and opposing team use the same policy module, with the opposing observation transformed into canonical team perspective and its action transformed back.
- Add a JAX-native path for the environment/trainer to consume opponent parameters without adding Python callbacks to the compiled rollout.
- Represent or pass the selected opponent identity so it remains stable across a match and changes only when a new match begins.
- Preserve stationary/random support for existing users and tests; it must not become a training source in the new experiment’s 100%-historical config.
- Tests: perspective round-trip; same policy can act for both sides; frozen opponent parameters do not change during an update; a new episode can select a different opponent; existing stationary/random behavior remains unchanged.

**Exit gate:** one PPO generation can run against a frozen policy snapshot end to end on JAX, with no Python work in the per-step compiled loop.

### Phase 2 — Snapshot archive, active pool, and selectors

- Add an experiment/lineage-scoped archive format for policy parameters and small metadata: schema version, experiment, lineage, generation, seed/architecture compatibility fields.
- Save generation 0 and every 10th generation. Load and validate snapshots without loading optimizer/RNG state.
- Implement selectors for all history, recent half, and latest snapshot; combine them using validated configurable weights. For the initial experiment require historical source weights to sum to 100%; reject invalid/negative weights and unavailable history states with actionable errors.
- Make the active pool and its optional maximum configurable. Start the new experiment with every archived snapshot eligible; ensure storage is not destructively pruned when the active subset is smaller.
- Tests: save/load parameter parity; lineage and experiment isolation; cadence including generation 0 and 10; deterministic selector behavior under controlled keys; weight validation and small-pool edge cases.

**Exit gate:** a lineage can persist and reload its pool, select each configured category correctly, and cannot see another lineage’s snapshots.

### Phase 3 — One-lineage historical PPO loop

- Integrate pool selection with vectorized environments, sampling one opponent for each episode and keeping it fixed for that episode.
- Run training in 10-generation chunks so the process can record metrics and save a snapshot at the agreed cadence; avoid a single opaque compile/run spanning the entire experiment.
- Add training metrics that identify generation, lineage, sampled selector/category, opponent snapshot generation, episode return/length, goals for/against, and elapsed time where available.
- Keep policy updates exclusive to the current learner; opponents are inference-only snapshots.
- Tests: learner changes after updates while archive snapshots remain immutable; per-episode opponents remain stable; sampling diversity appears across environments with a fixed test key; save/reload boundary does not alter subsequent policy behavior.

**Exit gate:** a short CPU smoke experiment initializes gen-0 history, trains through gen 10, saves and reloads snapshots, and shows opponent IDs from only its own pool.

### Phase 4 — Experiment configuration and independent lineages

- Move configuration and orchestration out of `scripts/train_jax.py`; keep the script as argument parsing + call into the training module.
- Define a JSON config schema for experiment identity, PPO settings, `num_lineages`, environments per lineage, per-lineage seed derivation, generation count, snapshot cadence, pool settings, opponent-selector weights, and output paths.
- Validate config before device allocation: unique experiment directory, legal weights, valid counts, snapshot cadence, and per-lineage seed determinism.
- Make one-lineage the default and provide the initial 16-lineage experiment JSON.
- Add a test/benchmark harness comparing separate JAX learner states batched on a lineage axis versus independent workers. Pick the implementation only after measuring; do not serialize/merge learner state across lineages.
- Test independence behaviorally: distinct seeds initialize distinct states, updating one lineage cannot alter another lineage’s learner or pool, and each lineage’s snapshots remain isolated.

**Exit gate:** the 16-lineage configuration starts with 16 independent generation-0 snapshots and all outputs land under the correct experiment/lineage scopes.

### Phase 5 — Separate 120-second evaluation

- Add a dedicated evaluation module and CLI, separate from training. Load a selected current policy and its historical pool from the experiment archive.
- Support configurable number of 120-second matches (1,800 control steps at 15 Hz), evaluation seed, and opponent selection. Do not assume fixed references exist; discover/validate compatible previous checkpoints and report missing/incompatible references clearly.
- Evaluate each trained lineage against its own historical distribution; additionally evaluate against compatible existing fixed references when available. Random/stationary may be evaluation references only if configured, never silently added to training.
- Record per-policy/per-opponent and aggregate W/D/L, win rate, goals for/against, and goal difference. Include seeds/config/snapshot identities in machine-readable output.
- Tests: episode horizon exactly 1,800 control steps, no training-state mutation, reproducibility for fixed policy/environment seeds, correct score aggregation, incompatible checkpoint handling.

**Exit gate:** evaluation can be run independently from a saved experiment and returns auditable metrics for all 16 lineages.

### Phase 6 — Throughput, memory, and first experiment

- Benchmark increasing `num_lineages`, environments per lineage, and active historical-pool sizes; separately track compile time, training throughput, host↔device transfer, and peak device memory.
- Compare all-snapshots active pool with configurable capped pools to quantify overhead; retain all snapshots on disk in both cases.
- Confirm cross-platform behavior: CPU/Mac development path and Ubuntu RTX 3060 GPU target. Do not make CUDA-only assumptions in the interface.
- Run a short end-to-end experiment before the full run; then execute the 16-lineage JSON config and final evaluation.
- Save resolved config and metrics beside the experiment artifacts so results can be compared with current PPO baselines and cited Brandão protocol without claiming exact reproduction.

## Testing strategy

- Start at the highest stable seams: test the existing PPO/environment interface, snapshot archive/selector public behavior, then experiment runner and evaluation CLI. Avoid asserting private JIT implementation details.
- Required checks: focused RL/environment tests; config and archive tests; a short single-lineage smoke run through generation 10; a small multi-lineage independence run; evaluation smoke with shortened test horizon (while production config remains 1,800 steps); full project tests.
- Keep performance checks separate from correctness tests. Benchmarks should report observed hardware/device, lineages, environments per lineage, pool size, compile time, steady-state env steps/sec, and peak memory.

## Open decisions to settle during implementation

- Confirm the recommended per-episode opponent sampling unit before Phase 1; prior decisions specify a weighted historical distribution but not when an opponent is resampled.
- Exact evaluation match count and whether to evaluate every checkpoint or only periodically plus final.
- Which old policies/checkpoints are compatible evaluation references.
- Whether JAX-axis lineages or independent worker processes is the practical default after benchmark.
- Final JSON/config and artifact directory naming conventions.
- Whether episode duration/reward semantics should remain at the simulator’s current defaults for the initial training experiment. The 120-second evaluation duration is already decided.

## Out of scope

- Implementing the separate sampler from the user’s partial work.
- VSSS-INF/VSSS-EMC opponents.
- Resumable full-training checkpoints containing optimizer, RNG, environment, and sampler state.
- Changing the centralized policy into per-robot policies.
- Changing reward function, initialization policy, or default training episode horizon without an explicit follow-up decision.
