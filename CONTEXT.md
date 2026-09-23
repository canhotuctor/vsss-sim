# VSSS Reinforcement-Learning Context

Glossary for training and evaluation concepts specific to reinforcement learning in this simulator.

## Training and self-play

**Generation**:
One PPO update: collect one rollout and apply the configured PPO epochs/minibatches once.
_Avoid_: using “generation” for an episode, a snapshot interval, or a full experiment run.

**Lineage**:
An independent training run with its own current policy, seed, environments, and historical policy pool.
_Avoid_: batch, when it implies several lineages update one shared policy.

**Policy snapshot**:
An inference-ready copy of one lineage’s policy parameters at a recorded generation, used as a frozen opponent. It does not contain the optimizer, environment, or RNG state needed to resume training.
_Avoid_: checkpoint, unless referring to a complete resumable training state.

**Historical pool**:
The lineage-scoped collection of that lineage’s policy snapshots available for opponent selection.
_Avoid_: global pool, unless intentionally discussing a pool shared across experiments (not the current design).

**Opponent selector**:
A configured rule that chooses a historical snapshot (for example, uniformly from all history, from the recent half, or the latest snapshot).
_Avoid_: opponent policy, when referring specifically to the rule rather than the selected policy.
