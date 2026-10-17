# Glossary

**Generation** — One PPO update: collect a rollout and apply the configured epochs and minibatches.

**Lineage** — An independent training run with its own policy, seed, environments, and historical snapshot pool.

**Policy snapshot** — Frozen policy parameters saved for inference as an opponent. It does not contain the optimizer or runtime state required to resume training.

**Historical pool** — The snapshots available to one lineage for opponent selection.

**Policy definition** — A model and its observation and action contracts.

**Observation layout** — The ordered meaning of the fields in a policy observation. Matching vector lengths alone do not establish compatibility.

**Reward profile** — The formula used to calculate a reward.

**Reward scope** — Whether the reward is one team value or a value for each robot.

**Experiment recipe** — The training settings for one run, including reward, horizon, opponent selection, and simulation timing.

**Command cadence** — How often the policy supplies commands during an experiment.

**Physics substeps** — The integration steps taken within one command interval.
