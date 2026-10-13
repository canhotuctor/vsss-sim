# Brandão reward profile

The experiment-level `reward_profile` selects one team reward rule:

- `default` preserves the existing goal, ball-progress, and goal-area crowding
  reward.
- `brandao_reward` implements the reward structure in Section VI-B,
  equations (6)-(16), of Brandão et al., *Multiagent Reinforcement Learning
  for Strategic Decision Making and Control in Robotic Soccer Through
  Self-Play* (IEEE Access, 2022).

Select the profile from a training command:

```bash
python scripts/train_jax.py --generations 30 --reward-profile brandao_reward
python scripts/train_jax.py --xp experiment26 --reward-profile brandao_reward
python scripts/train_jax.py --xp experiment26_brandao_reward
```

The repository includes `configs/experiments/experiment26_brandao_reward.json`
as a long-run historical self-play configuration. It uses a distinct
`experiment_name` so its snapshots and evaluation records do not overlap the
default-profile run. The selected
profile is saved in the resolved experiment configuration and in evaluation
results. `ExperimentConfig` defaults to `default` when older JSON files omit
the field.

## Team reward adaptation

With the existing `joint` architecture, the blue team receives one scalar
reward. The paper's separate positioning terms are therefore averaged over the
three blue robots in that mode:

```text
r = 0.7 * r_speed + 0.1 * r_dist
  + 0.1 * mean(r_off) + 0.1 * mean(r_def)
```

`r_speed` measures reduction in Euclidean distance from the ball to the
opponent goal, subtracts 0.05 m, divides by 0.14 m, and clips to `[-1, 1]`.
`r_dist` is the negative distance from the ball to the nearest blue robot,
capped at 1 m. Both angle terms use `angle / pi - 1`.

On a goal, the continuous reward is replaced by
`10 * goal * (1 + remaining_steps / max_episode_steps)`, where blue goals are
`+1` and yellow goals are `-1`. Reward is computed from the post-physics state
before kickoff, so the goal transition does not replace the positions used by
the reward. The next observation is built from the state carried into the next
step, including a post-goal kickoff when that environment uses in-match goals.

The simulator does not yet detect fouls, so this profile cannot emit the
paper's foul penalty. A future referee event can be added to the shared match
step facts and consumed by the reward profile when foul detection exists.

## Shared per-robot policy

`policy_architecture: "brandao_shared"` is available alongside the existing
`"joint"` policy. One actor-critic is shared across the three robots; it is
called once with each robot's local observation and returns that robot's two
wheel commands. Each robot gets its own offensive and defensive angle terms,
while the ball-progress, nearest-robot proximity, and goal terms are shared.
The action is held for exactly 0.1 seconds per decision.

The local observation follows the paper's listed inputs: normalized positions
of all robots and the ball; sine, cosine, and normalized arctangent for the
agent heading and seven specified bearings; eight distances; the team's six
previous wheel commands; and remaining episode time. It stacks eight readings,
each formed at 10 Hz, and applies the paper's stated position and heading
measurement noise. The actor uses 300/200/100 tanh layers and Beta wheel-speed
outputs; the critic uses 256/128/64 tanh layers. Training episodes end at the
first goal or after 40 seconds. At each snapshot checkpoint, a frozen opponent
snapshot is replaced when the current policy's goal difference per completed
game reaches 0.6.

The paper-scale configuration is
`configs/experiments/experiment26_brandao_shared_160m.json`. It uses 200
environments × 214 policy decisions × 3 robot transitions per update, and 1,246
updates, for **159,986,400 agent transitions** (just under the paper's 160
million). Train it and evaluate the final snapshot separately:

```bash
python scripts/train_jax.py --xp experiment26_brandao_shared_160m
python scripts/evaluate_jax.py --xp experiment26_brandao_shared_160m --lineage 0 --opponents all --matches 100
```

This is a close implementation of the paper's policy, reward, transition
counting, update size, and self-play opponent rule. It is not a simulator-level
reproduction: this repository's JAX physics, field/referee rules, actuator
model, and available evaluation opponents differ from the paper's MuJoCo and
Pequi Mecânico setup. The simulator also has no foul event, so the paper's
foul penalty is unavailable.
