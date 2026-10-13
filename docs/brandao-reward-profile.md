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

The paper gives each robot separate positioning rewards, while this simulator
trains the blue team with one shared scalar reward. The offensive and defensive
angle terms are therefore averaged over the three blue robots:

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
