# Brandão (2022) reward profile

The JAX experiment runner accepts `reward_profile` in experiment JSON:

- `default` preserves the simulator's current reward (default).
- `brandao_2022` selects the reward structure from Section VI-B, equations
  (6)-(16), of Brandão et al., *Multiagent Reinforcement Learning for Strategic
  Decision Making and Control in Robotic Soccer Through Self-Play* (IEEE Access,
  2022).

Run the supplied experimental configuration with:

```bash
python scripts/train_jax.py --xp brandao_2022_reward
```

The continuous reward, from blue's perspective, is

```text
r = 0.7 * r_speed + 0.1 * r_dist + 0.1 * mean(r_off) + 0.1 * mean(r_def)
```

`r_speed` measures reduction in Euclidean distance from the ball to the
opponent goal, subtracts 0.05 m, divides by 0.14 m, and clips to [-1, 1].
`r_dist` is the negative distance from the ball to the closest blue robot,
capped at 1 m. `r_off` and `r_def` are the paper's angle-at-ball and
angle-at-robot terms (`angle / pi - 1`), averaged over the three blue robots.
The averages give one shared team reward in this simulator's single-agent
JAX environment; the paper returns each agent's specific positioning terms.

On a goal, this profile replaces the continuous reward with
`10 * goal * (1 + remaining_steps / max_episode_steps)`, where blue goals are
`+1` and yellow goals are `-1`. The simulator currently has no foul detection,
so the paper's `-1` foul reward is not emitted. Goals still start an in-match
kickoff, matching the simulator's existing episode semantics.

The supplied JSON is a separate experiment configuration and output name. To
compare rewards, use the same training settings and seeds, changing only
`reward_profile` and `experiment_name` in a copy of the JSON.
