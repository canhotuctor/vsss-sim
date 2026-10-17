# Reward profiles

The default profile uses the blue team's goal result, ball progress, and goal-area crowding:

    reward = goal + 0.10 × change_in_ball_x + crowding_penalty

The crowding penalty is −0.02 when two or more blue robots occupy the same goal area.

The brandao_reward profile implements the reward terms from Section VI-B, equations (6)–(16), of Brandão et al., “Multiagent Reinforcement Learning for Strategic Decision Making and Control in Robotic Soccer Through Self-Play” (IEEE Access, 2022).

## Reward assignment

The joint policy receives one scalar team reward. Its Brandão reward averages the offensive and defensive angle terms across the three blue robots:

    team_reward = 0.7 × speed + 0.1 × proximity
                  + 0.1 × mean(offensive_angle)
                  + 0.1 × mean(defensive_angle)

The brandao_shared and brandao_sets policies receive one reward per robot. Each gets the shared speed and proximity terms plus its own offensive and defensive angle terms:

    robot_reward[i] = 0.7 × speed + 0.1 × proximity
                      + 0.1 × offensive_angle[i]
                      + 0.1 × defensive_angle[i]

The speed term is the clipped reduction in ball-to-opponent-goal distance, with a 0.05 m dead zone and 0.14 m scale. Proximity is the negative distance to the nearest blue robot, capped at 1 m. Each angle term is angle / π − 1.

On a goal, the continuous Brandão reward is replaced by:

    goal_reward = 10 × goal × (1 + remaining_steps / max_episode_steps)

The goal value is +1 for blue and −1 for yellow. In per-robot mode this same goal value is assigned to each robot. Reward uses the post-physics state before any kickoff; the next observation uses the state carried into the next step.

## Configuration

The training CLI accepts a reward profile:

    python scripts/train_jax.py --reward-profile brandao_reward
    python scripts/train_jax.py --xp experiment26 --reward-profile brandao_reward

Experiment configurations store the resolved profile. Older configurations without the field default to default for joint policies and brandao_reward for Brandão per-robot policies. The direct Jumanji environment requires brandao_reward when using either Brandão policy.

The simulator does not detect fouls, so the paper's foul penalty is not implemented.
