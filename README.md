# vsss-sim

IEEE Very Small Size Soccer (VSSS) 3v3 simulation for reinforcement learning.
This project implements VSSS, not RoboCup Small Size League (SSL).

## What it provides

- JAX-based physics for CPU and CUDA devices.
- Single-match and batched Gymnasium environments.
- A Jumanji environment and native JAX PPO trainer.
- Optional Pygame rendering.

## Install

Python 3.10 or newer is required.

    python -m pip install -e .
    python -m pip install -e '.[render]'   # optional renderer
    python -m pip install -e '.[jax-rl]'   # native JAX training

## Gymnasium quick start

    import gymnasium as gym
    import vsss_sim  # registers VSSS-v0

    env = gym.make("VSSS-v0")
    observation, info = env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    env.close()

The action contains six normalized wheel speeds for the blue team. The yellow
team uses the configured opponent policy.

## Train with native JAX PPO

    python scripts/train_jax.py --generations 30 --num-envs 256 --rollout-length 128

To run a saved experiment configuration:

    python scripts/train_jax.py --xp experiment26

The trainer compiles policy inference, environment steps, rollout collection,
and PPO updates into JAX programs. Startup compilation and execution time are
reported separately.

## Documentation

- [Structure](STRUCTURE.md): environment and training paths, transitions,
  policy data shapes, and current timing defaults.
- [Reward profiles](docs/brandao-reward-profile.md): current reward equations
  and how they are assigned.
- [CUDA setup](docs/cuda-setup.md): install and verify JAX GPU support.
- [Glossary](docs/glossary.md): reinforcement-learning terms used here.
