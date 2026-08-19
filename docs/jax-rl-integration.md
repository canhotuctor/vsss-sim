# JAX-native RL integration

Research checked on 2026-08-18 against the projects' official repositories and
documentation.

## Decision

Use a **Jumanji-compatible environment as the simulator boundary**, then build
a small project-owned PPO trainer following the **PureJaxRL execution pattern**.
This keeps the dependency set focused on JAX, Flax, and Optax.

`VSSJumanjiEnv` now provides a fully functional `reset(key)` / `step(state,
action)` interface. Environment state, observation construction, opponent
actions, reward shaping, goal kickoffs, scoring, and episode truncation all
remain in JAX. A trainer can therefore compile and vectorize the complete
rollout without the Gymnasium, NumPy, or PyTorch boundaries.

## Alternatives

| Option | Fit now | Strengths | Main drawback |
|---|---:|---|---|
| PureJaxRL-style local PPO | Best fit | Minimal JAX/Flax/Optax dependencies, fully compiled, easy to specialize for VSSS | The project owns correctness tests, checkpoints, and experiment tooling |
| [Stoix](https://github.com/EdanToledo/Stoix) | Best algorithm reference | Broad single-agent algorithm set, Anakin end-to-end JAX design, Jumanji-native | Explicitly not designed as an importable library; clone-and-adapt workflow and heavier Hydra configuration |
| [PureJaxRL](https://github.com/luchris429/purejaxrl) | Best execution reference | Small, understandable, fully compiled PPO | Not a library and intentionally single-file; adapt the pattern rather than add it as a dependency |
| [SBX](https://github.com/araffin/sbx) | Migration fallback | Familiar SB3 API with JAX policy updates and several continuous algorithms | Gymnasium environment boundary remains, so it does not unlock the raw JAX physics ceiling |
| [Mava](https://github.com/instadeepai/mava) | Future MARL path | End-to-end JAX IPPO/MAPPO and CTDE support, including continuous environments | Premature while the blue team is modeled as one centralized six-dimensional agent; also clone-oriented |
| [JaxMARL](https://github.com/FLAIROx/JaxMARL) | Future environment API/reference | Packaged environment API and standard IPPO/MAPPO baselines | Its published compatibility currently trails this repo's JAX version, and integration is more natural after exposing robots as separate agents |
| [Brax training](https://github.com/google/brax) | Rejected | Maintained and fast PPO/SAC implementations | The package pulls MuJoCo/MJX and a large simulation stack that this project does not need |

RLax is useful for loss and return primitives, but it is not an end-to-end
trainer. Acme is flexible, but its larger component/distributed stack and past
JAX dependency friction make it a weaker first connection here.

## Why Jumanji is the boundary

- It is a maintained and packaged JAX environment API.
- Its immutable state and `TimeStep` objects compose directly with `jit`,
  `vmap`, and `scan`.
- Stoix already treats Jumanji as its native environment API.
- The simulator remains independent of the trainer. A local PureJaxRL-style
  PPO, a Stoix experiment, or a future Mava multi-agent wrapper can share the same core
  environment semantics.

## Current usage

```bash
pip install -e ".[jax-rl]"
```

```python
import jax
import jax.numpy as jnp

from vsss_sim.envs.jumanji import VSSJumanjiEnv

env = VSSJumanjiEnv(opponent_policy="stationary", init_mode="random")
state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0))
state, timestep = jax.jit(env.step)(state, jnp.zeros(6, dtype=jnp.float32))

# Native batching, with no host copy:
keys = jax.random.split(jax.random.PRNGKey(1), 256)
states, timesteps = jax.jit(jax.vmap(env.reset))(keys)
```

The Flax/Optax PPO loop is implemented in `vsss_sim.rl.ppo`, with
`scripts/train_jax.py` as its training entry point. Stoix remains an
algorithm/reference comparison rather than the runtime training dependency.

```bash
python scripts/train_jax.py --total-timesteps 1000000 --num-envs 256
```

Each call to `PPO.update` compiles the rollout scan, truncation-aware GAE, and
all PPO minibatch epochs together. Python only coordinates updates and emits
periodic metrics; it is not part of the per-environment-step hot path.
