# vsss-sim

IEEE VSSS 3×3 simulator for Reinforcement Learning.

Implements the **Very Small Size Soccer** (VSSS) specification — 3 robots per
team, 150 × 130 cm field, golf ball — with a standard
[Gymnasium](https://gymnasium.farama.org/) interface and a switchable physics
engine (NumPy or JAX) designed for GPU-accelerated batched training.

> **VSSS, not SSL.** This project targets IEEE Very Small Size Soccer, not
> RoboCup Small Size League. Different field, different drivetrain, different
> action space. See [`CLAUDE.md`](CLAUDE.md) for the disambiguation.

---

## Features

| Feature | Details |
|---|---|
| **IEEE VSSS compliant** | Field 150×130 cm, goal 40 cm wide, 7.5 cm box robots, golf ball |
| **Gymnasium interface** | `step` / `reset` / `render` — drop into any RL framework |
| **Differential-drive** | Correct 2-D kinematics, OBB collisions, rolling friction |
| **Switchable physics backends** | `numpy_backend.py` (CPU, default) or `jax_backend.py` (CPU/GPU/Metal) |
| **Native batched Vector env** | `VSSVecEnv` runs N parallel matches in one `jit(vmap(step))` call |
| **Pluggable opponents** | `"stationary"`, `"random"`, or any callable policy |
| **Pygame renderer** | `"human"` window or headless `"rgb_array"` frame output |
| **Cross-platform** | macOS (Apple Silicon) · Ubuntu 22.04 with CUDA-capable GPU |

---

## Installation

```bash
# Core (no rendering)
pip install -e .

# With Pygame renderer
pip install -e ".[render]"

# With the JAX physics backend
pip install -e ".[jax]"

# Developer tools (tests + linter)
pip install -e ".[dev]"
```

Python ≥ 3.10 is required.

---

## Quick start

### Single env (default: NumPy backend)

```python
import gymnasium as gym
import vsss_sim          # registers "VSSS-v0"

env = gym.make("VSSS-v0", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(500):
    action = env.action_space.sample()   # 6 wheel speeds in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if truncated:
        obs, info = env.reset()

env.close()
```

### Single env on the JAX backend

```python
env = gym.make("VSSS-v0", backend="jax")
# Or globally: VSSS_PHYSICS_BACKEND=jax python my_script.py
```

### Batched vector env (JAX, ~30× CPU speedup at `num_envs=256`)

```python
envs = gym.make_vec(
    "VSSS-v0", num_envs=256, vectorization_mode="vector_entry_point",
)
obs, info = envs.reset(seed=0)            # obs shape: (256, 46) float32
obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
```

Or directly: `from vsss_sim.envs import VSSVecEnv; envs = VSSVecEnv(num_envs=256)`.

---

## Environment details

### Observation space — `Box(-5.0, +5.0, (46,), float32)`

All components are pre-normalised; the bound is a generous safety margin.

| Slice | Contents |
|---|---|
| `[0:4]` | Ball: `x/norm, y/norm, vx/norm, vy/norm` |
| `[4:25]` | Blue robots 0-2: `x/norm, y/norm, sin θ, cos θ, vx/norm, vy/norm, ω/norm` each |
| `[25:46]` | Yellow robots 0-2: same layout |

### Action space — `Box(-1, 1, (6,), float32)`

```
[ v_left_0, v_right_0,   v_left_1, v_right_1,   v_left_2, v_right_2 ]
```

Normalised wheel speeds for the **blue** (controlled) team.

### Reward

| Event | Reward |
|---|---|
| Blue scores | `+1.0` |
| Yellow scores | `−1.0` |
| Otherwise | `0.0` |

Goals trigger an in-episode kickoff; episodes truncate at `MAX_EPISODE_STEPS`
(default 1200 ≈ 20 s of sim at 60 Hz).

---

## Field layout

```
         ← 150 cm →
    ┌──────────────────────┐   ↑
 ║  │                      │  ║   130
 ║  │        ●             │  ║    cm
 ║  │                      │  ║   ↓
    └──────────────────────┘
  Blue goal              Yellow goal
  (−x end-line)          (+x end-line)
```

---

## Physics backends

The package ships with two backends, both implementing the same interface:

| Backend | File | Use when |
|---|---|---|
| **NumPy** (default) | `physics/numpy_backend.py` | Dev, debugging, visual smoke tests. Single env, CPU. |
| **JAX** | `physics/jax_backend.py` | Batched training, GPU/CUDA, Apple Metal (Metal currently blocked by jax-metal compatibility — CPU works). |

Selection priority: `backend=` kwarg on `VSSEnv` / `VSSVecEnv` &gt; `VSSS_PHYSICS_BACKEND` env var &gt; `"numpy"`.

The JAX backend is a pure-functional mirror of the NumPy backend: `SimState`
is a `NamedTuple` PyTree, `step()` is `jax.jit`-compiled with
`lax.fori_loop` substeps, and the function is `vmap`-ready (this is what
`VSSVecEnv` uses to batch). Trajectories match the NumPy backend within
`atol=1e-3` on positions over 20-step rollouts.

### Throughput (Apple M4 Pro CPU, 12 cores)

| Setup | Throughput | Speedup vs NumPy single-env |
|---|---|---|
| NumPy, raw physics single-env | ~3,500 fps | 1× |
| JAX (jit), raw physics single-env | ~32,000 fps | 9× |
| JAX `jax.vmap`, batch=256, raw physics | ~300,000 fps | ~88× (CPU ceiling) |
| `VSSVecEnv` (Gymnasium wrapper), batch=256 | ~120,000 fps | ~34× |
| `VSSVecEnv`, batch=1024 | ~155,000 fps | ~44× |

Reproduce with `python scripts/bench_backends.py`. The gap between raw vmap
and `VSSVecEnv` is the Gymnasium numpy↔JAX boundary cost (per-step Python
overhead, obs construction, opponent policy dispatch). On GPU this becomes
negligible.

---

## Scripts

- `scripts/smoke.py` — SB3 PPO smoke (single env). Flags: `--backend numpy|jax`, `--render`, `--fps`, `--timesteps`.
- `scripts/bench_backends.py` — throughput benchmark across all backends and batch sizes.
- `scripts/train.py` — MLflow-tracked PPO training run.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest                          # 137/137 expected
```

---

## Project structure

```
src/vsss_sim/
├── __init__.py            Gymnasium registration (VSSS-v0 single + vector entry points)
├── config.py              IEEE VSSS constants
├── agents/                stationary.py, random.py (pluggable opponents)
├── envs/
│   ├── base.py            VSSBaseEnv (spaces, observation builder)
│   ├── vsss_3v3.py        VSSEnv — single-env Gymnasium Env
│   └── vsss_vec.py        VSSVecEnv — batched JAX Gymnasium VectorEnv
├── physics/
│   ├── __init__.py        Backend resolver (get_backend)
│   ├── numpy_backend.py   Reference CPU backend (mutable SimState)
│   └── jax_backend.py     JAX backend (functional, jittable, vmap-ready)
└── rendering/pygame.py
tests/                     137 tests across agents, envs, physics
scripts/                   smoke.py, train.py, bench_backends.py
docs/superpowers/plans/    Implementation plans for major features
pyproject.toml
```

---

## Roadmap

- **SB3 ↔ VSSVecEnv adapter** — let `scripts/train.py` consume batched physics with PPO (unlocks the ~30× wall-clock training speedup on Mac CPU).
- **CUDA on Ubuntu RTX 3060** — `pip install -U "jax[cuda12]"` and verify; expected 1M+ env-steps/sec at batch=256.
- **JAX-native RL** — integration with `purejaxrl` / `Stoix` to remove the Gymnasium boundary entirely.
- **Pymunk backend** — possible third backend for sanity-checking the hand-rolled physics.

---

## License

MIT
