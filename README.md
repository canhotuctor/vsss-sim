# vsss-sim

IEEE VSSS 3×3 simulator for Reinforcement Learning.

Implements the **Very Small Size Soccer** (VSSS) specification — 3 robots per
team, 150 × 130 cm field, golf ball — with a standard
[Gymnasium](https://gymnasium.farama.org/) interface and a JAX physics engine
designed for GPU-accelerated batched training.

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
| **JAX physics engine** | One functional, JIT-compatible implementation for CPU and GPU |
| **Native batched Vector env** | `VSSVecEnv` runs N parallel matches in one `jit(vmap(step))` call |
| **Pluggable opponents + init** | Opponents: `"stationary"`, `"random"`, or callable; reset: `init_mode="kickoff"` (default) or `"random"` |
| **Pygame renderer** | OBB robot shapes, FPS overlay — `"human"` or `"rgb_array"` |
| **Cross-platform** | macOS (Apple Silicon) · Ubuntu 22.04 with CUDA-capable GPU |

---

## Installation

```bash
# Core (no rendering)
pip install -e .

# With Pygame renderer
pip install -e ".[render]"

# With the end-to-end JAX environment and PPO trainer
pip install -e ".[jax-rl]"

# Developer tools (tests + linter)
pip install -e ".[dev]"
```

Python ≥ 3.10 is required.

---

## Quick start

### Single environment

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

### Batched vector environment (~30× CPU speedup at `num_envs=256`)

```python
envs = gym.make_vec(
    "VSSS-v0", num_envs=256, vectorization_mode="vector_entry_point",
)
obs, info = envs.reset(seed=0)            # obs shape: (256, 46) float32
obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
```

Or directly: `from vsss_sim.envs import VSSVecEnv; envs = VSSVecEnv(num_envs=256)`.

### End-to-end JAX PPO

This path bypasses Gymnasium, NumPy, SB3, and PyTorch during training. Policy
inference, VMAP'd physics, rollout collection, GAE, gradients, and Optax updates
remain on the JAX device.

```bash
python scripts/train_jax.py \
    --generations 30 \
    --num-envs 256 \
    --rollout-length 128
```

Use `--opponent random --init-mode random` for the randomized starting setup.
All generations execute in one compiled `lax.scan`. The script reports
initialization, compilation, and device execution times separately.

### Training with SB3 PPO + batched env (~4–8× wall-clock speedup)

```python
from stable_baselines3 import PPO
from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3

env = VSSVecEnvToSB3(VSSVecEnv(num_envs=256, opponent_policy="stationary"))
model = PPO("MlpPolicy", env, n_steps=128, verbose=1)
model.learn(total_timesteps=1_000_000)
```

Or via the smoke script: `python scripts/smoke.py --num-envs 256 --timesteps 50000`.

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

Sparse goal signal plus a small dense shaping term (see `BALL_FORWARD_REWARD_COEF` in `config.py`):

| Event | Reward |
|---|---|
| Blue scores | `+1.0` |
| Yellow scores | `−1.0` |
| Ball advances toward yellow goal | `+0.10 × Δx_ball` per step |
| Otherwise | `0.0` |

The shaping term telescopes to at most ~0.15 per episode — well below a single goal — so scoring stays the primary incentive.

Goals trigger an in-episode kickoff; episodes truncate at `MAX_EPISODE_STEPS`
(default 300 control steps = 20 s at 15 Hz, with four 60 Hz physics substeps
per control step).

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

## Physics engine

The simulator has one pure-functional JAX physics implementation in
`physics/jax_backend.py`. `SimState` is a `NamedTuple` PyTree, `step()` uses
`lax.fori_loop` for physics substeps, and the function is JIT- and VMAP-ready.
The same engine powers the single Gymnasium environment, batched `VSSVecEnv`,
and fully device-resident Jumanji environment.

### Throughput

Measured with `python scripts/bench_backends.py --steps 5000 --sb3`.

**Hardware:** Apple M4 Pro (12-core CPU, JAX CPU backend) · Ubuntu 22.04, NVIDIA RTX 3060 Laptop GPU 6 GB (JAX CUDA 12 backend).

#### Raw physics ceiling — `jax.vmap(step)`, no Gymnasium overhead

| batch | Mac M4 Pro CPU | Ubuntu RTX 3060 GPU | GPU / Mac |
|------:|---------------:|--------------------:|----------:|
| 1 | 29,700 fps | 690 fps | 0.02× |
| 8 | 83,700 fps | 6,700 fps | 0.08× |
| 64 | 252,000 fps | 51,700 fps | 0.2× |
| 256 | **304,000 fps** | 228,000 fps | 0.7× |
| 1024 | 244,000 fps | **909,000 fps** | **3.7×** |

GPU overhead dominates at small batches. At batch=1024 the GPU delivers ~910k
env-steps/sec — 3.7× faster than Mac CPU.

#### VSSVecEnv — Gymnasium wrapper (what your RL trainer sees)

| batch | Mac M4 Pro CPU | Ubuntu RTX 3060 GPU |
|------:|---------------:|--------------------:|
| 1 | 817 fps | 90 fps |
| 8 | 6,355 fps | 733 fps |
| 64 | 45,025 fps | 5,807 fps |
| 256 | 120,930 fps | 23,468 fps |
| 1024 | **169,000 fps** | **92,400 fps** |

Mac wins across all batch sizes. The per-step CUDA↔CPU copy (obs array transfer at the Gymnasium boundary) costs more than the GPU gains from the physics kernel. JAX-native RL (PR 5) eliminates this boundary and is the path to realising the 909k ceiling.

#### End-to-end SB3 PPO — `n_steps=32`, PyTorch MLP policy + JAX physics

| `num_envs` | Mac M4 Pro CPU | Ubuntu RTX 3060 GPU | GPU speedup vs GPU `num_envs=1` |
|-----------:|---------------:|--------------------:|--------------------------------:|
| 1 | 614 fps | 77 fps | 1× |
| 8 | 4,367 fps | 616 fps | 8× |
| 64 | 23,600 fps | 4,903 fps | 64× |
| 256 | **46,600 fps** | **18,500 fps** | 242× |

Mac is faster end-to-end with SB3 — same boundary cost as VSSVecEnv, plus PyTorch MLP policies are not GPU-efficient at small batch sizes. The 242× GPU scaling within Ubuntu is driven by JAX physics; the PyTorch policy is the remaining bottleneck.

Reproduce with `python scripts/bench_backends.py --steps 5000 --sb3`. Use `--device cpu` or `--device gpu` (after `pip install -e ".[cuda]"`) to force a specific device type.

---

## Scripts

- `scripts/smoke.py` — SB3 PPO smoke with MLflow logging. Flags: `--num-envs`, `--init-mode`, `--generations`, `--forever`, `--n-steps`, `--max-episode-steps`, `--render`, `--timesteps`.
- `scripts/train.py` — full MLflow-tracked PPO run (`mlflow.db` + `mlruns/`).
- `scripts/visualize.py` — render-only inspection (`--init-mode`, `--fps`).
- `scripts/bench_backends.py` — throughput across single-state JAX, raw `jax.vmap`, and `VSSVecEnv`; `--sb3` for end-to-end PPO; `--device cpu|gpu`.
- `scripts/check_gpu.py` — JAX device diagnostics.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Project structure

```
src/vsss_sim/
├── __init__.py            Gymnasium registration (VSSS-v0 single + vector entry points)
├── config.py              IEEE VSSS constants, InitMode, reward coefs
├── agents/                stationary.py, random.py (pluggable opponents)
├── envs/
│   ├── base.py            VSSBaseEnv (spaces, observation builder)
│   ├── vsss_3v3.py        VSSEnv — single-env Gymnasium Env
│   ├── vsss_vec.py        VSSVecEnv — batched JAX Gymnasium VectorEnv
│   └── jumanji.py         VSSJumanjiEnv — fully JAX-resident RL interface
├── physics/
│   ├── __init__.py        Public physics API
│   └── jax_backend.py     Functional, jittable, vmap-ready physics engine
├── rendering/pygame.py
└── sb3_adapter.py         VSSVecEnvToSB3 — SB3 VecEnv around VSSVecEnv
tests/                     tests across agents, envs, physics, config
scripts/                   smoke, train, visualize, bench_backends, check_gpu
docs/superpowers/plans/    Implementation plans for major features
pyproject.toml
```

---

## Roadmap

- **CUDA on Ubuntu RTX 3060** ✓ — verified; 909k env-steps/sec at batch=1024 (raw physics ceiling).
- **JAX-native RL** ✓ — Jumanji environment plus a fully compiled Flax/Optax PPO loop in `scripts/train_jax.py`. See `docs/jax-rl-integration.md` for the library comparison.
- **Selector InitMode** — learned/heuristic placement (`InitMode.SELECTOR`); reset infrastructure is in place.

---

## License

MIT
