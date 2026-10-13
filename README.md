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

This path bypasses Gymnasium and NumPy during training. Policy inference,
VMAP'd physics, rollout collection, GAE, gradients, and Optax updates remain
on the JAX device.

```bash
python scripts/train_jax.py \
    --generations 30 \
    --num-envs 256 \
    --rollout-length 128
```

Use `--opponent random --init-mode random` for the randomized starting setup.
All generations execute in one compiled `lax.scan`. The script reports
initialization, compilation, and device execution times separately.

### PPO smoke check

```bash
python scripts/smoke.py
```

This runs one small compiled JAX PPO update. Increase `--generations`,
`--num-envs`, or `--rollout-length` to exercise a larger setup.

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

The shared match rules also provide the `brandao_reward` profile based on
Brandão et al. (2022). Select it for a JAX PPO run with
`--reward-profile brandao_reward`, or set `"reward_profile": "brandao_reward"`
in an experiment JSON. The same profile option is available on `VSSEnv`,
`VSSVecEnv`, and `VSSJumanjiEnv`; `default` preserves the reward above.
The included long-run configuration can be started with
`python scripts/train_jax.py --xp experiment26_brandao_reward`.
See [the profile notes](docs/brandao-reward-profile.md) for the terms and the
team-reward adaptation.

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

Measured with `python scripts/bench_backends.py --steps 5000`.

**Hardware:** Apple M4 Pro (12-core CPU, JAX CPU backend) · Ubuntu 22.04, NVIDIA RTX 3060 Laptop GPU 6 GB (JAX CUDA 12 backend).

The throughput values below are previously recorded CPU/GPU measurements; rerun the command
above to measure the current checkout.

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

The per-step CUDA↔CPU copy at the Gymnasium interface can cost more than the GPU gains from
the physics kernel. Native JAX PPO uses the Jumanji environment and avoids that per-step host
transfer during training.

End-to-end JAX PPO throughput is measured by `scripts/bench_self_play.py`;
`scripts/perf_bench.py` compares raw physics, the Gymnasium vector interface,
and native Flax policy inference. Use `--device cpu` or `--device gpu` (after
`pip install -e ".[cuda]")` to force a specific device type.

---

## Scripts

- `scripts/smoke.py` — one small native-JAX PPO training pass.
- `scripts/train_jax.py` — JAX PPO and historical self-play training.
- `scripts/visualize.py` — render a saved native-JAX policy snapshot.
- `scripts/bench_backends.py` — throughput across single-state JAX, raw `jax.vmap`, and `VSSVecEnv`; `--device cpu|gpu`.
- `scripts/bench_self_play.py` — end-to-end JAX self-play throughput and memory benchmark.
- `scripts/check_gpu.py` — JAX device diagnostics.

---

## Running tests

```bash
pip install -e ".[dev,jax-rl]"
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
└── rl/                     Flax/Optax PPO, self-play, evaluation, and CLI
tests/                     tests across agents, envs, physics, config, and PPO
scripts/                   native-JAX train, smoke, visualize, and benchmark tools
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
