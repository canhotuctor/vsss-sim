# VSSS-Sim: Project Development Report

**Date:** 2026-05-30  
**Project:** IEEE Very Small Size Soccer 3×3 Simulator for Reinforcement Learning  
**Repository:** `canhotuctor/vsss-sim` (branch: `master`)  
**Test status:** 174 passed, 4 skipped (CUDA tests, no GPU on current machine)

---

## 1. Context and Motivation

This project implements a high-performance 2D physics simulator for the **IEEE VSSS (Very Small Size Soccer)** competition format, targeting use as a Reinforcement Learning training environment. The VSSS format specifies 3-versus-3 teams of 7.5 cm differential-drive robots on a 150 × 130 cm walled field, using a golf ball (~43 mm diameter) and no kicking mechanism — distinct from RoboCup SSL (which uses omnidirectional robots with kickers on much larger fields).

The main research goal is to train a control policy for the blue team (6-dimensional normalized wheel-speed actions) capable of scoring goals and adapting to opponent behavior. A secondary engineering goal is maximizing RL training throughput via hardware-accelerated batched simulation to reduce wall-clock training time.

---

## 2. Physical Specification

| Parameter | Value |
|---|---|
| Field | 150 × 130 cm (walled) |
| Teams | 3 robots/side |
| Ball | Golf ball, 42.67 mm diameter, 46 g |
| Robot footprint | Max 7.5 cm × 7.5 cm |
| Drivetrain | Differential drive (2 wheels) |
| Max wheel speed | ~1.30 m/s (50 rad/s × 0.026 m radius) |
| Wheelbase | 53 mm |
| Simulation frequency | 60 Hz |
| Default episode length | 1 200 steps (20 s) |
| Robot inertia | 8.44 × 10⁻⁵ kg·m² (solid square approximation) |
| Wheel accel limit | ~9.81 m/s² (1 g) |

---

## 3. Repository Structure

```
vsss-sim/
├── src/vsss_sim/
│   ├── __init__.py            # Gymnasium VSSS-v0 registration
│   ├── config.py              # IEEE VSSS constants + InitMode enum
│   ├── agents/                # stationary.py, random.py (pluggable opponents)
│   ├── envs/
│   │   ├── base.py            # VSSBaseEnv (backend-agnostic step, obs build)
│   │   ├── vsss_3v3.py        # VSSEnv — single-env Gymnasium interface
│   │   └── vsss_vec.py        # VSSVecEnv — batched JAX VectorEnv
│   ├── physics/
│   │   ├── __init__.py        # Backend resolver (numpy / jax / env var)
│   │   ├── numpy_backend.py   # Reference CPU physics (float64)
│   │   └── jax_backend.py     # JAX pure-functional physics (float32)
│   ├── rendering/
│   │   └── pygame.py          # Pygame renderer with OBB shapes + HUD
│   └── sb3_adapter.py         # VSSVecEnvToSB3 — bridges Gymnasium↔SB3
├── scripts/
│   ├── smoke.py               # Quick sanity + MLflow-tracked PPO run
│   ├── train.py               # Full MLflow-tracked SB3 PPO training
│   ├── visualize.py           # Render-only visual inspection
│   ├── bench_backends.py      # Throughput benchmark (numpy vs JAX vs vmap)
│   └── check_gpu.py           # JAX device diagnostics
├── tests/                     # 174 tests across physics, envs, config
├── docs/superpowers/specs/    # Design specs for major features
├── mlruns/                    # MLflow artifact store
├── mlflow.db                  # MLflow SQLite tracking backend
└── pyproject.toml             # Extras: [dev], [render], [jax], [cuda], [stoix]
```

---

## 4. Development Chronology

### Phase 1 — Foundation (commits `84b3ebb` → `8a9f904`)
- Initial project scaffolding: Python 3.14 virtual environment, `src/` layout, `pyproject.toml` with optional extras, Gymnasium `VSSS-v0` registration.
- Robot model: circular collision, simple top-down rendering.
- `scripts/smoke.py` and `scripts/train.py` added; MLflow wired to a local SQLite backend (`mlflow.db`) for tracking training runs.
- Physics: fully vectorized NumPy implementation of differential-drive kinematics and axis-aligned wall/ball/robot collisions.

### Phase 2 — Simulation Fidelity (`eadc33b`, `40fc902`)
- **Robot body switched from circle to box.** Collision detection upgraded to Oriented Bounding Box (OBB) using the Separating Axis Theorem (SAT). Rendering updated to draw oriented squares with a direction indicator.
- MLflow callback refactored to log **per-episode** metrics (mean reward, episode length) rather than per-step scalars, producing cleaner learning curves.

### Phase 3 — JAX Physics Backend (`6867e62` → `fa85032`)
- `physics/__init__.py` backend resolver introduced: selects `numpy_backend` or `jax_backend` based on the `backend=` kwarg, `VSSS_PHYSICS_BACKEND` env var, or default.
- `jax_backend.py` is a **pure-functional mirror** of the NumPy backend:
  - `SimState` is a `NamedTuple` PyTree (immutable, register-free, `vmap`-compatible).
  - All arrays use `float32` (GPU default).
  - `step()` is `jax.jit`-compiled with `lax.fori_loop` for sub-steps.
  - Collision resolution uses `jax.vmap` over robot/ball pairs for fully vectorized collision detection.
  - Numerical parity with NumPy backend verified within `atol=1e-3`.
- `VSSEnv` gains `backend=` kwarg and smoke test gains `--backend` flag.

### Phase 4 — Batched VSSVecEnv (`62f52c1` → `9370bee`)
- `VSSVecEnv` added: a `gymnasium.VectorEnv` running `num_envs` independent matches in parallel via `jit(vmap(jb.step))` over a single batched `SimState`.
- Registered as the `vector_entry_point` for `VSSS-v0` so `gym.make_vec("VSSS-v0", num_envs=N, vectorization_mode="vector_entry_point")` returns one.
- Auto-reset semantics follow Gymnasium `AutoresetMode.NEXT_STEP`: goals trigger in-episode kickoffs; episode truncation resets on the *next* step.
- `scripts/bench_backends.py` added to measure throughput across all three paths.

**Throughput benchmark (Mac M4 Pro, CPU-only):**

| Path | Steps/sec |
|---|---|
| NumPy single env | ~2.5 k |
| JAX single env (jit) | ~50 k |
| Raw `jax.vmap` batch=256 | ~303 k |
| VSSVecEnv wrapper batch=256 | ~111 k |
| SB3 full PPO cycle (rollout+grads) | ~8–10 k |

The gap between raw `vmap` and the wrapper is Gymnasium boundary overhead (Python call per step, JAX→NumPy copy). The gap between VSSVecEnv and SB3 full cycle is PyTorch CPU gradient updates becoming the bottleneck.

### Phase 5 — SB3 Adapter (`5609e1e` → `ace0ac0`)
- `VSSVecEnvToSB3` adapter bridges `VSSVecEnv` to the SB3 `VecEnv` protocol:
  - Translates `reset()` returning `(obs, info)` → `obs` only (SB3 convention).
  - Translates `step()` returning `(obs, rewards, terminated, truncated, infos)` → `(obs, rewards, dones, list[dict])` with `dones = term | trunc`.
  - Injects `terminal_observation` and `episode={"r", "l", "t"}` into episode-end infos (SB3 monitor convention for reward tracking).
  - Eager auto-reset is handled inside the adapter on episode end.
- `scripts/smoke.py` gains `--num-envs N` that routes through the adapter, enabling batched PPO training directly from the smoke script.

### Phase 6 — CUDA / Ubuntu RTX 3060 (`3a8f73d` → `28b9cfb`)
- JAX `0.6.2` + `jax[cuda12]` verified on Ubuntu with driver `580`.
- `check_gpu.py` diagnostics script added.
- Conditional CUDA test suite (`tests/physics/test_cuda.py`) — skipped automatically when no GPU is present.
- `bench_backends.py` gains `--device cpu|gpu` flag.
- `[cuda]` optional dependency group added to `pyproject.toml`.

**Throughput on Ubuntu RTX 3060 (GPU):**

| Path | Steps/sec |
|---|---|
| Raw `jax.vmap` batch=1024 | ~909 k |
| VSSVecEnv batch=1024 | ~92 k |

### Phase 7 — Pluggable Initialization (`00b2ca2` → `d9ab2fd`)
- `InitMode` enum added to `config.py` with two modes:
  - `KICKOFF` (default): standard formation with small random jitter.
  - `RANDOM`: uniformly random positions, random headings, ball randomly placed in the inner 80% of the field.
- Both backends implement `reset_random()`; the JAX version is `vmap`-ready.
- `VSSEnv` and `VSSVecEnv` accept `init_mode=` kwarg (string or enum).
- Infrastructure designed for a future `SELECTOR` mode (learned placement model).
- `--init-mode` flag added to `smoke.py` and `visualize.py`.

### Phase 8 — Physics Improvements and Reward Tuning (`923ddc4` → `7b0b53c`)
- **Vectorized collision resolution**: robot-robot collision pairs now processed with a single vectorized pass rather than sequential loops; ball spawn moved to full-field range.
- **Ball-forward-progress reward shaping**: a small dense reward proportional to Δx of the ball each step (`BALL_FORWARD_REWARD_COEF = 0.10`) is added on top of the sparse ±1 goal signal. This telescopes to at most `0.10 × FIELD_LENGTH = 0.15` per episode, avoiding domination of the sparse signal.
- Coefficient tuned down from `0.30` to `0.10` after empirical observation that the denser signal was overpowering goal-scoring incentives.
- `smoke.py` upgraded with per-generation MLflow logging, `--generations`, `--forever`, and `--n-steps` flags, plus `--max-episode-steps` flag.

---

## 5. Architecture Decisions

### 5.1 Dual Physics Backend

The most distinctive architectural choice is the side-by-side `numpy_backend` / `jax_backend` with a runtime resolver. This was motivated by three constraints:

1. **Development on Mac, training on Ubuntu**: The NumPy backend serves as a readable, float64, debuggable reference that runs anywhere without GPU drivers.
2. **Throughput**: The JAX backend enables `jit`-compilation and `vmap`-based batching that is ~120× faster than NumPy at batch=256 on CPU.
3. **Functional purity requirement for `vmap`**: JAX's `vmap` requires stateless, side-effect-free functions. The NumPy backend uses a mutable dataclass (`SimState`). The JAX backend's `SimState` is a `NamedTuple` PyTree — immutable, hashable, and trivially batched by adding a leading dimension.

The resolver in `physics/__init__.py` makes the choice transparent at the environment level; callers never import a backend directly.

### 5.2 OBB Collision Detection

The switch from circular to OBB collision was driven by accuracy: 7.5 cm square robots in a 150 × 130 cm field have a significant coverage fraction, and circular approximations introduce noticeable ghost collisions (triggering when corners are still clear) and missed collisions (missing corner-to-edge contacts). The SAT-based OBB resolver handles this correctly at negligible extra cost.

### 5.3 SB3 Adapter Pattern

Rather than modifying `VSSVecEnv` to comply with SB3's `VecEnv` protocol directly, a thin adapter class `VSSVecEnvToSB3` translates between the two APIs. This keeps the Gymnasium-native `VSSVecEnv` clean and usable outside SB3, while enabling SB3 PPO without any changes to its internals. The translation includes:
- API shape mismatch (`reset` returns tuple vs single array)
- Termination/truncation semantics (`done = term | trunc`)
- Episode info injection (SB3's monitor convention for `ep_reward`, `ep_len`)

### 5.4 Reward Design

The reward is intentionally sparse at its core (±1 on goal), with a small ball-forward-progress shaping term added as a training scaffold:

```
r(t) = goal_signal(t) + 0.10 × Δball_x(t)
```

The coefficient (0.10) was chosen so the total dense shaping over an episode (bounded by `0.10 × 1.50 m = 0.15`) is smaller than a single goal event (+1.0), ensuring the agent is never incentivized to push the ball endlessly without scoring. This is a known tradeoff in reward shaping for goal-scoring tasks.

---

## 6. Observation and Action Spaces

**Observation** (`Box(-inf, inf, (46,), float32)`):

| Slice | Content |
|---|---|
| `[0:4]` | Ball: x/norm, y/norm, vx/norm, vy/norm |
| `[4:46]` | 6 robots × 7 features: x/norm, y/norm, sin θ, cos θ, vx/norm, vy/norm, ω/norm |

Normalization: positions by half-field dimensions, velocities by `1.5 × max_wheel_speed`, angular velocity by the implied max omega from the wheelbase.

**Action** (`Box(-1, 1, (6,), float32)`):
6 values `[vl₀, vr₀, vl₁, vr₁, vl₂, vr₂]` — normalized left/right wheel speeds for all 3 blue robots. Denormalized to physical units inside the backend.

---

## 7. Training Infrastructure

- **MLflow** (SQLite backend at `mlflow.db`, artifacts at `mlruns/`) tracks all training runs. Logged metrics: per-generation mean/min/max reward, episode length, generation index (used as the MLflow x-axis for aligned multi-run charts).
- **SB3 PPO** is the current training algorithm, accessed via `VSSVecEnvToSB3` + `VSSVecEnv(num_envs=N)`.
- `scripts/smoke.py` supports `--forever` mode for continuous generational training with periodic policy snapshots saved to disk.
- `scripts/train.py` is the full training entrypoint with configurable hyperparameters.

---

## 8. Test Coverage

174 tests pass across the full suite (Mac M4 Pro, no GPU):

| Module | Tests |
|---|---|
| `tests/physics/test_physics.py` | NumPy backend physics correctness |
| `tests/physics/test_jax_backend.py` | JAX backend + parity vs NumPy |
| `tests/physics/test_random_init.py` | InitMode.RANDOM placement coverage |
| `tests/physics/test_backend_resolver.py` | Backend selection logic |
| `tests/physics/test_cuda.py` | CUDA-specific (skipped without GPU) |
| `tests/envs/test_env.py` | Gymnasium API compliance, spaces, reset/step |
| `tests/envs/test_vec_env.py` | VSSVecEnv batched step, autoreset |
| `tests/envs/test_sb3_adapter.py` | SB3 adapter protocol compliance |
| `tests/agents/` | Stationary and random opponent policies |
| `tests/test_config.py` | Constants + InitMode enum |

---

## 9. Current Status and Open Work

### Done
- [x] Full VSSS physics (diff-drive, OBB collisions, goal detection)
- [x] Switchable NumPy/JAX backends with runtime resolver
- [x] Batched `VSSVecEnv` (JAX `vmap`) for parallel simulation
- [x] SB3 PPO training via `VSSVecEnvToSB3` adapter
- [x] CUDA support verified on Ubuntu RTX 3060 (909 k fps raw, 92 k fps wrapped)
- [x] Pluggable `InitMode` (kickoff and random)
- [x] Dense ball-forward-progress reward shaping
- [x] MLflow experiment tracking (per-generation metrics, artifacts)
- [x] Pygame renderer with OBB shapes, FPS overlay, episode counter

### Next Directions

**JAX-native RL (Stoix integration — designed, not yet implemented)**  
The Stoix library (`EdanToledo/Stoix`) implements PPO in a fully JAX-resident Anakin architecture. A design spec (`docs/superpowers/specs/2026-05-30-stoix-rl-integration-design.md`) exists. The plan is:
1. Wrap `jax_backend.step` as a `jumanji.Environment` (`VSSStoixEnv`) with Stoix-compatible `observation_spec` / `action_spec`.
2. Add `scripts/train_stoix.py` using `stoix.systems.ppo.ff_ppo`.
3. MLflow logging via a Stoix logger protocol implementation.
4. The SB3 path remains unchanged; Stoix is additive.

Expected benefit: close the 10× gap between `VSSVecEnv` (~92k fps) and raw `vmap` ceiling (~909k fps on GPU) by eliminating the Gymnasium ↔ NumPy ↔ PyTorch boundary.

**Selector InitMode**  
A learned or heuristic placement model (`InitMode.SELECTOR`) that outputs robot placement distributions. Infrastructure is in place; requires defining the selector callable interface and wiring it into `_batched_reset`.

**Adversarial opponents**  
Yellow robots are currently stationary. A self-play or curriculum-trained opponent policy would significantly improve the difficulty landscape.

**Multi-agent RL (IPPO/MAPPO)**  
Observation and action construction are currently factored as swappable functions, forward-compatible with per-robot egocentric observations. Stoix has multi-agent variants and JaxMARL provides reference environments. This is explicitly deferred until the single-agent training baseline stabilizes.

---

## 10. Dependency Summary

| Group | Key packages |
|---|---|
| Core | `gymnasium>=0.29`, `numpy`, `pygame` (optional) |
| JAX path | `jax>=0.4.30`, `jaxlib` |
| CUDA | `jax[cuda12]` |
| Training | `stable-baselines3`, `mlflow` |
| Stoix (planned) | `stoix`, `jumanji`, `optax`, `flax`, `chex` |
| Dev | `pytest`, `ruff` (line-length=100) |

Python 3.14; `.venv/` at repo root.

---

*Generated 2026-05-30 from git log (`84b3ebb..caffc2f`, 60 commits) and source inspection.*
