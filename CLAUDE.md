# vsss-sim

IEEE Very Small Size Soccer 3×3 simulator built for Reinforcement Learning.

## IMPORTANT: VSSS vs SSL — don't conflate them

This project targets **IEEE VSSS (Very Small Size Soccer)** *only*. It is **not** RoboCup SSL (Small Size League). The names look similar and the two leagues share a lot of literature, but the physical specs and the action space are different enough that mixing them up will silently produce wrong code.

| | **IEEE VSSS** (this project) | **RoboCup SSL** (NOT this project) |
|---|---|---|
| Team size | 3 | up to 11 (commonly 6) |
| Robot | 7.5 cm cube, **differential drive** (2 wheels), **no kicker, no dribbler** | 180 mm Ø × 150 mm tall, **omnidirectional** (4 omni wheels), **with kicker + dribbler** |
| Field | 150 × 130 cm, **walled** | 605 × 405 cm, no walls |
| Vision | Per-team overhead camera | Shared `SSL-Vision` server |
| Action space (RL) | 2 wheel speeds → 6-dim for 3 robots | 3-DOF body velocity + kick + dribble |

**What transfers** (useful when reading SSL prior art): RL methodology (curriculum, self-play, sim-to-real, domain randomization), multi-agent RL patterns (CTDE, role assignment), strategy primitives (positional play, intercept geometry), and general physics building blocks. The `rSoccer` framework (`robocin/rSoccer`) is the canonical cross-league reference — it provides Gym envs for *both* VSSS and SSL on a shared simulator.

**What does NOT transfer**: drivetrain dynamics, action space, anything involving the kicker or dribbler, field-topology assumptions (walls vs no walls), and the coordination scale of 3v3 vs 6v6+. A policy or reward-shaping rule that assumes kicking is meaningless in VSSS.

**Heuristic:** if a paper, repo, or simulator mentions `grSim`, `SSL-Vision`, kickers, or omnidirectional drive, it's SSL. Treat it as adjacent literature, not as a drop-in.

See `~/dev/personal/docs/ssl-vs-vsss.md` in the Obsidian vault for the longer write-up.

## What this project is

- IEEE VSSS spec: 3 robots/team, 150×130 cm field, golf ball, 7.5 cm differential-drive robots
- Gymnasium env id: `VSSS-v0` (registered in `src/vsss_sim/__init__.py`)
- Observation: `Box(-inf, inf, (46,), float32)` — ball (4) + 6 robots × 7 features (pos, sin/cos θ, vel, ω)
- Action: `Box(-1, 1, (6,), float32)` — normalised wheel speeds for the blue (controlled) team
- Reward: `+1` blue goal, `-1` yellow goal, else `0`
- GitHub: `canhotuctor/vsss-sim` — default branch is **`master`** (renamed from `main`; use `master` in all git commands)

## Hardware targets

Code targets macOS development and Ubuntu with an NVIDIA GPU. Training and physics use JAX;
install the CUDA extra on supported Linux machines when GPU execution is needed.

## Repo layout

```
vsss-sim/
├── src/vsss_sim/
│   ├── config.py          # field, timing, initialization, reward profiles
│   ├── envs/              # Gymnasium single/vector and Jumanji interfaces
│   ├── physics/           # pure-functional JAX engine
│   ├── rl/                # Flax/Optax PPO, snapshots, self-play, evaluation
│   └── rendering/         # Pygame rendering
├── configs/experiments/   # serializable native-JAX experiment profiles
├── tests/                 # environment, rules, physics, CUDA, and PPO tests
├── scripts/               # JAX training, evaluation, visualization, benchmarks
├── pyproject.toml
└── requirements.txt
```

## Environment setup

```bash
source .venv/bin/activate
pip install -e ".[dev,render,jax-rl]"
pytest
python scripts/smoke.py
python scripts/train_jax.py --xp experiment26_brandao_reward
python scripts/evaluate_jax.py --xp experiment26_brandao_reward --lineage 0
```

The Gymnasium environments support interactive use and external clients. Training uses the
Jumanji environment and the repository's native Flax/Optax PPO implementation. To inspect a
saved experiment policy, pass a `.msgpack` snapshot to `scripts/visualize.py`.

The active physics implementation is JAX-only. CUDA device selection can be forced for benchmark
scripts with `--device cpu` or `--device gpu`.

## What's been built (chronological)

> Entries below are historical and may describe retired code paths. Use the setup and entry
> points above for the current workflow.


1. Initial scaffolding — venv, pyproject extras, src layout, Gymnasium registration.
2. Branch rename `main` → `master` (local + GitHub default + `origin/HEAD`).
3. MLflow integration in `scripts/train.py` (SQLite tracking + `mlruns/`).
4. `scripts/smoke.py` for quick visual checks, with `--render`, `--timesteps`, `--fps` flags.
5. Pygame renderer + optional FPS cap.
6. Robot model switched from circle to **box** — OBB collision + square render with direction indicator (commit `eadc33b`).
7. MLflow callback changed to log **per-episode** metrics instead of per-step (commit `40fc902`).
8. **Switchable physics backends**: `physics/__init__.py` resolver picks between `numpy_backend.py` and `jax_backend.py`. JAX backend is a pure-functional mirror — `SimState` as a `NamedTuple` PyTree, `step()` is `jax.jit`-compiled with `lax.fori_loop` substeps and `vmap`-ready shapes. Parity tested vs the NumPy backend within `atol=1e-3` for positions.
9. **Batched `VSSVecEnv`** (Gymnasium `VectorEnv`): runs `num_envs` matches in parallel via `jit(vmap(jb.step))` over a single batched `SimState`. Registered as the `vector_entry_point` for `VSSS-v0` so `gym.make_vec("VSSS-v0", num_envs=N, vectorization_mode="vector_entry_point")` returns one. `NEXT_STEP` autoreset on truncation; goals trigger in-episode kickoffs (no step-counter reset). Bench at batch=256 on Mac M4 Pro CPU: ~111k env-steps/sec via wrapper, ~303k via raw `jax.vmap` — gap is the Gymnasium boundary overhead.
10. A former adapter connected the Gymnasium vector environment to an external PPO trainer. It
    was retired after the native JAX training path was implemented.


11. **CUDA support on Ubuntu RTX 3060** — JAX 0.6.2 + `jax[cuda12]`, driver 580. GPU verified: 909k env-steps/sec at batch=1024 (raw physics ceiling). `scripts/check_gpu.py` for diagnostics; `--device gpu|cpu` flag on `bench_backends.py`; conditional CUDA test suite in `tests/physics/test_cuda.py`.

12. **`InitMode` — pluggable placement strategy** (`src/vsss_sim/config.py::InitMode`): `KICKOFF` (standard formation + jitter, default) and `RANDOM` (uniform random positions in each team's half, random headings, ball random in inner 80% of field). Both backends implement `reset_random`; the JAX version is `vmap`-ready. `VSSEnv` and `VSSVecEnv` accept `init_mode=` kwarg (string or enum). Infrastructure is designed for a future `SELECTOR` mode (learned/heuristic placement model).

    ```python
    # Random init — single env
    env = gym.make("VSSS-v0", init_mode="random")

    # Random init — batched
    envs = VSSVecEnv(num_envs=256, init_mode="random")
    ```

## Open threads / next directions

- **CUDA on Ubuntu RTX 3060** ✓ — verified 2026-05-27; 909k fps at batch=1024.
- **Alternative JAX RL libraries** — `Stoix`, `JaxMARL`, and `RLax` remain references for future algorithm and multi-agent needs. The project already has a native Flax/Optax PPO trainer; benchmark it with `scripts/bench_self_play.py`.
- **Simulator efficiency optimizations (PR 6, optional)** — profile and reduce the VSSVecEnv↔numpy boundary overhead. Candidates: batch obs construction in JAX before copy, fused autoreset, fewer Python calls per step.
- **Selector init mode** — learned/heuristic model that outputs robot placement distributions; extend `InitMode` and wire a callable into `_batched_reset`.
- **Pymunk backend** mentioned as a possible alternative once user evaluates it. Isaac Gym / PhysX explicitly deferred.
- Adversarial opponents are stationary today — could be upgraded once training is stable.

## Conventions

- Use `master`, not `main`, in git operations.
- Don't add features beyond what's asked — user prefers iterative, small, committed changes.
- Commit between tasks; user often asks "commit changes before proceeding".
- When user asks to brainstorm or plan something non-trivial, use the brainstorming skill first.
