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

Code must run on both:
- **Ubuntu laptop with NVIDIA RTX 3060** — primary target for GPU training
- **Mac M4 Pro (Apple Silicon)** — primary dev machine; can fall back to slower CPU backends if Mac GPU support isn't available

When picking GPU libraries, prefer cross-platform (JAX with Metal, PyTorch MPS) over CUDA-exclusive (Isaac Gym) unless the user confirms Ubuntu-only is acceptable.

## Repo layout

```
vsss-sim/
├── src/vsss_sim/
│   ├── __init__.py        # registers VSSS-v0
│   ├── config.py          # IEEE VSSS constants
│   ├── agents/            # stationary.py, random.py (pluggable opponents)
│   ├── envs/              # base.py, vsss_3v3.py (VSSEnv), vsss_vec.py (VSSVecEnv)
│   ├── physics/           # numpy_backend.py + jax_backend.py (resolver in __init__.py)
│   └── rendering/         # pygame.py
├── tests/
│   ├── agents/
│   ├── envs/test_env.py
│   ├── physics/test_physics.py
│   └── test_config.py
├── scripts/
│   ├── smoke.py           # quick run with --render, --timesteps, --fps
│   └── train.py           # MLflow-tracked training
├── mlruns/                # MLflow artifacts
├── mlflow.db              # MLflow SQLite backend
├── pyproject.toml
└── requirements.txt
```

Last known test state: 147/147 passing (numpy + jax backends + VSSVecEnv + SB3 adapter).

## Environment setup

Python 3.14 venv at `.venv/`. On Mac, pygame needs SDL2 from Homebrew (`brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf`).

```bash
source .venv/bin/activate
pip install -e ".[dev,render]"     # includes pytest, ruff, pygame
# optional extras: [gpu] for torch, [jax] for the JAX physics backend
pytest                              # run tests
python scripts/smoke.py --render    # visual smoke test (numpy)
python scripts/smoke.py --backend jax   # smoke test on the JAX backend
python scripts/bench_backends.py    # numpy vs JAX vs VSSVecEnv throughput
python scripts/train.py             # MLflow-tracked training
```

The active physics backend can also be set via the `VSSS_PHYSICS_BACKEND`
environment variable (`numpy` or `jax`); the `backend=` kwarg on `VSSEnv` /
`gym.make("VSSS-v0", backend=...)` overrides the env var.

For batched training-throughput envs:

```python
import gymnasium as gym
import vsss_sim  # noqa: F401 — registers VSSS-v0

envs = gym.make_vec(
    "VSSS-v0", num_envs=256, vectorization_mode="vector_entry_point",
)
obs, info = envs.reset(seed=0)            # obs: (256, 46) float32
obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
```

Or directly: `from vsss_sim.envs import VSSVecEnv; envs = VSSVecEnv(num_envs=256, ...)`.

For SB3 (PPO etc.) with batched physics, use the adapter:

```python
from stable_baselines3 import PPO
from vsss_sim.envs import VSSVecEnv
from vsss_sim.sb3_adapter import VSSVecEnvToSB3

env = VSSVecEnvToSB3(VSSVecEnv(num_envs=256, opponent_policy="stationary"))
model = PPO("MlpPolicy", env, n_steps=128, verbose=1)
model.learn(total_timesteps=1_000_000)
```

Or via `scripts/smoke.py --num-envs 256 --timesteps 50000`.

`pyproject.toml` uses ruff (`line-length=100`), pytest configured for `tests/`.

## What's been built (chronological)

1. Initial scaffolding — venv, pyproject extras, src layout, Gymnasium registration.
2. Branch rename `main` → `master` (local + GitHub default + `origin/HEAD`).
3. MLflow integration in `scripts/train.py` (SQLite tracking + `mlruns/`).
4. `scripts/smoke.py` for quick visual checks, with `--render`, `--timesteps`, `--fps` flags.
5. Pygame renderer + optional FPS cap.
6. Robot model switched from circle to **box** — OBB collision + square render with direction indicator (commit `eadc33b`).
7. MLflow callback changed to log **per-episode** metrics instead of per-step (commit `40fc902`).
8. **Switchable physics backends**: `physics/__init__.py` resolver picks between `numpy_backend.py` and `jax_backend.py`. JAX backend is a pure-functional mirror — `SimState` as a `NamedTuple` PyTree, `step()` is `jax.jit`-compiled with `lax.fori_loop` substeps and `vmap`-ready shapes. Parity tested vs the NumPy backend within `atol=1e-3` for positions.
9. **Batched `VSSVecEnv`** (Gymnasium `VectorEnv`): runs `num_envs` matches in parallel via `jit(vmap(jb.step))` over a single batched `SimState`. Registered as the `vector_entry_point` for `VSSS-v0` so `gym.make_vec("VSSS-v0", num_envs=N, vectorization_mode="vector_entry_point")` returns one. `NEXT_STEP` autoreset on truncation; goals trigger in-episode kickoffs (no step-counter reset). Bench at batch=256 on Mac M4 Pro CPU: ~111k env-steps/sec via wrapper, ~303k via raw `jax.vmap` — gap is the Gymnasium boundary overhead.
10. **SB3 ↔ VSSVecEnv adapter** (`src/vsss_sim/sb3_adapter.py::VSSVecEnvToSB3`): translates between SB3's `VecEnv` interface and `VSSVecEnv`'s Gymnasium `VectorEnv`. Bridges (a) `reset()` returning only obs, (b) `step → (obs, rewards, dones, list[dict])` with `dones = term|trunc`, (c) eager auto-reset with `terminal_observation` and `episode={"r","l","t"}` per SB3 convention. `scripts/smoke.py` gained `--num-envs N` that routes through the adapter. End-to-end SB3 PPO throughput on Mac M4 Pro CPU: 600 fps (num_envs=1) → 47,500 fps (num_envs=256) — **~80× wall-clock training speedup.**

## Open threads / next directions

- **CUDA on Ubuntu RTX 3060** — `pip install -U "jax[cuda12]"`, then verify `VSSVecEnv` runs on the GPU. Code is already ready; mostly setup/verification work. Expected boost: 1M+ env-steps/sec at batch=256 for raw physics; SB3 ceiling depends on policy size.
- **JAX-native RL libraries** — `purejaxrl`, `Stoix`, `JaxMARL`, `Brax` (env + RL), `RLax` (building blocks only). These would skip the Gymnasium + SB3 boundary entirely and recover the gap between VSSVecEnv (~111k fps batch=256 on Mac CPU) and raw vmap ceiling (~303k fps). Biggest win on GPU.
- **Pymunk backend** mentioned as a possible alternative once user evaluates it. Isaac Gym / PhysX explicitly deferred.
- Adversarial opponents are stationary today — could be upgraded once training is stable.

## Conventions

- Use `master`, not `main`, in git operations.
- Don't add features beyond what's asked — user prefers iterative, small, committed changes.
- Commit between tasks; user often asks "commit changes before proceeding".
- When user asks to brainstorm or plan something non-trivial, use the brainstorming skill first.
