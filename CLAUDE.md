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
│   ├── envs/              # base.py, vsss_3v3.py (VSSEnv, backend-switchable)
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

Last known test state: 117/117 passing (numpy + jax backends).

## Environment setup

Python 3.14 venv at `.venv/`. On Mac, pygame needs SDL2 from Homebrew (`brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf`).

```bash
source .venv/bin/activate
pip install -e ".[dev,render]"     # includes pytest, ruff, pygame
# optional extras: [gpu] for torch, [jax] for the JAX physics backend
pytest                              # run tests
python scripts/smoke.py --render    # visual smoke test (numpy)
python scripts/smoke.py --backend jax   # smoke test on the JAX backend
python scripts/train.py             # MLflow-tracked training
```

The active physics backend can also be set via the `VSSS_PHYSICS_BACKEND`
environment variable (`numpy` or `jax`); the `backend=` kwarg on `VSSEnv` /
`gym.make("VSSS-v0", backend=...)` overrides the env var.

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

## Open threads / next directions

- **Batched JAX env (`VSSVecEnv`)** — PR 2 of the JAX work. Will expose a Gymnasium `VectorEnv` that holds `num_envs` states, batches `jax.vmap(step)`, and ports the opponent policies to operate on batched observations. The single-env `VSSEnv(backend='jax')` is already vmap-ready; the wrapper is mostly env-layer plumbing.
- **Pymunk backend** mentioned as a possible alternative once user evaluates it. Isaac Gym / PhysX explicitly deferred.
- **JAX-native RL libraries** — user asked for the SB3 analog in JAX; question was open at end of last session. Candidates: `purejaxrl`, `Stoix`, `JaxMARL`, `Brax` (env + RL), `RLax` (building blocks only).
- Adversarial opponents are stationary today — could be upgraded once training is stable.

## Conventions

- Use `master`, not `main`, in git operations.
- Don't add features beyond what's asked — user prefers iterative, small, committed changes.
- Commit between tasks; user often asks "commit changes before proceeding".
- When user asks to brainstorm or plan something non-trivial, use the brainstorming skill first.
