# Handoff: curriculum and opponent prioritization against overfitting

Status: research done, nothing implemented. Branch base: `master` at `50e0cee`.

## Goal

Evaluate prioritized-level-replay (PLR) style techniques to stop the PPO policy from
overfitting. The user suggested reverse curriculum generation applied to **only a fraction of
the vectorized envs**. Look for something that is low effort in this codebase.

Still open: the user has not said which kind of overfitting they see.
- **Start-state overfitting:** strong from kickoff, weak from other placements. Fix with item 2 below.
- **Opponent overfitting:** exploits the snapshot pool, weak against unseen policies. Fix with item 1.

Ask before building both.

## Findings in the code (verified on master `50e0cee`)

- **Reset distribution is narrow.**
  - `physics.reset_kickoff` (`src/vsss_sim/physics/jax_backend.py:66`) uses one fixed formation
    with ±2 cm jitter and the ball at the centre.
  - The only alternative is `reset_random` (`:93`), which places everything uniformly.
  - `VSSJumanjiEnv.reset` (`src/vsss_sim/envs/jumanji.py:183`) picks between them via `init_mode`.
  - With `terminate_on_goal` (the Brandão default, `jumanji.py:136`), every episode starts at kickoff.
- **Per-env autoreset is already inside the rollout scan.** `PPO._maybe_reset`
  (`src/vsss_sim/rl/ppo.py:796`) runs `lax.cond(done, reset, keep)` per env under `vmap`, and also
  resamples the opponent. This is the hook point for "reset a fraction of envs from a buffer".
- **Opponent sampling is category-based and ignores difficulty.**
  - `PPO._sample_opponent` (`ppo.py:715`) first samples one of five categories
    (all / recent-half / latest / random / stationary). It then picks a uniform index inside the category.
  - The weights come from `OpponentSamplingWeights` (`src/vsss_sim/rl/self_play.py:31`).
  - There is no per-snapshot weighting.
- **Brandão runs (`_brandao_opponent_pools`, `self_play.py:458`) train against a single promoted
  opponent**, not a pool.
  - Promotion compares goal margin against that one opponent (`self_play.py:1060-1150`).
  - So opponent prioritization only applies to the historical-pool path (`_active_pools`, `self_play.py:433`).
- **Rollout transitions already carry `opponent_generation` and `goal`** (`ppo.py` `_Transition`).
  Per-snapshot win rates can be aggregated from training rollouts without extra evaluation.
- **Brandão observations include a time-remaining feature and an 8-frame history**
  (`src/vsss_sim/envs/brandao_observation.py`, `initialize_history` at `:129`).
  A reset from an arbitrary state must rebuild the history the same way `reset` does.

## Research summary

- **Reverse Curriculum Generation** (Florensa et al., CoRL 2017, https://arxiv.org/abs/1707.05300)
  - Start from states near success.
  - Keep "good starts": return between `R_min` and `R_max`.
  - Expand the set with short random-action ("Brownian") rollouts from good starts.
  - Requires reset-to-any-state, which the functional JAX `SimState` gives us.
- **PLR / Robust PLR / ACCEL** keep a level buffer scored by regret proxies (positive value loss, MaxMC).
  JAX reference implementations:
  - JaxUED, https://arxiv.org/abs/2403.13091, code https://github.com/DramaCow/jaxued
  - minimax, https://github.com/facebookresearch/minimax
- **Sampling For Learnability (SFL)** (Rutherford et al., NeurIPS 2024, https://arxiv.org/abs/2408.15099,
  code https://github.com/amacrutherford/sampling-for-learnability)
  - PLR/ACCEL regret proxies often fail to beat plain domain randomisation.
  - Scoring a level by `p·(1−p)` (p = success rate) and mixing those levels with random ones beats
    DR, PLR and ACCEL.
  - Designed for binary-outcome tasks, which `terminate_on_goal` episodes are.
- **PFSP** (AlphaStar prioritized fictitious self-play): sample opponents by `f(win rate)`.
  - `(1−p)²` focuses on hard opponents.
  - `p(1−p)` focuses on even matches.

## Recommended plan (cheapest first)

### 1. PFSP over the historical pool

- Add a per-snapshot probability vector, fed to `_sample_opponent` alongside or replacing the
  "all history" category.
- Weights come from a moving average of win rate per `opponent_generation`, aggregated host-side
  from rollout metrics after each generation.
- Keep a uniform floor (about 20%) so the pool cannot collapse.
- Only meaningful for the non-Brandão historical-pool runs unless Brandão also moves to a pool.

### 2. Start-state buffer for a fraction ρ of resets (SFL scoring + reverse-curriculum seeding)

All state lives in `RunnerState`; `SimState` is about 55 floats, so the buffer is cheap.

- **Buffer per lineage:** K slots (e.g. 1024) of `SimState`, a moving-average success `p`, and a visit count.
- **Reset:**
  - In `_maybe_reset`, if `uniform < ρ` (start at 0.2–0.3), sample a slot weighted by `p(1−p)`
    plus a uniform mix, and reset from it.
  - Otherwise call the normal `env.reset`.
  - Track the slot id per env (`-1` for a normal reset).
- **Seeding (reverse curriculum):**
  - Keep a ring buffer of the last H `SimState`s per env.
  - When a goal happens, insert the state from k steps earlier. Do this for blue and yellow goals
    so defence is covered too.
  - Increase k as the mean `p` rises.
  - Replace the lowest-scoring slot.
- **Optional Brownian expansion:** perturb a sampled slot with a few `physics.step` calls using
  uniform random wheels before starting.
- **Score update:** on episode end for a slot-started env, update the slot's `p`
  (1 = blue goal, else 0).
- **Env hook:** add `VSSJumanjiEnv.reset_from(simulation, key)`, mirroring `reset`:
  - zero `score` and `t`
  - `initial_referee_state()`
  - `step_count = 0`
  - zeroed `previous_actions`
  - rebuild Brandão history via `build_agent_features` + `initialize_history`

  Keep the curriculum logic in the trainer, so `VSSEnv`, `VSSVecEnv` and `VSSJumanjiEnv` match
  semantics stay identical, as `CLAUDE.md` requires.
- **Joint architecture:** it does not terminate on goals. Define success as "blue scores first
  within N steps", or restrict the buffer to `terminate_on_goal` runs.

### Not recommended now

Full PLR⊥ with value-loss scores, ACCEL and PAIRED. They need more machinery, and SFL indicates the
payoff is doubtful.

## Evaluation protocol

- **Generalization gap:** evaluate with `init_mode=kickoff` vs `init_mode=random`
  (`src/vsss_sim/rl/evaluate.py:118`, `evaluate_experiment`).
- **Unseen opponents:** play lineage A vs lineage B snapshots. This is the key self-play overfitting
  signal and is not yet a built-in mode.
- **Sanity floors:** random and stationary opponents.
- **Ablation, 3 seeds each:**
  - baseline
  - baseline + domain randomisation (e.g. 25% `reset_random`, or wider kickoff jitter); DR is a strong baseline per SFL
  - + PFSP
  - + start buffer
  - + both

## Suggested skills

- `superpowers:brainstorming`: confirm with the user which overfitting mode they see and what scope to build.
- `superpowers:writing-plans`: turn the chosen item(s) into a step plan before touching `ppo.py`.
- `superpowers:test-driven-development`: unit-test `reset_from`, buffer sampling and score
  updates under `jit`/`vmap`.
- `superpowers:verification-before-completion`: run tests and a short smoke training before claiming results.

## User preferences

- Commit messages: one-line conventional commits, no body.
- Analysis reports normally go in chat, not new `.md` files. This file exists because the user
  explicitly asked for a committed handoff.
