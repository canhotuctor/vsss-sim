# Simulator structure

This document is a visual map of the simulator as it exists today. Read it top-to-bottom
for progressively more detail: public entry points, one environment step, both batched training
paths, data shapes, physics, and reset semantics.

> This is **IEEE VSSS 3v3**: six differential-drive robots, two wheel commands per robot,
> no kicker or dribbler, and a walled 150 x 130 cm field.

## 1. System map

The simulator exposes three paths over shared physics: a flexible single-match Gymnasium path,
a Gymnasium/SB3 compatibility path, and a fully JAX-native PPO path. The native path uses a
functional Jumanji contract so complete rollouts and optimizer generations can be compiled.

```mermaid
flowchart LR
    user["Policy / user code"]
    gym["Gymnasium registration<br/>VSSS-v0"]

    subgraph single["Single-match path"]
        env["VSSEnv<br/>Gymnasium Env"]
        base["VSSBaseEnv<br/>spaces + observation builder"]
        opponent["Opponent policy<br/>stationary / random / callable"]
        renderer["Pygame renderer<br/>human / rgb_array"]
        physics["JAX physics<br/>functional state, JIT"]

        env --> base
        env --> opponent
        env --> renderer
        env --> physics
    end

    subgraph batch["Compatibility training path"]
        adapter["VSSVecEnvToSB3<br/>API + autoreset adapter"]
        vec["VSSVecEnv<br/>Gymnasium VectorEnv"]
        adapter --> vec --> physics
    end

    subgraph native["Fully JAX-native training path"]
        trainjax["scripts/train_jax.py"]
        ppojax["Flax/Optax PPO<br/>nested lax.scan"]
        jumanji["VSSJumanjiEnv<br/>functional JAX state"]

        trainjax --> ppojax --> jumanji --> physics
    end

    sb3["Stable-Baselines3 PPO"]
    mlflow["MLflow metrics + artifacts"]

    user --> gym
    gym --> env
    gym --> vec
    sb3 --> adapter
    sb3 --> mlflow
    user --> trainjax
```

All three paths share the same functional JAX physics step. `VSSVecEnv` batches it with `vmap`
and compiles the result with `jit`. `VSSJumanjiEnv` represents one pure match; PPO applies
`vmap` for the environment batch and keeps the policy, rollout, GAE, gradients, and optimizer
state on the JAX device.

## 2. One control step in `VSSEnv`

The controlled policy only commands blue. The environment obtains yellow's commands from the
configured opponent, combines both teams, advances physics, and turns the result into the RL
transition.

```mermaid
sequenceDiagram
    participant P as Blue policy
    participant E as VSSEnv
    participant O as Opponent policy
    participant X as JAX physics engine
    participant R as Renderer

    P->>E: action (6,) in [-1, 1]
    E->>E: build current observation (46,)
    E->>O: current observation
    O-->>E: yellow wheels (3, 2)
    E->>E: reshape blue to (3, 2)<br/>stack all actions to (2, 3, 2)
    E->>X: step(state, all_actions)
    X-->>E: new state + goal event
    E->>E: update score and step count
    E->>E: reward = goal + 0.10 * delta ball-x<br/>+ goal-area crowding penalty
    E->>E: build obs, terminated, truncated, info
    opt goal scored
        E->>E: reset placement for in-match kickoff
    end
    opt rendering enabled
        E->>R: rgb_array after control step<br/>human after every physics sub-step
    end
    E-->>P: obs, reward, terminated, truncated, info
```

`terminated` is always false: a match has no absorbing mid-episode state. The time limit sets
`truncated`; goals update the score and trigger a kickoff without ending the episode.

## 3. Batched PPO data paths

### Gymnasium/SB3 compatibility path

The batched design keeps simulation state on the JAX device, but SB3/PyTorch consumes NumPy
arrays on the host. That boundary explains both the large physics speedup and the remaining
end-to-end overhead.

```mermaid
flowchart LR
    ppo["SB3 PPO<br/>PyTorch policy"]
    sb3api["VSSVecEnvToSB3<br/>SB3 VecEnv contract"]

    subgraph host["Python / host"]
        actions["Blue actions<br/>(B, 6)"]
        opponents["Built-in opponent actions<br/>(B, 3, 2)"]
        bookkeeping["Per-env counters<br/>rewards, done flags,<br/>episode statistics"]
        numpyobs["Writable NumPy observations<br/>(B, 46)"]
    end

    subgraph device["JAX device"]
        state["Batched SimState<br/>leading axis B"]
        vmapped["jit(vmap(step))"]
        obsbuilder["JAX observation builder"]
        state --> vmapped --> state
        state --> obsbuilder
    end

    ppo --> sb3api --> actions
    actions --> vmapped
    opponents --> vmapped
    vmapped --> bookkeeping
    obsbuilder -->|"device to host copy"| numpyobs
    bookkeeping --> sb3api
    numpyobs --> sb3api --> ppo
```

Built-in stationary and random opponents are assembled as batches. A custom callable receives
one observation at a time through a Python loop, so it is intentionally the slow path.

The adapter exists because Gymnasium and SB3 use different vector-environment contracts:

```mermaid
flowchart LR
    gymvec["VSSVecEnv<br/>reset -> obs, info<br/>step -> obs, reward, term, trunc, info<br/>NEXT_STEP autoreset"]
    adapter["VSSVecEnvToSB3<br/>merge term OR trunc<br/>build per-env infos<br/>track episode stats<br/>eagerly reset done envs"]
    sb3vec["SB3 VecEnv<br/>reset -> obs<br/>step -> obs, reward, done, infos<br/>terminal_observation in info"]

    gymvec --> adapter --> sb3vec
```

### Fully JAX-native path

The native path removes the per-step Python framework boundary. One compiled executable owns
the policy, environment batch, rollout buffer, advantage calculation, and optimizer updates.

```mermaid
flowchart LR
    config["Python configuration"] --> compile["XLA compile once"]

    subgraph device["One JAX executable"]
        generations["generation lax.scan"]
        rollout["rollout lax.scan"]
        policy["Flax actor-critic<br/>(B,46) -> (B,6) + value"]
        env["vmap(VSSJumanjiEnv.step)"]
        gae["reverse lax.scan<br/>truncation-aware GAE"]
        epochs["epoch + minibatch lax.scan"]
        optax["Optax gradients + Adam"]

        generations --> rollout
        rollout --> policy --> env --> rollout
        rollout --> gae --> epochs --> optax --> generations
    end

    compile --> generations
    generations --> metrics["Final metric history to host"]
```

The Gymnasium path remains useful for SB3 compatibility, rendering-adjacent workflows, and
external RL libraries. The Jumanji path is the high-throughput training implementation.

## 4. One fully compiled PPO generation

With the defaults, one generation collects `B x T = 256 x 128 = 32768` transitions and then
performs four PPO epochs over eight minibatches. The shapes remain static, allowing XLA to
compile the entire sequence.

```mermaid
sequenceDiagram
    participant S as RunnerState
    participant P as Flax policy/value
    participant E as VMAP Jumanji envs
    participant G as GAE
    participant O as Optax PPO update

    loop T rollout steps inside lax.scan
        S->>P: observations (B, 46)
        P-->>S: tanh actions, raw actions, log-probs, values
        S->>E: states + actions (B, 6)
        E-->>S: next states, rewards, discounts, LAST flags
        S->>S: lax.cond autoreset per ended match
    end
    S->>G: trajectory (T, B, ...)
    G-->>O: normalized advantages + value targets
    loop PPO epochs and minibatches inside nested lax.scan
        O->>O: clipped policy/value loss<br/>gradients + Adam update
    end
    O-->>S: next TrainState and scalar metrics
```

Time-limit truncation has two signals: `discount=1` bootstraps the value from the final
observation, while `LAST` stops GAE recursion before the reset episode. Goals are not terminal;
they update the score and trigger an in-match kickoff.

## 5. Data model and RL interface

The physics state is richer than the policy observation. Wheel speeds, score, and time remain
internal; positions and velocities are normalized into one flat policy vector.

```mermaid
flowchart TB
    subgraph input["Policy action: 6 float32 values"]
        a0["robot 0: left, right"]
        a1["robot 1: left, right"]
        a2["robot 2: left, right"]
    end

    stack["Add opponent actions<br/>all_actions shape = (2 teams, 3 robots, 2 wheels)"]

    subgraph simstate["SimState"]
        ball["ball (4,)<br/>x, y, vx, vy"]
        robots["robots (2, 3, 6)<br/>x, y, theta, vx, vy, omega"]
        wheels["wheel_speeds (2, 3, 2)<br/>last applied speeds"]
        score["score (2,)"]
        time["t scalar"]
    end

    subgraph observation["Policy observation: 46 float32 values"]
        oball["ball [0:4]<br/>normalized x, y, vx, vy"]
        oblue["blue robots [4:25]<br/>3 x 7 features"]
        oyellow["yellow robots [25:46]<br/>3 x 7 features"]
        features["per robot<br/>x, y, sin theta, cos theta, vx, vy, omega"]
    end

    a0 --> stack
    a1 --> stack
    a2 --> stack
    stack --> wheels
    wheels --> robots
    ball --> oball
    robots --> oblue
    robots --> oyellow
    features -.-> oblue
    features -.-> oyellow
```

For `VSSVecEnv`, every state and interface shape above gains a leading batch dimension `B`.
For native PPO, `jax.vmap` adds that dimension to `VSSJumanjiEnv.State`, while `lax.scan` adds
the rollout dimension `T` to stored transitions.

## 6. Physics step: 15 Hz control, 4 sub-steps

A policy selects target wheel speeds once per 1/15-second control step. The physics engine
splits that interval into four smaller steps for smoother acceleration and more reliable
collision handling.

```mermaid
flowchart TD
    command["Normalized wheel commands<br/>clip to [-1, 1]"]
    target["Scale by max wheel speed"]
    repeat{{"Repeat 4 physics sub-steps"}}
    slew["Slew current wheel speeds<br/>toward targets using acceleration limit"]
    drive["Differential-drive kinematics<br/>left/right wheels -> vx, vy, omega"]
    integrateRobots["Integrate robot position + heading"]
    friction["Apply rolling friction to ball"]
    integrateBall["Integrate ball position"]
    walls["Resolve robot-wall collisions"]
    robotChamfers["Resolve robot-corner chamfers"]
    ballwalls["Resolve ball-wall collisions<br/>and detect goals"]
    ballChamfers["Resolve ball-corner chamfers"]
    ballrobots["Resolve ball-robot collisions<br/>circle vs oriented box"]
    robotrobots["Resolve robot-robot collisions<br/>oriented box SAT"]
    done["Advance simulation time by 1/15 s<br/>return first goal event"]

    command --> target --> repeat
    repeat --> slew --> drive --> integrateRobots --> friction --> integrateBall
    integrateBall --> walls --> robotChamfers --> ballwalls --> ballChamfers --> ballrobots --> robotrobots
    robotrobots -->|"next sub-step"| repeat
    repeat -->|"after 4"| done
```

State is an immutable float32 `NamedTuple` PyTree. `step` returns a new state, uses
`lax.fori_loop` for substeps, and can be transformed directly with `jax.jit` and `jax.vmap`.

## 7. Match and reset lifecycle

There are two different resets: a goal creates a new kickoff *inside* the same match, while the
time limit ends the episode and causes a full reset.

```mermaid
stateDiagram-v2
    [*] --> EpisodeReset
    EpisodeReset --> Playing: kickoff or random placement<br/>step_count = 0
    Playing --> Playing: ordinary control step
    Playing --> InMatchKickoff: goal<br/>update score + emit reward
    InMatchKickoff --> Playing: new placement<br/>keep episode alive
    Playing --> Truncated: step_count reaches 300
    Truncated --> EpisodeReset: caller reset or autoreset
```

`InitMode.KICKOFF` uses the standard formation with jitter. `InitMode.RANDOM` samples the ball
and all robots across the field with random headings; overlaps are allowed and left for physics
to resolve.

### Current implementation boundaries

These details matter when comparing trajectories across execution paths:

- `VSSEnv` supports rendering; `VSSVecEnv` currently does not.
- Human rendering advances and draws each physics sub-step. Non-human stepping runs one engine
  call for the whole control step.
- Reset helpers return a fresh state whose score and simulation time are zero. The single-env
  in-match kickoff therefore resets those fields; the Jumanji path explicitly preserves them.
- Physics resolves the field's 45-degree corner chamfers for both robots and the ball.
- On a goal, single `VSSEnv` builds the returned observation before its kickoff reset. Batched
  `VSSVecEnv` performs the kickoff before exporting the returned observation.
- The single path calculates goal-area crowding from the post-physics state. On a goal, the
  batched path performs its kickoff first and then calculates crowding from the new placement.
- Gymnasium vector mode uses next-step autoreset after truncation. The SB3 adapter changes this
  to eager autoreset and preserves the terminal observation in `info`.
- `VSSJumanjiEnv` is functional and renderer-free. Its built-in stationary and random opponents
  stay in JAX; arbitrary Python opponent callables are intentionally unsupported in this path.
- The Jumanji goal kickoff explicitly preserves match score and simulation time. Native PPO
  autoresets ended episodes through `lax.cond` and bootstraps timeout values from the final
  observation before replacing it with the reset observation.

## 8. Source map

```mermaid
flowchart TD
    registration["src/vsss_sim/__init__.py<br/>VSSS-v0 registration"]
    config["config.py<br/>field, robot, timing, rewards, InitMode"]
    base["envs/base.py<br/>spaces + single observation encoding"]
    single["envs/vsss_3v3.py<br/>single-match orchestration"]
    vector["envs/vsss_vec.py<br/>batched orchestration + observation encoding"]
    jumanji["envs/jumanji.py<br/>pure state, TimeStep, reward + truncation semantics"]
    ppo["rl/ppo.py<br/>Flax actor-critic, GAE, PPO scans"]
    physicsapi["physics/__init__.py<br/>public physics API"]
    jax["physics/jax_backend.py<br/>JAX physics engine"]
    agents["agents/<br/>opponent policies"]
    render["rendering/pygame.py"]
    adapter["sb3_adapter.py"]
    trainjax["scripts/train_jax.py<br/>compile, execute, benchmark"]
    scripts["scripts/<br/>Gymnasium/SB3 training, benchmark, visualize"]
    tests["tests/<br/>env, adapter, physics, CUDA, reset, PPO"]
    report["docs/jax-native-ppo-report-2026-08.md"]

    registration --> single
    registration --> vector
    config --> jumanji
    config --> base
    config --> single
    config --> vector
    config --> jax
    base --> single
    agents --> single
    single --> physicsapi --> jax
    vector --> jax
    jumanji --> jax
    ppo --> jumanji
    trainjax --> ppo
    single --> render
    adapter --> vector
    scripts --> registration
    tests -. verify .-> single
    tests -. verify .-> vector
    tests -. verify .-> jumanji
    tests -. verify .-> ppo
    tests -. verify .-> jax
    report -. documents .-> ppo
```
