# Simulator structure

This document is a visual map of the simulator as it exists today. Read it top-to-bottom
for progressively more detail: public entry points, one environment step, the Gymnasium vector
interface, native PPO data flow, data shapes, physics, reset semantics, and referee rules.

> This is **IEEE VSSS 3v3**: six differential-drive robots, two wheel commands per robot,
> no kicker or dribbler, and a walled 150 x 130 cm field.

## 1. System map

The simulator provides Gymnasium single-match and batched interfaces for interactive use and
external clients, plus a fully JAX-native PPO path for training. All three use the same JAX
physics implementation; native PPO runs the policy, environment, rollout, and optimizer on device.

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

    subgraph vector["Batched Gymnasium path"]
        vec["VSSVecEnv<br/>Gymnasium VectorEnv"]
        vec --> physics
    end

    subgraph native["Native JAX PPO path"]
        trainjax["scripts/train_jax.py"]
        ppojax["Flax/Optax PPO<br/>nested lax.scan"]
        jumanji["VSSJumanjiEnv<br/>functional JAX state"]
        trainjax --> ppojax --> jumanji --> physics
    end

    user --> gym
    gym --> env
    gym --> vec
    user --> trainjax
```

`VSSEnv` provides rendering and Python opponent callables. `VSSVecEnv` batches physics with
`vmap` and exposes the standard Gymnasium VectorEnv interface, including NumPy observations.
`VSSJumanjiEnv` represents a pure match; PPO batches it with `vmap` and keeps policy, rollout,
GAE, gradients, and optimizer state on the JAX device.

## 2. One control step in `VSSEnv`

The controlled policy only commands blue. The environment obtains yellow's commands from the
configured opponent, combines both teams, advances physics, and delegates reward, score, kickoff,
referee, restart, and observation semantics to the shared `MatchRules` module.

```mermaid
sequenceDiagram
    participant P as Blue policy
    participant E as VSSEnv
    participant O as Opponent policy
    participant X as JAX physics engine
    participant M as Shared MatchRules
    participant R as Renderer

    P->>E: action (6,) in [-1, 1]
    E->>E: build current observation (46,)
    E->>O: current observation
    O-->>E: yellow wheels (3, 2)
    E->>E: reshape blue to (3, 2)<br/>stack all actions to (2, 3, 2)
    opt match is STOPPED after a whistle
        E->>E: zero all actions
    end
    E->>X: step(state, all_actions)
    X-->>E: post-physics state + goal event
    E->>E: increment step count
    E->>M: MatchStepInput(before, after, goal, step count, kickoff key)
    M->>M: apply configured reward profile
    M->>M: resolve referee events and timers
    M->>M: update score and optionally reset kickoff placement
    M->>M: place restart formation after a whistle
    M-->>E: carried state, observation, reward, referee state
    E->>E: package terminated, truncated, info
    opt rendering enabled
        E->>R: rgb_array after control step<br/>human after every physics sub-step
    end
    E-->>P: obs, reward, terminated, truncated, info
```

In `VSSEnv`, `terminated` is always false: a match has no absorbing mid-episode state. The time
limit sets `truncated`; goals update the score and trigger a kickoff without ending the episode.
`VSSJumanjiEnv` can instead end the episode on a goal (`terminate_on_goal`), which the Brandão
policies use by default.

## 3. Batched Gymnasium path

`VSSVecEnv` keeps the batched simulation state on the JAX device and exposes observations and
step results through Gymnasium's VectorEnv contract. This interface is useful for rendering-
adjacent tooling and external clients that consume Gymnasium directly. Native PPO uses
`VSSJumanjiEnv` instead, avoiding per-step host transfers while keeping the same match rules.

```mermaid
flowchart LR
    actions["Blue actions<br/>(B, 6) in NumPy"]
    opponents["Built-in opponent actions<br/>(B, 3, 2)"]

    subgraph device["JAX device"]
        state["Batched SimState<br/>leading axis B"]
        step["jit(vmap(step))"]
        observation["JAX observation builder"]
        state --> step --> state
        state --> observation
    end

    exported["Writable NumPy observations<br/>(B, 46)"]
    actions --> step
    opponents --> step
    observation -->|"device to host copy"| exported
```

Stationary and random opponents are assembled as batches. A custom callable receives one
observation at a time through a Python loop, so it is intended for development rather than
maximum training throughput. Gymnasium vector mode uses next-step autoreset after truncation.

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
observation, while `LAST` stops GAE recursion before the reset episode. With the joint policy,
goals are not terminal; they update the score and trigger an in-match kickoff. The Brandão
per-robot policies terminate on a goal and use `(B, 3, 424)` observations, `(B, 3, 2)` actions,
and one reward per robot.

## 5. Data model and RL interface

The physics state is richer than the policy observation. Wheel speeds, ball spin and height,
score, and time remain internal; positions and velocities are normalized into one flat policy
vector.

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
        spin["ball_spin (3,)<br/>angular velocity wx, wy, wz"]
        vertical["ball_vertical (2,)<br/>height z, vertical velocity vz"]
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

## 6. Physics step: 10 Hz control, 6 sub-steps

A policy selects target wheel speeds once per 1/10-second control step. The physics engine
splits that interval into six smaller steps (60 Hz physics) for smoother acceleration and more
reliable collision handling. Every policy architecture uses these defaults. Environments and
experiment configurations set both values explicitly with `control_rate_hz` and
`physics_substeps`.

```mermaid
flowchart TD
    command["Normalized wheel commands<br/>clip to [-1, 1]"]
    target["Scale by max wheel speed"]
    repeat{{"Repeat 6 physics sub-steps"}}
    slew["Slew current wheel speeds<br/>toward targets using acceleration limit"]
    drive["Differential-drive kinematics<br/>left/right wheels -> vx, vy, omega"]
    integrateRobots["Integrate robot position + heading"]
    friction["Ball floor contact: skid until rolling,<br/>rolling resistance, or flight + bounce"]
    integrateBall["Integrate ball position"]
    walls["Resolve robot-wall collisions"]
    robotChamfers["Resolve robot-corner chamfers"]
    ballwalls["Resolve ball-wall collisions<br/>with friction, detect goals"]
    ballChamfers["Resolve ball-corner chamfers"]
    ballrobots["Resolve ball-robot collisions<br/>circle vs oriented box, with friction"]
    robotrobots["Resolve robot-robot collisions<br/>oriented box SAT"]
    done["Advance simulation time by 1/10 s<br/>return first goal event"]

    command --> target --> repeat
    repeat --> slew --> drive --> integrateRobots --> friction --> integrateBall
    integrateBall --> walls --> robotChamfers --> ballwalls --> ballChamfers --> ballrobots --> robotrobots
    robotrobots -->|"next sub-step"| repeat
    repeat -->|"after 6"| done
```

State is an immutable float32 `NamedTuple` PyTree. `step` returns a new state, uses
`lax.fori_loop` for substeps, and can be transformed directly with `jax.jit` and `jax.vmap`.
`step_with_frames` runs the same sub-steps with `lax.scan` and also returns the state after each
one, which human rendering draws.

### Ball spin, skidding, and pop-ups

The ball is a 3D sphere (moment of inertia `0.4·m·r²`) moving over the 2D field. Its spin and
height are tracked, but robots and walls stay planar. Coefficients in `config.py` describe a golf
ball on the rulebook's wooden board, which has a ping-pong-table finish, against painted wooden
walls and PLA robot covers; their comments cite the measurements behind them.

- **Skidding and rolling.** While the floor contact point slips, sliding friction
  (`BALL_SLIDING_FRICTION`) slows the ball and spins it up until it rolls. A robot face strikes
  the ball at its centre height and adds no topspin, so every kick starts a skid that lasts
  longer the harder the hit; it ends rolling at 5/7 of the kicked speed. A rolling ball slows by
  rolling resistance (`BALL_FRICTION`).
- **Wall and robot friction.** Walls and robot faces apply friction (`BALL_WALL_FRICTION`,
  `BALL_ROBOT_FRICTION`) as well as restitution. A rolling ball's front surface moves down, so
  friction pushes it up: it pops up about 3 mm at 1 m/s and 1.5 cm at 2 m/s. A skidding ball has
  no topspin and does not pop up. Glancing hits trade sideways speed for spin about the vertical
  axis, which the floor slowly removes (`BALL_SPIN_DECEL`). Robot faces move with their robot,
  including its spin.
- **Flight.** A ball in the air follows its exact ballistic arc, ignores floor friction, and
  bounces on landing (`BALL_FLOOR_RESTITUTION`) until a bounce cannot outlast one sub-step. It
  still hits walls and robots, which are taller than any pop-up. Rendering draws its shadow on the
  floor and the ball raised, with height exaggerated 3x.

Friction can at most stop the slip at a contact, so contacts never add energy. Restarts and
kickoffs place the ball with no spin or height.

## 7. Match and reset lifecycle

There are two different resets: a goal creates a new kickoff *inside* the same match, while the
time limit ends the episode and causes a full reset. When `terminate_on_goal` is set, a goal ends
the episode instead. Referee whistles stop play without resetting the episode.

```mermaid
stateDiagram-v2
    [*] --> EpisodeReset
    EpisodeReset --> Playing: kickoff or random placement<br/>step_count = 0
    Playing --> Playing: ordinary control step
    Playing --> InMatchKickoff: goal<br/>update score + emit reward
    InMatchKickoff --> Playing: new placement<br/>keep episode alive
    Playing --> Terminated: goal with terminate_on_goal
    Playing --> Stopped: referee whistle<br/>place restart formation
    Stopped --> Playing: one frozen control step<br/>actions zeroed, physics discarded
    Playing --> Truncated: step_count reaches max_episode_steps
    Terminated --> EpisodeReset: caller reset or autoreset
    Truncated --> EpisodeReset: caller reset or autoreset
```

`max_episode_steps` is 200 for the joint policy (20 s) and 400 for the Brandão policies (40 s).

`InitMode.KICKOFF` uses the standard formation with jitter. `InitMode.RANDOM` samples the ball
and all robots across the field with random headings; overlaps are allowed and left for physics
to resolve.

### Current implementation boundaries

These details matter when comparing trajectories across execution paths:

- `VSSEnv` supports rendering; `VSSVecEnv` currently does not.
- Every mode runs one engine call per control step. Human rendering uses `step_with_frames`
  to draw each physics sub-step of that call, paced against wall-clock deadlines so a slow
  frame shortens the following ones rather than slowing playback.
- A goal kickoff creates fresh placement but preserves match score and simulation time in all
  environment adapters. Episode step counts continue through in-match kickoffs.
- Shared `MatchRules` computes reward from the post-physics state before kickoff, then returns an
  observation of the state carried forward after any kickoff. Single, vector, and Jumanji paths
  use this same ordering.
- Physics resolves the field's 45-degree corner chamfers for both robots and the ball.
- The policy observation omits ball spin and height; policies see only the ball's 2D position
  and velocity.
- `VSSVecEnv` uses Gymnasium next-step autoreset after truncation.
- `VSSJumanjiEnv` is functional and renderer-free. Its built-in stationary and random opponents
  stay in JAX; arbitrary Python opponent callables are intentionally unsupported in this path.
- Native PPO autoresets ended episodes through `lax.cond` and bootstraps timeout values from the
  final observation before replacing it with the reset observation.

### Referee events and restarts

`match_referee.py` infers rulebook calls from the 2D state before and after physics. It carries
per-match timers in `RefereeState`, and the PPO metrics count each event per rollout. Every rule
timer is 5 s, and a ball at or below 0.1 m/s counts as stuck (VSSReferee's threshold). When
several calls apply in the same step, the first row below wins.

| Event | Condition | Restart |
|---|---|---|
| `GOAL` | ball crosses a goal line | kickoff for the conceding team, or episode end |
| `DEFENDER_GOAL_AREA` | a second defender takes part in play inside its own goal area; the attackers play on until the attack ends without a goal, 5 s pass, or another call stops play | penalty kick for the attackers |
| `ATTACKER_GOAL_AREA` | two attackers in the opponent's goal area during a contested ball | goal kick for the defenders |
| `GOALKEEPER_BLOCK` | an attacker in the goal area commits a personal foul (next row) on the goalkeeper; ordinary contact with the goalkeeper plays on | goal kick for the defenders |
| `GOALKEEPER_TIMEOUT` | goalkeeper holds the ball uncontested for 5 s | penalty kick for the attackers |
| `PERSONAL_FOUL` | an avoidable, one-sided collision anywhere: a fresh impact at 0.6 m/s or more on a robot within 20 cm of the ball, or at 0.9 m/s or more anywhere, or a push or pin lasting 1 s; never during a ball dispute | free kick for the fouled team |
| `GOAL_AREA_IMPASSE` | contested ball stuck in a goal area for 5 s | goal kick for the area's team |
| `GOAL_AREA_STOPPED_BALL` | uncontested ball stuck in a goal area for 5 s | penalty kick against the area's team |
| `FIELD_IMPASSE` | ball stuck outside the goal areas for 5 s | free ball in the ball's quadrant |

A personal foul also needs the victim to have spent 0.2 s in the offender's forward lane, so a
robot that cuts across a driving opponent cannot draw it. Drive speeds come from wheel speeds.

`RefereeMode` selects which calls apply. `full` (default) uses every row above. `simplified`
keeps only the two-robot goal-area fouls and frees a ball stuck anywhere, including inside a goal
area, with `FIELD_IMPASSE`. `off` stops play only for goals. All adapters and experiment
configurations accept `referee_mode`.

`match_restarts.py` places deterministic formations for each restart. A free kick is taken from
the TL mark, which is also the penalty mark. Its formation is spread so no robot starts within
10 cm of the ball, any two robots start at least one robot diagonal apart, and every robot is
inside the walls and corner chamfers. After a whistle the match is `STOPPED` for one control step: all wheel commands are zeroed, that
step's physics result is discarded, and reward is zero.

Under the Brandão reward profile, fouls (both goal-area violations, goalkeeper timeout, personal
foul, goalkeeper block, and goal-area stopped ball) replace the offending team's step reward
with −1. The default profile instead adds −1 when the team concedes a penalty kick. Only the
offender is penalized: the fouled team keeps its ordinary reward, since the restart is its
gain. A defender foul's reward arrives when its penalty is awarded, after the advantage.

## 8. Source map

```mermaid
flowchart TD
    registration["src/vsss_sim/__init__.py<br/>VSSS-v0 registration"]
    config["config.py<br/>field, robot, timing, rewards, InitMode"]
    base["envs/base.py<br/>spaces + single observation encoding"]
    single["envs/vsss_3v3.py<br/>single-match orchestration"]
    vector["envs/vsss_vec.py<br/>batched orchestration + observation encoding"]
    jumanji["envs/jumanji.py<br/>pure state, TimeStep, reward + truncation semantics"]
    matchrules["envs/match_rules.py<br/>shared step: reward, score, kickoff, referee, observation"]
    referee["envs/match_referee.py<br/>events, timers, RefereeState"]
    restarts["envs/match_restarts.py<br/>restart formations"]
    geometry["envs/match_geometry.py<br/>goal-area and contact geometry"]
    brandaoobs["envs/brandao_observation.py<br/>per-robot features + 8-frame history"]
    selfplay["rl/self_play.py<br/>experiments, lineages, snapshot archive"]
    evaluate["rl/evaluate.py<br/>historical snapshot evaluation"]
    ppo["rl/ppo.py<br/>Flax actor-critic, GAE, PPO scans"]
    physicsapi["physics/__init__.py<br/>public physics API"]
    jax["physics/jax_backend.py<br/>JAX physics engine"]
    agents["agents/<br/>opponent policies"]
    render["rendering/pygame.py"]
    trainjax["scripts/train_jax.py<br/>compile, execute, benchmark"]
    scripts["scripts/<br/>JAX training, benchmark, visualize"]
    tests["tests/<br/>env, physics, CUDA, reset, PPO"]
    report["docs/performance-memory-report.md"]

    registration --> single
    registration --> vector
    config --> jumanji
    matchrules --> single
    matchrules --> vector
    matchrules --> jumanji
    referee --> matchrules
    restarts --> matchrules
    geometry --> referee
    brandaoobs --> jumanji
    selfplay --> ppo
    evaluate --> selfplay
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
    trainjax --> selfplay
    single --> render
    scripts --> registration
    tests -. verify .-> single
    tests -. verify .-> vector
    tests -. verify .-> jumanji
    tests -. verify .-> ppo
    tests -. verify .-> jax
    report -. documents .-> selfplay
```
