# JAX-native PPO implementation and benchmark report

Date: 2026-08-19

Machine: Apple M1 MacBook Air, 8 CPU cores (4 performance + 4 efficiency),
8 GB RAM, JAX CPU backend

## Scope

This change closes the JAX training loop for the centralized VSSS team policy.
The native path keeps the functional Jumanji environment, Flax policy, rollout,
GAE, PPO losses, gradients, and Optax state inside JAX. Complete generations
run in a compiled outer `jax.lax.scan`; Python is used only for setup,
compilation, executable launch, timing, and final metric reads.

This report compares that path with the compatibility path:

```text
native:         Jumanji environment -> Flax PPO -> Optax
compatibility:  VSSVecEnv -> Gymnasium/SB3 adapter -> PyTorch PPO
```

All results below were remeasured on the M1. The earlier version of this report
incorrectly identified the current machine as the M4 Pro used for the thesis
measurements. Consequently, the old absolute comparison was not controlled.

## Timing semantics: 60 Hz physics is not 60 Hz PPO control

The simulator advances at two rates:

- physics: 60 Hz;
- policy/control: 15 Hz; and
- physics substeps per policy action: 4.

Therefore, five simulated seconds contain 300 raw physics frames but only 75
PPO transitions. A PPO `rollout_length=300` contains 1,200 raw physics frames
and represents 20 simulated seconds per environment.

Both interpretations were measured:

- `T=75`: the physically correct five-second rollout; and
- `T=300`: the requested literal 300-transition rollout.

## Network and hyperparameter audit

### Neural networks

The networks are exactly the same size. Runtime inspection found 14,797
trainable parameters in both implementations.

| Component | Native Flax PPO | SB3 `MlpPolicy` |
|---|---|---|
| Observation | 46 floats | 46 floats |
| Actor | 46 -> 64 -> 64 -> 6 | 46 -> 64 -> 64 -> 6 |
| Critic | 46 -> 64 -> 64 -> 1 | 46 -> 64 -> 64 -> 1 |
| Hidden activation | tanh | tanh |
| Actor/critic sharing | separate towers | separate towers |
| Action scale | 6 learned log standard deviations | 6 learned log standard deviations |
| Initialization | orthogonal; hidden gain sqrt(2), actor 0.01, critic 1.0 | same |
| Parameter count | 14,797 | 14,797 |

Network size does not explain the limited speedup.

### Controlled benchmark values

The following numerical values were set explicitly for both trainers.

| Setting | Five-second run | 300-transition run |
|---|---:|---:|
| Parallel environments `B` | 256 | 256 |
| Rollout transitions per environment `T` | 75 | 300 |
| Samples per generation `B x T` | 19,200 | 76,800 |
| PPO epochs | 4 | 4 |
| Minibatches per epoch | 8 | 8 |
| Samples per minibatch | 2,400 | 9,600 |
| Measured generations | 10 | 10 |
| Learning rate | 3e-4 | 3e-4 |
| Discount `gamma` | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Policy clip | 0.2 | 0.2 |
| Value coefficient | 0.5 | 0.5 |
| Entropy coefficient | 0.01 | 0.01 |
| Maximum gradient norm | 0.5 | 0.5 |
| Adam epsilon | 1e-5 | 1e-5 |
| Adam beta values | 0.9, 0.999 | 0.9, 0.999 |
| Opponent / placement | stationary / kickoff | stationary / kickoff |
| Seed | 0 | 0 |
| Device | CPU | CPU |

Each path performs exactly 32 gradient steps per generation: four epochs times
eight minibatches.

### Remaining algorithmic differences

The numerical settings and network sizes match, but the implementations are
not yet mathematically identical:

| Behavior | Native implementation | SB3 implementation |
|---|---|---|
| Action bounds | samples a Gaussian, applies `tanh`, stores raw Gaussian action | samples a Gaussian, clips only the action sent to the environment |
| Advantage normalization | once over the complete 76,800-sample batch | independently inside every minibatch |
| Value clipping | maximum of clipped and unclipped squared losses | MSE of the clipped prediction when `clip_range_vf` is enabled |
| Timeout bootstrap | evaluates the complete actor-critic on every next observation | evaluates the final observation and additional terminal observations only |
| Rollout execution | one compiled JAX scan | Python loop calling PyTorch and the vector environment |

For this rerun, SB3's `clip_range_vf` was explicitly set to 0.2 so the numeric
threshold matched the native configuration. Its loss formula still differs as
shown above. These differences matter for learning equivalence and some of them
also affect latency. The results should therefore be called a controlled
performance comparison, not bit-for-bit PPO equivalence.

## Remeasured end-to-end results on the M1

Compilation and the one-generation SB3 warm-up are excluded from steady-state
throughput. All JAX timings synchronize with `jax.block_until_ready`.

| Rollout | Path | Total steps | Execution | Throughput | Native speedup |
|---|---|---:|---:|---:|---:|
| 75 control steps / 5 simulated seconds | JAX native | 192,000 | 4.006 s | 47,926 steps/s | 1.52x |
| 75 control steps / 5 simulated seconds | VSSVecEnv + SB3 | 192,000 | 6.080 s | 31,578 steps/s | baseline |
| 300 control steps / 20 simulated seconds | JAX native | 768,000 | 13.622 s | 56,381 steps/s | 1.36x |
| 300 control steps / 20 simulated seconds | VSSVecEnv + SB3 | 768,000 | 18.459 s | 41,605 steps/s | baseline |

For the actual five-second rollout, native JAX supplies 51.8% more steps per
second and reduces time by 34.1%. For the longer rollout it supplies 35.5% more
steps per second and reduces time by 26.2%.

The gain becomes smaller at `T=300` because the much larger 9,600-sample
minibatch makes the small PyTorch network efficient, while fixed Python and
Gymnasium costs are amortized across more samples.

## Latency decomposition for the 300-transition run

One generation contains 76,800 PPO transitions and 307,200 raw physics
substeps. The tables below combine directly measured complete stages with
standalone microbenchmarks. Independently compiled stages do not add perfectly
because the full JAX program can fuse operations and avoid some intermediate
materialization.

### Native JAX path

| Stage | Median latency | Increment over preceding stage | Share of 1.358 s generation |
|---|---:|---:|---:|
| Raw physics scan | 705.4 ms | 705.4 ms | 51.9% |
| Functional Jumanji environment scan | 816.6 ms | 111.3 ms | 60.1% cumulative |
| Complete rollout collection | 1,073.9 ms | 257.2 ms | 79.1% cumulative |
| GAE only | 0.12 ms | 0.12 ms | less than 0.01% |
| Four-epoch PPO optimization | 300.9 ms | 300.9 ms | 22.2% |
| Complete compiled generation | 1,358.2 ms | n/a | 100% |

The ten-generation outer scan averaged 1,362.2 ms per generation. Its small
difference from the isolated 1,358.2 ms median is normal run-to-run variation.

#### 1. Raw physics: 705.4 ms

This is the irreducible shared simulation work: four substeps per control
transition, wheel acceleration, differential-drive integration, friction,
walls and corner chamfers, ball-robot collisions, robot-robot SAT collisions,
and goal detection. The scan delivered 108,878 control transitions/s.

Because raw physics alone consumes about 52% of a native generation, even a
zero-cost policy, rollout buffer, and optimizer could improve the current
native result by no more than about 1.93x.

#### 2. Environment logic: another 111.3 ms

The difference between raw physics and `VSSJumanjiEnv.step` covers action
assembly, opponent actions, normalized 46-value observations, reward shaping,
crowding penalties, score and step counters, goal kickoffs, truncation flags,
and Jumanji `TimeStep` construction. Keeping these operations in JAX limits the
increment to roughly 8% of the full generation.

#### 3. Policy inference and rollout collection: another 257.2 ms

The complete rollout adds Gaussian sampling, log probabilities, actor and
critic inference, PRNG splitting, timeout bootstrapping, autoreset, episode
metrics, and materialization of the `(T, B, ...)` trajectory.

Standalone measurements over changing observations took:

- 59.8 ms for the sampled actor-critic rollout forward passes; and
- 14.9 ms for the second complete actor-critic evaluation used to obtain
  bootstrap values.

The remainder is primarily PRNG/control flow and writing approximately 76,800
observations, raw actions, values, log probabilities, rewards, flags, and
metrics into the rollout PyTree. These standalone kernels are explanatory and
must not be added mechanically to the fused generation timing.

The second actor-critic evaluation is avoidable. Standard rollout storage can
reuse the following transition's critic value and evaluate only true timeout
terminal observations plus the final rollout observation.

#### 4. GAE: 0.12 ms

The reverse `lax.scan` that computes deltas, generalized advantages, and value
targets is negligible. GAE is not the performance bottleneck.

#### 5. Policy update: 300.9 ms

Each generation shuffles 76,800 samples four times and executes 32 gradient
steps. Every step performs actor and critic forward passes, clipped policy and
value losses, reverse-mode differentiation, global-gradient clipping, and an
Adam update. This consumes roughly 22% of the native generation.

### VSSVecEnv + SB3 path

| Component | Approximate latency per generation | Share of 1.846 s generation |
|---|---:|---:|
| Shared raw physics | 705 ms | 38.2% |
| `VSSVecEnv` observation/reward/host boundary | 360 ms | 19.5% |
| PyTorch policy forward passes | 91 ms | 4.9% |
| SB3 rollout loop, sampling, clipping, buffer and callbacks | 348 ms | 18.9% |
| PyTorch PPO optimization | 329 ms | 17.8% |
| Residual/timing composition | about 13 ms | 0.7% |

The supporting microbenchmarks at batch 256 were:

- raw `jit(vmap(physics.step))`: 108,856 transitions/s;
- `VSSVecEnv.step`: 72,089 transitions/s; and
- pure SB3 policy inference: 845,874 observations/s.

The wrapper figure includes building NumPy actions, JAX conversion, physics,
device synchronization, host-side scoring and counters, observation export,
and writable NumPy copies. The SB3 rollout then adds observation-to-tensor
conversion, Gaussian sampling, action clipping, rollout-buffer writes,
timeout handling, and Python callback dispatch.

The decomposition is approximate because the wrapper microbenchmark uses zero
actions while a PPO rollout samples actions and may trigger different goal and
reset paths. The directly measured aggregate values are 1,504 ms for SB3
rollout collection, 329 ms for its update, and 1,846 ms per complete
generation.

## Why the relative gain is not larger

1. **Both paths already share batched JAX physics.** Native PPO removes the
   framework boundary; it does not remove the 705 ms physics scan.
2. **The comparison is CPU-to-CPU.** JAX, NumPy, and PyTorch all use host
   memory on this M1. There is no PCIe transfer to eliminate, unlike the
   expected RTX 3060 case.
3. **The learner is small and its minibatches are large.** With only 14,797
   parameters and 9,600 samples per minibatch, PyTorch needs 329 ms for the
   update while JAX needs 301 ms: only a 1.09x update-stage advantage.
4. **The compatibility environment was already optimized.** `VSSVecEnv`
   performs one VMAP'd physics launch for all 256 matches; it is not equivalent
   to 256 Python Gymnasium environments.
5. **The current native rollout performs redundant work.** It evaluates the
   full actor-critic again on every next observation for bootstrap values,
   even though most values can be shifted from the next transition.
6. **Longer rollouts favor SB3 amortization.** Moving from 75 to 300 transitions
   raises SB3 throughput by 31.8%, versus 17.6% for native JAX, reducing the
   speedup from 1.52x to 1.36x.

A useful Amdahl bound is the ratio between the complete SB3 generation and raw
physics: `1.846 / 0.705 = 2.62x`. No implementation that preserves the current
physics can exceed that speedup against this SB3 run. The measured 1.36x is
therefore about 43% of the way from the SB3 time to the raw-physics floor when
measured as eliminated latency.

## Correct comparison with the thesis

The thesis measurements were made on an Apple M4 Pro, whereas these runs were
made on an 8 GB M1 MacBook Air. The machines must not be compared as if they
were a before/after software benchmark.

| Measurement | Thesis M4 Pro | Current M1 |
|---|---:|---:|
| Raw physics, batch 256 | 298,564 steps/s | 108,856 steps/s |
| `VSSVecEnv`, batch 256 | 160,353 steps/s | 72,089 steps/s |
| End-to-end SB3, selected setup | 49,285 steps/s | 41,605 steps/s in the controlled `T=300` run |
| Native JAX PPO | not implemented in TG-1 | 56,381 steps/s at `T=300` |

Hardware is now a major confound for the 2.7x raw-physics difference.
Corner-chamfer physics, goal-area bookkeeping, dependency versions, rollout
length, and timing method have also changed, but their individual effects
cannot be inferred from this cross-machine comparison. The earlier report's
claim that the raw kernel itself had regressed was not supported.

The thesis's headline two-to-three-order-of-magnitude gain compares the modern
batched system with legacy ITAndroids ODE training at 63--378 steps/s. It does
not predict an order-of-magnitude native-JAX-over-SB3 gain. The new native
result remains approximately 149x faster than 378 steps/s and 895x faster than
63 steps/s.

A controlled historical comparison requires either running the current commit
on the original M4 Pro or running the thesis commit and current commit on this
same M1 with one dependency lock and benchmark script.

## Recommended next optimizations

In priority order:

1. remove the full per-transition bootstrap actor-critic pass and reuse shifted
   critic values;
2. add a critic-only evaluation path for rollout-final and timeout states;
3. make action bounding, advantage normalization, and value-loss semantics
   configurable so an exact SB3-equivalent mode can be benchmarked;
4. profile trajectory writes and PRNG/control-flow overhead inside the compiled
   rollout; and
5. repeat the controlled comparison on the RTX 3060, where keeping the learner
   and simulator on the same device should provide a larger relative gain.

## Verification

- Focused Jumanji/PPO tests: 11 passed.
- Full suite at implementation time: 185 passed, 4 CUDA-only tests skipped.
- Runtime versions for this rerun: JAX 0.11.1, Flax 0.12.9, Optax 0.2.8,
  PyTorch 2.13.0, Gymnasium 1.3.0, SB3 2.9.0, and Jumanji 1.1.2.
