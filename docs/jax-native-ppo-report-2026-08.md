# JAX-native PPO implementation and benchmark report

Date: 2026-08-19

Machine: Apple M4 Pro, JAX CPU backend

## Scope

This change closes the JAX training loop for the centralized VSSS team policy.
The new path no longer passes each transition through Gymnasium, NumPy,
Stable-Baselines3, or PyTorch. It consists of:

- `VSSJumanjiEnv`: a functional JAX environment with on-device observations,
  rewards, opponents, goal kickoffs, counters, and truncation;
- a Flax actor-critic with six tanh-bounded wheel commands and one value head;
- VMAP execution across matches;
- rollout collection and autoreset through `jax.lax.scan`;
- truncation-aware generalized advantage estimation (GAE);
- clipped PPO policy/value losses and Optax updates; and
- an outer `lax.scan` over complete PPO generations, compiled as one XLA
  executable by `scripts/train_jax.py`.

Only setup, XLA compilation, executable launch, timing, and final metric reads
remain in Python.

## Verification

- Full test suite: 185 passed, 4 CUDA-only tests skipped on the Mac.
- PPO-specific tests cover bounded actions, parameter updates, timestep counts,
  multi-generation compilation, and correct value bootstrapping at a time-limit
  truncation without leaking GAE into the reset episode.

## Measurements

Compilation is excluded from steady execution throughput. Every timing calls
`jax.block_until_ready` before stopping the clock.

### Small fully JAX smoke benchmark

| Setting | Value |
|---|---:|
| Parallel environments | 8 |
| Rollout length | 32 |
| PPO generations | 5 |
| PPO epochs / minibatches | 4 / 8 |
| Total environment steps | 1,280 |
| Initialization | 0.830 s |
| XLA compilation | 1.701 s |
| Device execution | 0.043 s |
| End-to-end throughput | 29,434 steps/s |

This workload is intentionally too small for peak throughput; fixed launch and
learner costs are only weakly amortized.

### Matched end-to-end PPO comparison

Both paths used 256 environments, 128 rollout steps, 10 measured generations,
four PPO epochs, eight minibatches of 4,096 transitions, 64x64 actor/critic
networks, and matching learning rate, GAE, clipping, and entropy settings. The
SB3 path received one warm-up generation before measurement.

| Training path | Steps | Execution | Throughput |
|---|---:|---:|---:|
| Fully JAX: Jumanji + Flax + Optax | 327,680 | 5.958 s | 55,002 steps/s |
| Compatibility: VSSVecEnv + SB3/PyTorch | 327,680 | 9.387 s | 34,906 steps/s |

The fully JAX path is 1.58x faster, supplies 57.6% more steps per second, and
reduces measured execution time by 36.5%.

### Current simulator ceilings at batch 256

A separate 1,000-step diagnostic used the current repository code with zero
actions after warm-up:

| Path | Throughput | Fraction of current raw ceiling |
|---|---:|---:|
| Raw `jit(vmap(physics.step))` | 106,876 steps/s | 1.00 |
| Gymnasium `VSSVecEnv` only | 67,553 steps/s | 0.63 |
| Fully JAX PPO | 55,002 steps/s | 0.51 |
| VSSVecEnv + SB3 PPO | 34,906 steps/s | 0.33 |

These component measurements use a simpler timing method than the thesis
(total chained execution rather than per-call trimmed statistics), so they are
diagnostic rather than replacement thesis measurements.

## Why the gain over VSSVecEnv + SB3 is 1.58x, not an order of magnitude

The thesis reports two different comparisons that are easy to conflate.

1. The headline **two-to-three-order-of-magnitude gain** compares the modern
   batched training stack (49,285 steps/s on the M4 Pro in Chapter 5) with old
   ITAndroids ODE training at 63--378 steps/s. It does not compare JAX-native
   PPO against SB3. At 55,002 steps/s, the new trainer is still approximately
   145x faster than 378 steps/s and 873x faster than 63 steps/s, fully consistent
   with that headline claim.
2. The thesis's approximately 298,564 steps/s at batch 256 is a **raw physics
   ceiling**. It excludes observation construction, policy inference, stochastic
   action sampling, rollout storage, GAE, minibatch shuffling, backpropagation,
   and optimizer updates. End-to-end PPO cannot be compared directly with it.

Several factors limit the incremental speedup on the present Mac:

- Both compared PPO systems already use the same VMAP'd JAX physics. The new
  path replaces only the wrapper/learner portion, not the whole workload.
- On an M4 CPU, JAX arrays and PyTorch/NumPy arrays use host memory. There is no
  PCIe device-to-host penalty. The removed boundary consists mainly of Python
  dispatch, array conversions, and framework overhead. The expected advantage
  should be larger on the RTX 3060, where the SB3 path crosses GPU/CPU memory.
- A batch of 256 amortizes fixed Gymnasium overhead well. The thesis measured
  `VSSVecEnv` at roughly 54% of its raw ceiling at that batch; the current
  diagnostic obtains 63%.
- PPO still performs real learner work. Four optimization epochs, policy/value
  forward passes, gradients, and Adam updates do not disappear when expressed
  in JAX; they are only fused and kept resident more effectively.
- Amdahl's law bounds the result. In the matched run, one SB3 generation costs
  about 0.939 s and one native generation about 0.596 s. Moving the environment
  and learner into one XLA program removes roughly 0.343 s, not the full
  generation.

## Why current absolute numbers differ from the thesis tables

The thesis records older measurements of 298,564 raw physics steps/s, 160,353
`VSSVecEnv` steps/s, and 49,285 end-to-end SB3 steps/s at batch 256. The current
runs obtain 106,876, 67,553, and 34,906 respectively. These are not a controlled
before/after experiment, but the evidence indicates benchmark drift:

- the simulator has since gained corner-chamfer collision handling
  (`b82c636`, adding collision work inside every physics step);
- goal-area handling and reward bookkeeping were expanded (`7e6c17c`);
- JAX, PyTorch, SB3, and Python versions have changed; and
- the current run aggregates ten generations after one warm-up generation,
  whereas the thesis reports per-generation medians from twelve generations.

The current raw kernel is about 2.8x slower than the historical table, so a
large part of the missing absolute throughput predates the new PPO trainer and
cannot be attributed to Jumanji or Flax. A precise attribution would require a
commit-by-commit benchmark using the same dependency lock and measurement
script.

## Conclusions and next measurements

The JAX-native implementation succeeds at its architectural objective and
improves matched CPU training by 1.58x. It does not make policy optimization or
the now-heavier physics free, and the thesis never claimed an order-of-magnitude
gain over SB3 on the M4.

The next meaningful measurement is the same matched benchmark on the RTX 3060.
That experiment will test the principal expected advantage: keeping physics,
policy inference, rollout data, and gradients in GPU memory without the SB3
host round trip. Separately, the raw-kernel regression should be bisected before
updating thesis performance tables.
