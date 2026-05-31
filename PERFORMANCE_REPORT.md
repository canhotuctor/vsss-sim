# vsss-sim Performance Report

**Date:** 2026-05-31  
**Platform:** macOS 15.7.7 · Apple M4 Pro · arm64  
**Python:** 3.14.4 · JAX backend: CPU (no Metal/GPU on this run)  
**Benchmark:** `scripts/perf_bench.py --reps 2000 --steps 3000 --gens 12`  
**Note:** All numbers on this machine only. Ubuntu RTX 3060 GPU numbers from prior verified run are included in Section 7 for context.

---

## 1. NumPy Backend — Component Latencies

| Subroutine | min (μs) | median (μs) | p95 (μs) |
|---|---|---|---|
| `reset_kickoff` | 13.46 | 15.12 | 15.50 |
| `reset_random` | 8.21 | 9.29 | 9.71 |
| **`step` (full)** | **273.62** | **295.88** | **317.08** |
| `_diff_drive` | 1.42 | 1.62 | 1.71 |
| `_ball_wall_collisions` | 0.50 | 0.62 | 0.67 |
| `_robot_wall_collisions` | 12.21 | 13.79 | 14.12 |
| `_ball_robot_collisions` | 8.38 | 9.54 | 10.12 |
| `_robot_robot_collisions` | 33.04 | 37.42 | 40.50 |

**NumPy step budget breakdown (of the ~296 μs median step):**

| Phase | median (μs) | % of step |
|---|---|---|
| `_robot_robot_collisions` (15 SAT pairs) | 37.42 | 12.6% |
| `_robot_wall_collisions` | 13.79 | 4.7% |
| `_ball_robot_collisions` | 9.54 | 3.2% |
| `_diff_drive` | 1.62 | 0.5% |
| `_ball_wall_collisions` | 0.62 | 0.2% |
| Python overhead + integration in `step()` | ~233 | 78.8% |

The vast majority of the NumPy step time is Python function-call overhead and NumPy array allocation/copy within `step()` — not the collision math itself. The collision sub-routines account for only ~21% of the total step latency.

---

## 2. JAX Backend — Component Latencies

### JIT Compile Times (one-time startup cost)

| Function | Compile time |
|---|---|
| `jb.step` (full physics step) | **162.5 ms** |
| `jb.reset_kickoff` | 80.8 ms |
| `jb.reset_random` | 97.2 ms |

These are paid once per process. At training scale (millions of steps), they are negligible.

### Warm Timings (after JIT, `block_until_ready` per call)

| Function | min (μs) | median (μs) | p95 (μs) |
|---|---|---|---|
| `step` | 14.33 | 30.08 | 36.71 |
| `reset_kickoff` | 7.62 | 8.75 | 9.25 |
| `reset_random` | 8.21 | 9.67 | 12.67 |
| `_diff_drive` (jit'd alone) | 4.50 | 5.04 | 5.50 |

The JAX step median of **30 μs** vs NumPy's **296 μs** is a **9.8× speedup** for a single env. The min of 14 μs reflects the true compute time when the OS scheduler cooperates; the median is slightly higher due to Python dispatch overhead even for a jit'd call.

Reset operations are nearly free in JAX (~9 μs) thanks to JIT compilation — essentially the same cost as a step.

---

## 3. Single-Env Throughput (reference baseline)

| Backend | fps | μs/step | Compile |
|---|---|---|---|
| NumPy | 3,340 | 299.4 | — |
| JAX (jit) | 31,766 | 31.5 | 162.5 ms one-time |
| **Speedup** | **9.5×** | | |

---

## 4. Raw `jax.vmap(step)` — Physics Ceiling

This is the theoretical upper bound on training throughput: JAX's pure physics computation with no Python overhead.

| batch | compile (ms) | per_call (μs) | per_env fps | total fps | vs numpy |
|---|---|---|---|---|---|
| 1 | 157 | 30.9 | 32,374 | 32,374 | 9.7× |
| 8 | 166 | 44.0 | 22,728 | 181,823 | 54.4× |
| 32 | 175 | 135.7 | 7,368 | 235,775 | 70.6× |
| 64 | 174 | 271.4 | 3,684 | 235,775 | 70.6× |
| 128 | 167 | 500.9 | 1,997 | 255,559 | 76.5× |
| 256 | 174 | 879.7 | 1,137 | 290,995 | 87.1× |
| 512 | 167 | 1711.7 | 584 | **299,110** | 89.6× |
| 1024 | 173 | 3430.2 | 292 | 298,526 | 89.4× |

**Key observations:**

- **Throughput plateaus at batch≈512 (~299k fps).** Doubling to 1024 yields no gain, indicating CPU memory-bandwidth saturation on the M4 Pro.
- **Efficiency cliff at batch=32→64:** total fps stays flat (235k) even as the batch doubles. The M4 Pro's CPU cores are fully saturated around batch=32; further scaling is purely memory-bound.
- **Compile time is nearly constant (~170 ms)** regardless of batch size — XLA traces the same graph shape.
- **vmap overhead at small batches:** batch=8 takes 44 μs vs 31 μs for batch=1 (42% overhead for 8×). By batch=32, the compute dominates and overhead becomes irrelevant.

---

## 5. VSSVecEnv (Gymnasium Wrapper) — What RL Libraries See

This includes the full Python boundary: JAX→NumPy array copies, observation normalization, reward computation, opponent policy dispatch, and Gymnasium protocol overhead.

| batch | per_call (μs) | per_env fps | total fps | vs numpy |
|---|---|---|---|---|
| 1 | 1,306.6 | 765 | 765 | 0.2× |
| 8 | 1,283.3 | 779 | 6,234 | 1.9× |
| 32 | 1,402.8 | 713 | 22,812 | 6.8× |
| 64 | 1,541.1 | 649 | 41,530 | 12.4× |
| 128 | 1,828.7 | 547 | 69,994 | 21.0× |
| 256 | 2,346.7 | 426 | 109,090 | 32.7× |
| 512 | 3,264.5 | 306 | 156,841 | 47.0× |
| 1024 | 5,227.7 | 191 | **195,881** | 58.7× |

### Wrapper Overhead vs Raw vmap (same batch)

| batch | vmap fps | VSSVecEnv fps | overhead factor |
|---|---|---|---|
| 1 | 32,374 | 765 | **42.3×** |
| 8 | 181,823 | 6,234 | **29.2×** |
| 32 | 235,775 | 22,812 | **10.3×** |
| 64 | 235,775 | 41,530 | **5.7×** |
| 128 | 255,559 | 69,994 | **3.6×** |
| 256 | 290,995 | 109,090 | **2.7×** |
| 512 | 299,110 | 156,841 | **1.9×** |
| 1024 | 298,526 | 195,881 | **1.5×** |

**Key observations:**

- **The fixed Python overhead per `VSSVecEnv.step()` call is ~1,300 μs**, nearly independent of batch size (1,283 μs at batch=1 vs slightly higher at larger batches). This is dominated by: JAX `block_until_ready` + device-to-host copy + NumPy obs/reward construction + Python dispatch.
- **Wrapper overhead halves roughly every 2× increase in batch size** above batch=32, trending toward 1× (zero overhead) asymptotically — never reached on CPU.
- **At batch=1024, the wrapper gap closes to 1.5×**, meaning ~34% of the VSSVecEnv call time is still Python overhead even at 1024 parallel envs.
- **VSSVecEnv at batch=1 is actually slower than a single NumPy env (765 vs 3,340 fps)** — the JAX→NumPy boundary at small batches costs more than the physics computation itself.

### Where the ~1,300 μs Fixed Overhead Goes

The fixed overhead per call includes (estimated, not individually profiled):

| Source | Estimated cost |
|---|---|
| `jax.block_until_ready()` + device→host copy | ~200–400 μs |
| NumPy obs array construction (shape `(B, 46)`) | ~100–300 μs |
| Reward computation, autoreset logic, Python loops | ~200–400 μs |
| Gymnasium VectorEnv protocol overhead | ~100–200 μs |
| **Total** | **~600–1,300 μs** |

---

## 6. End-to-End SB3 PPO — Per-Generation Timing (12 generations each)

### Summary Table

| num_envs | n_steps | rollout_med (ms) | update_med (ms) | total fps | vs 1 env |
|---|---|---|---|---|---|
| 1 | 512 | 718.9 | 76.8 | 640 | 1.0× |
| 8 | 512 | 724.4 | 116.9 | 4,845 | 7.6× |
| 32 | 512 | 810.6 | 214.0 | 15,925 | 24.9× |
| 64 | 256 | 522.3 | 160.5 | 24,019 | 37.6× |
| 128 | 256 | 627.6 | 328.8 | 34,260 | 53.6× |
| **256** | **128** | **424.6** | **341.7** | **42,900** | **67.1×** |

### Per-Generation Detail: num_envs=256, n_steps=128

| Gen | rollout (ms) | update (ms) | rollout fps |
|---|---|---|---|
| 1 | 419.4 | 329.1 | 78,127 |
| 2 | 426.9 | 338.2 | 76,764 |
| 3 | 425.9 | 347.2 | 76,940 |
| 4 | 425.1 | 340.0 | 77,085 |
| 5 | 423.6 | 343.0 | 77,356 |
| 6 | 418.4 | 347.3 | 78,311 |
| 7 | 415.2 | 334.0 | 78,918 |
| 8 | 424.2 | 344.4 | 77,250 |
| 9 | 434.0 | 349.9 | 75,508 |
| 10 | 411.3 | 341.7 | 79,664 |
| 11 | 426.2 | 339.0 | 76,892 |
| 12 | 429.9 | — | 76,227 |
| **median** | **424.6** | **341.7** | **77,200** |

**Key observations:**

- **Extremely stable across generations.** Rollout time variance is ±10ms (~2.4%) and update time variance is ±10ms (~3%) — the JAX JIT is fully warm after 1 generation of warmup. There are no second-gen warmup artifacts.
- **At peak config (256 envs, 128 steps), the generation time splits ~55% rollout / 45% gradient update.** This is the Amdahl limit for the SB3+PyTorch CPU path.
- **Rollout throughput is ~77k fps inside SB3** (vs VSSVecEnv standalone at ~109k fps at batch=256). The ~30% gap is SB3's observation buffering, policy forward pass (PPO inference during rollout), and callback overhead.
- **The gradient update (341 ms) is the new bottleneck** at 256 envs: processing 32,768 samples × 10 epochs of PyTorch SGD on CPU. Reducing `n_epochs` or moving to GPU gradient computation are the levers.
- **Scaling efficiency from 1 → 256 envs: 67×** out of an ideal 256× — 26% Amdahl efficiency. The gradient update, which doesn't scale with num_envs, is the primary cause.

### Rollout vs Update Fraction by Scale

| num_envs | rollout % of gen | update % of gen |
|---|---|---|
| 1 | 90.4% | 9.6% |
| 8 | 86.1% | 13.9% |
| 32 | 79.1% | 20.9% |
| 64 | 76.5% | 23.5% |
| 128 | 65.6% | 34.4% |
| 256 | 55.4% | 44.6% |

As num_envs grows, the gradient update becomes the dominant cost. Extrapolating: at ~512 envs, update and rollout would be roughly equal; beyond that, the gradient update dominates.

---

## 7. GPU Numbers (Ubuntu RTX 3060 — prior verified run)

Included here for comparison. Not re-run in this session.

| Path | fps |
|---|---|
| Raw `jax.vmap` batch=1024 (GPU) | ~909,000 |
| VSSVecEnv batch=1024 (GPU) | ~92,000 |
| SB3 PPO num_envs=64, n_steps=512 | ~8,000–10,000 |

The GPU raw vmap ceiling is ~3× higher than CPU (909k vs 299k). The VSSVecEnv wrapper is ~2.1× slower on GPU than CPU because the JAX→NumPy device transfer (GPU→CPU copy) is more expensive than a CPU→CPU copy. The SB3 path on GPU is limited by the same PyTorch CPU gradient bottleneck unless the neural network is moved to CUDA too.

---

## 8. Bottleneck Summary

### Current bottlenecks by training path

| Layer | Bottleneck | Cost | Fix |
|---|---|---|---|
| Physics (NumPy) | Python overhead in `step()` | ~233 μs of 296 μs | Use JAX backend |
| Physics (JAX, single) | Python dispatch per JIT call | ~30 μs; min 14 μs | Use vmap |
| Physics ceiling (vmap CPU) | CPU memory bandwidth | Plateaus ~300k fps at batch≥512 | GPU |
| VSSVecEnv wrapper | Fixed ~1,300 μs Python overhead | 42× at batch=1, 1.5× at batch=1024 | JAX-native env (Stoix) |
| SB3 PPO rollout | JAX→NumPy boundary + policy inference | ~77k fps at batch=256 | JAX-native RL |
| SB3 PPO gradient update | PyTorch CPU SGD on large batch | 341 ms at 256×128=32k samples | CUDA PyTorch or JAX optimizer |

### The three walls

1. **Physics wall (CPU vmap):** ~299k fps. Eliminated with GPU (→909k fps).
2. **Gymnasium boundary wall:** fixed ~1,300 μs per call drops peak from 299k to 196k fps on CPU (and from 909k to 92k fps on GPU). Eliminated with JAX-native env (Stoix/jumanji).
3. **Gradient update wall:** at 256 envs, the PyTorch CPU gradient update (342 ms) roughly equals the rollout (424 ms), capping overall training at ~43k fps. Eliminated with GPU PyTorch or JAX-native RL (Stoix).

### Realistic throughput at each stage

```
NumPy single env          :      3,340 fps   (baseline)
JAX single env            :     31,766 fps   ( 9.5× over NumPy)
VSSVecEnv batch=256       :    109,090 fps   (32.7× over NumPy)
VSSVecEnv batch=1024      :    195,881 fps   (58.7× over NumPy)
Raw vmap batch=512        :    299,110 fps   (89.6× — physics ceiling on CPU)
SB3 PPO, 256 envs (end-to-end) :  42,900 fps   (12.8× over NumPy)

GPU raw vmap batch=1024 [RTX 3060] :  ~909,000 fps  (272× over NumPy)
GPU VSSVecEnv batch=1024  [RTX 3060]:   ~92,000 fps
```

---

## 9. Implications for Next Steps

### Stoix (JAX-native RL) — expected impact

Replacing `VSSVecEnv + SB3` with a `jumanji.Environment + Stoix PPO` path eliminates:
- The JAX→NumPy device transfer per step
- The Python Gymnasium protocol overhead
- The PyTorch CPU gradient update (replaced by `optax` running in JAX/XLA)

Expected outcome: throughput approaches the raw vmap ceiling (~300k fps on CPU, ~909k fps on GPU) with the gradient update co-located on the same device.

### Where to focus for each goal

| Goal | Action |
|---|---|
| Faster iteration on Mac (CPU) | Stoix: eliminates 1.5–42× wrapper overhead |
| Maximum training speed (GPU) | Stoix + CUDA: approaches 909k fps ceiling |
| Reduce VSSVecEnv overhead without Stoix | Batch obs construction in JAX before copy; fused autoreset |
| Reduce gradient bottleneck in SB3 | Reduce `n_epochs` (10→4); reduce `batch_size`; or PyTorch MPS |

---

*Benchmark script: `scripts/perf_bench.py` · All timings measured with `time.perf_counter_ns()` and `jax.block_until_ready()` · 2000 reps for component timings, 3000 steps for throughput, 12 PPO generations per config.*
