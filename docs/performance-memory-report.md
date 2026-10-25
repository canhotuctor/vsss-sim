# Throughput and memory report: brandao_shared self-play

Measured on 2026-10-23, branch codex/referee-transition-fixes (commit 01e23d5), on an Apple Mac (12 CPU cores, 24 GB RAM) with JAX 0.10.0, Flax 0.12.7 and Optax 0.2.8. Every measurement ran on the CPU backend. Memory figures come from XLA's `memory_analysis()` and buffer-assignment dumps of the compiled `PPO._train_many` executable that `run_experiment` builds. GPU buffer assignment differs somewhat, but the dominant buffers are the same. A GPU also adds a few hundred MB for the CUDA context and cuBLAS workspace.

Unless stated otherwise, the configuration is `experiment26_brandao_shared_160m`: brandao_shared, 200 envs per lineage, rollout_length 214, update_epochs 5, num_minibatches 8, 10 Hz control, 6 physics sub-steps, max_episode_steps 400.

The scripts used are reproduced in the appendix.

## 1. Where the memory goes

Peak memory (arguments + outputs + temporaries) of one compiled training call:

| Configuration | Peak XLA buffers |
|---|---|
| 1 lineage | 644 MB |
| 2 lineages | 1,289 MB |
| 3 lineages | 1,932 MB |
| 4 lineages | 2,577 MB |
| 1 lineage, rollout_length 400 | 1,192 MB |
| 1 lineage, 400 envs | 1,280 MB |
| 1 lineage, mirror_augmentation | 656 MB |
| 2 lineages, mirror_augmentation | 1,312 MB |
| 1 lineage, 50 updates per chunk | 653 MB |
| 1 lineage, num_minibatches 4 | 672 MB |
| 1 lineage, num_minibatches 16 | 644 MB |

Rule of thumb:

    peak ≈ 3 × lineages × envs × rollout_length × 5,088 bytes

The 5,088 bytes is one environment step's stacked observation: 3 robots × 424 floats × 4 bytes. Memory grows linearly with lineages, envs and rollout length. Mirror augmentation, chunk length and minibatch count barely change it.

The buffer-assignment dump shows three full 217.8 MB copies of the stacked observation buffer (200 × 214 × 3 × 424 × float32) live at peak:

1. The rollout `scan` output (`trajectory.observation`).
2. A defensive copy XLA makes of that buffer.
3. The shuffled batch produced by `x[permutation]` in `PPO._optimize` (`src/vsss_sim/rl/ppo.py`).

Everything else is small by comparison:

| Buffer | Size per lineage |
|---|---|
| Network parameters | 1.4 MB |
| Adam optimizer state | about 4 MB |
| Environment state | 2.0 MB |
| Whole runner state | 7.1 MB |
| One minibatch of observations | 27 MB |
| Minibatch activations | about 60 MB |

### Interpreting a 2.6 GiB reading

A 2.6 GiB peak matches about 4 lineages at rollout_length 214, or 2 lineages at rollout_length 400 plus CUDA overhead. It can also be JAX's default GPU preallocation: JAX reserves 75% of total VRAM on the first operation, and `nvidia-smi` reports that reservation whatever the program actually uses.

To tell them apart, run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and read `jax.devices()[0].memory_stats()["peak_bytes_in_use"]`. The package defaults to the CPU backend (`src/vsss_sim/__init__.py`), so GPU runs need `JAX_PLATFORMS=cuda`.

All lineages are vmapped into one program, so memory grows strictly linearly with the lineage count. On a card with about 3 GB usable, 3 lineages at rollout_length 214 is borderline, and at rollout_length 400 it runs out of memory.

## 2. Memory improvements

### M1. Store one frame per step instead of the 8-frame history

Each stored observation stacks 8 frames of 53 features, and consecutive steps share 7 of them. Only the newest frame is new data. Instead, store:

- the newest frame of each step, `(T, N, 3, 53)`;
- the 8 frames of the observation at rollout start;
- the start index of each episode, derived from `episode_end`.

Then rebuild each stacked observation per minibatch. With `G` the concatenation of the initial 8 frames and the newest frames of steps 1..T−1:

    frame k of obs_t = G[max(t − 7 + k, episode_start_t) + 7]

`episode_start_t` is −7 for steps still in the episode that was running at rollout start. After a reset, `initialize_history` fills all 8 frames with the reset frame, so the `max` clamp reproduces it.

### M2. Gather minibatches by index

Do not materialize a shuffled copy of the whole batch. Reshape the permutation to `(num_minibatches, minibatch_size)`, use it as the `scan` input, and gather `batch[idx]` inside the minibatch step.

### Prototype results

`proto.py` (appendix) implements M2 alone ("A") and M1 together with M2 ("A+B") as a PPO subclass.

The rebuilt observations match the stored ones exactly: max abs diff 0.0, with 8 envs, 64 steps, max_episode_steps 10 and 48 episode boundaries in the window.

| Variant | Lineages | Rollout | Peak |
|---|---|---|---|
| Baseline | 1 | 214 | 644 MB |
| A (M2) | 1 | 214 | 339 MB |
| A+B (M1 + M2) | 1 | 214 | 159 MB |
| Baseline | 4 | 214 | 2,577 MB |
| A (M2) | 4 | 214 | 1,736 MB |
| A+B (M1 + M2) | 4 | 214 | 651 MB |
| A+B (M1 + M2) | 1 | 400 | 291 MB |

Generation time on CPU is unchanged: baseline 3.634 s, A 3.639 s, A+B 3.672 s.

The prototype's `_update` is simplified. It omits mirror augmentation and most metrics, so it only supports the memory and timing measurements and is not a drop-in replacement.

### M3. Shrink minibatches if needed

After M1 and M2, minibatch activations are the largest remaining buffer: about 16k robot-rows × about 1,000 hidden units. Raising `num_minibatches` reduces them linearly. This changes a PPO hyperparameter, so record it in experiment reports.

### M4. Make the lineage tradeoff a setting

`jax.lax.map(update, runners, batch_size=k)` (available in JAX 0.10) vectorizes k lineages at a time and loops over the rest. Memory then scales with k rather than the total lineage count, and results stay bit-identical. The lineage count becomes a time cost only.

This is preferable to one process per experiment, because each process pays for its own CUDA context and the processes compete for the GPU. Lineages already have separate keys, parameters and optimizer state, so they stay independent either way.

### M5. Smaller items

- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=.9`, or disable preallocation, to recover headroom.
- Add `donate_argnums` to the `train_lineages` jit. It saves only about 7 MB per lineage.
- Stop returning `opponent_generations` with shape (T, N) per update. It is constant for brandao_shared and costs about 8.6 MB per lineage per 50-generation chunk.
- Storing observations in bfloat16 would halve observation memory again, but it changes observation precision. Avoid it for scientific comparisons.

## 3. Throughput

CPU time for one generation at 1 lineage (3.6 s, about 12,000 env-steps/s):

| Component | Time per generation |
|---|---|
| Full update (rollout + PPO) | 3.589 s |
| Rollout only | 2.125 s |
| Environment-only scan, 214 steps | 0.664 s |
| Physics, 6 sub-steps | 0.543 s |
| Goal-area membership, one call per step | 0.154 s |
| Full referee (calls membership twice) | 0.306 s |
| Brandão feature construction | 0.013 s |
| `env.reset` | 0.011 s |
| 3 MLP forward passes, with Beta sampling | 0.693 s |
| 3 MLP forward passes, without Beta sampling | 0.207 s |
| Beta sampling only (learner + opponent) | 0.555 s |

Physics stages, each timed over 214 × 6 sub-steps:

| Stage | Time |
|---|---|
| Robot–robot collisions | 0.141 s |
| Walls and chamfers | 0.056 s |
| Ball–robot collisions | 0.042 s |

The PPO update takes about 1.5 s of the generation. Beta sampling costs more than all three network forward passes. Goal-area polygon clipping accounts for nearly all of the referee's cost.

### GPU: the rollout is limited by launch latency

The rollout runs 214 sequential steps, and each step multiplies tiny 600-row matrices, so per-step launch and synchronization overhead dominates on a GPU.

The compiled vmapped `env.step` contains 16 while loops and no conditionals: every `lax.cond` becomes a `select` that executes both branches under `vmap`. The full training program contains 71 while loops. 57 have a known trip count, and 14 run on every rollout step with a data-dependent trip count:

- 8 come from `jax.random.beta`, whose gamma sampler uses rejection loops. There are four draws per step (α and β for the learner and the opponent).
- 6 come from `_clip_polygon` in `src/vsss_sim/envs/match_geometry.py`, because its `fori_loop(0, count)` has a traced `count`.

On XLA:GPU, each iteration of a data-dependent while loop copies its predicate back to the host and synchronizes. Replacing such loops with static-length masked loops has been reported to give large GPU speedups ([SOLVAX #139](https://github.com/uwplasma/SOLVAX/pull/139), [jax discussion #32791](https://github.com/jax-ml/jax/discussions/32791)). This report did not measure that effect on a GPU; confirm it with `jax.profiler` or `nsys`.

### Throughput improvements

- **T1. Branchless, fixed-length goal-area clipping.** Square-versus-rectangle clipping has a small known maximum vertex count. Unroll the loop in Python and replace `lax.cond` with `jnp.where`. The result is identical and no data-dependent loops remain. The opponent evaluator benefits too, since it uses the same `env.step`.
- **T2. One gamma draw for all Beta samples.** Concatenate the learner and opponent α and β, draw `X = gamma([α_l, β_l, α_o, β_o])` once, and form Beta = X_α / (X_α + X_β). The distribution is unchanged and there are a quarter of the rejection loops. The random stream changes, so earlier runs will not replay bit-for-bit. A fixed number of proposals would remove the loops entirely but adds a small bias. A Kumaraswamy distribution would change the method.
- **T3. Larger batches are nearly free on a GPU.** Because the rollout is latency-bound, more envs or lineages add little wall time until the GPU saturates. Memory is the current limit, so M1 and M2 also raise throughput: about 4× more envs × lineages fit in the same VRAM.
- **T4. One loss for mirror augmentation.** `_optimize` currently calls `value_and_grad` twice per minibatch. Concatenating the original and mirrored minibatches into one loss of twice the size gives the same averaged gradient with one larger matrix multiply.
- **T5. Dense robot–robot collisions.** `_robot_robot_collisions` scatter-adds over 15 pairs, and `_sat_square_overlap` picks the contact axis with a dynamic `axes[argmin]` lookup. A dense 6×6 masked formulation with a one-hot axis select avoids the scatters and gathers on both CPU and GPU.
- **T6. Vectorized opponent evaluation.** `run_experiment` evaluates lineages one at a time in a Python loop. Vmapping the evaluator over lineages needs one call per snapshot.

### Checked and low value

- **Both-branch resets and restarts.** The always-executed `env.reset` and restart branches cost about 0.02 s per generation.
- **Bootstrap critic.** Reusing the next step's value instead of re-evaluating the critic for `bootstrap_value` would save about 20% of rollout network FLOPs. Handling truncations exactly makes this more involved.

## 4. Suggested order

1. Measure the true GPU peak with preallocation disabled.
2. Implement M2.
3. Implement M1, using the exact-match check from `proto.py` as a test.
4. Implement T1 and T2, then profile on a GPU.
5. Add M4 so the lineage count is limited by time rather than memory.

With M1 and M2, the budget that holds 4 lineages today should hold about 16 at the same configuration.

## 5. 2026-10-25 implementation validation

M2 was implemented first in `PPO._optimize`: the shuffled permutation is
reshaped to `(num_minibatches, minibatch_size)` and each scan iteration gathers
only its own rows. No shuffled copy of the full training batch is materialized.

M1 then changes Brandao rollout storage. The rollout keeps the newest
53-feature frame for each robot rather than the complete 424-feature history.
At optimization time, it combines the rollout-start eight-frame history with
those stored frames and derives each episode start from the previous step's
`episode_end`. This preserves the repeated reset frame after every autoreset.
The non-Brandao PPO path continues to retain its dense observation.

The before/after measurements below ran on 2026-10-25 on this Apple Silicon
macOS machine (10 logical CPUs reported by Python), with JAX 0.11.2, Flax
0.12.10 and Optax 0.2.8. Both runs forced the CPU backend and used the same
one-lineage `brandao_shared` workload: 200 environments, rollout length 214,
five update epochs and eight minibatches. Memory is XLA executable
`memory_analysis()`; timing is the best of three warmed executions.

| Metric | Before | M1 + M2 | Change |
|---|---:|---:|---:|
| Estimated XLA peak buffers | 644.4 MB | 162.3 MB | -74.8% |
| Temporary buffers | 629.2 MB | 147.1 MB | -76.6% |
| Full PPO update | 3.886 s | 3.794 s | -2.4% |
| Full-update throughput | 11,014 env-steps/s | 11,281 env-steps/s | +2.4% |
| Rollout only | 2.098 s | 2.059 s | -1.9% |

The small throughput delta is within the expected variation of a CPU timing
run; the meaningful outcome is that rebuilding minibatch histories did not
materially slow the update while reducing the compiled memory estimate by about
482 MB per lineage.

Correctness coverage includes an exact reconstruction test with six reset
boundaries across two environments and a reversed minibatch order. The targeted
PPO and self-play suites passed after the change, including the mirrored
Brandao-set update path.

## Appendix: measurement scripts

Save the scripts in one scratch directory, for example /tmp/vsss_perf. Several scripts import from that directory, and `hlo2.py` reads the XLA dump written there. Run them from the repository root with the project virtual environment:

    .venv/bin/python /tmp/vsss_perf/mem.py '{"L":1}' '{"L":2}' '{"L":4}' '{"L":1,"mirror":true}' '{"L":1,"rollout":400}'
    XLA_FLAGS="--xla_dump_to=/tmp/vsss_perf/dump" .venv/bin/python /tmp/vsss_perf/mem.py '{"L":1}'
    .venv/bin/python /tmp/vsss_perf/hlo.py
    .venv/bin/python /tmp/vsss_perf/hlo2.py
    .venv/bin/python /tmp/vsss_perf/time.py
    .venv/bin/python /tmp/vsss_perf/time2.py
    .venv/bin/python /tmp/vsss_perf/time3.py
    .venv/bin/python /tmp/vsss_perf/time4.py
    .venv/bin/python /tmp/vsss_perf/proto.py
    .venv/bin/python /tmp/vsss_perf/proto_time.py

To list the largest values inside the temporary buffer, find the `preallocated-temp` allocation number in the dump (1431 in the run above):

    f=$(ls -S /tmp/vsss_perf/dump/*train_many*buffer-assignment.txt | head -1)
    grep -E "^allocation [0-9]+: size [0-9]+" "$f" | sort -t' ' -k4 -rn | head
    awk '/^allocation 1431:/{p=1;next} /^allocation [0-9]+:/{p=0} p' "$f" \
      | grep -oE "value: <[0-9]+ [^>]+> \(size=[0-9]+,offset=[0-9]+\)" \
      | awk -F'size=' '{split($2,a,","); print a[1], $0}' | sort -rn | head -25

The `nn_inference_no_sampling_s` value printed by `time.py` is invalid: XLA hoisted the loop-invariant forward pass out of the scan. `time2.py` repeats that measurement with observations that change every step, and the throughput table uses its results.

### mem.py

```python
import sys, time, json
import jax, jax.numpy as jnp
from vsss_sim.rl.self_play import ExperimentConfig, _make_env
from vsss_sim.rl.ppo import PPO

def measure(L, envs=200, rollout=214, mirror=False, updates=1, minibatches=8, arch="brandao_shared"):
    cfg = ExperimentConfig(experiment_name="m", num_generations=1, num_lineages=L,
        num_envs_per_lineage=envs, rollout_length=rollout, update_epochs=5,
        num_minibatches=minibatches, policy_architecture=arch, mirror_augmentation=mirror)
    env = _make_env(cfg); tr = PPO(env, cfg.ppo_config())
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(L, dtype=jnp.uint32))
    init = jax.jit(jax.vmap(tr._initialize))(keys)
    pools = jax.tree_util.tree_map(lambda v: v[:, None], init.train_state.params)
    probs = jnp.tile(jnp.asarray([[0,0,1.,0,0]], jnp.float32), (L,1))
    gens = jnp.zeros((L,1), jnp.int32)
    runners = tr.initialize_lineages(keys, pools, probs, gens)
    t=time.perf_counter()
    exe = tr._compiled_train_many.lower(runners, pools, probs, gens, num_updates=updates).compile()
    ct=time.perf_counter()-t
    m = exe.memory_analysis()
    nbytes = lambda tree: sum(x.nbytes for x in jax.tree_util.tree_leaves(tree))
    MB=2**20
    out = dict(L=L, envs=envs, rollout=rollout, mirror=mirror, updates=updates, mb=minibatches, arch=arch,
        compile_s=round(ct,1), arg_MB=round(m.argument_size_in_bytes/MB,1), out_MB=round(m.output_size_in_bytes/MB,1),
        temp_MB=round(m.temp_size_in_bytes/MB,1), alias_MB=round(m.alias_size_in_bytes/MB,1),
        runner_MB=round(nbytes(runners)/MB,2), params_MB=round(nbytes(init.train_state.params)/MB/L,2),
        env_state_MB=round(nbytes(runners.env_state)/MB,2))
    out["peak_est_MB"] = round(out["arg_MB"]+out["out_MB"]+out["temp_MB"]-out["alias_MB"],1)
    print(json.dumps(out), flush=True)
    return exe, (runners, pools, probs, gens)

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        kw = json.loads(spec); measure(**kw)
```

### hlo.py

```python
import re, jax, jax.numpy as jnp
from vsss_sim.rl.self_play import ExperimentConfig, _make_env
cfg = ExperimentConfig(experiment_name="m", num_generations=1, policy_architecture="brandao_shared")
env = _make_env(cfg)
N=200
keys = jax.random.split(jax.random.PRNGKey(0), N)
states, ts = jax.vmap(env.reset)(keys)
act = jnp.zeros((N,3,2)); mode = jnp.zeros((N,), jnp.int32)
f = jax.jit(jax.vmap(lambda s,a,o,m: env.step(s,a,opponent_action=o,opponent_mode=m)))
txt = f.lower(states, act, act, mode).compile().as_text()
print("while ops in one vmapped env.step:", len(re.findall(r"= .*? while\(", txt)))
print("conditional ops:", len(re.findall(r" conditional\(", txt)))
# trip counts
for m in re.finditer(r'while\(.*?known_trip_count=\{n=(\d+)\}', txt): pass
print("known_trip_count annotations:", re.findall(r'known_trip_count[^}]*\}', txt)[:20])
```

### hlo2.py

```python
import re
txt = open([__import__('glob').glob('/tmp/vsss_perf/dump/*jit__train_many.cpu_after_optimizations.txt')][0][0]).read()
whiles = re.findall(r'^\s*(\S+) = .*? while\((.*)$', txt, re.M)
known = [w for w in whiles if 'known_trip_count' in w[1]]
unknown = [w for w in whiles if 'known_trip_count' not in w[1]]
print("total while:", len(whiles), "known trip:", len(known), "data-dependent:", len(unknown))
for name, rest in unknown:
    m = re.search(r'op_name="([^"]+)"', rest)
    print("  ", name, (m.group(1)[:160] if m else rest[:160]))
```

### time.py

```python
import time, json, sys
import jax, jax.numpy as jnp
sys.path.insert(0, "/tmp/vsss_perf")
from mem import ExperimentConfig, _make_env, PPO

def bench(fn, *args, reps=3):
    out = fn(*args); jax.block_until_ready(out)
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); out=fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter()-t)
    return min(ts)

L=1; envs=200; T=214
cfg = ExperimentConfig(experiment_name="m", num_generations=1, num_lineages=L, num_envs_per_lineage=envs,
    rollout_length=T, update_epochs=5, num_minibatches=8, policy_architecture="brandao_shared")
env=_make_env(cfg); tr=PPO(env, cfg.ppo_config())
keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(L, dtype=jnp.uint32))
init = jax.jit(jax.vmap(tr._initialize))(keys)
pools = jax.tree_util.tree_map(lambda v: v[:, None], init.train_state.params)
probs = jnp.tile(jnp.asarray([[0,0,1.,0,0]], jnp.float32), (L,1)); gens=jnp.zeros((L,1), jnp.int32)
runners = tr.initialize_lineages(keys, pools, probs, gens)

full = jax.jit(jax.vmap(tr._update))
roll = jax.jit(jax.vmap(lambda r,p,q,g: tr._collect_rollout(r,p,q,g)[1].reward))
r0 = jax.tree_util.tree_map(lambda x: x[0], runners)
# env only: scan T steps with zero actions, same opponent API
step = jax.vmap(lambda s,a,o,m: env.step(s,a,opponent_action=o,opponent_mode=m))
def env_only(state):
    a=jnp.zeros((envs,3,2)); m=jnp.zeros((envs,),jnp.int32)
    def body(s,_):
        s2,ts=step(s,a,a,m); return s2, ts.reward
    return jax.lax.scan(body, state, None, length=T)[1]
env_only=jax.jit(env_only)
# policy inference only (learner sample + opponent beta + bootstrap) on fixed obs
params = r0.train_state.params
def infer(params, obs, key):
    def body(k,_):
        k,k1,k2=jax.random.split(k,3)
        a,raw,lp,v = tr._sample_policy(params, obs, k1)
        f,s,_ = tr.model.apply(params, obs); oa = jax.random.beta(k2,f,s)
        _,_,bv = tr.model.apply(params, obs)
        return k, (a.sum()+oa.sum()+bv.sum()+lp.sum())
    return jax.lax.scan(body, key, None, length=T)[1]
infer=jax.jit(infer)
def infer_mean(params, obs, key):
    def body(k,_):
        k,k1=jax.random.split(k)
        f,s,v = tr.model.apply(params, obs); f2,s2,_ = tr.model.apply(params, obs)
        _,_,bv = tr.model.apply(params, obs)
        return k, (f.sum()+s.sum()+f2.sum()+bv.sum())
    return jax.lax.scan(body, key, None, length=T)[1]
infer_mean=jax.jit(infer_mean)

res = {}
res["full_update_s"] = bench(full, runners, pools, probs, gens)
res["rollout_only_s"] = bench(roll, runners, pools, probs, gens)
res["env_only_scan_s"] = bench(env_only, r0.env_state)
res["nn_inference_with_beta_s"] = bench(infer, params, r0.observation, jax.random.PRNGKey(1))
res["nn_inference_no_sampling_s"] = bench(infer_mean, params, r0.observation, jax.random.PRNGKey(1))
res["env_steps_per_s_full"] = envs*T/res["full_update_s"]
print(json.dumps({k: round(v,3) for k,v in res.items()}))
```

### time2.py

```python
import time, json, sys
import jax, jax.numpy as jnp
sys.path.insert(0, "/tmp/vsss_perf")
from time import perf_counter
from mem import ExperimentConfig, _make_env, PPO
def bench(fn, *a, reps=3):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t=perf_counter(); jax.block_until_ready(fn(*a)); ts.append(perf_counter()-t)
    return min(ts)
envs=200; T=214
cfg = ExperimentConfig(experiment_name="m", num_generations=1, num_envs_per_lineage=envs, rollout_length=T, update_epochs=5, policy_architecture="brandao_shared")
env=_make_env(cfg); tr=PPO(env, cfg.ppo_config())
r0 = jax.jit(tr._initialize)(jax.random.PRNGKey(0)); params=r0.train_state.params; obs0=r0.observation
def make(sample):
    def f(params, key):
        def body(c,_):
            k,obs=c; k,k1,k2=jax.random.split(k,3)
            f1,s1,v = tr.model.apply(params, obs); f2,s2,_ = tr.model.apply(params, obs*0.999)
            _,_,bv = tr.model.apply(params, obs*1.001)
            if sample:
                a=jax.random.beta(k1,f1,s1); b=jax.random.beta(k2,f2,s2)
            else:
                a=f1/(f1+s1); b=f2/(f2+s2)
            obs = obs + 1e-3*(a.mean()+b.mean()+bv.mean())
            return (k,obs), None
        return jax.lax.scan(body, (key, obs0), None, length=T)[0][1]
    return jax.jit(f)
def beta_only(key):
    a=jnp.full((envs,3,2),1.5)
    def body(c,_):
        k,acc=c; k,k1=jax.random.split(k); return (k, acc+jax.random.beta(k1,a+acc*1e-6,a).sum()), None
    return jax.lax.scan(body,(key,0.0),None,length=T)[0][1]
res=dict(nn_fwd_x3_with_beta=bench(make(True),params,jax.random.PRNGKey(1)),
         nn_fwd_x3_no_beta=bench(make(False),params,jax.random.PRNGKey(1)),
         beta_sampling_only_2x=2*bench(jax.jit(beta_only),jax.random.PRNGKey(2)))
print(json.dumps({k:round(v,3) for k,v in res.items()}))
```

### time3.py

```python
import json, sys
import jax, jax.numpy as jnp
from time import perf_counter
sys.path.insert(0, "/tmp/vsss_perf")
from mem import ExperimentConfig, _make_env
from vsss_sim.physics import jax_backend as physics
from vsss_sim.envs.match_rules import MatchStepInput
from vsss_sim.envs.match_geometry import _goal_area_membership
from vsss_sim.envs.brandao_observation import build_agent_features
from vsss_sim.envs.match_referee import _resolve_referee
def bench(fn, *a, reps=3):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t=perf_counter(); jax.block_until_ready(fn(*a)); ts.append(perf_counter()-t)
    return min(ts)
N=200; T=214
cfg = ExperimentConfig(experiment_name="m", num_generations=1, policy_architecture="brandao_shared")
env=_make_env(cfg)
st,_ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0),N))
sim = st.simulation
acts = jax.random.uniform(jax.random.PRNGKey(3),(N,2,3,2),minval=-1,maxval=1)
def scan_of(fn):
    def f(sim):
        def body(s,_):
            s2 = fn(s); return s2, None
        return jax.lax.scan(body, sim, None, length=T)[0]
    return jax.jit(f)
phys = jax.vmap(lambda s,a: physics.step(s,a,dt=0.1,sub_steps=6)[0])
def p(s): return phys(s, acts)
def area(s):
    m = jax.vmap(_goal_area_membership)(s.robots)
    return s._replace(robots=s.robots + 1e-4*m.sum((-1,-2,-3))[:,None,None,None].astype(jnp.float32))
def ref(s):
    d = jax.vmap(lambda a,b,r: _resolve_referee(a,b,r,jnp.int32(0),jnp.float32(0.1)))(s, s, st.referee_state)
    return s._replace(robots=s.robots + 1e-4*d.event[:,None,None,None].astype(jnp.float32))
def feat(s):
    f = jax.vmap(lambda x,k: build_agent_features(x, jnp.zeros((2,3,2)), jnp.float32(1.), k))(s, jax.random.split(jax.random.PRNGKey(1),N))
    return s._replace(robots=s.robots + 1e-6*f.sum((-1,-2,-3))[:,None,None,None])
def reset(s):
    r,_ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2),N))
    return s._replace(robots=s.robots + 1e-6*r.simulation.robots)
res = {k: round(bench(scan_of(fn), sim),3) for k,fn in dict(physics_6substeps=p, goal_area_membership_x1=area, referee_full=ref, brandao_features=feat, env_reset=reset).items()}
print(json.dumps(res))
```

### time4.py

```python
import json, sys
import jax, jax.numpy as jnp
from time import perf_counter
sys.path.insert(0, "/tmp/vsss_perf")
from mem import ExperimentConfig, _make_env
from vsss_sim.physics import jax_backend as P
def bench(fn, *a, reps=3):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t=perf_counter(); jax.block_until_ready(fn(*a)); ts.append(perf_counter()-t)
    return min(ts)
N=200; S=214*6
env=_make_env(ExperimentConfig(experiment_name="m", num_generations=1, policy_architecture="brandao_shared"))
st,_ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0),N)); sim=st.simulation
def scan_of(fn):
    def f(sim):
        def body(s,_):
            s2 = jax.vmap(fn)(s); return s2._replace(robots=s2.robots*0.9999+1e-5), None
        return jax.lax.scan(body, sim, None, length=S)[0]
    return jax.jit(f)
parts = dict(
  robot_robot=P._robot_robot_collisions, ball_robot=P._ball_robot_collisions,
  walls_chamfers=lambda s: P._ball_chamfer_collisions(P._ball_wall_collisions(P._robot_chamfer_collisions(P._robot_wall_collisions(s)))[0]),
  identity=lambda s: s)
print(json.dumps({k: round(bench(scan_of(f), sim),3) for k,f in parts.items()}))
```

### proto.py

```python
"""Scratch prototype: (A) index-gathered minibatches, (B) frame-deduplicated rollout storage."""
import inspect, textwrap, json, sys
import jax, jax.numpy as jnp
sys.path.insert(0, "/tmp/vsss_perf")
from vsss_sim.rl import ppo as ppo_mod
from vsss_sim.rl.ppo import PPO, _TrainingBatch, compute_gae
from vsss_sim.envs.brandao_observation import mirror_x_axis_observation, FRAME_FEATURES as F, HISTORY_LENGTH as H
from vsss_sim.rl.self_play import ExperimentConfig, _make_env

# --- (B) rollout that stores only the newest frame ---------------------------
src = textwrap.dedent(inspect.getsource(PPO._collect_rollout))
assert "observation=carry.observation," in src
src = src.replace("observation=carry.observation,", "observation=carry.observation[..., -%d:]," % F)
ns = dict(vars(ppo_mod)); exec(src, ns); _collect_frames = ns["_collect_rollout"]

def reconstruct(G, starts, t, n):
    """G: (T+7, N, 3, F) frames; starts: (T, N); t, n: (M,) -> (M, 3, H*F)."""
    k = jnp.arange(H)
    src_t = jnp.maximum(t[:, None] - (H - 1) + k[None, :], starts[t, n][:, None]) + (H - 1)
    frames = G[src_t, n[:, None]]                      # (M, H, 3, F)
    return jnp.swapaxes(frames, 1, 2).reshape(t.shape[0], 3, H * F)

class ProtoPPO(PPO):
    dedupe = False
    def _collect_rollout(self, runner, *a):
        return (_collect_frames if self.dedupe else PPO._collect_rollout)(self, runner, *a)

    def _update(self, runner, opponent_pool=None, opponent_probabilities=None, opponent_generations=None):
        initial_obs = runner.observation
        rollout_runner, traj = self._collect_rollout(runner, opponent_pool, opponent_probabilities, opponent_generations)
        T, N = traj.episode_end.shape
        ee = traj.episode_end[..., None]; disc = traj.discount[..., None]
        adv, tgt = compute_gae(traj.reward, traj.old_value, traj.bootstrap_value, disc, ee, self.config.gamma, self.config.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if self.dedupe:
            init_frames = jnp.moveaxis(initial_obs.reshape(N, 3, H, F), 2, 0)   # (8, N, 3, F)
            G = jnp.concatenate([init_frames, traj.observation[1:]], axis=0)    # (T+7, N, 3, F)
            def scan_start(s, x):
                t, prev_end = x
                s = jnp.where(prev_end, t, s); return s, s
            prev_end = jnp.concatenate([jnp.zeros((1, N), bool), traj.episode_end[:-1]], 0)
            _, starts = jax.lax.scan(scan_start, jnp.full((N,), -(H - 1), jnp.int32), (jnp.arange(T, dtype=jnp.int32), prev_end))
            obs_fn = lambda t, n: reconstruct(G, starts, t, n)
        else:
            obs_fn = lambda t, n: traj.observation[t, n]
        small = dict(raw_action=traj.raw_action, old_log_prob=traj.old_log_prob, old_value=traj.old_value, advantage=adv, target=tgt)
        B, nmb = self.config.batch_size, self.config.num_minibatches
        def mb_step(state, idx):
            t, n = idx // N, idx % N
            mb = _TrainingBatch(observation=obs_fn(t, n), **{k: v[t, n] for k, v in small.items()})
            (_, m), g = jax.value_and_grad(self._loss, has_aux=True)(state.params, mb)
            return state.apply_gradients(grads=g), m
        def epoch(c, _):
            state, k = c; k, pk = jax.random.split(k)
            perm = jax.random.permutation(pk, B).reshape(nmb, B // nmb)
            state, m = jax.lax.scan(mb_step, state, perm); return (state, k), m
        (ts, key), m = jax.lax.scan(epoch, (rollout_runner.train_state, rollout_runner.key), None, length=self.config.update_epochs)
        return rollout_runner._replace(train_state=ts, key=key), jax.tree_util.tree_map(jnp.mean, m)

def build(cls, L, envs, T, eps=400, dedupe=False):
    cfg = ExperimentConfig(experiment_name="m", num_generations=1, num_lineages=L, num_envs_per_lineage=envs,
        rollout_length=T, update_epochs=5, num_minibatches=8, policy_architecture="brandao_shared", max_episode_steps=eps)
    env = _make_env(cfg); tr = cls(env, cfg.ppo_config()); tr.dedupe = dedupe
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(L, dtype=jnp.uint32))
    init = jax.jit(jax.vmap(tr._initialize))(keys)
    pools = jax.tree_util.tree_map(lambda v: v[:, None], init.train_state.params)
    probs = jnp.tile(jnp.asarray([[0, 0, 1., 0, 0]], jnp.float32), (L, 1)); gens = jnp.zeros((L, 1), jnp.int32)
    return tr, tr.initialize_lineages(keys, pools, probs, gens), pools, probs, gens

def check_reconstruction():
    tr, runners, pools, probs, gens = build(ProtoPPO, 1, 8, 64, eps=10)
    r = jax.tree_util.tree_map(lambda x: x[0], runners); p = jax.tree_util.tree_map(lambda x: x[0], pools)
    tr.dedupe = False; _, full = jax.jit(lambda r: tr._collect_rollout(r, p, probs[0], gens[0]))(r)
    tr.dedupe = True;  _, fr = jax.jit(lambda r: tr._collect_rollout(r, p, probs[0], gens[0]))(r)
    T, N = full.episode_end.shape
    init_frames = jnp.moveaxis(r.observation.reshape(N, 3, H, F), 2, 0)
    G = jnp.concatenate([init_frames, fr.observation[1:]], 0)
    prev_end = jnp.concatenate([jnp.zeros((1, N), bool), fr.episode_end[:-1]], 0)
    _, starts = jax.lax.scan(lambda s, x: (jnp.where(x[1], x[0], s),) * 2, jnp.full((N,), -(H - 1), jnp.int32), (jnp.arange(T, dtype=jnp.int32), prev_end))
    tt, nn = jnp.meshgrid(jnp.arange(T), jnp.arange(N), indexing="ij")
    rec = reconstruct(G, starts, tt.reshape(-1), nn.reshape(-1)).reshape(full.observation.shape)
    print(json.dumps(dict(episodes_in_window=int(full.episode_end.sum()), max_abs_diff=float(jnp.abs(rec - full.observation).max()))))

def mem(cls, L, dedupe, envs=200, T=214):
    tr, runners, pools, probs, gens = build(cls, L, envs, T, dedupe=dedupe)
    exe = tr._compiled_train_many.lower(runners, pools, probs, gens, num_updates=1).compile()
    m = exe.memory_analysis(); MB = 2**20
    print(json.dumps(dict(variant=("A+B" if dedupe else "A") if cls is ProtoPPO else "baseline", L=L, envs=envs, T=T,
        temp_MB=round(m.temp_size_in_bytes / MB, 1), peak_MB=round((m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes) / MB, 1))), flush=True)

if __name__ == "__main__":
    check_reconstruction()
    for L in (1, 4):
        mem(ProtoPPO, L, False); mem(ProtoPPO, L, True)
    mem(ProtoPPO, 1, True, envs=200, T=400)
```

### proto_time.py

```python
import sys, json, jax
from time import perf_counter
sys.path.insert(0, "/tmp/vsss_perf")
from proto import ProtoPPO, build
from vsss_sim.rl.ppo import PPO
def t(cls, dedupe, L=1):
    tr, r, p, q, g = build(cls, L, 200, 214, dedupe=dedupe)
    f = jax.jit(jax.vmap(tr._update)); jax.block_until_ready(f(r, p, q, g)); ts=[]
    for _ in range(3):
        s=perf_counter(); jax.block_until_ready(f(r, p, q, g)); ts.append(perf_counter()-s)
    return round(min(ts),3)
print(json.dumps(dict(baseline=t(PPO, False), A=t(ProtoPPO, False), A_plus_B=t(ProtoPPO, True))))
```
