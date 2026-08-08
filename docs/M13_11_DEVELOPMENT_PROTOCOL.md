# M13.11 development protocol: random-init masked semi-Markov PPO

Status: **development-only, non-promotional; frozen before execution**.

M13.11 compares a deterministic NumPy PPO actor-critic to the exact
random-initialized, duration-aware M13.10 Double-DQN control.  It changes only
the learner topology.  It does not use imitation labels, a warm start, an
oracle action loss, M13.9/M13.10 probe data, or any screen/confirmation/audit
partition.

## Fixed boundary

Both learned arms retain the 30 public features, eight macros,
policy-owned memory, complementary mask, 200-decision mechanics, compiler,
M13.9 potential reward, and `0.99 ** duration` transition semantics.  Private
`info`, task/reset identifiers, layouts, and environment internals are outside
inference; `info` is reward/evaluation sidecar input only.

## Fresh development partitions

| Partition | Seeds | Layouts | Access |
| --- | --- | --- | --- |
| `development_fit` | 6000–6019 | `m1311_fit_*` | training and the capped smoke only |
| `development_check` | 6020–6039 | `m1311_check_*` | one declared post-fit development evaluation |

The ledger has only those partitions and is ordered `fit -> check`.
M13.8 `4900–5119`, M13.9 `5300–5519`, and M13.10 `5700–5919` are neither
referenced nor readable by this harness.  Any future screen must be a separate
frozen protocol with fresh `6100+` splits and explicit authorization.

## Arms, learner, and equal budget

Training seeds are `20261311` through `20261314`.  Each seed runs:

1. `masked_semimarkov_ppo_candidate`: PCG64 He-initialized
   `30 -> 64 -> 64` ReLU trunk, masked categorical eight-action actor and
   scalar critic; Adam `3e-4`, betas `.9/.999`, epsilon `1e-8`; clip `.20`,
   value coefficient `.50`, entropy coefficient `.01`, GAE lambda `.95`,
   rollout 2048, four update epochs, normalized advantages, max gradient norm
   `.50`, no reward or observation normalization.
2. `random_init_duration_dqn_control`: the M13.10 random-init duration-aware
   Double-DQN update, reward, mask, network, replay and epsilon schedule.

Each learned job receives exactly 400,000 macro decisions as its stopping
target.  Jobs finish an already-started normal terminal/truncation episode, so
the deterministic overshoot is recorded; paired jobs may differ by no more
than one 200-decision episode.  Evaluation also reports mask-aware random and
the scripted public oracle ceiling, but their actions never enter PPO fitting.

For a transition duration `d`, PPO uses:

```text
delta = r + (1-done) * .99**d * V(next) - V(current)
A = delta + (1-done) * .99**d * .95 * A(next)
return = A + V(current)
```

## Execution and gates

All eight learned jobs are submitted before any result is awaited with spawned
workers. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` are set to one; worker
count, PIDs, timings, decisions, episodes, updates, artifact hashes, and strict
replay count are recorded. The pre-training gates are layout freshness,
coverage, contract/reward parity, masking, GAE, gradients, serialization, and
the full test suite.

The smoke is fit-only, uses no more than 32 episodes per learned job, records
to a caller-selected smoke directory, and cannot touch the canonical ledger.
After the scoped implementation commit, the declared development evaluation
opens `development_check` once and strictly replays every serialized-policy
trace.  Its rows are diagnostic only. No success promotes M13, M14, or M15.
