# M13.6 protocol — parallel update-ratio seed-stability replication

Protocol version: `m13.6-update-ratio-stability-r2`

## Revision and causal boundary

The unimplemented `m13.6-duration-discount-r1` proposal is withdrawn. It
would have changed `.99` from a per-macro discount to a per-second discount
through `.99 ** duration_seconds`, changing the effective planning horizon for
long travel macros. That is a different RL objective, not a minimal causal
repair of M13.5's seed instability, so it is neither implemented nor run.

M13.5 establishes that update-every-4 can learn persistent maintenance under
the frozen public masked-DQN interface, whereas update-every-256 and random
cannot. It does not establish reproducibility: only one of three dense-update
replicas passes every condition. M13.6 therefore changes no policy behavior.
It replicates the exact M13.5 candidate/control comparison over eight fresh,
predeclared paired training seeds to measure whether dense optimization is
reliably sufficient or whether its apparent success was a lucky seed.

Both arms retain M13.5's complementary public mask, 30 public features,
policy-owned memory, eight macros/durations, M13.3 reward including the
recovery-safe blocked penalty, M10 mechanics/horizon, Double-DQN target
`reward + .99 * (not_done) * bootstrap`, replay capacity/warmup/batch,
network, Adam/Huber, epsilon schedule, and target copy every 1,000 optimizer
updates. The null differs only in update cadence: one update every 256 macro
decisions rather than every 4. No input, memory, reward, discount, mask,
macro/compiler, private-state access, demonstration, curriculum, or hierarchy
may change. M13.5 models, seeds/layouts, holdouts, and selected successful seed
are forbidden.

## Fresh splits and mechanics ceiling

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 4100–4139 | `m136_dev_{northeast,southwest,northwest,southeast}` | Fitting and diagnostics only. |
| Validation | 4200–4219 | `m136_validation_{northeast,southwest,northwest,southeast}` | Open once only after a frozen passing development matrix. |
| Sealed audit | 4300–4319 | `m136_audit_{northeast,southwest,northwest,southeast}` | Open once only after validation passes. |

Conditions remain persistent reference, renewal/morphology, event relocation,
and compound. M10 public scan coverage and the scripted state-oracle ceiling
must pass 40/40 in every development condition before candidate scoring. They
are mechanics checks only and never enter policy inference or fitting.

## Arms, budget, and candidate-first parallel execution

Use paired PCG64 training seeds `20260818` through `20260825` inclusive. For
each seed, run exactly:

1. **Dense-update candidate:** update every 4 macro decisions.
2. **Sparse-update null:** update every 256 macro decisions.
3. **Mask-aware macro random:** no fitting; uniform over the same public mask.

Every learned fit is exactly 12,000 episodes and visits each development
condition/seed pair 75 times. Candidate and null share initial weights and
action/RNG seed within each pair. All 16 learned fits and all eight random
evaluations are reported; no seed, checkpoint, or model may be selected after
results are seen.

Run up to eight spawned worker processes, with one numerical-library thread
per worker. Submit all eight dense candidates before any sparse nulls, so the
expensive jobs occupy the first worker wave. Submit sparse jobs as workers
free. Report submission order, job start/finish elapsed time, worker count,
start method, and numerical-thread cap. This scheduling is execution-only.

No timing-only training preflight is run. M13.5 already measured the same dense
candidate configuration; M13.6's fresh larger matrix is the useful budget.
The expected wall-clock envelope is 30–45 minutes. Failure to run dense jobs
concurrently is recorded as an execution failure, not hidden by reducing the
frozen budget.

## Development gate

Coverage and scripted ceiling must pass 40/40 in every development row. Every
dense-update candidate replicate must reach, per condition, at least 36/40
survival, 36/40 maintenance completion, `.85` mean safe-drive fraction, and
36/40 completed recovery chains in relocation rows. It must record zero public
mask violations and zero redundant selected `GO_*` macros.

For every paired seed, candidate minus sparse null must be at least `.10`
pooled maintenance and `.10` pooled survival over the 160 development
episodes, and candidate minus mask-aware random pooled survival must be at
least `.20`. Every one of the eight candidates must pass. A passing subset
does not open validation, and failure is final for this protocol.

## Replay, validation, audit, and limits

Every trace records policy-visible observation/features, raw Q vector, public
mask, eligible macros, compiled action/duration, memory before/after, reward,
outcome, ordinary/exempt blocked classification, counters, terminal fields,
RNG/model/source hashes, update cadence, target-copy count, one-way split
marker, and scheduling/job metadata. Strict replay recomputes the constrained
learned action and memory; random replay restores its pre-choice PCG64 state.
Evaluation always uses serialized weights, never a retrained policy.

Validation and audit use the same per-condition and paired-delta gates on their
fresh splits. Validation opens once only after all eight development replicas
pass; audit opens once only after validation passes. Any failure is published
unchanged.

M13.6 can establish whether M13.5's dense-update effect is reproducible under
the frozen public interface. It cannot identify the cause of a failed
replication, justify M13.5 holdout reuse, or claim a robust baseline unless
every gate passes.
