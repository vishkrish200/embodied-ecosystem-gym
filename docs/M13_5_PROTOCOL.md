# M13.5 protocol — parallel training-dynamics diagnostic

Protocol version: `m13.5-update-ratio-r1`

## Purpose and causal boundary

M13.4 removed public local macro errors—every 3,040 trace replayed with zero
mask violations and zero redundant selected `GO_*` actions—yet no training
replicate established reliable maintenance. Its three full policies instead
collapsed into food-heavy/no-rest, rest/wait starvation, and partially balanced
but terminally unsafe routines. This is evidence of unstable value learning,
not evidence that the M10 environment, public action interface, or public
state boundary is insufficient: the scripted public-state ceiling passes.

M13.5 tests only whether the current DQN receives enough optimization updates
per experienced macro decision. M13.4's full fits processed 211,714–654,275
decisions but performed only 824–2,552 minibatch updates, because one update is
made every 256 decisions. That also produced only zero to two target copies.

M13.5 preserves M13.4's complementary public mask, 30 public features,
policy-owned memory, eight macros and their durations, M13.3 reward including
the recovery-safe blocked penalty, M10 mechanics/horizon, replay capacity,
warmup/batch, network, Adam/Huber, epsilon, discount `.99`, and target-copy
every 1,000 optimizer updates. It changes only the update cadence for the
candidate from one minibatch every 256 macro decisions to one every 4 macro
decisions. Since the target-copy unit stays optimizer updates, this is a
declared training-dynamics bundle: both optimizer exposure and target refresh
per environment time change. It is not evidence isolating target cadence alone.

No reward, input, mask, macro, duration, network, demonstration, curriculum,
hierarchy, or private resource/event signal may change. M13.4 validation and
audit remain forbidden.

## Fresh split, conditions, and ceilings

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 3500–3539 | `m135_dev_northeast`, `m135_dev_southwest`, `m135_dev_northwest`, `m135_dev_southeast` | Fitting and diagnostics only. |
| Validation | 3600–3619 | `m135_validation_northeast`, `m135_validation_southwest`, `m135_validation_northwest`, `m135_validation_southeast` | Open once only after a frozen passing development matrix. |
| Sealed audit | 3700–3719 | `m135_audit_northeast`, `m135_audit_southwest`, `m135_audit_northwest`, `m135_audit_southeast` | Open once only after validation passes. |

Conditions are persistent reference, renewal/morphology, event relocation, and
compound. M10 public scan coverage and the scripted public-state ceiling must
pass 40/40 in every development condition before candidate scoring. They are
mechanics evidence only and never enter policy inference or fitting.

## Arms, training budget, and parallel execution

Use the paired PCG64 training seeds `20260812`, `20260813`, and `20260814`.
For each seed, run exactly:

1. **Update-ratio candidate:** complementary-mask DQN, update every 4 macro
   decisions.
2. **Cadence-null control:** identical complementary-mask DQN, update every
   256 macro decisions.
3. **Mask-aware macro random:** no fitting; uniform over the same public mask.

Every learned fit stays at 12,000 episodes and visits each development
condition/seed pair 75 times. Candidate and cadence-null have the same initial
weight seed and action/RNG seed for each paired replicate; they diverge only as
their declared update schedules begin learning. All three candidate policies
and all three controls are reported. No seed, checkpoint, or model is selected
after results are seen.

Training jobs are independent and run in up to eight isolated spawned worker
processes. Each job writes only its own `(training-seed, arm)` directory; the
parent process checks every return value, fingerprints the source/config, then
runs evaluation/replay. Parallel scheduling is an execution optimization, not
a policy input or a result-selection mechanism. Workers set numerical library
thread counts to one so up to eight training jobs use separate physical cores
without nested-thread oversubscription. The worker count, execution mode, and
per-job artifacts are reported; a different worker count must not reuse the
same report path.

## Development gate

Before validation, coverage and scripted ceiling must pass 40/40 in every row.
Every update-ratio candidate replicate must reach per-condition at least 36/40
survival, 36/40 maintenance completion, `.85` mean safe-drive fraction, and
36/40 completed recovery chains in relocation rows. It must have zero mask
violations and zero redundant selected `GO_*` macros.

For every paired seed, the candidate must exceed its cadence-null control by
at least `.10` maintenance and `.10` survival over the pooled 160 development
episodes, and exceed mask-aware random survival by `.20`. The report gives
per-seed and cross-replicate values; one passing seed never opens validation.
Failure is final for this protocol and leaves validation/audit unopened.

## Replay, validation, and audit

Every trace records the policy-visible observation, features, raw Q vector,
public mask, eligible macros, macro/action, memory before/after, reward,
outcome, ordinary/exempt blocked classification, counters, terminal fields,
RNG/model/source hashes, update cadence, target-copy count, and one-way split
marker. Strict replay recomputes the constrained learned action and memory;
random replay restores its pre-choice PCG64 state. Evaluation uses serialized
development weights, never a retrained policy.

Validation and audit use the same per-condition survival, maintenance,
safe-drive, recovery, and paired-delta gates on their fresh splits. Audit is
opened once only after validation passes. Any failure is published unchanged.

## Limits

M13.5 can show that greater optimizer exposure under this fixed public macro
interface helps or does not help state-oracle maintenance. It cannot separate
update ratio from more frequent target refresh in environment time, establish
learned perception/motor control, or prove long-horizon planning beyond the
frozen simulator task.
