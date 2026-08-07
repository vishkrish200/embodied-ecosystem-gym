# M13.6 protocol — duration-aware semi-MDP discount diagnostic

Protocol version: `m13.6-duration-discount-r1`

## Purpose and causal boundary

M13.5 establishes two facts on fresh development data. Increasing optimization
from one update per 256 macro decisions to one per 4 is necessary for this
public masked DQN to learn maintenance: the dense-update arm substantially
beats its paired sparse-update and random comparators in every seed. It is not
sufficient for a reproducible baseline: one of three dense-update replicas
passes all rows, one collapses in compound/persistent-reference, and one
misses a safe-drive row. M13.5 validation and audit remain unopened.

The current Double-DQN target discounts every macro transition by the same
`.99`, although compiled public macros span different simulated durations:
`PICK_UP` and `CONSUME` are `.1` seconds, `PLAY`, `REST`, and `WAIT` are `1`
second, and `GO_*` duration is public distance divided by walk speed. Thus a
multi-second walk and a `.1`-second interaction have identical bootstrap
discount. That is an incorrect time model for this semi-Markov action surface
and is a plausible source of timing-sensitive seed instability.

M13.6 changes only the target discount. The candidate uses
`gamma_effective = .99 ** (duration_seconds / 1.0)`; the paired null retains
the current per-decision `.99`. One second is the declared reference decision
interval, so existing one-second macros retain `.99`. The reward remains
unchanged and already continues to charge the environment's duration-scaled
step penalty. The duration is taken from the selected typed macro action; it
is public to the policy/trainer and adds no observation feature, memory field,
or private-state access.

Both arms retain M13.5's now-adequate update-every-4 budget. Consequently the
comparison isolates target time-discounting, not optimizer exposure, target
copy cadence, reward, masking, action semantics, network size, demonstrations,
curriculum, hierarchy, or observability. It does not reuse any M13.5 model,
development seed, validation seed, audit seed, checkpoint, or selected
replicate.

## Fresh split, timing-only preflight, and ceilings

| Partition | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Timing-only preflight | 3720–3723 | `m136_timing_{northeast,southwest,northwest,southeast}` | Runtime measurement only; never evaluated or scored. |
| Development | 3800–3839 | `m136_dev_{northeast,southwest,northwest,southeast}` | Fitting and diagnostics only. |
| Validation | 3900–3919 | `m136_validation_{northeast,southwest,northwest,southeast}` | Open once only after the frozen development matrix passes. |
| Sealed audit | 4000–4019 | `m136_audit_{northeast,southwest,northwest,southeast}` | Open once only after validation passes. |

The four scored conditions stay persistent reference, renewal/morphology,
event relocation, and compound. M10 public scan coverage and the scripted
state-oracle ceiling must pass 40/40 in every development condition before
candidate scoring. These are mechanics checks only; the learned policy never
receives their task options, IDs, or results.

Before the full matrix, run one 750-episode candidate-only timing preflight
over the timing-only split using training seed `20260815`, the final batch,
replay, and update-every-4 configuration. It records elapsed wall time,
decisions, updates, numerical-thread count, worker start method, and a source
fingerprint, but produces no scored policy, episode evaluation, or candidate
selection. Its conservative full-fit projection is
`preflight_elapsed_seconds * 16 * 1.25`. Proceed only if that projection is at
most 35 minutes per learned fit. Otherwise publish the timing result and
redesign execution only; do not reduce episodes, choose a faster seed, or run
the scored matrix under an unbounded budget.

## Arms, training budget, and parallel execution

Use paired PCG64 training seeds `20260815`, `20260816`, and `20260817`. For
each seed, run exactly:

1. **Duration-aware candidate:** M13.5 complementary-mask DQN with
   update-every-4 and `gamma_effective = .99 ** duration_seconds`.
2. **Unit-discount null:** identical M13.5 complementary-mask DQN with
   update-every-4 and `gamma_effective = .99`.
3. **Mask-aware macro random:** no fitting; uniform over the same public mask.

Every learned fit is exactly 12,000 episodes and visits each development
condition/seed pair 75 times. Candidate and null use identical initial weights
and action/RNG seed for their paired replicate. All six learned fits are
reported; no seed, checkpoint, or model may be selected after results are
seen.

Run the six fits in independent spawned worker processes with up to eight
workers, assigning every numerical library one thread per worker. The parent
serializes weights, validates fingerprints, then evaluates and strictly
replays them. This parallelism is an execution optimization only. A different
worker count, thread count, batch size, episode count, or update cadence must
not reuse the same artifact path.

## Development gate

Before scoring, coverage and the scripted ceiling must pass 40/40 in every
development row. Every duration-aware candidate replicate must reach, in each
condition, at least 36/40 survival, 36/40 maintenance completion, `.85` mean
safe-drive fraction, and 36/40 completed forced-recovery chains in relocation
rows. It must record zero public-mask violations and zero redundant selected
`GO_*` macros.

For every paired seed, candidate minus unit-discount null must be at least
`.10` pooled maintenance and `.10` pooled survival over the 160 development
episodes. Candidate minus mask-aware random pooled survival must be at least
`.20`. Every predeclared candidate replica must pass; a passing seed does not
open validation, and a failure is final for this protocol.

## Replay, validation, and audit

Every learned replay transition stores the selected action duration, effective
discount, reference interval, target-copy count, raw Q vector, selected masked
macro, public mask, policy-visible observation/features, memory before/after,
reward, outcome, terminal fields, RNG/model/source hashes, and one-way split
marker. Strict replay recomputes the public macro/action, duration, effective
discount, masked action, and policy memory. Random replay restores its
pre-choice PCG64 state. Evaluation always uses serialized development weights,
never a retrained policy.

Validation and audit use the identical per-condition and paired-delta gates on
their respective fresh splits. Validation opens once only after all development
gates pass; audit opens once only after validation passes. Any failure is
published unchanged.

## Limits

M13.6 can show whether semantically correct duration-aware bootstrapping makes
the dense-update public masked DQN reproducible on this simulator task. It
cannot prove that target discounting is the only remaining instability source,
establish learned perception/motor control, or justify use of M13.5's sealed
validation/audit splits.
