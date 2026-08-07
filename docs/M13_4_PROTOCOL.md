# M13.4 protocol — public macro-precondition-constrained DQN

Protocol version: `m13.4-public-macro-preconditions-r1`

## Purpose and causal boundary

M13.3's fresh development run established that charging ordinary blocked
actions removes the repeated cheap-blocked-pickup loop, but does not establish
persistent maintenance: the full policy completed zero maintenance episodes in
every row. Its remaining dominant behavior was successful but useless
navigation: it repeatedly selected `GO_*` while already inside a target's
interaction radius. M13.4 tests one new, narrower explanation: a value learner
may be spending too much of a 160-decision horizon selecting macros whose
public typed-skill preconditions are already false, including navigation after
arrival.

M13.4 keeps M13.3's M10 mechanics, state-oracle observation boundary,
30-feature `30→64→64→8` Double-DQN, macro compiler/durations, policy memory,
M13.3 reward (including the ordinary `-0.10` blocked-action penalty), training
budget, replay/optimizer settings, and cycle/survival/recovery gates. The one
change is a deterministic **public macro-precondition mask** used when selecting a
macro, both during epsilon exploration and when taking the Double-DQN maximum.
The network still emits eight Q values; the mask is not appended to its input
vector or stored as private policy state.

This is a policy change, not an environment or reward change. It therefore can
show learned long-horizon choice *among currently feasible typed skills*. It
cannot show that the learner discovered the elementary physical preconditions
of those typed skills from trial and error.

## Exact public complementary mask

For observation `o`, fixed public configuration `c`, and the compiled action
duration of a macro, the eligible set is:

| Macro | Eligibility rule |
| --- | --- |
| `GO_FOOD` | `||agent_xy - food_xy|| > c.pickup_radius`. |
| `PICK_UP` | `||agent_xy - food_xy|| <= c.pickup_radius` and `holding_food == false`. |
| `CONSUME` | `holding_food == true`. |
| `GO_TOY` | `||agent_xy - toy_xy|| > c.toy_interaction_radius`. |
| `PLAY` | `||agent_xy - toy_xy|| <= c.toy_interaction_radius`. |
| `GO_REST` | `||agent_xy - rest_xy|| > c.rest_interaction_radius`. |
| `REST` | `||agent_xy - rest_xy|| <= c.rest_interaction_radius`. |
| `WAIT` | Always eligible. |

No rule may inspect `food_available`, respawn timers, disturbances, task/layout
IDs, reset controls, `info`, authoritative counters, reward terms, private
state, segmentation, or past artifacts. In particular, food availability is
not observable to this policy, so a pickup near a location with unavailable
food remains eligible and can legitimately receive the existing blocked
outcome/reward.

The mandatory forced relocation transition is preserved. Before that first
pickup, the public observation still places the agent within the food pickup
radius, so `PICK_UP` is eligible. Relocation occurs inside the environment only
after macro selection; neither the mask nor the DQN sees an event label.

Masked greedy choice is the lowest-index maximum among eligible Q values.
Masked epsilon choice is uniform over eligible macros. The Double-DQN target
uses the same mask recomputed from the public next observation. `WAIT` is
always eligible. A mask never rewrites an
action, changes its duration, or grants a successful outcome.

## Fresh split, conditions, and ceilings

No M13.4 use overlaps M10–M13.3, including M13.3's opened development result
or every earlier validation/audit suite.

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 3200–3239 | `m134_dev_northeast`, `m134_dev_southwest`, `m134_dev_northwest`, `m134_dev_southeast` | Fixed fitting and diagnostics only. |
| Validation | 3300–3319 | `m134_validation_northeast`, `m134_validation_southwest`, `m134_validation_northwest`, `m134_validation_southeast` | Open once after frozen development gate pass. |
| Sealed audit | 3400–3419 | `m134_audit_northeast`, `m134_audit_southwest`, `m134_audit_northwest`, `m134_audit_southeast` | Open once, only after passing validation. |

The four controls remain persistent reference, renewal/morphology, event
relocation, and compound. Their public scan coverage and the scripted
state-oracle M10 ceiling must pass each split. The ceiling is mechanics
evidence only, never a policy input or training trace.

## Training arms and frozen budget

Each learned arm trains for 12,000 episodes under each paired PCG64 training
seed `20260809`, `20260810`, and `20260811`, visiting every development
condition/seed pair 75 times per fit. No replicate may be selected after its
result is seen: all three frozen fitted weights are evaluated and reported.
Epsilon is exactly
M13.3's 1.00-to-.05 inclusive linear schedule over episodes 0–9000, followed
by .05. Replay capacity/warmup/batch/update cadence, target copies, Double-DQN,
Adam, Huber, architecture, feature compiler, macro compiler, reward, and
source revision remain M13.3's frozen values.

The required six arms are:

1. **Complementary-mask full DQN** — the M13.4 candidate.
2. **Interaction-only mask DQN** — `GO_*` remains always eligible, while only
   the interaction rules from the table apply. This records whether removing
   redundant near-target navigation, rather than masking interaction attempts
   alone, is necessary.
3. **Mask-null full DQN** — identical code, reward, weights seed, budget, and
   public inputs, but every macro is eligible. This is the causal comparator;
   M13.3's result is context, not a substitute for a fresh paired arm.
4. **Complementary-mask no-drive DQN** — the candidate with only its three drive features
   zeroed.
5. **Complementary-mask no-memory DQN** — the candidate with policy-owned memory cleared
   after each step.
6. **Seeded complementary-mask macro-random** — uniform among the same eligible macros.

All initialization, action/RNG stream, mask implementation, feature/compiler
implementation, Adam beta/epsilon, Huber delta, reward config, split data,
and source hashes are fingerprinted. No demonstrations, curriculum, extra
training, network expansion, or tuning against validation/audit is allowed.
The M13.3 reward remains fixed in every arm. A reward-null arm is not repeated:
M13.3 already isolated that reward repair; M13.4's fresh same-split controls
isolate the mask parameterization instead.

## Development gate

Before validation, every development row must have coverage and scripted
ceiling 40/40. Every complementary-mask full replicate must have, in every row, at least 36/40
survival, 36/40 maintenance completion, mean safe-drive fraction `.85`, and
at least 36/40 completed recovery chains in relocation rows. It must also
exceed interaction-only, mask-null, no-drive, and no-memory maintenance by at
least `.10` over its 160 paired development episodes and exceed complementary-mask
random survival by at least `.20`. The report additionally gives the across-replicate
mean and minimum; a single passing replicate does not authorize validation.

Mask conformance is an independent hard gate: every recorded selected macro
must be eligible under the recorded public observation and recomputed mask;
every forced relocation row must retain its stale-pickup opportunity; and no
mask computation may read non-permitted fields. The protocol tests assert the
float32 `<=`/`>` boundary exactly and assert that no drive threshold is used:
drive thresholds decide whether an interaction is useful or cycle-eligible,
which remains the learned scheduling problem. Ordinary blocked outcomes are
reported separately, but are not automatically errors because unavailable food
is intentionally hidden.

Failure publishes the full development report unchanged and leaves validation
and audit unopened.

## Validation, replay, and audit

Validation runs all six arms once per frozen training replicate over the 20
fresh seeds per condition. It
requires the existing per-condition 18/20 survival, maintenance, safe-drive,
and relocation-recovery gates for every complementary-mask full replicate, plus
the same full-arm deltas versus interaction-only/mask-null/no-drive/no-memory
and the `.20` survival delta versus complementary-mask random. It must load the exact serialized development weights; no
retraining occurs on validation.

Every trace records policy-visible observation, 30-vector, Q vector, mask bit
vector, eligible macro names, selected macro/action, memory before/after,
reward/outcome, ordinary versus forced-exempt blocked classification, counters,
terminal fields, policy/model/RNG/source hashes, and a one-way split-open
marker. Strict learned-policy replay recomputes features, mask, constrained action,
environment transition, and memory from a clean reset. Random replay restores
the recorded pre-choice PCG64 state and recomputes the eligible-set draw.

Only a passing validation may open audit. Audit runs all three frozen
complementary-mask full replicas and the scripted ceiling
and scripted ceiling once on 3400–3419, replays every trace, and publishes any
failure unchanged.

## Limits

M13.4 would establish only learned state-oracle scheduling over a fixed,
publicly feasibility-constrained macro interface. It would not establish
learned perception, raw motor control, discovery of action preconditions,
physical manipulation, or broad embodiment.
