# M13.3 protocol — recovery-safe blocked-action reward

Protocol version: `m13.3-blocked-action-r1`

## Purpose and causal boundary

M13.2 is a development-only negative diagnostic. Its public-state Double-DQN
collapsed to repeated cheap blocked pickups: blocked actions paid only their
duration cost, unlike ordinary invalid outcomes. M13.3 tests precisely that
reward/action-contract flaw. It retains M13.2's continuous 30-feature
`30→64→64→8` Double-DQN, M13.1 cycle rewards, M10 mechanics, state boundary,
memory role, eight macros/compiler, horizon, and success gates. It does not
add an action mask, change macro durations, enlarge the network, change replay
or optimizer settings, add demonstrations, or use earlier learned artifacts.

The only behavioral reward difference is `-0.10` on a `BLOCKED` outcome. The
one forced stale pickup that triggers M10 food relocation is exempt: that
single transition remains measurable as the recovery-chain start, but every
later blocked action is penalized. The exception is an environment reward rule
evaluated after the transition; it is not a policy input or an event label.

Before fitting, M13.3 also corrects prior protocol-conformance defects without
adding information: policy food cooldown is exact elapsed selected duration,
not `ceil(duration)` buckets; play/rest memory uses public pre-action drives
evolved by the frozen public configuration over the selected duration, matching
environment eligibility; and epsilon reaches .05 exactly at episode 9,000.

## Fresh split, conditions, and ceilings

No M13.3 use overlaps any M10–M13.2 seed/layout, opened validation, or sealed
audit.

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 2900–2939 | `m133_dev_northeast`, `m133_dev_southwest`, `m133_dev_northwest`, `m133_dev_southeast` | Fixed fitting and development diagnostics only. |
| Validation | 3000–3019 | `m133_validation_northeast`, `m133_validation_southwest`, `m133_validation_northwest`, `m133_validation_southeast` | Open once after code and development policy are frozen. |
| Sealed audit | 3100–3119 | `m133_audit_northeast`, `m133_audit_southwest`, `m133_audit_northwest`, `m133_audit_southeast` | Open once only after passing validation. |

The four condition names and reset controls are exactly M13.1's. M10 public
scan coverage and the scripted public-state ceiling must pass each split. The
ceiling is offline mechanics evidence and never a policy or training input.

## Inference and reward contract

Policy input remains only `agent_xy`, `food_xy`, `toy_xy`, `rest_xy`, drives,
`holding_food`, `prior_outcome`, and declared serializable policy memory. Its
fixed 30-vector and eight output macros remain M13.2's. Task/layout IDs,
seeds, reset controls, `info`, reward components, authoritative counters,
resource flags/timers, segmentation, private state, and prior run artifacts
are prohibited.

The scalar reward is M13.1's time/invalid/terminal/recovery reward plus `.25`
only for each authoritative feed/play/rest cycle increment, and additionally:

```text
-0.10  if outcome == BLOCKED
  0.00  instead, only if this exact transition triggered forced food relocation
```

Environment defaults for this new penalty are zero so M0–M13.2 behavior is
unchanged. M13.3 reports ordinary blocked actions and exempt forced stale
pickups separately. A policy never receives either classification.

## Training, arms, and development gate

Training stays 12,000 episodes, 75 visits to each development condition/seed
pair, PCG64 base seed `20260808`, epsilon 1.00 linearly to .05 inclusive over
episodes 0–9000 then .05, M13.2 replay capacity/warmup/batch/update cadence,
Double-DQN, Adam, Huber, and target-copy settings. All policy initialization,
sampling, schedule, features, optimizer settings, compiler, reward config,
and source revision are fingerprinted.

Required arms are full M13.3; reward-null full DQN (identical code/config but
blocked penalty zero); no-drive M13.3; no-memory M13.3; and seeded macro
random. The reward-null arm is required because conformance repairs make
historical M13.2 an insufficient causal comparison.

Before validation, each development condition must show ceiling/coverage 40/40
and the full arm must reach at least 36/40 survival, 36/40 maintenance, mean
safe-drive fraction .85, and 36/40 recovery chains where required. It must
have zero blocked-loop episodes, predeclared as eight or more consecutive
identical non-exempt blocked macros. Failure leaves validation/audit unopened.

## Validation, replay, and audit

Validation preserves M13.1's per-condition 18/20 survival, maintenance,
safe-drive, and relocation-recovery gates and full-policy deltas versus each
ablation (+.10 maintenance; +.20 survival versus random). It writes ceiling
and all five-arm traces, including policy-visible observation/30-vector,
memory before/after, macro, action, reward, outcome, ordinary/exempt blocked
classification, resource/cycle fields, terminal flags, policy/RNG/model hashes
and one-way split-open markers. Strict replay recomputes each learned action
from permitted input and frozen weights; random replay restores recorded RNG
state. Audit runs only the frozen full arm plus ceiling, once, after a passing
validation. Any failure is published unchanged and does not authorize a retry.

## Limits

M13.3 can establish only learned state-oracle macro maintenance under these
fixed mechanics. It cannot establish learned perception, raw motor control,
physical manipulation, or broad embodiment.
