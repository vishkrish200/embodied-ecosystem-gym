# M13.1 protocol — cycle-transition reward revision

Protocol version: `m13.1-cycle-reward-r1`

## Decision and bounded change

The original M13 state-oracle tabular-Q result is a completed **negative
baseline**: it survived its opened validation suite but completed no maintenance
episodes. Its sealed 2200–2219 audit remains unopened. M13.1 tests one narrow
hypothesis: the failure is caused by M13's sparse, horizon-only maintenance
reward, rather than by its observation boundary, macro interface, or tabular
representation.

M13.1 changes only the scalar training reward. It retains the M10 world and
mechanics, 160-step horizon, M13 public `state_oracle` adapter, eight macros,
typed-skill compiler, policy-owned memory, state encoder and bins, tabular
Q-learning algorithm, and all gates. It is not an M13 retry or a hidden policy
change. M12 remains a separate RGB behavior-cloning diagnostic and supplies no
data, weights, labels, thresholds, or scored rows to M13.1.

## Fresh split and condition contract

No M13.1 policy selection, coverage check, trajectory, or score may reuse M13
seeds 2000–2219, M12 seeds 1700–1819, or historical M10 suites. The four
condition controls and their meanings are exactly M13's; only the newly named
layouts and seed ranges below may be used.

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 2300–2339 | `m131_dev_northeast`, `m131_dev_southwest`, `m131_dev_northwest`, `m131_dev_southeast` | Fixed 12,000-episode fitting and implementation diagnostics only. |
| Validation | 2400–2419 | `m131_validation_northeast`, `m131_validation_southwest`, `m131_validation_northwest`, `m131_validation_southeast` | One final frozen-policy check, after source and development choices are frozen. |
| Sealed audit | 2500–2519 | `m131_audit_northeast`, `m131_audit_southwest`, `m131_audit_northwest`, `m131_audit_southeast` | Open exactly once only after a passing M13.1 validation. |

Conditions are `persistent_reference`, `renewal_and_morphology`,
`event_relocation`, and `compound`, with the same reset controls as the M13
protocol. Before scoring any split, the M10 public scan-coverage check must
pass for its layouts and seeds, including replenished food and relocation /
distractor visibility where applicable. Coverage is a privileged offline
mechanics check, never a policy input. The scripted state-oracle mechanics
ceiling must complete every scored split before learned scores are accepted.

## Frozen policy and action boundary

At each decision the policy sees only M13's public observation:
`agent_xy`, `food_xy`, `toy_xy`, `rest_xy`, three drives, `holding_food`, and
`prior_outcome`, plus the same deterministic serializable policy-owned memory
(capped feed/play/rest counts, food-cooldown bucket, last macro, and pending
recovery bit). It may not read task/layout IDs, seeds, reset controls, `info`,
reward components, private state, food timers, cycle counters, segmentation,
or any M10/M12/M13 trajectory or label.

Its only discrete choices remain `GO_FOOD`, `PICK_UP`, `CONSUME`, `GO_TOY`,
`PLAY`, `GO_REST`, `REST`, and `WAIT`. The existing fixed compiler maps them
to the same guarded `walk_to`, `pick_up`, `consume`, `run_around`, `rest`, and
`idle` typed skills with unchanged targets and durations. The learner still
does not receive a controller that repairs an invalid action.

The encoder, bins, Q-learning update, epsilon schedule, and training schedule
are copied verbatim from M13: 12,000 episodes; PCG64 seed `20260806`; four
development conditions in order; each 2300–2339 seed/condition pair exactly
75 times; `alpha=0.20`, `gamma=0.99`, epsilon 1.00 to 0.05 over 9,000 episodes
then 0.05. Full, no-drive, and no-memory tables use streams `+0`, `+1`, and
`+2`; the random macro ablation uses `+3` and has no fitted table or memory.
There is no early stopping, validation selection, or reward-tuning run.

## Reward version `m13.1-cycle-reward-r1`

M13.1 retains every M13 reward term unchanged:

```text
-0.01 * selected typed-skill duration
-0.10 for an invalid (not-success and not-blocked) action
+1.00 only for surviving the horizon with all maintenance requirements met
+0.05 for the existing disturbance recovery event
-1.00 on satiety or energy depletion
```

It adds three guarded rewards, computed by the environment **after** the typed
skill executes and only when the authoritative persistent counter actually
increments:

```text
+0.25 if feed_cycles increases (a successful persistent CONSUME)
+0.25 if play_cycles increases (a successful eligible PLAY at boredom >= 0.60)
+0.25 if rest_cycles increases (a successful eligible REST at energy <= 0.45)
```

There is no reward for a movement action, proximity, a pickup attempt, an
interaction attempt, a predicted counter increment, a raw drive value,
safe-drive time, a private availability flag, or an `info` field. These are
transition rewards, not additional observations: the policy learns from the
returned scalar after acting, but never sees an individual reward component or
the authoritative counter. Config defaults are zero, preserving all prior
tasks and the original M13 reward exactly.

## Validation, replay, and decision rule

Each 20-seed condition must independently reach: at least 18/20 survival,
18/20 maintenance completion, mean safe-drive fraction at least 0.85, and, in
the two relocation conditions, at least 18/20 forced stale-pickup then later
consume recoveries. Across matched 80 episodes, full M13.1 must exceed every
random/no-drive/no-memory ablation by 0.10 maintenance completion and random
by 0.20 survival. The thresholds are not tunable after validation opens.

Every validation or audit decision writes a replayable trace containing reset
metadata, config and protocol fingerprints, policy-visible observation,
encoded state, memory before/after, macro, compiled action, scalar reward,
outcome, resource/cycle fields, and terminal flags. Generic replay must match
the environment transition; strict replay must also reproduce every policy
macro/action from its permitted input and memory. Manifest hashes, policy
table hashes, source revision, training schedule hash, and all trace hashes
are required. Privileged metric sidecars are kept separate from policy inputs.

Validation opens once. If it fails, record all rows and leave the M13.1 audit
unopened; start M13.2 on a new protocol and new splits. If it passes, freeze
the exact full table and source revision, run coverage and the ceiling, then
run the M13.1 audit exactly once over its 80 episodes. An audit failure is
reported without retry. M13.2 is authorized only by a clean M13.1 validation
failure with working ceiling and replay, and will change representation (not
this reward version) on a separately frozen split family.

## Acceptance and limits

M13.1 is accepted only when validation and the one-shot audit both pass with
complete replay evidence. A passing result establishes only reward-trained
state-oracle macro maintenance under these fixed mechanics. It does not show
learned perception, raw control, physical manipulation, or broad embodiment.
