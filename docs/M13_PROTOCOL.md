# M13 protocol — state-oracle persistent-maintenance RL baseline

Protocol version: `m13-state-oracle-persistent-rl-r1`

## Status, purpose, and bounded claim

M13 is the first learned baseline for M10's `persistent_maintenance` task. It
tests whether a reward-trained state-oracle macro policy can survive a full
160-step horizon while repeatedly feeding, playing, resting, and recovering
from an event-triggered stale pickup. It is **not** an RGB, hybrid, VLM/VLA,
raw-control, or contact-physics result.

M12 remains an exploratory RGB behavior-cloning diagnostic. Its 80/80 survival
and recovery evidence, and its 61/80 maintenance-completion result (including
3/20 in the compound row), are useful failure evidence only. The M10
development and validation suites used by M12 (1700–1819) are retired from M13
training, policy selection, and scoring. M13 does not train from M12 labels,
traces, weights, failure classifications, or validation results.

M13 keeps M10's deterministic world, persistent task mechanics, fixed 160-step
horizon, guarded typed-skill executor, and completion requirement of at least
three feed cycles, three play cycles, and two rest cycles. It changes neither
the environment mechanics nor the action executor.

## Frozen M13 task, split, and layout contract

The implementation must add the following M13-only layout identifiers. They
must have no identifiers, seed ranges, or generated traces in common with M10,
M11, or M12. Each condition is evaluated on each listed seed, so validation
and audit each contain 80 episodes.

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 2000–2039 | `m13_dev_northeast`, `m13_dev_southwest`, `m13_dev_northwest`, `m13_dev_southeast` | Training, algorithm design, and development diagnostics only. |
| Validation | 2100–2119 | `m13_validation_northeast`, `m13_validation_southwest`, `m13_validation_northwest`, `m13_validation_southeast` | One frozen final-checkpoint evaluation after the M13 implementation and training choices are frozen. Never train, tune, or select a checkpoint on it. |
| Sealed audit | 2200–2219 | `m13_audit_northeast`, `m13_audit_southwest`, `m13_audit_northwest`, `m13_audit_southeast` | Open exactly once, only after validation passes. No retuning, retraining, or second audit after it is opened. |

The old M10 audit suite (1900–1919) is not reused. It remains historical M10
mechanics evidence, rather than becoming an M13 model-selection shortcut.

The four condition names and reset controls below are frozen. Only the layout
identifier changes by split. The new layout specifications must retain M10's
bounded room/task semantics while providing independent seeded spawn ranges.

| Condition | Layout suffix | Frozen reset controls beyond `task_id=persistent_maintenance` and `layout_id` |
| --- | --- | --- |
| `persistent_reference` | `northeast` | `food_variant=orange`, `toy_variant=ball`, `camera_control=scan_v2`, `initial_scan_sector=north` |
| `renewal_and_morphology` | `southwest` | `food_variant=purple`, `food_shape_variant=capsule`, `toy_variant=cube`, `agent_shape_variant=capsule`, `camera_control=scan_v2`, `initial_scan_sector=south` |
| `event_relocation` | `northwest` | `food_variant=blue`, `food_shape_variant=box`, `toy_variant=capsule`, `lighting_variant=dim`, `camera_control=scan_v2`, `initial_scan_sector=east`, `event_relocation_on_first_pickup=true` |
| `compound` | `southeast` | `food_variant=red`, `food_shape_variant=capsule`, `toy_variant=cube`, `agent_shape_variant=box`, `dynamics_variant=grippy`, `blocked_distractor=true`, `distractor_xy=[-0.04, 0.04]`, `camera_control=scan_v2`, `initial_scan_sector=west`, `event_relocation_on_first_pickup=true` |

Before any split is scored, the M10 offline segmentation-coverage procedure
must pass for that split's four layouts: initial and replenished food, toy, and
rest target must be visible in a four-sector public scan for every seed; the
relocated food and compound distractor must additionally be visible in every
applicable episode. Segmentation belongs only to this offline mechanics check;
it is never an M13 policy input. Development coverage may be checked while
developing. Validation coverage is run immediately before its one evaluation;
audit coverage is the first step of opening the audit.

## Permitted state-oracle policy boundary

M13 uses a versioned `state_oracle` observation adapter, not environment
methods. Its complete live input at decision time is:

```text
agent_xy: float32[2]
food_xy: float32[2]
toy_xy: float32[2]
rest_xy: float32[2]
drives: float32[3]                 # [satiety, energy, boredom]
holding_food: {0, 1}
prior_outcome: ActionOutcome index
policy_owned_memory: fixed M13 memory state only
```

`rest_xy` is a required public addition to the M10 `state_oracle` observation
space before M13 training. It must be exposed through the normal observation
contract and trajectory records, never fetched through `_rest_xy`, `_state`,
or another private environment method. The other listed fields retain their
normal state-oracle meanings. `food_xy` remains a coordinate even while food
is in its deterministic renewal cooldown; the policy must infer availability
from its own prior consume outcome and elapsed chosen durations, not read a
private food flag or timer.

The policy may persist only this deterministic, serializable memory, updated
from the preceding permitted observation, its chosen macro/skill and duration,
and the next `prior_outcome`:

```text
feed_count: 0..3                 # cap at 3 after a successful CONSUME
play_count: 0..3                 # cap at 3 after a successful PLAY begun at boredom >= 0.60
rest_count: 0..2                 # cap at 2 after a successful REST begun at energy <= 0.45
food_cooldown_bucket: 0..5       # set to 0 after successful CONSUME; add each chosen duration,
                                 # clamp at 5; 5 means "respawn may now be available"
last_macro: START or one M13 macro
pending_recovery: {false, true}  # set after a blocked PICK_UP; clear after successful CONSUME
```

The outcome is associated with the previously selected macro. A blocked
`PICK_UP` triggered by M10's relocation is therefore recoverable from the next
fresh `food_xy` observation, but it is not given a disturbance label. No other
memory, environment write, `info` field, private state, reward component, or
hidden timer is permitted.

The following are prohibited at inference and during action selection:

- task IDs, layout IDs, seed, reset options, condition names, or split labels;
- `info`, reward decomposition, terminal-cause fields, event/resource flags,
  cycle counters, food availability/respawn timers, simulator coordinates not
  named above, segmentation, renderer state, or private environment objects;
- M10/M11/M12 traces, M12 teacher labels or weights, and any scripted policy
  decision;
- direct mutation of the world, controller, configuration, or trajectory.

The scalar reward returned by `env.step` is available to the training update
only after a transition. It is not an extra inference feature. Offline reports
may inspect privileged mechanics fields to calculate the declared metrics, but
they must be recorded separately from the policy-visible transition stream.

## Macro and typed-action representation

The learned action is one of eight discrete macros:

```text
GO_FOOD, PICK_UP, CONSUME, GO_TOY, PLAY, GO_REST, REST, WAIT
```

A fixed, non-learned compiler maps a macro to the existing typed skill API:

| Macro | Typed skill | Target and duration rule |
| --- | --- | --- |
| `GO_FOOD` | `walk_to` | `food_xy`; duration is `max(0.1, distance(agent_xy, food_xy) / walk_speed_per_second)` |
| `PICK_UP` | `pick_up` | zero target, `0.1` seconds |
| `CONSUME` | `consume` | zero target, `0.1` seconds |
| `GO_TOY` | `walk_to` | `toy_xy`; same bounded travel-duration rule |
| `PLAY` | `run_around` | zero target, `1.0` second |
| `GO_REST` | `walk_to` | `rest_xy`; same bounded travel-duration rule |
| `REST` | `rest` | zero target, `1.0` second |
| `WAIT` | `idle` | zero target, `1.0` second |

The compiler neither corrects an invalid macro nor substitutes an interaction
after travel. Guarded outcomes remain observable through `prior_outcome` on the
next decision. `SCAN`, relative-coordinate skills, and action parameters not
listed above are out of the M13 action space. This keeps the learned result on
the same M10 typed-skill substrate while making recovery a fresh state-oracle
decision after a blocked pickup rather than a hidden scripted repair.

## Reward and training specification

M13 uses M10 reward version `m13-persistent-reward-r1`, whose immutable
configuration is the M10 persistent configuration:

```text
max_episode_steps=160
satiety_decay_per_second=0.018
energy_decay_per_second=0.012
boredom_gain_per_second=0.020
play_success_boredom_threshold=0.60
play_boredom_reduction=0.55
eat_satiety_gain=0.55
food_respawn_seconds=5.0
rest_cycle_energy_threshold=0.45
rest_energy_gain=0.65
task_success_reward=1.0
step_penalty_per_second=0.01
invalid_action_penalty=0.1
disturbance_recovery_reward=0.05
```

For avoidance of doubt, learning receives the unmodified environment scalar:
the time penalty, the existing invalid-action penalty, `+1.0` only for
horizon maintenance completion, `+0.05` for the existing recovery event, and
the terminal depletion penalty. There are no M13 macro bonuses, cycle-count
bonuses, hand-authored priority rewards, teacher labels, demonstrations, or
potential-based shaping. Any change to this list, the reward calculation, the
adapter, macro compiler, split table, or state encoding creates a new protocol
version and invalidates M13 comparison.

The baseline is deterministic tabular Q-learning over a fixed encoded state.
Coordinates are represented only as relative target vectors
`food_xy-agent_xy`, `toy_xy-agent_xy`, and `rest_xy-agent_xy`, each coordinate
clipped to `[-1.8, 1.8]` then bucketed into nine equal-width bins (edges
`[-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8]`). Drives use five
bins with edges `[0.0, 0.15, 0.45, 0.60, 0.90, 1.0]`. The encoded state also
includes `holding_food`, `prior_outcome`, and the permitted memory values
above. This representation, including bin edges and capped memory values, must
be serialized and hashed.

The complete reproducible training budget is 12,000 episodes with no early
stopping or validation checkpoint selection. Use Q-learning with
`alpha=0.20`, `gamma=0.99`, and epsilon-greedy exploration linearly decayed
from `1.00` to `0.05` over the first 9,000 episodes and held at `0.05` for the
last 3,000. Use NumPy PCG64 seed `20260806`. Episode `e` (zero based) uses
condition `e mod 4` in the condition-table order and seed
`2000 + ((e // 4) mod 40)`; each development condition/seed pair is therefore
visited exactly 75 times. The exported final Q table after episode 12,000 is
the only candidate scored on validation.

## Required ablations

All ablations use the same macro set, compiler, M13 configuration, 12,000
episode schedule, development condition/seed schedule, and their own fixed
PCG64 stream derived from the base seed by adding 1, 2, or 3. They receive no
teacher labels.

| Policy | Difference from full M13 policy |
| --- | --- |
| `macro_random` | Uniformly samples the eight macros at each decision. It receives the same compiler but has no fitted values or memory. |
| `no_drive_q` | Zeros all three drives both when encoding training transitions and when selecting at evaluation. Other permitted fields and the same Q-learning budget remain. |
| `no_memory_q` | Resets all policy-owned memory to its canonical empty state before every decision, during both training and evaluation. Current observation and `prior_outcome` remain available. |

Randomness, learned tables, adapter definition, compiler, and configuration
are fingerprinted before and after each frozen evaluation. No policy object may
be modified while validation or audit episodes are running.

## Measurements and frozen gates

For each condition independently, report 20 episode records and the following
metrics with Wilson 95% intervals for binary rates:

- survival: horizon reached without satiety or energy depletion;
- maintenance completion: survived horizon with at least 3 feed, 3 play, and
  2 rest cycles; report each episode's three counts and their per-condition
  minimum and mean;
- safe-drive fraction: fraction of post-action steps satisfying satiety
  `> 0.15`, energy `> 0.15`, and boredom `< 0.90`;
- event/resource counts and terminal-cause counts;
- recovery chain for `event_relocation` and `compound`: forced first pickup
  produces `blocked`, then a later successful `consume` of relocated food
  occurs before termination. The chain is action-agnostic: state-oracle M13
  is not required to perform RGB scans. Its intermediate actions, outcomes,
  and fresh food-relative vector must be recorded.

The full M13 policy passes validation only if every one of the four conditions
independently satisfies all applicable gates:

| Gate | `persistent_reference` / `renewal_and_morphology` | `event_relocation` / `compound` |
| --- | ---: | ---: |
| Survival | at least 18/20 | at least 18/20 |
| Maintenance completion | at least 18/20 | at least 18/20 |
| Mean safe-drive fraction | at least 0.85 | at least 0.85 |
| Forced stale pickup then later consume recovery | not applicable; report 0 required episodes | at least 18/20 required episodes |

Additionally, across the matched 80 validation condition/seed pairs, full M13
must exceed each of `macro_random`, `no_drive_q`, and `no_memory_q` by at least
0.10 in maintenance-completion rate, and must exceed `macro_random` by at
least 0.20 in survival rate. Report paired per-episode deltas for all three
comparisons; an aggregate may never substitute for a failed per-condition
gate. An ablation tie or a failed full-policy gate is a failed M13 validation,
not a reason to change thresholds or reuse validation for tuning.

After validation passes, the identical frozen full policy (not a new fit) is
run once on all 80 sealed-audit episodes. The audit reports the same condition
rows and applies the same gates. It need not rerun ablations. A failed audit is
reported as a failed audit; it does not authorize a retry on the same audit
suite.

## Reproducibility, replay, and audit record

The report must contain a protocol fingerprint over this document's versioned
constants, M10 mechanics/config fingerprint, M13 state-adapter schema,
encoder/bin edges, macro compiler, all split/condition tables, algorithm
hyperparameters, and source revision. It must separately contain hashes for
the final policy table, each ablation table, the training schedule, and every
trajectory file. A result missing any required fingerprint is unscorable.

Every scored validation or audit episode writes a schema-versioned JSONL trace
with the complete reset metadata and configuration, policy fingerprint,
policy-visible observation/encoded state, policy-owned memory before and after
the decision, selected macro, compiled typed action, scalar reward, outcome,
termination flags, and the existing replay-critical resource/cycle fields.
Privileged offline metric fields are stored in a separate sidecar keyed by
episode/step and are never handed back to the policy.

`replay_and_validate` (or a stricter successor) must replay every scored
validation and audit trace from clean reset and verify exact actions,
policy-visible observations, rewards, outcomes, resource events, cycle
counters, and terminal flags. The evaluator must also rerun policy inference
from the recorded permitted observation and memory and require the recorded
macro/action on every step. Any mismatch invalidates that split's report.

The audit procedure is ordered and one-way:

1. Freeze and hash the final validation-passing policy and source revision.
2. Run audit coverage and the M10 privileged mechanics ceiling on the audit
   split; if either fails, record the failure and do not score the learned
   policy.
3. Run the frozen full policy once across all 80 audit episodes and write all
   traces before examining aggregate scores.
4. Replay every trace, publish per-condition results and failures, and retain
   the immutable artifact directory. Do not retrain, edit the policy, or rerun
   the audit seeds.

## Acceptance and limits

M13 is accepted only when the M10 mechanics/coverage ceiling and every M13
validation and sealed-audit gate above pass with complete replay evidence. The
only justified claim is that a reward-trained state-oracle macro policy,
restricted to the declared public state adapter and typed skills, maintained
the M10 needs under the new fixed conditions. It does not establish learned
perception, hidden-state robustness beyond the declared memory, general
physical manipulation, or broad embodied competence.
