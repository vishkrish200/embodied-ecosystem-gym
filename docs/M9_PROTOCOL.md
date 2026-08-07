# M9 protocol — integrated RGB drive maintenance

## Purpose and bounded claim

M9 tests the smallest missing v1 capability: **one deterministic, learned
RGB-and-drive policy can complete a single longer-lived maintenance episode by
feeding and playing, while it has to scan, handle food relocation, and reject a
food-like blocked distractor.**  It is a simulator result over typed,
kinematic skills, not a claim of end-to-end learning, semantic object
understanding, collision-aware navigation, or contact-rich manipulation.

This is an integration milestone, not another M8 food-grounding audit.  M8.7
learns local food grounding, whereas M7 uses a separate fixed RGB adapter and
an authored scan/retry control path; neither existing task requires food and
toy completion in one episode.  The relevant seams are the RGB-only public
observation contract in `ecosystem_gym/env.py:433-450`, the one-success task
termination in `ecosystem_gym/env.py:350-385`, M7's fixed colour adapter and
macro-to-skill mapping in `ecosystem_gym/experiments/m7.py:61-195`, and M8.7's learned
food/agent grounder in `ecosystem_gym/experiments/m87.py:159-195`.

## Frozen protocol

### Environment contract

Add one `maintain_needs` task.  The episode ends successfully only after the
same world has both `food_consumed` and `toy_played`; an early feed or play is
an observable intermediate transition, never an invisible success.  The task
uses the existing `scan_v2`, `walk_relative`, `pick_up_relative`, `consume`,
and `run_around` skills and their explicit outcomes.  Its RGB observation
remains exactly `{rgb, drives, holding_food, prior_outcome}`.  The policy may
keep its own action/outcome history, but receives no task id, reset options,
`info`, camera sector, coordinates, segmentation, or private state.

The evaluation shell must require a complete public four-sector scan before
its first target-directed walk and again after a relocation.  Relocation occurs
immediately after that first target-directed approach, so the old local target
produces an explicit failed pickup and the policy must start a new public scan
cycle. The policy makes the `scan` choice itself; this is a scored behavioural
requirement, not a controller prelude. An offline MuJoCo-segmentation coverage audit may inspect
food, toy, distractor, and agent geoms, but it runs before fitting/scoring and
is unavailable to every policy method.  This follows the M8.4 visibility
boundary in `ecosystem_gym/experiments/m84.py:159-207`.

### One policy definition

`IntegratedRgbDrivePolicy` is one inference object with a single `act`
method.  It owns all three policy decisions:

1. A learned RGB grounding stack inside the one policy combines the frozen
   M8.7 food-candidate model with an M9 multiclass toy/agent/distractor model,
   then maps component centroids to local skill offsets.  Food-like distractors
   share the food-candidate appearance and must be resolved through public
   outcomes and policy memory. Segmentation labels are development-only teacher
   labels.
2. A learned sequential macro selector takes those policy-produced component
   features, public drives, holding state, prior outcome, and policy-owned
   scan/action memory, then chooses `scan`, walk-to-food, walk-to-toy,
   `pick_up_relative`, `consume`, or `run_around`.
3. The same selector chooses recovery after a `blocked` pickup or relocation;
   it may not delegate scanning, target selection, or retry to an authored
   episode controller.  Converting its chosen macro and learned local offset
   into the existing typed Gym action is action serialization, not a second
   policy.

The policy is structured behaviour cloning, with learned perception and
learned macro/recovery selection.  It is deliberately not an end-to-end
pixel-to-joint learner.  A state-oracle controller is permitted only as a
separately named ceiling and teacher-label source; its coordinates must not
cross the RGB policy boundary.

### Splits and rows

All splits, layout ids, seed ranges, condition dictionaries, trainer
hyperparameters, and report schema are constants versioned in `m9.py` before
the first fit.  Development data is the only source of teacher labels and
model selection.  Validation may choose between predeclared policy variants;
the audit is never used for training, threshold adjustment, or model choice.

| Split | Seeds | Role |
| --- | --- | --- |
| Development | 1300–1339 | Fit the multiclass grounder and macro selector on development-only layouts. |
| Validation | 1400–1419 | Select only among predeclared variants and enforce the development gate. |
| Sealed audit | 1500–1519 | One post-selection external score on disjoint layouts; do not tune after it. |

Each validation and audit row has 20 seeds.  Every matrix includes all three
elements—food, toy, and four-sector scanning—and each adverse condition is
required rather than averaged away:

| Row | Initial drives | Required adverse condition | Held-out variation |
| --- | --- | --- | --- |
| `hungry_bored_reference` | low satiety, high boredom | none | new layout and spawn distribution |
| `bored_then_feed_blocked` | high satiety, high boredom | food-like blocked distractor; a blocked pickup must lead to re-observation/retry | toy shape and camera pose |
| `hungry_bored_relocation` | low satiety, high boredom | food moves after the first target-directed approach; policy must scan again before eating | lighting and food appearance |
| `balanced_drive_compound` | moderate satiety, high boredom | blocked distractor plus relocation | disjoint layout, dynamics, food/agent morphology |

The layouts are disjoint from M8.4–M8.9 and from one another across
development, validation, and audit.  Per-row coverage requires that a full
public scan sees food and toy at reset, and that the relocated food remains
visible after its disturbance.  If coverage fails, the benchmark raises before
any RGB score is emitted.

## Metrics and gates

The report records per row and aggregate values for the learned policy and the
state-oracle ceiling:

- `completion_rate`, Wilson 95% interval, successes, and seeds for the full
  feed-and-play task.
- `feed_rate`, `play_rate`, survival rate, mean steps, and terminal outcomes,
  so aggregate task success cannot hide one unmet need.
- `scan_before_first_target_rate`, `post_relocation_rescan_rate`, blocked
  distractor encounters, and successful blocked-distractor recovery rate.
- coverage counts for food, toy, and relocated food; all must be 20/20 in each
  row before policy scoring.
- exact replay result for one adverse viewer trace, and model/protocol
  fingerprints in the report.

The learned policy passes validation only if every coverage assertion passes,
the state-oracle ceiling completes 20/20 in every row, aggregate learned
completion is at least 75%, each learned row is at least 70% (14/20), each
required scan/recovery action rate is at least 75% on the rows where it
applies, and no evaluation action or policy feature uses private data.  The
audit uses the same gates and reports a negative result unchanged if any one
fails.  These thresholds are intentionally lower than a product guarantee:
they prevent a single easy condition from masking an integration collapse and
make the 20-seed uncertainty visible, rather than claiming solved autonomy.

## Implementation plan

1. Extend `tasks.py` with disjoint M9 layouts and `maintain_needs`, then extend
   `env.py` so food and toy are persistent intermediate achievements and the
   task succeeds only after both.  Add policy-visible trajectory fields only
   when they already exist in the public observation; preserve M0–M8 replay.
2. Implement `m9.py` with constants for the frozen protocol, an offline
   multi-object coverage audit, deterministic teacher collection, the learned
   integrated grounding stack, `IntegratedRgbDrivePolicy`, oracle ceiling, report,
   fingerprints, and replayable viewer demo.  The module must reject custom
   validation/audit seeds just as M8.7 does.
3. Add `m9-benchmark` and `m9-demo` CLI commands.  Keep the viewer unchanged:
   it must continue forwarding the exact environment API and trajectory writer
   (`ecosystem_gym/viewer.py:81-184`).
4. Add focused environment, protocol-boundary, leakage, coverage, split,
   gate, and replay tests.  Then run the full suite, validation report, one
   sealed audit, and a deterministic adverse-condition replay.
5. Update the milestone record and README with achieved metrics and the exact
   limitation: M9 is one structured learned RGB policy over a bounded simulator
   action interface, not end-to-end embodied intelligence.

## Acceptance criteria

- `maintain_needs` cannot terminate on feed alone or play alone, and it
  replays exactly through both `replay_and_validate` and `ViewerSession`.
- The RGB policy's public input test rejects task ids, reset options, `info`,
  state-oracle coordinates, camera sector, and segmentation; only offline
  development labels may use segmentation.
- Every validation/audit row has exactly 20 frozen seeds, disjoint layouts,
  and complete food/toy/relocated-food visibility before scoring.
- One policy instance produces scan, food, toy, blocked-distractor recovery,
  and post-relocation recovery actions in the same episode; no M7 or M8 shell
  controls its scan or retry sequence.
- The report emits per-row Wilson intervals, an explicit oracle ceiling,
  coverage evidence, fingerprints, and an unambiguous pass/fail verdict.
- Existing M8.9 artifacts and all current tests remain valid.
