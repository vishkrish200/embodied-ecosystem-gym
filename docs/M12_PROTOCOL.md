# M12 protocol — learned persistent-maintenance successor

> **Status: exploratory diagnostic, not an accepted policy milestone.** M12
> established useful RGB survival and recovery behavior but did not pass the
> compound maintenance gate. Its validation rows were subsequently inspected
> while diagnosing variants, so this document and its report must not be used
> to select a final policy or justify a sealed M12 audit. M13 returns to the
> project's state-oracle RL baseline before another visual policy is attempted.

## Claim and training boundary

M12 fits one deterministic structured policy only from oracle-teacher macro
labels and public observations collected on four frozen M12 development
conditions using M10's two development layouts and seeds 1700–1719. The four
rows cover reference, renewable resources plus morphology, event relocation,
and compound blocked-distractor recovery. The teacher is development-only supervision: the deployed policy
does not receive its coordinates, task ID, reset controls, `info`, segmentation,
or M11 failure labels. The initial fit preceded use of M10 validation seeds
1800–1819; later diagnosis inspected those rows, so they are no longer an
independent model-selection holdout. M10 audit seeds 1900–1919 remain reserved
for M13's final protocol, not a retry of M12.

The policy object may contain learned visual grounding and learned macro
selection weights. At inference it receives only RGB, drives, holding-food,
prior outcome, and policy-owned scan/candidate/cycle memory; it is not passed
an environment object. Typed kinematic skills remain the action substrate.

## Learned scope

Development trajectories label scan, food, toy, rest, pickup, consume, play,
and rest macros. The selector is fitted from those trajectories rather than an
exhaustive authored decision table. A fixed public execution shell completes
each four-view scan and turns a successful walk into exactly one corresponding
interaction attempt; it then retires all agent-relative offsets and scans
again. Within a four-view scan, candidate order is derived from the learned
local displacement rather than the view that happened to be processed first.
Candidate retirement after a blocked pickup, scan-cycle memory,
repeated resource-cycle memory, and recovery are state transitions in the
same learned-policy rollout contract. The M11 failure atlas is evaluation
evidence only and never provides M12 training labels.

The validation layouts, seeds, and exact reset rows are disjoint from this
development suite; a pass is bounded transfer, not an in-distribution score.

## Frozen evaluation and ablations

Each of M10's four validation conditions runs all 20 frozen seeds. M12 compares
the full policy with equal-training-budget no-memory and no-drive policies. The
no-memory policy resets scan/candidate/cycle state after every action; the
no-drive policy receives zeroed drive features during both fitting and
inference. The frozen M11 M9 baseline is evaluated on the same condition/seed
pairs only after M12 fitting completes.

Paired improvement is the mean of `survived_M12 - survived_M11` across all 80
matched condition/seed episodes. It must be at least 0.20 and its deterministic
10,000-resample percentile-bootstrap 95% lower bound must exceed zero.

## Gates

Every validation condition must independently reach all of:

- at least 15/20 horizon survivals;
- at least 15/20 maintenance-complete episodes, where every success has the
  environment's required 3 feed, 3 play, and 2 rest cycles;
- mean per-episode safe-drive fraction of at least 0.80, using M11's bands:
  satiety > 0.15, energy > 0.15, boredom < 0.90;
- for each relocation row, at least 15/20 complete stale-pickup -> four-scan
  rescan -> later-consume recovery chains.

The full policy must exceed both ablations on aggregate survival and
maintenance completion. The report must pin the protocol, fitted-policy, and
training-data fingerprints; replay one adverse trace exactly. M12 has no
sealed-audit score.

## Recorded result

The best frozen implementation result reached 80/80 horizon survival, complete
forced recovery chains, and a positive paired survival advantage over M11. It
completed maintenance in 61/80 episodes, however, with only 3/20 compound
episodes meeting the required feed/play/rest cycle counts. The failure is a
long-horizon scheduling limit: the agent remains safe but exhausts the horizon
before completing its final cycle. This is evidence to preserve, not a gate to
relax or tune against.
