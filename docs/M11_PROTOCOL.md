# M11 protocol — frozen M9 persistent-maintenance baseline

## Purpose

M11 measures the capability gap between M9's frozen single-cycle RGB policy and
M10's validated persistent-maintenance environment. It does not fit, tune, or
select a policy. The exact deterministic M9 parameter fingerprint is pinned in
`ecosystem_gym/experiments/m11.py`; an evaluation fails if the reconstructed policy differs.

## Frozen evaluation

M11 uses M10's four validation conditions and seeds 1800–1819 unchanged. M10's
coverage and privileged-oracle gate must pass before the frozen M9 policy runs.
The policy may receive only RGB, drives, holding state, prior outcome, and its
own memory. Offline reporting may inspect the policy's own RGB detections, but
does not give the policy segmentation, coordinates, task IDs, reset options, or
private environment state.

## Measures and failure atlas

Safe drive time means satiety above 0.15, energy above 0.15, and boredom below
0.90 after an action. Each condition reports 20-seed survival curves, completed
feed/play/rest cycles, safe-drive time, forced-recovery chains, resource and
disturbance interventions, and terminal causes.

Every step persists drives before and after the action, frozen-policy RGB
candidate offsets, selected rank and local offset, action outcome, scan state,
replenishment, relocation, and cycle counters. Each non-successful episode gets
one first observed failure classification tied to a recorded step. These labels
are diagnostic evidence for M12, not training labels and not a causal claim
beyond the recorded sequence.

## Acceptance criteria

- M10 validation still passes before baseline scoring.
- The frozen M9 fingerprint is identical before and after all 80 episodes.
- All four conditions and all 20 M10 validation seeds are reported.
- Every non-successful episode has a linked first-failure classification.
- No M10 development or sealed-audit seed is scored, and no M9 sealed audit is opened.
