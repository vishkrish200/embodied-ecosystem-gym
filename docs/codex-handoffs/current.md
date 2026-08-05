# Codex Handoff: M9 integrated RGB agent

Updated: 2026-08-06
Repo/path: /Users/vishnukrishnan/.codex/worktrees/fe6c/embodied-ecosystem-gym
Branch/worktree: `codex/m89-visual-morphology-audit` at `e5f3003` (pushed to origin). `origin/main` is at M8.8, so continue from this branch until M8.9 is merged.

## Current Goal

- Re-center the project on the v1 PRD claim: demonstrate one policy, using only public RGB observations plus drives, that can maintain needs and complete simple grounded tasks through a longer episode with food, toy, scanning, relocation, and a blocked distractor.
- Do not start another isolated M8 food-grounding audit. The next work should be an integration milestone, provisionally M9.

## Current State

- The project has a deterministic MuJoCo Gym, typed skill actions/outcomes, `state_oracle`, `hybrid`, and RGB observation modes, deterministic JSONL replay, and a thin viewer sharing the exact environment loop.
- M6 demonstrates state-oracle drive arbitration. M7 demonstrates RGB-driven macro selection with a fixed colour adapter and teacher-labelled behaviour cloning. M8.7 learns only local RGB target grounding; scanning, pickup, retry, and coordinate calibration remain authored.
- M8.9 is a sealed external visual-morphology audit of frozen M8.7: 75/80 learned successes and 80/80 privileged-oracle successes. Rows are 20/20 blue capsule food, 20/20 red box food plus relocation, 19/20 blue box-agent plus blocked distractor, and 16/20 purple capsule-agent plus landmark.

## Completed

- M0–M5: reproducible environment/replay/viewer foundation.
- M6–M7: bounded drive-arbitration evidence with explicit limits.
- M8–M8.9: observability repair, sealed evaluation protocols, a genuine M8.5 transfer failure, M8.6 diagnosis, frozen M8.7 successor, and two held-out audits.
- M8.2 remains an invalid legacy artifact: the target was invisible in all public scan sequences. Do not reuse it as either a negative result or training data.

## Files Touched Or Investigated

- `docs/PRD.md`: original product claim and release criteria.
- `docs/PROJECT_CONTEXT.md`: source-of-truth Gym/viewer boundary and policy-input constraints.
- `docs/MILESTONES.md`: current evidence and limits through M8.9.
- `ecosystem_gym/env.py`: MuJoCo environment and public observation contract.
- `ecosystem_gym/m7.py`: RGB macro arbitration with fixed colour grounding.
- `ecosystem_gym/m84.py`, `ecosystem_gym/m87.py`, `ecosystem_gym/m89.py`: learned local RGB grounder and sealed audits.
- `ecosystem_gym/viewer.py`, `ecosystem_gym/trajectory.py`: thin viewer and replay boundary.

## Commands And Checks

- `uv run pytest`: passed after M8.9.
- `uv run python -m ecosystem_gym m89-benchmark --output artifacts/reports/m89-sealed-visual-morphology-audit.json`: 75/80 learned, 80/80 oracle, all coverage gates pass.
- `uv run python -m ecosystem_gym m89-demo --trace artifacts/trajectories/m89-sealed-visual-morphology-audit-demo.jsonl`: successful five-step replayable rollout.

## Known Failures Or Blockers

- The current system is not an end-to-end learned embodied agent. It has isolated learned pieces joined by authored control.
- The environment is not contact-rich or collision-aware: M4 reports a kinematic controller, and M8 geometry/morphology changes are primarily rendered visual shifts.
- The toy room is a correct thin viewer, not a developed user-facing virtual-pet product.
- The PRD says all six task families should have at least 20 fixed evaluation seeds. M7's visual-held-out matrix uses 10 seeds per row, so the release criterion is not cleanly closed as written.

## Decisions And Constraints

- Preserve deterministic replay and the thin viewer boundary.
- Policies must not receive task IDs, reset options, `info`, oracle coordinates, segmentation, or other private state. Offline teacher labels are permitted only when explicitly isolated from inference.
- Freeze an M9 protocol before fitting/selecting its policy. Require coverage auditing before any RGB score, per-condition results with confidence intervals, a privileged oracle ceiling, and a fresh held-out audit.
- Do not broaden into frontend work or claim real-world robotics, semantic object understanding, contact-rich physics, or end-to-end learning unless implemented and tested.

## Do Not Repeat

- Do not create another score-only M8.x audit of the frozen M8.7 food grounder; it would not materially test the PRD claim.
- Do not treat aggregate success as valid when target visibility has not been audited. M8.2's 0/80 was an impossible-observation protocol, not policy evidence.
- Do not call M7 or M8 end-to-end learned perception: M7 uses a fixed colour adapter and M8.7 uses offline segmentation labels during training plus authored control at deployment.

## Next Steps

1. Read the PRD, project context, milestones, and this handoff; inspect current `env.py`, M7, M8.7, viewer, and trajectory code. Summarize the current state and identify the smallest integrated M9 claim that would close the real v1 gap.
2. Write and review an M9 protocol before coding: a longer-lived RGB-only food/play environment with explicit initial-drive scenarios, public scanning, relocation, blocked distractor, held-out layout/appearance/dynamics rows, 20 seeds, coverage checks, confidence intervals, and an oracle ceiling. Specify whether one learned policy must own macro choice, target grounding, and search/recovery.
3. Implement only after that protocol is fixed. Preserve existing M8 results and use fresh development/validation/audit splits; publish a negative result rather than tuning against the final audit.

## Reactivation Prompt

We are continuing from this handoff: `/Users/vishnukrishnan/.codex/worktrees/fe6c/embodied-ecosystem-gym/docs/codex-handoffs/current.md`.
Read it first, inspect the current repo state, verify what still applies, summarize the state in 5 bullets, then continue from the Next Steps without relying on the old chat context.
