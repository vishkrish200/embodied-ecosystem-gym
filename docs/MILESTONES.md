# Milestones

## M0 — Contract and scaffold

Define schemas, package layout, action and observation contracts, deterministic configuration, and tests that enforce the Gymnasium API. Exit when `reset`/`step` work in a minimal non-physics reference environment and a seed-replay test passes.

**Status: complete (2026-08-01).** The repository now has a deterministic state-oracle reference environment, typed actions and outcomes, versioned configuration, a Gymnasium contract check, seeded-reset coverage, and an explicit Find and eat transition test. The M0 environment is intentionally non-physics; M1 replaces its transition layer with MuJoCo while retaining the public contract.

## M1 — Find and eat vertical slice

Implement one MuJoCo room, a creature, food, satiety, `walk_to`, `pick_up`, and `consume`. Add a scripted policy, a fixed set of seeds, a trajectory writer, and a regression video. Exit when the scripted policy clears 95% of fixed in-distribution seeds and trajectories replay.

**Status: complete (2026-08-02).** The M1 environment replaces M0's reference transition layer with a deterministic MuJoCo room while preserving its public state-oracle observation and typed skill contract. A 20-seed scripted policy reaches 20/20 task successes, each episode writes a versioned JSONL trace, all traces replay against clean resets, and `python -m ecosystem_gym record-video` produces the seed-7 regression MP4 from the same Gym API.

## M2 — Benchmark and baseline

Add task registration, metrics, reward configuration, held-out layouts, and an oracle-state RL baseline. Exit when the benchmark CLI produces one comparable report and the learned baseline beats random.

**Status: complete (2026-08-03).** The registered `find_and_eat` task has two fixed training spawn layouts and two held-out spawn layouts, each evaluated across 20 fixed seeds. `python -m ecosystem_gym benchmark --output ...` writes one versioned report with the exact reward configuration, terminal-reason counts, and comparable train/held-out success, reward, and step metrics. With 400 deterministic Q-learning training episodes, the learned state-oracle policy reaches 40/40 held-out successes while the raw random-action policy reaches 1/40; the report labels this as same-room spawn-distribution generalization, not visual or geometry generalization.

## M3 — Perception and recovery

Add egocentric rendering, hybrid observation mode, RGB behavior cloning data, camera/layout variation, and disturbance recovery. Exit when oracle/hybrid/RGB results are separately reported and failure reasons are inspectable.

**Status: complete (2026-08-03).** The environment now supports state-oracle, hybrid, and agent-mounted RGB observations. Hybrid food detections are deterministic red-pixel segmentations of its RGB input, never simulator coordinates. `perception-benchmark` reports all three modes separately over two training and two held-out spawn layouts, three camera offsets, and clean versus food-relocation conditions; it records disturbance events, recovery actions, and named action failure reasons. `collect-bc` writes 64x64 RGB frames with privileged teacher action labels, including recovery examples. The M3 visual results are deliberately labeled calibrated color-servo baselines. A follow-on learned-RGB gate fits a deterministic NumPy local-target regressor on fixed RGB candidate-geometry features from red/orange demonstrations, measures a 0/40 random-RGB baseline, and passes its declared blue/purple held-out threshold (28/40 successes); this is a color-shift gate in the same room, not broad end-to-end visual generalization.

## M4 — Drives and generalization

Add boredom, toys, competing drives, and held-out dynamics/object distributions. Exit when task success, survival, collisions, and robustness are reported together.

**Status: complete (2026-08-03).** The MuJoCo room now has a visible toy with explicit location and shape variants. `play_when_bored` requires `run_around` near that toy and reduces boredom; `competing_drives` starts with both low satiety and high boredom, uses drive values to prioritize food, and rejects a play-before-food route. `m4-benchmark` evaluates fixed train and held-out spawn, food-color, toy-shape, and bounded-dynamics conditions, reporting task success, survival, reward, boundary-contact/clipping counts, and each condition separately. The current kinematic controller has no physical contact dynamics, so boundary contacts are reported rather than mislabeling them as physical collisions; food color and toy shape are visual variants with shared interaction semantics. The included drive-aware oracle controller reaches 40/40 held-out episodes per task.

## M5 — Toy room and adapter

Build a thin replay/live viewer over the Gym. Optionally translate compatible skill actions to an external virtual-pet runtime. Exit when the same seed and action sequence produces the same logged trajectory through both CLI and viewer.

**Status: complete (2026-08-03).** `python -m ecosystem_gym viewer` serves a dependency-free local browser panel over a `ViewerSession`. Its controls, frame rendering, live drives/outcomes, recording, and replay all call the wrapped `EcosystemEnv` API. Viewer traces use trajectory schema 0.2 with an episode-metadata header containing reset options and the full environment configuration; replay reconstructs those conditions exactly and remains compatible with schema 0.1 step-only traces. No external virtual-pet adapter is included because it is optional and would add a second runtime boundary.

## M6 — Learned drive arbitration

Train a learned policy that chooses food versus play from state-oracle observations and drives, then compare it with fair ablations and replay it through the viewer. Exit when it clears the full M4 held-out matrix, demonstrates a counterfactual food/play decision, and records an exactly replayable viewer trace.

**Status: complete (2026-08-03).** `m6-benchmark` trains one deterministic tabular Q-learning policy across M4 train tasks and conditions. The policy selector receives public state-oracle fields only; it never receives task IDs, reset variants, `info`, or private environment values. It achieves 40/40 held-out successes for both M4 tasks, while the same-budget no-drive ablation reaches 28/40 on competing drives and macro-random reaches 12/40 there. A same-geometry counterfactual probe selects food under low satiety/high boredom and play under high satiety/high boredom. `m6-demo` records its fixed held-out competing-drives rollout through `ViewerSession`, then replays the trace exactly. This remains a state-oracle macro-control result, not learned RGB multi-drive control.

## M7 — Learned RGB drive arbitration

Train a learned selector from RGB-derived local object evidence and public drives, then compare it with no-drive and macro-random ablations. Exit when it passes an appearance/dynamics-held-out M4 matrix, changes food/play action under a same-frame drive counterfactual, and records an exactly replayable viewer trace.

**Status: complete (2026-08-04).** `m7-benchmark` fits a deterministic tabular behaviour-cloning selector from task-teacher macro labels. At inference it receives only RGB, drives, holding state, the prior outcome, and policy-owned interaction memory; a fixed colour adapter supplies local food/toy geometry and the additive `walk_relative` skill grounds it without oracle object positions or global pose. On the fixed 10-seed visual-held-out matrix, it reaches 40/40 play successes and 39/40 competing-drive successes, while the no-drive ablation is 0/40 and 39/40 respectively and macro-random is 28/40 and 13/40. The same RGB frame selects food at low satiety and play at higher satiety. The gate covers appearance and dynamics changes on each task's seen layout; spatial-layout scores are diagnostics because this policy has no learned relocalization or search policy. `m7-demo` records a held-out RGB rollout through `ViewerSession` and replays it exactly. This is learned macro arbitration with a fixed visual adapter, not end-to-end learned vision or reinforcement learning.

## M8 — Benchmark validity and sequential RGB recovery

Freeze the evaluation protocol before controller work, add public camera scanning and local candidate pickup, and measure sequential RGB recovery without global coordinates. Exit when all named adversarial conditions run on 20 fixed seeds with Wilson confidence intervals, an explicitly privileged oracle ceiling, replayable adverse traces, and no policy pass gate that can hide a failed condition.

**Status: complete (2026-08-05).** `m8-benchmark` freezes `range(20)` across reference, camera-sector occlusion, blocked food-like distractor, relocation, unseen visual geometry, and camera-pose conditions. The fixed RGB colour-component scan/recovery baseline receives only RGB, drives, holding state, prior outcome, and policy-owned memory; its 95% Wilson intervals are reported beside a separately named state-oracle ceiling. It reaches 20/20 on reference, occlusion, blocked distractor, and camera pose, 19/20 post-relocation completions after 20 recorded relocations, and 0/20 on withheld visual geometry. That 0/20 result is retained as a visual-geometry limit, not averaged away or treated as a hidden policy failure. `post_disturbance_completion` is action-agnostic, so the sequential `walk_relative` route is measured without changing M3's legacy `recovery_action` field. `m8-demo` records scan, disturbance, and post-disturbance-completion fields through the unchanged `ViewerSession` and replays exactly. This is a benchmark-validity and fixed-baseline result, not learned perception, collision-aware navigation, or physics robustness.

## M8.1 — Memory as the sequential-recovery question

**Status: complete (2026-08-05).** M8.1 preserves M8 v1 and trains equal-output RGB ridge behavior-cloning classifiers on disjoint seeds 100–119, then evaluates seeds 0–19 across reference, occlusion, blocked-distractor, and relocation conditions. The recurrent classifier receives only its prior public action/outcome in addition to the feed-forward policy's RGB input. It wins 74 paired held-out episodes missed by feed-forward and loses none, a predeclared net advantage of 0.925 over 80 pairs; the visible `m81_landmark` scene change is diagnostic-only and excluded from that verdict. Both learned policies use M8's fixed RGB component adapter to ground local targets, so this is evidence for learned action selection with short-horizon memory, not end-to-end learned vision or general geometry reasoning.

## M8.2 — Sealed external-validity audit

Freeze the M8.1 train configuration and fitted weight fingerprint before defining a separate post-result suite. Evaluate the same fixed M8 policy, feed-forward RGB BC policy, recurrent RGB BC policy, and privileged state-oracle ceiling on 20 new seeds per condition. Exit to M9 only if the recurrence-minus-feed-forward paired advantage remains at least 0.10 and its deterministic 95% bootstrap lower bound exceeds zero; otherwise record the negative result and do not tune on this suite.

**Status: complete (2026-08-05, negative result).** `m82-benchmark` seals M8.1's train seeds 100–119, original test seeds 0–19, all result/diagnostic conditions, ridge setting, a SHA-256 protocol fingerprint, and a fitted-weight fingerprint, then evaluates new seeds 200–219. The four new conditions combine new northwest/southeast agent-food quadrants, blue/purple food, dim illumination that keeps the cyan agent visible, different scan sectors, a new landmark placement, a blocked distractor, relocation, and one bounded motion variant. The fixed controller scores 0/20, 0/20, 0/20, and 1/20, while feed-forward and recurrent RGB BC each score 0/20 in all four and the state oracle scores 20/20 throughout. The paired recurrent advantage is 0.0 over 80 episodes with deterministic bootstrap 95% CI [0.0, 0.0], so the predeclared stop rule rejects M9. This is evidence that the M8.1 recurrence result does not transfer to the new spatial/appearance suite; it does not justify classifier tuning, end-to-end perception claims, or geometry/physics claims.

## M8.3 — One-factor failure diagnosis

Before changing the policy, run the frozen M8.1 policies on new seeds with one factor changed at a time and report initial fixed-adapter visibility, first policy action, and end-to-end success. Keep M8.2 sealed; these diagnostics may diagnose M8.1 but may not be used as M8.2 training data.

**Status: complete (2026-08-05).** `m83-diagnostics` freezes M8.1's original protocol and weights again, then tests seeds 300–319. The recurrent policy remains 20/20 on isolated camera-sector, purple-appearance, dim-lighting, agent-spawn, and landmark changes, but falls to 7/20 when only the food quadrant moves northwest; the fixed baseline is 17/20 there and the state oracle remains 20/20. The M8.2 failure is therefore not evidence that colour alone broke the controller. It exposes a narrow spatial-grounding and compound-shift problem in a classifier that emits actions from a fixed pooled-RGB projection; the next policy must learn visual target grounding and be trained/evaluated closed-loop without opening M8.2.
