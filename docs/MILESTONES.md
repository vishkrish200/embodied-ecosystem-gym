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
