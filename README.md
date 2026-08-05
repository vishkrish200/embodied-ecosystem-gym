# Embodied Ecosystem Gym

A reproducible embodied-agent benchmark with a first-party virtual-toy experience. The Gym owns MuJoCo state transitions, tasks, evaluation, and trajectory logs; the toy room is a thin human-facing client over the same environment.

Milestone 1 provides one deterministic MuJoCo room, one creature, one food item, normalized satiety/energy/boredom drives, and a complete `walk_to` -> `pick_up` -> `consume` task. The public Gymnasium contract remains state-oracle only for now; hybrid and RGB observations arrive in M3.

Milestone 2 turns that vertical slice into a small benchmark. It evaluates a raw random-action policy against a learned tabular state-oracle Q-learning policy over fixed training and held-out spawn-layout splits, then writes one comparable JSON report.

Milestone 3 adds an agent-mounted RGB camera, a hybrid mode whose food detection is computed from those pixels, controlled camera offsets, and a deterministic food-relocation disturbance. Its report keeps state-oracle, hybrid, and RGB results separate. The M3.5 gate trains a deterministic NumPy behavior-cloning controller from RGB-derived local image features, then evaluates red/orange training appearances separately from blue/purple held-out appearances.

Milestone 4 adds a visible toy, boredom-relieving play, a competing-drives task, and fixed train/held-out appearance, object, dynamics, and spawn-layout conditions. Milestone 5 provides a local browser viewer that forwards every reset, action, render, and trace write to the same Gym session; it does not own simulation state.

Milestone 6 closes the remaining control gap with one learned state-oracle policy that chooses between pursuing food and pursuing play from public object positions, held state, and drives. It is evaluated against a same-budget no-drive Q-learning ablation, a macro-random baseline, and the scripted M4 oracle; this is learned drive arbitration, not RGB-driven multi-drive control.

Milestone 7 moves that macro decision to RGB observations. A fixed, inspectable colour adapter extracts local food/toy offsets and grounds them through relative walking, while a compact tabular classifier learns food-versus-play selection from task-teacher macro labels and public drive values at inference. Its visual-held-out gate changes object appearance and movement dynamics on each task's seen layout; held-out spawn layouts are reported as a diagnostic rather than claimed as a navigation-generalization result.

Milestone 8 freezes a 20-seed sequential-RGB recovery protocol before selecting a policy. It reports camera-sector occlusion, a blocked food-like distractor, mid-episode food relocation, a withheld visual-geometry shift, and camera-pose variation with Wilson confidence intervals and a separately labeled state-oracle ceiling. Its RGB controller is a fixed colour-component scan/recovery baseline, not learned perception or physics robustness.

M8.1 asks the project’s actual research question: on disjoint training seeds, does prior action/outcome memory improve RGB recovery over an equal-output feed-forward behavior-cloning policy? It compares those two small NumPy learned action classifiers against the M8 fixed controller and an oracle ceiling on frozen 20-seed conditions. The colour-component target adapter remains fixed, so this is learned action selection with short-horizon memory, not end-to-end representation learning.

M8.2 is the external-validity audit. It freezes and fingerprints the M8.1 trainer and weights, then tests a new 20-seed suite of unseen spawn quadrants, camera sectors, blue/purple appearance, dim illumination, a new landmark placement, blocked distractors, and relocation. Its predeclared rule advances only if recurrence keeps a paired advantage of at least 0.10 with a positive 95% bootstrap lower bound; the current result is a negative transfer result, so there is no M9 implementation gate.

```bash
uv sync --group dev
uv run pytest

# Run the fixed 20-seed scripted baseline and retain replayable JSONL logs.
uv run python -m ecosystem_gym evaluate --trajectory-dir artifacts/trajectories/m1

# Validate any saved trace against a clean reset of the same seed.
uv run python -m ecosystem_gym replay artifacts/trajectories/m1/find-eat_seed-0007.jsonl

# Record the MuJoCo regression video through the public skill API.
uv run python -m ecosystem_gym record-video artifacts/regression/find-and-eat_seed-0007.mp4

# Train/evaluate both baselines and write the M2 benchmark artifact.
uv run python -m ecosystem_gym benchmark --output artifacts/reports/m2-find-and-eat.json

# Report state-oracle, hybrid, and RGB performance under camera/layout variation and recovery.
uv run python -m ecosystem_gym perception-benchmark --output artifacts/reports/m3-perception.json

# Write RGB frames and privileged teacher labels, then run the learned appearance gate.
uv run python -m ecosystem_gym collect-bc --output artifacts/datasets/m35-rgb-bc.npz
uv run python -m ecosystem_gym learned-rgb-gate --dataset artifacts/datasets/m35-rgb-bc.npz --output artifacts/reports/m35-learned-rgb.json

# Evaluate play/competing-drive tasks across held-out conditions.
uv run python -m ecosystem_gym m4-benchmark --output artifacts/reports/m4-drive-benchmark.json

# Train and evaluate learned food-versus-play arbitration, then record its viewer trace.
uv run python -m ecosystem_gym m6-benchmark --output artifacts/reports/m6-learned-drives.json
uv run python -m ecosystem_gym m6-demo --trace artifacts/trajectories/m6-competing-demo.jsonl

# Train/evaluate RGB drive arbitration and record one replayable RGB viewer trace.
uv run python -m ecosystem_gym m7-benchmark --output artifacts/reports/m7-rgb-drives.json
uv run python -m ecosystem_gym m7-demo --trace artifacts/trajectories/m7-competing-demo.jsonl

# Run the frozen sequential RGB-recovery protocol and replay one adverse trace.
uv run python -m ecosystem_gym m8-benchmark --output artifacts/reports/m8-rgb-recovery.json
uv run python -m ecosystem_gym m8-demo --trace artifacts/trajectories/m8-rgb-recovery-demo.jsonl

uv run python -m ecosystem_gym m81-benchmark --output artifacts/reports/m81-rgb-memory.json
uv run python -m ecosystem_gym m81-demo --trace artifacts/trajectories/m81-recurrent-demo.jsonl

uv run python -m ecosystem_gym m82-benchmark --output artifacts/reports/m82-external-validity.json
uv run python -m ecosystem_gym m82-demo --trace artifacts/trajectories/m82-frozen-rnn-demo.jsonl

# Diagnose one factor at a time before training any successor policy.
uv run python -m ecosystem_gym m83-diagnostics --output artifacts/reports/m83-one-factor-diagnostics.json

# Open the same environment loop in a local browser, with optional replayable logging.
uv run python -m ecosystem_gym viewer --trace artifacts/trajectories/viewer.jsonl
```

The M1 evaluation command reports a 1.0 success rate on its fixed in-distribution seed suite. The M2 report records registered layout IDs, the exact reward configuration, fixed evaluation seeds, terminal-reason counts, and train/held-out metrics for both baselines. M3's original visual policies remain calibrated color-servo baselines, while the separate learned-RGB gate is intentionally modest: it learns the image-geometry-to-local-target mapping over fixed RGB candidate features, so it proves the declared color shift rather than broad end-to-end visual generalization. M4's drive-aware oracle controller proves task/reset-condition mechanics. M6 is the learned counterpart: its selector receives no task ID, reset options, `info`, or private environment state, and a counterfactual same-geometry probe checks food-first versus play-first behavior. M7 keeps task IDs out of selector inference too, but its offline macro labels come from the training-task teacher, so it is a learned RGB macro classifier rather than end-to-end RL. M8 intentionally does not publish a policy pass gate: its fixed RGB baseline is 20/20 in the reference, camera-sector-occlusion, and blocked-distractor conditions, 19/20 after relocation, and 0/20 on the withheld visual geometry condition, with 95% Wilson intervals on every rate. The state-oracle ceiling is separate, and the 0/20 geometry result is a limit rather than a failure hidden by aggregation. M8.2 preserves that standard: its four held-out external conditions score 0/20, 0/20, 0/20, and 1/20 for the fixed controller, while both learned RGB classifiers are 0/20 in every condition and the oracle remains 20/20. The recurrence-minus-feed-forward paired advantage is 0.0 with a 95% bootstrap interval of [0.0, 0.0], so its sealed stop rule rejects M9. Viewer traces retain disturbance, post-disturbance-completion, and camera-sector evidence while `replay` continues to accept legacy 0.1 step-only traces.

Read [the PRD](docs/PRD.md) for the product boundary and [the milestones](docs/MILESTONES.md) for the delivery plan.
