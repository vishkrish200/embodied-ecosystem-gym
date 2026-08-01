# Milestones

## M0 — Contract and scaffold

Define schemas, package layout, action and observation contracts, deterministic configuration, and tests that enforce the Gymnasium API. Exit when `reset`/`step` work in a minimal non-physics reference environment and a seed-replay test passes.

**Status: complete (2026-08-01).** The repository now has a deterministic state-oracle reference environment, typed actions and outcomes, versioned configuration, a Gymnasium contract check, seeded-reset coverage, and an explicit Find and eat transition test. The M0 environment is intentionally non-physics; M1 replaces its transition layer with MuJoCo while retaining the public contract.

## M1 — Find and eat vertical slice

Implement one MuJoCo room, a creature, food, satiety, `walk_to`, `pick_up`, and `consume`. Add a scripted policy, a fixed set of seeds, a trajectory writer, and a regression video. Exit when the scripted policy clears 95% of fixed in-distribution seeds and trajectories replay.

## M2 — Benchmark and baseline

Add task registration, metrics, reward configuration, held-out layouts, and an oracle-state RL baseline. Exit when the benchmark CLI produces one comparable report and the learned baseline beats random.

## M3 — Perception and recovery

Add egocentric rendering, hybrid observation mode, RGB behavior cloning data, camera/layout variation, and disturbance recovery. Exit when oracle/hybrid/RGB results are separately reported and failure reasons are inspectable.

## M4 — Drives and generalization

Add boredom, toys, competing drives, and held-out dynamics/object distributions. Exit when task success, survival, collisions, and robustness are reported together.

## M5 — Toy room and adapter

Build a thin replay/live viewer over the Gym. Optionally translate compatible skill actions to an external virtual-pet runtime. Exit when the same seed and action sequence produces the same logged trajectory through both CLI and viewer.
