# M13 final synthesis and decision

Date: 2026-08-19

Status: **M13 state-oracle RL research lane closed as a negative result**.

## Decision

Stop the M13.x sequence. Do not run another adjacent reward, optimizer,
decomposition, seed-selection, or replay-plumbing variant. Do not open any
M13.14-r3 confirmation or audit partition.

The project has established that persistent maintenance is representable and
solvable through the public state interface, but the tested model-free RL
methods do not retain or discover that behavior reliably enough across seeds
and conditions. The useful engineering baseline is teacher imitation plus a
deterministic safety/commitment shield, not another M13.x learner tweak.

Any future RL research should begin as a new benchmark version with genuine
room for learning: suboptimal teachers, changing or unknown dynamics, resource
scarcity, partial observability, or longer delayed trade-offs. It must not tune
on the frozen M13 check traces.

## Evidence summary

| Milestone | Primary result | Decision |
| --- | --- | --- |
| M13.10 | imitation-only `64/64`; subsequent DQN fine-tuning `0/64` | representation is sufficient; RL destroys retention |
| M13.11-r2 | corrected PPO `33/320`; DQN `9/320` | correct implementation, unstable learner |
| M13.12 | PPO `1e-3` `179/320` vs `3e-4` `110/320`; candidate seeds `73,80,26,0/80` | aggregate improvement, all-replica failure |
| M13.13 | coverage abort before policy evaluation | no policy claim |
| M13.14 | teacher-minibatch validation abort before fitting | no policy claim |
| M13.14-r2 | all fits and policy replays completed; random diagnostic fingerprint abort before report | no official gate |
| M13.14-r3 | modular+anchor+shield `258/320`; monolithic+anchor+shield `273/320`; no-anchor `45/320`; no-shield `57/320` | development gate failed; no confirmation |

M13.14-r3 candidate replicas scored `49/80`, `67/80`, `73/80`, and
`69/80`. None passed every condition. All 1,440 traces replayed, the scripted
ceiling reached `80/80`, random reached `0/80`, and there were zero integrity,
replay, or unsafe-WAIT violations.

The r3 report has one non-promoting verifier defect: artifact summaries include
a valid `schema_version` field omitted from the verifier's canonical
reconstruction. All common artifact fields, hashes, loaded policy
fingerprints, and evaluation specs match. The development gate is already
false, so this defect cannot authorize confirmation or change the conclusion.

## What the experiments established

1. **The public boundary is expressive.** A 30-feature policy and the eight
   macros can express near-perfect persistent maintenance.
2. **Teacher anchoring matters.** Removing persistent anchor updates reduced
   r3 performance from `258/320` to `45/320`.
3. **The shield matters.** Removing the acting-loop shield reduced performance
   to `57/320`.
4. **Modular decomposition did not help.** The matched monolithic anchored
   control scored `273/320`, above the modular candidate's `258/320`.
5. **Seed and condition stability remain unsolved.** Candidate failures moved
   between compound, relocation, renewal, and persistent-reference conditions.
6. **The bottleneck is not replay or WAIT behavior.** R3 had complete replay,
   strong safety, complete relocation accounting, and zero unsafe-WAIT rate.

## Frozen split index

Opened evidence must never be repurposed as fresh development data:

- M13.10: `5600--5627`
- M13.11: `6000--6039` (implementation-invalid original)
- M13.11-r2: `6040--6079`
- M13.12: `6100--6139`
- M13.13: `6200--6239`
- M13.14: `6600--6639`
- M13.14-r2: `6720--6759`
- M13.14-r3: `6840--6879`

Every later confirmation/audit family remains sealed and unavailable. Unused
gaps also remain unavailable for opportunistic tuning under these protocols.

## Frozen evidence index

- [M13.10 result](M13_10_SCREEN_RESULTS.md)
- [M13.11-r2 protocol](M13_11_R2_DEVELOPMENT_PROTOCOL.md)
- [M13.12 result](M13_12_DEVELOPMENT_RESULTS.md)
- [M13.13 protocol](M13_13_POLICY_FAMILIES_PROTOCOL.md)
- [M13.13 abort](M13_13_DEVELOPMENT_ABORT.md)
- [M13.14 protocol](M13_14_MODULAR_ANCHORED_PROTOCOL.md)
- [M13.14 coverage preflight](M13_14_DEVELOPMENT_PREFLIGHT.md)
- [M13.14 abort](M13_14_DEVELOPMENT_ABORT.md)
- [M13.14-r2 protocol](M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md)
- [M13.14-r2 coverage preflight](M13_14_R2_DEVELOPMENT_PREFLIGHT.md)
- [M13.14-r2 abort](M13_14_R2_DEVELOPMENT_ABORT.md)
- [M13.14-r3 protocol](M13_14_R3_MODULAR_ANCHORED_PROTOCOL.md)
- [M13.14-r3 coverage preflight](M13_14_R3_DEVELOPMENT_PREFLIGHT.md)
- [M13.14-r3 result](M13_14_R3_DEVELOPMENT_RESULTS.md)

Large generated datasets, policy files, and traces remain Git-ignored locally.
Their hashes are recorded in the corresponding result/abort documents. This
document freezes the conclusion, not a new experiment surface.

## Product and research fork

The recommended product path is teacher imitation plus the deterministic
shield, followed by hybrid/RGB perception work with honest wording: this is
teacher-distilled maintenance, not a passed state-RL baseline.

If a learned-RL contribution remains the goal, create a new benchmark version
where the teacher is imperfect and adaptation is necessary. Do not continue
the decimal M13 ladder.

