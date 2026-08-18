# M13.12 development protocol: one-factor PPO learning-rate comparison

Status: **development-only, non-promotional; frozen before execution**.

M13.12 tests one narrow hypothesis suggested by M13.11-r2's near-zero clip
fraction and small approximate KL: corrected PPO updates may have been too
conservative at the inherited `3e-4` learning rate. This does not reinterpret
M13.11-r2, which remains a clean negative.

## Frozen intervention and unchanged boundary

Two paired corrected-PPO arms start from identical weights and sampling seeds:

1. `ppo_lr_3e4_baseline`: unchanged M13.11-r2 learning rate `3e-4`.
2. `ppo_lr_1e3_candidate`: learning rate `1e-3`.

The learning rate is the only changed factor. Both arms retain the corrected
actor gradient, 2,048-row multi-episode rollouts, 30 public features, eight
macros, explicit policy memory, complementary mask, M13.9 reward and duration
semantics, `30 -> 64 -> 64` actor-critic, normalized advantages, four epochs,
clip `.20`, entropy `.01`, GAE lambda `.95`, gamma `.99`, max gradient norm
`.50`, and a 400,000-decision target. No grid, adaptive schedule, recurrence,
warm start, label, oracle loss, reward change, or environment change is allowed.

## Fresh partitions

| Partition | Seeds | Layouts | Access |
| --- | --- | --- | --- |
| `development_fit` | 6100--6119 | `m1312_fit_*` | canonical training |
| `development_check` | 6120--6139 | `m1312_check_*` | one post-fit evaluation |

Training seeds `20261321`--`20261324` are fresh and paired across arms. The
canonical ledger is ordered `fit -> check` and fingerprints the protocol,
sources, layouts, seeds, and rates. The capped plumbing smoke uses only the
already-open M13.11-r2 fit partition 6040--6059 and cannot access the canonical
M13.12 ledger. No 6060--6079 result may be used for tuning.

## Gate and stop rule

The primary gate is unchanged in spirit from M13.11-r2: the `1e-3` candidate
must achieve at least one full-objective success in every training-seed
replica. Per-seed, per-condition, safety, entropy, KL, clipping, action,
gradient, artifact, ledger, coverage, and strict replay evidence are reported.
The paired `3e-4` arm is a fresh-data baseline, not a tuning arm.

Failure stops the hypothesis as a clean development negative. A pass permits
only a proposal for a separately frozen fresh screen. It does not authorize a
screen, confirmation, audit, M14, or M15.
