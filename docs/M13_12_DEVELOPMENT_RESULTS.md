# M13.12 development result: higher-rate PPO remains unstable

Status: **clean development negative (2026-08-10)**.

M13.12 ran exactly one canonical comparison under the frozen
[protocol](M13_12_DEVELOPMENT_PROTOCOL.md). Both corrected-PPO arms used fresh
6100--6119 fit data, one 6120--6139 development check, fresh paired training
seeds, and 400,000 macro decisions. The only difference was learning rate:
`3e-4` baseline versus `1e-3` candidate.

## Primary result

| Arm | 20261321 | 20261322 | 20261323 | 20261324 | Total | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo_lr_3e4_baseline` | 80/80 | 4/80 | 0/80 | 26/80 | 110/320 | diagnostic baseline |
| `ppo_lr_1e3_candidate` | 73/80 | 80/80 | 26/80 | 0/80 | 179/320 | **fail** |

The candidate failed the predeclared all-replica gate because seed 20261324
had zero full-objective successes. Its paired per-seed differences from the
baseline were `-7`, `+76`, `+26`, and `-26`. The aggregate increase therefore
does not establish a stable learning-rate benefit.

Candidate successes were distributed across persistent-reference `59/80`,
renewal-and-morphology `40/80`, event-relocation `40/80`, and compound `40/80`,
but the seed-level failure remained decisive. Seed 20261324 had partial
maintenance and survival behavior in every condition without satisfying the
full objective once.

## Optimizer and integrity evidence

The intervention changed optimization in the intended direction. Candidate
approximate KL was `0.001406--0.002595` and clip fraction
`0.008206--0.020590`, versus baseline KL `0.000266--0.000378` and clip fraction
exactly zero. Candidate entropy was `0.6488--0.8857`; baseline entropy was
`0.9732--1.1252`. Every action was sampled in every training job.

All coverage checks passed, all eight jobs completed 196 update batches, and
all 640 serialized-policy traces strictly replayed. The canonical report is
`artifacts/reports/m1312-development.json`; the opened ledger fingerprint is
`25aab7623fa4a3edd4226e122cbcfad1a882dae66c8c0c79ecb0337b2570f0e4`,
and the report SHA-256 is
`a4a1d6a59b24d4aa3d56c1a393ccd0bef150121fc7b57079769950d625c7eff1`.

## Decision

Preserve this as a clean negative and stop. The result supports the narrow
claim that `1e-3` engaged PPO's trust-region mechanics more than `3e-4`; it
does not support a stable policy improvement. Do not select favorable seeds,
tune either rate on 6120--6139, or launch a screen, confirmation, audit, M14,
or M15 from this result.
