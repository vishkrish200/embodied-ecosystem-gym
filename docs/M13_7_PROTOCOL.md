# M13.7 protocol — fail-fast target-cadence screen

Version: `m13.7-target-cadence-screen-r1`

## Purpose

M13.5/M13.6 establish that dense updates are necessary but yield only 2/8 fully passing replicas. Dense update-every-4 also increased target refreshes in environment time. M13.7 tests that remaining training-dynamics confound only: hard target-copy cadence.

## Fresh screen partitions

| Partition | Seeds | Use |
| --- | --- | --- |
| Screen training | 4400–4419 | Fit only. |
| Screen probe | 4420–4427 | One-shot screen scoring only. |
| Future confirmation development | 4500–4539 | Reserved; unopened. |
| Future confirmation validation | 4600–4619 | Reserved; unopened. |
| Future confirmation audit | 4700–4719 | Reserved; unopened. |

Screen layouts are a new `m137_screen_{northeast,southwest,northwest,southeast}` family; probe layouts are `m137_probe_*`. No M13.5/M13.6 layout, seed, policy, validation, or audit is reused.

## Fixed boundary and arms

Keep M13.6's public 30 features/memory, complementary mask, eight macros/durations, reward, gamma `.99`, replay/warmup/batch, network, optimizer, epsilon schedule, and update-every-4 fixed. Use paired seeds `20260826` and `20260827`; each arm has identical initial/action RNG within a seed.

Run exactly 4,000 episodes for each of:

1. Control: hard target copy every 1,000 optimizer updates.
2. Candidate A: hard target copy every 250 updates.
3. Candidate B: hard target copy every 100 updates.

Run six spawned one-thread jobs concurrently. A short run is a rejector, not evidence of a robust policy.

## Promotion and stop rule

Score serialized weights once on the 32-episode probe matrix (four conditions x eight seeds). Reject a candidate if either paired seed is at least `.10` below its 1,000-update control in pooled maintenance or survival, or if replay/mask checks fail. Promote at most one candidate only if it is non-inferior on both seeds and has the larger pooled maintenance delta; ties choose the slower cadence (250). If neither candidate promotes, stop: no 12,000-episode run.

A promoted setting earns, but does not prove, a future eight-seed 12,000-episode confirmation on the reserved 4500–4719 splits. Its all-replica gate remains M13.6's original gate.

## Replay and limits

Record target cadence/count, traces, source/model/RNG hashes, and screen/probe marker. Strict learned/random replay is required. M13.7 cannot establish a baseline or open confirmation holdouts; it only decides whether a target-cadence candidate warrants confirmation.
