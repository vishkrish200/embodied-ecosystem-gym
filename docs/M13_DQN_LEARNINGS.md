# M13 Double-DQN learning index

This is a navigation index, not a rewrite of frozen evidence.  Every M13
source module, test, protocol, result, command, artifact, and split ledger
remains executable in `ecosystem_gym/experiments/` and `artifacts/`.

| Study | Frozen isolated change | Opened / permanently unavailable | Outcome and supported learning | Prohibited retry or interpretation |
| --- | --- | --- | --- | --- |
| [M13](M13_PROTOCOL.md) | Sparse tabular state-oracle Q baseline | 2000–2039 dev, 2100–2119 validation; 2200–2219 audit unopened | Sparse reward did not establish maintenance. | Do not call state coordinates sufficient by themselves. |
| [M13.1](M13_1_PROTOCOL.md) | Cycle-transition rewards | 2300–2519 dev; validation/audit unopened | Encouraging fit did not survive the frozen validation. | Do not reopen its audit. |
| [M13.2](M13_2_PROTOCOL.md) | 30→64→64 Double-DQN representation | 2600–2819 development only | Continuous Q values still collapsed to blocked pickup loops. | Do not treat features as a planning solution. |
| [M13.3](M13_3_PROTOCOL.md) | Blocked penalty and public-memory timing | 2900–3119 development only | Removed repeated blocked loops, not long-horizon maintenance. | Do not score validation/audit. |
| [M13.4](M13_4_PROTOCOL.md) | Complementary public macro mask | 3200–3419 development only | Removed local invalid/redundant macros but was unstable. | Do not claim masks solve scheduling. |
| [M13.5](M13_5_PROTOCOL.md) | Dense update cadence | 3500–3719 development only | Strong mean effect, failed all-replica stability requirement. | Do not select the favorable seed. |
| [M13.6](M13_6_PROTOCOL.md) | Eight-seed stability replication | 4100–4319 development only | Dense updates were not a reproducible baseline. | Do not reinterpret duration discount post hoc. |
| [M13.7](M13_7_PROTOCOL.md) | Target-copy cadence screen | 4400–4419 fit, 4420–4427 probe; 4500–4719 sealed | Screen rejected; mechanics and replay passed. | No confirmation or audit. |
| [M13.8](M13_8_PROTOCOL.md), [result](M13_8_SCREEN_RESULTS.md) | Capped bounded cycle reward | 4800–4827 opened; **4900–5119 sealed** | Quota-then-WAIT failure; 1/64 full objective. | Never reuse 4900–5119. |
| [M13.9](M13_9_PROTOCOL.md), [result](M13_9_SCREEN_RESULTS.md) | Duration-aware public-drive potential | 5200–5227 opened; **5300–5519 sealed** | Potential/replay mechanics passed; unsafe WAIT rejected both replicas. | Never tune on 5220–5227 or use 5300–5519. |
| [M13.10](M13_10_PROTOCOL.md), [result](M13_10_SCREEN_RESULTS.md) | Imitation initialization then matched DQN | 5600–5627 opened; **5700–5919 sealed** | Imitation-only generalized 32/32; DQN erased it. | No M13.10 retry, confirmation, audit, or imitation claim for PPO. |
| M13.11 (r1, invalid implementation evidence) | Random-init masked semi-Markov PPO versus DQN | 6000–6039 opened | The recorded PPO collapse is not an algorithm result: the actor gradient sign was reversed and rollout updates flushed per episode. | Preserve r1 code/report/ledger/traces unchanged; never reinterpret it as a PPO negative. |
| [M13.11-r2](M13_11_R2_DEVELOPMENT_PROTOCOL.md), [result](../artifacts/reports/m1311r2-development.json) | Correct actor gradient and 2,048-row multi-episode rollout only | 6040–6079 opened; **6080–6099 unused** | Corrected PPO reached 33/320 full objectives but failed all-replica stability (0/80, 3/80, 20/80, 10/80); DQN reached 9/320. All 640 traces replayed. | Stop with this clean development negative. Do not open screen/confirmation/audit or M14/M15; any screen would require a separately frozen 6100+ protocol and authorization. |
| [M13.12](M13_12_DEVELOPMENT_PROTOCOL.md), [result](M13_12_DEVELOPMENT_RESULTS.md) | PPO learning rate `3e-4` versus `1e-3` only | 6100–6139 opened | The higher rate engaged KL/clipping and reached 179/320 versus 110/320, but remained seed-unstable (73/80, 80/80, 26/80, 0/80). All 640 traces replayed. | Preserve the clean negative. Do not select seeds, tune on 6120–6139, or launch later gates. |

The concise implication is not that the benchmark lacks an executable public
strategy: M13.10's frozen imitation-only arm demonstrates the public
representation can express one.  It is that the current off-policy DQN update
has not produced a stable random-initialized long-horizon learner.  M13.11 is
therefore a development-only optimizer comparison on fresh 6000–6039 data.
