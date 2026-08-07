# M13.8 fresh screen result

Status: **rejected; confirmation not run**

Frozen protocol fingerprint:
`c67e8afeefba6d0f3ea90dec1120f98f4c68e4d75449c5b6b213482e9994c6ca`

Primary report: [`artifacts/reports/m138-screen.json`](../artifacts/reports/m138-screen.json)

## Integrity and preflight

- The public scan coverage, 80/80 scripted full-gate ceiling, specialist-return
  ranking, and strict preflight replay all passed.
- The balanced oracle beat every fixed feed-only, play-only, and rest/wait
  specialist by at least `6.91` mean candidate return in every condition.
- All four learned policies were serialized and reloaded before the one-shot
  probe opened.
- All `512/512` preflight and probe traces replayed without a conformance,
  inference, memory, raw-environment, or shaped-reward mismatch.
- Both learned arms in each pair had byte-identical initial parameter hashes.
- The report content hash is
  `af9cb7ae17f1f42feae49ad3c632c25ee711fb7e49df69bf2cfec6b098486b80`.

## Paired probe result

Each row pools four conditions x eight fresh probe seeds.

| Training seed | Arm | Survival | 3/3/2 maintenance | Decision-safe | Duration-safe | Full gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260901` | bounded candidate | `32/32` | `32/32` | `0.541` | `0.560` | `0/32` |
| `20260901` | legacy control | `31/32` | `31/32` | `0.932` | `0.910` | `29/32` |
| `20260902` | bounded candidate | `29/32` | `21/32` | `0.691` | `0.709` | `1/32` |
| `20260902` | legacy control | `32/32` | `32/32` | `0.845` | `0.820` | `11/32` |

The candidate passed the required `+0.20` survival delta over matched random;
both random replicas survived `0/32`. It failed every paired-control promotion
comparison. Candidate-minus-control deltas were:

| Training seed | Survival | Maintenance | Decision-safe | Duration-safe |
| --- | ---: | ---: | ---: | ---: |
| `20260901` | `+0.031` | `+0.031` | `-0.391` | `-0.351` |
| `20260902` | `-0.094` | `-0.344` | `-0.154` | `-0.112` |

Neither pair met the positive-effect rule. The bounded candidate therefore
failed both replica gates and the global screen promotion gate.

## What actually failed

This reward did stop feed farming: candidate feed means were roughly `8–13`
per condition instead of the control's roughly `28–36`. But the removed
ongoing cycle incentive was not replaced by a learnable recurring-maintenance
signal.

- In six of eight candidate seed/condition cells, `WAIT` occupied roughly
  `51–66%` of decisions. Drives then spent long periods unsafe, mainly from
  low satiety and high boredom.
- Seed `20260901` completed every required cycle and survived every episode,
  yet all `32/32` episodes missed the terminal safety gate.
- Seed `20260902` reached good safety in morphology (`0.901/0.900`) but
  completed maintenance `0/8`; it selected `PLAY` 565 times while producing
  only one mean play cycle. This is a credit/threshold-timing failure, not
  cycle farming.
- The candidate terminal `+5` was reached in only `1/64` probe episodes. Under
  per-decision `gamma=.99`, that distant outcome is weak and depends on
  cumulative safety/recovery state absent from the frozen 30-vector.
- The legacy control's repeated cycles were not merely cosmetic reward
  hacking: the repeated actions also restored drives and produced much better
  safe-band occupancy on this probe.

## Decision

The tested redesign did **not** improve maintenance. It was materially worse
than the contemporaneous 200-step legacy control. The original diagnosis was
only partly right: uncapped cycles misstate the declared quota objective, but
simply capping them plus adding a cost-only safety term destroys the recurring
incentive that keeps drives maintained. Long-horizon credit assignment and the
non-Markov terminal gate are now the more likely bottlenecks.

Per the frozen stop rule, no coefficient was changed, the opened probe was not
reused, and the 12,000-episode x eight-seed confirmation and audit splits remain
unopened.
