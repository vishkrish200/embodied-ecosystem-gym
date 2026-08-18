# M13.9 screen result

Status: **screen rejected; confirmation and audit not run**

The exact frozen M13.9 screen ran from implementation commit `f9585e2` with
protocol fingerprint
`2e98c688db3710aa7a6016d7d517d85f6ce122b2e3a94ee82bff504f8bc05adb`.
It is a clean negative result, not an implementation failure.

## Execution

- Command: `uv run python -m ecosystem_gym experiment m139-screen`
- Wall time: `193.28` seconds (`user 472.60`, `sys 14.11`).
- Training: six simultaneous spawned workers, one numerical-library thread per
  worker, three arms by two seeds, 2,000 episodes per fit, 12,000 learned
  episodes total.
- All six jobs were submitted before results were collected. Individual fit
  times were 58.58–62.30 seconds.
- Preflight coverage, scripted ceiling, potential diagnostics, specialists,
  quota-then-wait rejection, and replay gates passed.
- Probe coverage and scripted ceiling passed before learned scoring.
- All 688 generated traces replayed exactly: 400 preflight, 32 probe-mechanics,
  and 256 scored probe traces.
- The report, six serialized policy artifacts, their hashes, and every scored
  trace-derived gate summary were independently reverified after completion.

## Frozen screen result

| Training seed | Candidate survival | Maintenance | Full objective | Decision safe | Duration safe | Unsafe WAIT | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260911 | 100.0% | 78.1% | 28.1% | 80.8% | 83.3% | 62.2% | reject |
| 20260912 | 93.8% | 93.8% | 56.2% | 87.2% | 87.4% | 43.3% | reject |

Neither replica passed every per-condition gate. In particular, every
candidate condition in both replicas exceeded the frozen 15% screen unsafe-WAIT
limit. Seed `20260911` also failed compound maintenance, persistent-reference
safety, and the static-control improvement bundle. Seed `20260912` failed
persistent-reference survival and legacy non-inferiority: its pooled
full-objective rate was 56.2% versus the legacy guardrail's 65.6%, a `-9.4`
percentage-point delta against the frozen `-3` point floor.

The result is not uniformly negative. Seed `20260912` strongly improved over
its static-cost control: full-objective success increased from 0% to 56.2%,
decision-safe occupancy by 11.0 points, duration-safe occupancy by 10.0 points,
and unsafe-WAIT fraction improved by 17.3 points. But the protocol requires
both replicas to pass every rule, and seed `20260911` did not reproduce that
effect. The remaining unsafe-WAIT rates are also above the declared screen limit.

## Split state and conclusion

The canonical ledger opened only `screen_fit` (`5200–5219`) and `screen_probe`
(`5220–5227`). Confirmation fit/evaluation and audit ranges `5300–5519` remain
unopened. The rejection does not authorize confirmation, coefficient changes,
replacement seeds, or a rerun. Preserve M13.9 as a negative screen result and
pause this M13 reward-tuning line unless a separately frozen successor protocol
is justified.

The full ignored report is at `artifacts/reports/m139-screen.json`; its content
hash is `7d3e38d2bba8975c0abb7eed99b5f2eb4a40f21fb2820ff7782c40b788dc08ed`.

See [`M13_9_FAILURE_ANALYSIS.md`](M13_9_FAILURE_ANALYSIS.md) for the read-only
trace diagnosis and the evidence that the public network can represent the
oracle strategy.
