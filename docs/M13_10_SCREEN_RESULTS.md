# M13.10 screen result: imitation succeeds, RL destroys retention

Status: **frozen negative result (2026-08-08)**

The exact M13.10 screen completed and rejected. This is not an implementation
or replay failure: fit coverage, the balanced-oracle ceiling, both supervised
fit gates, probe mechanics, policy serialization, artifact hashes, and all 368
strict trace replays passed.

The narrow causal claim failed. Imitation-only policies generalized almost
perfectly to the fresh screen probe, but the identical imitation initializations
followed by 2,000 episodes of Double-DQN lost the strategy and performed worse
than matched random-initialized RL.

## Execution and split state

The canonical command was:

```text
/usr/bin/time -p uv run python -m ecosystem_gym experiment m1310-screen \
  --output artifacts/reports/m1310-screen.json
```

- wall time: `321.52` seconds;
- user time: `754.35` seconds;
- system time: `20.05` seconds;
- one spawned wave, six workers, six distinct worker PIDs, one numerical thread
  per worker;
- 16,000 public teacher rows and 80 supervised epochs per imitation fit;
- four RL fits × 2,000 episodes = 8,000 RL training episodes; and
- 368 trace files, all hash-valid and strictly replayed.

Only screen-fit `5600–5619` and screen-probe `5620–5627` were opened. M13.10
confirmation-fit, evaluation, and audit ranges `5700–5919` remain sealed and
must not be opened after this rejection.

## Probe result

Each pooled row contains four conditions × eight fresh probe seeds.

| Training seed | Arm | Survival | Maintenance | Full objective | Decision safe | Duration safe | Unsafe WAIT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260921 | imitation + RL | 56.2% | 0.0% | 0.0% | 43.5% | 59.6% | 61.3% |
| 20260921 | random-init RL | 84.4% | 68.8% | 18.8% | 73.8% | 69.1% | 52.1% |
| 20260921 | imitation only | 100% | 100% | 100% | 99.0% | 97.9% | 0.09% |
| 20260922 | imitation + RL | 100% | 6.2% | 0.0% | 26.1% | 26.0% | 89.4% |
| 20260922 | random-init RL | 87.5% | 87.5% | 71.9% | 91.8% | 86.0% | 41.3% |
| 20260922 | imitation only | 100% | 100% | 100% | 99.1% | 98.4% | 0.07% |

Both imitation fits passed their frozen integrity gates. Their fit accuracies
were `93.36%` and `93.06%`; non-`WAIT` accuracies were `98.37%` and `99.14%`.
Candidate and imitation-only weights were byte-identical immediately after
supervision for each seed. The divergence therefore occurred during the
declared RL phase.

Every imitation-only condition passed every hard screen gate. Every
imitation-plus-RL condition failed. The candidate's full-objective deltas
against random-init RL were `-18.75` and `-71.875` percentage points, opposite
the required positive `15` points. Retention deltas against imitation-only were
also far outside the allowed band.

## Conclusion

M13.10 establishes two useful facts:

1. the public 30-feature network and balanced teacher generalize across the
   fresh screen layout/seed split; and
2. the current off-policy RL update does not retain that policy, even with a
   strong initialization and a screen-aligned exploration schedule.

Do not retry M13.10, open confirmation, or tune against `5620–5627`. If another
state-policy experiment is justified, it needs fresh splits and a frozen
retention mechanism chosen using fit-only evidence—for example a supervised
anchor or constrained update—not another change to M13.9 reward coefficients.

The ignored canonical report is `artifacts/reports/m1310-screen.json`. Its
canonical content hash is
`325ca595173908f3f2a1c3d425de17b765c45f1cb91a2169a28a1b8a45365644`.
