# M13.14-r2 development abort: random diagnostic fingerprint

Date: 2026-08-19

Status: **implementation abort after policy evaluation but before report sealing**.

## Authorized command

After the r2 coverage preflight passed, the user separately authorized the exact
frozen r2 development stage:

```text
uv run python -m ecosystem_gym experiment m1314r2-development \
  --manifest artifacts/manifests/m1314r2-modular-anchored.json \
  --preflight-report artifacts/reports/m1314r2-development-preflight.json \
  --output artifacts/reports/m1314r2-development.json
```

Protocol fingerprint:
`47e6f7b4d7bd508407e43f901d22bd72906c7efbaf24f926dd6e4df2270fd5a6`.

## What completed

1. The manifest and sealed passing preflight verified.
2. The append-only ledger opened `development_preflight`, `development_fit`,
   then `development_check`.
3. A fresh valid 16,000-row public teacher dataset was collected.
4. All 12 declared fits completed: four training seeds across the modular
   anchored candidate, monolithic anchored control, and modular no-anchor
   control.
5. All 12 policy artifacts serialized successfully.
6. All 1,280 scored policy episodes were written and strictly replayed.
7. All 80 scripted-ceiling diagnostic episodes were written and replayed.

The no-shield control reused the candidate weights as declared and did not add
a fourth fit.

## Abort

The first mask-random diagnostic episode was written, then strict replay raised:

```text
ValueError: M13.9 random trace policy fingerprint mismatch
```

`_evaluate_one` always called `run_m139_episode(..., policy_fingerprint=None)`.
That is valid for the deterministic and shield-wrapped policy surfaces, but
`replay_m139_trace` requires a `SeededRandomM139Policy` trace header to contain
`pcg64-<seed>`. The runner had the correct fingerprint in its evaluation spec
but failed to pass it to the trace writer.

This is diagnostic replay plumbing failure, not evidence for or against any
learned policy. The development gate and report were never sealed.

## Preserved evidence

- policy artifacts: `12`
- trace files: `1,361`
  - scored policy traces: `1,280`
  - scripted ceiling traces: `80`
  - incomplete random diagnostic surface: `1/80`
- development report: absent
- confirmation report: absent
- audit report: absent

SHA-256 aggregates:

- policy-artifact hash list:
  `168df33c67c2892fcb96c011b61f8c707f3a9291214a063a5c7316c6d7f19090`
- trace hash list:
  `d096da00b9f9e7e16e492f9274ea70a451e7d97015bc85b7e2433a1ebba86136`
- teacher dataset:
  `0093aa0aa7590414995e2990c7677e93a85ccc6195e97551d03bf0e786ca8e1a`
- stage ledger:
  `8de04eeaad4a8f8344f03989639c1d8b053259ddf9b57047926b4b14ac548b6c`

The ledger contains exactly:

```text
development_preflight -> development_fit -> development_check
```

Confirmation and audit remain unopened.

## Boundary and required correction

Do not resume r2, reconstruct an unofficial gate from its traces, patch its
frozen manifest, or open confirmation. A corrected additive successor must:

1. use a new protocol fingerprint, manifest, ledger, milestone identity, and
   fresh fit/check splits;
2. preserve the r2 teacher data, policies, traces, ledger, and this abort;
3. pass `pcg64-<seed>` to `run_m139_episode` for mask-random diagnostics;
4. add a non-mocked trace/replay regression for `SeededRandomM139Policy`;
5. verify the complete diagnostic matrix before report sealing; and
6. require separate explicit authorization before any corrected preflight or
   development run.

