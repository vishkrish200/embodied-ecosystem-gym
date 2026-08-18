# M13.14-r2 development coverage preflight

Date: 2026-08-19

Status: **passed; corrected development fitting and evaluation remain unopened**.

## Authorized scope

The user authorized only:

1. creation of the frozen M13.14-r2 manifest; and
2. the r2 development coverage preflight over its exact fresh fit/check
   partitions.

No r2 ledger, teacher collection, fit, policy evaluation, replay, confirmation,
or audit was authorized or run.

## Commands

```text
uv run python -m ecosystem_gym maintenance-policy-manifest-m1314r2 \
  --output artifacts/manifests/m1314r2-modular-anchored.json

uv run python -m ecosystem_gym experiment m1314r2-development-preflight \
  --manifest artifacts/manifests/m1314r2-modular-anchored.json \
  --output artifacts/reports/m1314r2-development-preflight.json
```

Protocol fingerprint:
`47e6f7b4d7bd508407e43f901d22bd72906c7efbaf24f926dd6e4df2270fd5a6`.

Sealed preflight content hash:
`98daea099f62f051c700933af46ffe028f034ba6a95056ffb56702de315a5928`.

File SHA-256 values:

- manifest: `1bbeebb7786a1c9e9174fd8ddbefd1e0710cb23b1e002b27af427bcd823b832f`
- preflight report: `0400417f3d3a9d6e7e5cd4798b00f8e487443278781e5e08d5c2db899126b7ed`

## Result

Both exact r2 partitions passed:

- `development_fit`: seeds `6720--6739`
- `development_check`: seeds `6740--6759`

For every partition and each of the four conditions, all 20 episodes exposed:

- initial food: `20/20`
- replenished food: `20/20`
- toy: `20/20`
- rest target: `20/20`

Both relocation conditions exposed relocated food `20/20`. The compound
condition exposed the blocked distractor `20/20`. Conditions without a
relocation or distractor correctly reported zero required episodes.

The stored report was independently verified against its protocol fingerprint,
manifest fingerprint, exact partitions/conditions, episode counts, and content
hash.

## Boundary after preflight

The canonical r2 manifest and sealed coverage report now exist locally under
Git-ignored `artifacts/`. There is still:

- no M13.14-r2 ledger;
- no r2 teacher dataset;
- no r2 learned policy artifact;
- no r2 development fit or evaluation report;
- no r2 trace or replay result; and
- no r2 confirmation or audit access.

The next possible action is the exact frozen `m1314r2-development` command. It
requires separate explicit authorization after review of this preflight. That
authorization would cover fresh r2 teacher collection, the declared
development fits, and one development evaluation only; it would not authorize
confirmation or audit.

