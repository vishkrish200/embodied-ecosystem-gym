# M13.14 development coverage preflight

Date: 2026-08-18

Status: **passed; development fitting and evaluation remain unopened**.

## Authorized scope

The user authorized only:

1. creation of the frozen M13.14 manifest; and
2. the development coverage preflight over the exact fresh fit/check
   partitions.

No teacher collection, fit, policy evaluation, replay, confirmation, or audit
was authorized or run.

## Commands

```text
uv run python -m ecosystem_gym maintenance-policy-manifest-m1314 \
  --output artifacts/manifests/m1314-modular-anchored.json

uv run python -m ecosystem_gym experiment m1314-development-preflight \
  --manifest artifacts/manifests/m1314-modular-anchored.json \
  --output artifacts/reports/m1314-development-preflight.json
```

Protocol fingerprint:
`5c215736ba2194f502f09600f6ce5f287543deb237f77546088eb7279490cfb8`.

Sealed preflight content hash:
`fcaa88d8c62c7730ae47c974b6809d22912cec2629deff84c0e59c866810c88f`.

File SHA-256 values:

- manifest: `d61d767268f3b2f788cba97adc689ac891ab5900e8877876d07ae3f1c30811ce`
- preflight report: `ef474ba032615d3c05a16d6f63982fd5168bf291a4ccd9216fa8d9a29e27e20c`

## Result

Both exact partitions passed:

- `development_fit`: seeds `6600--6619`
- `development_check`: seeds `6620--6639`

For every partition and each of the four conditions, all 20 episodes exposed:

- initial food: `20/20`
- replenished food: `20/20`
- toy: `20/20`
- rest target: `20/20`

Both relocation conditions exposed relocated food `20/20`. The compound
condition also exposed the blocked distractor `20/20`. Conditions without a
relocation or distractor correctly reported zero required episodes.

The stored report was independently verified against its protocol fingerprint,
manifest fingerprint, expected partitions/conditions, episode counts, and
content hash.

## Boundary after preflight

The canonical manifest and sealed coverage report now exist locally under
`artifacts/`, which remains Git-ignored. There is still:

- no M13.14 ledger;
- no teacher dataset;
- no learned policy artifact;
- no development fit or evaluation report;
- no M13.14 evaluation trace or replay result; and
- no confirmation or audit access.

The next possible action is the exact frozen `m1314-development` command. It
requires separate explicit authorization after review of this preflight. That
authorization would cover teacher collection, the declared development fits,
and the one development evaluation only; it would not authorize confirmation
or audit.

