# M13.14-r3 development coverage preflight

Date: 2026-08-19

Status: **passed; r3 development fitting and evaluation remain unopened**.

## Authorized scope

The user authorized only the frozen r3 manifest and development coverage
preflight. No r3 ledger, teacher collection, fit, policy evaluation, replay,
confirmation, or audit was authorized or run.

## Commands

```text
uv run python -m ecosystem_gym maintenance-policy-manifest-m1314r3 \
  --output artifacts/manifests/m1314r3-modular-anchored.json

uv run python -m ecosystem_gym experiment m1314r3-development-preflight \
  --manifest artifacts/manifests/m1314r3-modular-anchored.json \
  --output artifacts/reports/m1314r3-development-preflight.json
```

Protocol fingerprint:
`d4c309c0e9637727f555abc44064e206f6f11a29f58113da9884afd0cb3f38bb`.

Sealed preflight content hash:
`d6878992c24855a9a5122dc770512b6d93f4ad05928cd6ce7983c9b628bbb237`.

File SHA-256 values:

- manifest: `6035f51b6b8db37415bb0522c46a98d2e67be4b9ad8e85b58beb32c16c653cde`
- preflight report: `42f7da671fa43731b6f927580cb0f08d1c7457dda41126df91fdb650849bdc49`

## Result

Both exact r3 partitions passed:

- `development_fit`: seeds `6840--6859`
- `development_check`: seeds `6860--6879`

For every partition and each of the four conditions, all 20 episodes exposed
initial food, replenished food, the toy, and the rest target. Both relocation
conditions exposed relocated food `20/20`; compound also exposed the blocked
distractor `20/20`.

The stored report independently verified against its protocol and manifest
fingerprints, exact partitions/conditions, episode counts, and content hash.

## Boundary after preflight

The canonical r3 manifest and sealed coverage report now exist locally under
Git-ignored `artifacts/`. There is still no r3 ledger, teacher dataset, learned
policy, development report, evaluation trace, confirmation, or audit.

The next possible action is the exact frozen `m1314r3-development` command. It
requires separate explicit authorization and would cover fresh teacher
collection, the declared development fits, and one development evaluation
only. It would not authorize confirmation or audit.

