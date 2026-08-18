# M13.13 development abort: public coverage gate

Date: 2026-08-18

Status: **aborted before policy evaluation**. This is neither a policy-family
success nor a policy-family negative result.

## Authorized command

The user authorized the exact frozen M13.13 manifest and development run. The
canonical history was first merged to `main` at `957accd`. The following frozen
commands were then executed once:

```text
uv run python -m ecosystem_gym maintenance-policy-manifest \
  --output artifacts/manifests/m1313-policy-families.json

uv run python -m ecosystem_gym experiment m1313-development \
  --manifest artifacts/manifests/m1313-policy-families.json \
  --output artifacts/reports/m1313-development.json
```

The manifest protocol fingerprint was
`aa8180a744e6f6b969732467530e9db1399f10eb699181cab50ac0350b8d8252`.

## What happened

1. `development_fit` (`6200--6219`) opened and its public coverage gate passed.
2. Four corrected-PPO source actors completed their declared 400,000-decision
   fits, and the deterministic and shielded/unshielded policy artifacts were
   serialized.
3. `development_check` (`6220--6239`) opened.
4. Its offline public scan audit returned at least one failing visibility row.
5. The runner stopped with
   `RuntimeError: M13.13 development-check public coverage failed` before any
   policy-family evaluation, diagnostic evaluation, trace generation, replay,
   gate calculation, or report write.

The append-only ledger therefore contains exactly `development_fit` followed
by `development_check`. `confirmation_fit`, `confirmation_evaluation`, and
`audit` remain unopened. Seeds `6140--6199` and `6240--6299` remain unused.

No `artifacts/reports/m1313-development.json` exists. No performance claim is
supported for urgency/commitment, model-based scheduling, or the shielded
learned actor.

## Evidence limitation

`m10_scan_coverage` returned its detailed per-condition counts in memory, but
the M13.13 runner raised before persisting them. Re-running the opened check
partition merely to reconstruct those counts is prohibited. The durable fact
is limited to the hard-gate failure: at least one declared check condition did
not expose every required public target across all 20 seeds and four scan
sectors.

The pre-run tests checked layout freshness, protocol constants, synthetic
gates, serialization, ledger ordering, and CLI dispatch. They did not execute
the canonical M13.13 layout families through `m10_scan_coverage`. The runtime
gate caught that missing integration check, but only after the expensive fit.

## Preserved local artifacts

The ignored local artifacts are retained under `artifacts/`. Their hashes make
the exact aborted state inspectable without adding learned weights to Git:

- manifest: `e068e52619a359d2fe89e4a7722881f5dcffeb11f6b6719e434ee91b05905a32`
- ledger: `26e8698d34b688dff7cb1a597946f9089158eefd65c50e15101cdb857e57d617`
- PPO source seeds `20261331--20261334`:
  `31a8b9350198222bff187934e67ae7b3cb5ac9c0a133b0f1c26767ea4f36b427`,
  `332b8136cdc9ef6913bd1f0a9b99ce566ce3f546a0f809c0af9ba6fc6e9e85f9`,
  `4d89d97a4d4b266dcbf55219557ad7f0946c627e6c346765560f796e6c259405`,
  `083aecff5733e6874a7db66d35741ff8eb194e3d9b1e74325e66b153d864597b`

The 12 derived candidate/control artifacts remain in
`artifacts/reports/m1313-development-artifacts/policies/`. They may be inspected
for integrity, but they must not be scored on `6220--6239` or carried into a
successor as pretrained selection data.

- model candidate/control:
  `47a4b2cf67ff9d4a19c5cd5d36c2462d2c94ce5370468d7393da5a0e5d55b52f`,
  `d1582de49f174bbee741d2a0600a02b817ee389cc00fab4b7fed98efd7c1dd74`
- shielded candidates for seeds `20261331--20261334`:
  `6b73289d1c139660e8dc8cf1cadb03df2db532dec62b1b5b86ec3ed5651288a4`,
  `92531c9fa995b698e1bbffc4db3497dd11f72f83165b5f261a0076906e8890f8`,
  `2734c49579ba19456732f44bb5b54cc8fa5e3bd7ca6b08c3a3355213a6f4db39`,
  `a4b69c3f6ddb7e9825dbe37270eccf98066f8201bbec2177046e908587521923`
- unshielded controls for seeds `20261331--20261334`:
  `98310eeeaef4dab0bcb6f09829c8f3746a511fb384bad6d16502274cec0db4ca`,
  `f581b6b93a9164ea188b5b75ad34d30314bc7193b9d2a07f006352a41f83f6f1`,
  `fe03108b2c8212840932b4f0b0ac31cf10256afb4c0ebfc4f2746fef1c20ff84`,
  `b19e650d76b3ef6233de2affcd2278eb1f268c46763868eed00acef3e2a7b98e`
- urgency candidate/control:
  `8ad0ab2155eef88b46691281b8c4b1861aec70d17fb53218812f1954613dd246`,
  `b6f1ab83b9a328de173b42c3d72b267c47228aa03798540662672a1584fb507a`

## Required successor boundary

Do not rerun M13.13, edit its frozen thresholds/layouts in place, fabricate a
development report, or proceed to confirmation. A successor must:

1. use a new protocol fingerprint, manifest, ledger, milestone name, and fresh
   split family;
2. preserve M13.13 source, manifest, ledger, actors, and this abort record;
3. run and persist public coverage for all development partitions before any
   expensive learner fit;
4. stop before fitting if coverage fails;
5. keep the three policy hypotheses, one-factor controls, public 30-feature /
   eight-macro boundary, budgets, and independent gates unchanged unless a
   separately justified protocol explicitly changes them; and
6. require new explicit authorization before opening the successor splits.
