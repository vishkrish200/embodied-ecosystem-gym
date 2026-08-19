# M13.14-r3 frozen protocol: random-diagnostic fingerprint correction

Status: **implemented, frozen, and unopened**. M13.14-r3 is an additive
successor to the frozen M13.14-r2 lane. It preserves every r2 source file,
teacher dataset, artifact, trace, ledger, and abort conclusion. No r3
manifest, preflight, ledger marker, teacher dataset, fit, evaluation, replay,
confirmation, or audit has been authorized or run by this protocol.

## Purpose and correction

M13.14-r2 completed its scored development episodes and then stopped before
report sealing when the first random diagnostic trace omitted the exact
`pcg64-<seed>` policy fingerprint required by strict replay. That is a runner
and trace-plumbing failure, not a learner result. r3 fixes that binding and
adds a non-mocked random-trace replay regression while keeping the frozen r2
modular-Q learner classes unchanged.

The r3 lane owns new source hashes, schema identifiers, paths, layouts, data
partitions, training seeds, manifest, and ledger. An r2 teacher dataset,
artifact, report, trace, or ledger cannot be used as r3 input.

## Frozen public boundary and arms

Every arm reads the existing 30 public maintenance features and emits the
same eight macros: `GO_FOOD`, `PICK_UP`, `CONSUME`, `GO_TOY`, `PLAY`,
`GO_REST`, `REST`, and `WAIT`. Task IDs, reset options, `info`, authoritative
counters, environment internals, and old milestone bytes remain forbidden.
Private observation keys are rejected at the teacher and learner boundary.

The candidate and matched controls remain the r2 comparison:

| Arm | Declared factor |
| --- | --- |
| `modular_anchor_shield_candidate` | three drive-specific Q heads, persistent teacher margin, shield on |
| `monolithic_anchor_shield_control` | matched-budget monolithic head |
| `modular_no_anchor_shield_control` | teacher margin removed after byte-identical supervised initialization |
| `modular_anchor_no_shield_control` | candidate Q parameters, shield disabled only for acting |

The candidate fit is reused byte-for-byte only for the no-shield control.
The monolithic and no-anchor controls receive separate fresh fits. No
cross-arm winner selection or replacement seed is allowed.

## Fresh partitions, layouts, and training seeds

The r3 environment-seed partitions are disjoint from every previous lane:

| Partition | Seeds | Layout prefix | Use |
| --- | --- | --- | --- |
| development fit | `6840--6859` | `m1314r3_fit_*` | public teacher collection and development fit |
| development check | `6860--6879` | `m1314r3_check_*` | development gate |
| confirmation fit | `6880--6919` | `m1314r3_confirm_fit_*` | fresh confirmation fits |
| confirmation evaluation | `6920--6939` | `m1314r3_confirm_eval_*` | confirmation gate |
| audit | `6940--6959` | `m1314r3_audit_*` | candidate-only no-refit audit |

The four development training seeds are `20261435--20261438`. The eight
confirmation training seeds are `20261439--20261446`. All 20 r3 layouts are
additive and coordinate-disjoint from the earlier registered layouts.

## Preflight-first ledger

The append-only stage order is:

`development_preflight -> development_fit -> development_check ->
confirmation_preflight -> confirmation_fit -> confirmation_evaluation ->
audit_preflight -> audit`

Before each fit or evaluation family, its preflight scans and persists every
coverage row for the exact declared partitions. A failed row is still sealed
in a failed report and blocks that family. The ledger is source/protocol
fingerprinted, one-way, and permits resumption only of an interrupted fit
stage. No preflight opens merely because this document exists.

The source-hashed manifest binds the runner, CLI, active core, frozen r2
learner, r3 teacher/artifact helpers, support ledger, task layouts, and this
protocol documentation. A manifest or report with a mismatched source or
protocol fingerprint is rejected before ledger or environment work.

## Teacher, artifact, and diagnostic boundaries

The teacher dataset is public-only, mask-legal, immutable, and must cover all
eight macros at the persisted dataset level. Sampled minibatches may be
partial. It is written only through
`ecosystem_gym/maintenance/teacher_data_m1314r3.py` with schema
`m1314r3-teacher-dataset-v1`.

Policy serialization is owned by
`ecosystem_gym/maintenance/modular_q_artifacts_m1314r3.py` with schema
`m1314r3-modular-q-artifact-v1`. Its path rules reject M13.14, M13.14-r2,
and M13.13 namespaces. The r2 learner source is frozen and reused only as
code; r2 data and artifacts are never loaded.

Every scored and diagnostic episode is strictly replayed. The two shared
diagnostic surfaces remain separate from scored policy rows. In particular,
the mask-random policy trace header must carry `pcg64-<seed>` before replay;
the r3 regression exercises a non-canonical seed through the real trace
writer and replay verifier.

## Budgets and gates

Each fit has a 400,000 macro-decision budget. Each policy replica has four
conditions x 20 environment seeds = 80 scored episodes. Development has
1,280 scored policy episodes, confirmation 2,560, and the candidate-only
audit has 640 scored policy episodes plus separate diagnostics.

Every candidate replica must pass each condition with survival, maintenance,
and full-objective success at least 18/20; mean decision-safe and
duration-safe fractions at least .85; required recovery at least 18/20 in
relocation conditions; unsafe-WAIT at most .10; and zero coverage,
conformance, artifact, ledger, and replay violations. The candidate must
also satisfy the declared `.10`, `.10`, and `.15` pooled advantages against
the matched controls, with byte-identical Q parameters for the no-shield
comparison. Audit scoring is candidate-only; no control comparison is run in
the audit gate.

## Exact future commands

```text
uv run python -m ecosystem_gym maintenance-policy-manifest-m1314r3 --output artifacts/manifests/m1314r3-modular-anchored.json
uv run python -m ecosystem_gym experiment m1314r3-development-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --output artifacts/reports/m1314r3-development-preflight.json
uv run python -m ecosystem_gym experiment m1314r3-development --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-development-preflight.json --output artifacts/reports/m1314r3-development.json
uv run python -m ecosystem_gym experiment m1314r3-confirmation-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --development-report artifacts/reports/m1314r3-development.json --output artifacts/reports/m1314r3-confirmation-preflight.json
uv run python -m ecosystem_gym experiment m1314r3-confirmation --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-confirmation-preflight.json --development-report artifacts/reports/m1314r3-development.json --output artifacts/reports/m1314r3-confirmation.json
uv run python -m ecosystem_gym experiment m1314r3-audit-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --confirmation-report artifacts/reports/m1314r3-confirmation.json --output artifacts/reports/m1314r3-audit-preflight.json
uv run python -m ecosystem_gym experiment m1314r3-audit --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-audit-preflight.json --confirmation-report artifacts/reports/m1314r3-confirmation.json --output artifacts/reports/m1314r3-audit.json
uv run python -m ecosystem_gym experiment m1314r3-replay --manifest artifacts/manifests/m1314r3-modular-anchored.json --trace <trace.json> --policy <artifact.json>
```

## No-run boundary

`No M13.14-r3 preflight, fit, evaluation, replay, confirmation, or audit
command is authorized by this manifest alone.`

Manifest creation and source-hash calculation are pure. A separate explicit
authorization is required before development preflight; confirmation and
audit each require their own later authorization after verifying the prior
sealed reports. This implementation creates no manifest, ledger, teacher
dataset, artifact, training output, evaluation output, or trace.
