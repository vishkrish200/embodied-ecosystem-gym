# M13.14-r2 frozen protocol: corrected modular anchored homeostatic-Q successor

Status: **implemented, frozen, and unopened**. As of Tuesday, August 18, 2026,
no M13.14-r2 manifest, preflight report, ledger marker, teacher dataset, fit,
evaluation, replay, confirmation, or audit command has been authorized or run.

M13.14-r2 is additive. It does not repair M13.14 in place, does not reopen the
M13.14 frozen lane, and does not reuse M13.14 or M13.13 teacher datasets,
policy artifacts, check data, or report paths.

## Why this successor exists

- M13.14 froze a viable modular-Q successor shape, but its teacher boundary and
  artifact boundary were still too permissive for a fresh unopened lane.
- The r2 maintenance core now separates whole-dataset validation from sampled
  minibatch validation, so the persisted teacher dataset must cover all eight
  macros while legal sampled minibatches may be partial.
- M13.14-r2 also hardens artifact and teacher isolation: new schema families,
  new source hashes, and explicit rejection of legacy `m1314` and `m1313`
  teacher/artifact paths.

The old M13.14 code, manifests, ledgers, layouts, docs, and conclusions remain
frozen.

## Frozen public boundary

Every M13.14-r2 arm must:

- read exactly the existing 30 public maintenance features;
- emit exactly the same eight maintenance macros;
- ignore task IDs, reset options, `info`, authoritative counters, environment
  internals, M13.14 teacher datasets and policy artifacts, and all M13.13
  frozen bytes or report paths; and
- preserve the existing deterministic macro legality and semantic safety mask.

No private observation key may enter the learner or the teacher dataset.

## Declared arms

The candidate and three matched controls all share the same public observation
surface, macro compiler, duration-aware backup, fit budget, evaluation matrix,
and fresh training seeds.

| Arm | Change relative to candidate | Purpose |
| --- | --- | --- |
| `modular_anchor_shield_candidate` | baseline | three drive heads, persistent anchor, shield on |
| `monolithic_anchor_shield_control` | replace the three drive heads with one matched-budget monolithic head | isolate modular decomposition |
| `modular_no_anchor_shield_control` | zero the teacher margin after byte-identical supervised initialization | isolate persistent anchoring |
| `modular_anchor_no_shield_control` | disable the shield during action selection only | isolate the acting-loop shield |

Per training seed, the modular anchored learner is fit exactly once. The
candidate and no-shield control must serialize byte-identical Q parameters; the
shield flag is the only difference between them. The monolithic and no-anchor
controls receive their own declared fits.

## Fresh partitions and layouts

M13.14-r2 owns only the fresh `6720--6839` split family:

| Partition | Seeds | Layout prefix | Purpose |
| --- | --- | --- | --- |
| `development_fit` | `6720--6739` | `m1314r2_fit_*` | teacher collection and development fitting |
| `development_check` | `6740--6759` | `m1314r2_check_*` | one development evaluation |
| `confirmation_fit` | `6760--6799` | `m1314r2_confirm_fit_*` | rebuild all confirmation fits from scratch |
| `confirmation_evaluation` | `6800--6819` | `m1314r2_confirm_eval_*` | one confirmation evaluation |
| `audit` | `6820--6839` | `m1314r2_audit_*` | one no-refit audit |

All 20 `m1314r2_*` layouts are additive, disjoint from every earlier
registered layout, and remain inside the previously coverage-safe envelope.
Canonical coverage is still deferred to the separately authorized preflight
commands.

Fresh training seeds are also frozen:

- development modular/monolithic/no-anchor fits: `20261423--20261426`
- confirmation modular/monolithic/no-anchor fits: `20261427--20261434`

## Preflight-before-fit order

M13.14-r2 adds a durable append-only ledger with this exact stage order:

`development_preflight -> development_fit -> development_check ->
confirmation_preflight -> confirmation_fit -> confirmation_evaluation ->
audit_preflight -> audit`

Each preflight stage scans and persists the exact partitions named below before
any later fit or evaluation stage may open:

- `development_preflight`: `development_fit` and `development_check`
- `confirmation_preflight`: `confirmation_fit` and `confirmation_evaluation`
- `audit_preflight`: `audit`

Each preflight report is sealed with:

- the exact M13.14-r2 protocol fingerprint;
- the exact manifest fingerprint;
- the stage name;
- the exact partitions scanned;
- every persisted coverage count for every condition; and
- a content hash over the canonical JSON payload.

If any coverage row fails, the report must still be written, marked failed, and
used to block every later fit, evaluation, serialization, worker spawn, and
replay call for that stage family.

## Budgets and gates

The fit budget is frozen at `400,000` macro decisions per declared fit.

Evaluation stays on the same `4 conditions x 20 env seeds` matrix per replica:

- 80 scored episodes per arm replica;
- 1,280 scored development episodes across the four declared arms;
- 2,560 scored confirmation episodes across the four declared arms; and
- 640 scored audit policy episodes for the candidate only
  (`8 confirmation seeds x 4 conditions x 20 env seeds`).

Audit still persists shared diagnostics separately. Those diagnostic rows are
not counted inside the 640 scored policy episodes.

Every candidate replica must independently pass every condition:

- survival at least `18/20`;
- maintenance at least `18/20`;
- full objective at least `18/20`;
- mean decision-safe fraction at least `.85`;
- mean duration-safe fraction at least `.85`;
- required recovery at least `18/20` in each relocation condition;
- unsafe-WAIT fraction at most `.10`; and
- zero coverage, conformance, artifact, ledger, or replay violations.

The candidate must also beat each matched control on its declared comparison:

- versus monolithic: at least `.10` pooled full-objective advantage;
- versus no-anchor: at least `.10` pooled full-objective advantage; and
- versus no-shield: at least `.15` pooled full-objective advantage while the
  Q-parameter fingerprint remains byte-identical.

No cross-arm winner selection is allowed. There is no replacement-seed escape
hatch after a failed gate.

## Teacher and artifact boundaries

The teacher dataset must:

- come only from the fresh fit partition through the public oracle;
- remain public-only, legal under serialized masks, and immutable;
- cover all eight macros at the persisted whole-dataset level;
- allow sampled legal minibatches to omit some macros;
- persist throughout the off-policy run as a dedicated replay surface; and
- reject legacy M13.14 and M13.13 paths.

The sole artifact owner is
`ecosystem_gym/maintenance/modular_q_artifacts_m1314r2.py`. It must reject
legacy `m1314` and `m1313` paths, bind the new schema family, and enforce
byte-stable reloads, source-hash checks, and parameter fingerprints.

## Exact future command names

The implemented but unopened command names are:

- `maintenance-policy-manifest-m1314r2`
- `m1314r2-development-preflight`
- `m1314r2-development`
- `m1314r2-confirmation-preflight`
- `m1314r2-confirmation`
- `m1314r2-audit-preflight`
- `m1314r2-audit`
- `m1314r2-replay`

The future command order is:

```text
uv run python -m ecosystem_gym maintenance-policy-manifest-m1314r2 --output artifacts/manifests/m1314r2-modular-anchored.json
uv run python -m ecosystem_gym experiment m1314r2-development-preflight --manifest artifacts/manifests/m1314r2-modular-anchored.json --output artifacts/reports/m1314r2-development-preflight.json
uv run python -m ecosystem_gym experiment m1314r2-development --manifest artifacts/manifests/m1314r2-modular-anchored.json --preflight-report artifacts/reports/m1314r2-development-preflight.json --output artifacts/reports/m1314r2-development.json
uv run python -m ecosystem_gym experiment m1314r2-confirmation-preflight --manifest artifacts/manifests/m1314r2-modular-anchored.json --development-report artifacts/reports/m1314r2-development.json --output artifacts/reports/m1314r2-confirmation-preflight.json
uv run python -m ecosystem_gym experiment m1314r2-confirmation --manifest artifacts/manifests/m1314r2-modular-anchored.json --preflight-report artifacts/reports/m1314r2-confirmation-preflight.json --development-report artifacts/reports/m1314r2-development.json --output artifacts/reports/m1314r2-confirmation.json
uv run python -m ecosystem_gym experiment m1314r2-audit-preflight --manifest artifacts/manifests/m1314r2-modular-anchored.json --confirmation-report artifacts/reports/m1314r2-confirmation.json --output artifacts/reports/m1314r2-audit-preflight.json
uv run python -m ecosystem_gym experiment m1314r2-audit --manifest artifacts/manifests/m1314r2-modular-anchored.json --preflight-report artifacts/reports/m1314r2-audit-preflight.json --confirmation-report artifacts/reports/m1314r2-confirmation.json --output artifacts/reports/m1314r2-audit.json
uv run python -m ecosystem_gym experiment m1314r2-replay --manifest artifacts/manifests/m1314r2-modular-anchored.json --trace <trace.json> --policy <artifact.json>
```

## No-run boundary

This implementation does not authorize execution. The frozen boundary is:

`No M13.14-r2 preflight, fit, evaluation, replay, confirmation, or audit command is authorized by this manifest alone.`

Manifest generation is pure and source-hashed. Opening `development_preflight`
still requires separate explicit authorization. Confirmation and audit each
require their own later authorization after verifying the preceding sealed
reports.
