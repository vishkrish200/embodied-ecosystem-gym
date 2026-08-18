# Codex Handoff: M13.14-r2 modular-anchored successor

Updated: 2026-08-19
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch: `codex/m1314r2-minibatch-fix` from M13.14 abort commit `d11dcf7`

## Outcome

M13.14 remains a frozen warm-start abort. Its exact development coverage preflight passed all fit/check visibility rows on `6600--6639`. The separately authorized development stage then opened fit and wrote a valid public teacher dataset, but aborted during the first warm-start minibatch because minibatches were incorrectly required to contain all eight macro classes. No learner fit or policy evaluation completed.

M13.14-r2 corrected the minibatch bug and advanced through fresh teacher
collection, all 12 fits, policy serialization, 1,280 scored policy replays, and
80 scripted-ceiling replays. It then aborted before report sealing because the
first mask-random diagnostic trace omitted the `pcg64-<seed>` fingerprint
required by strict replay. No official development gate exists. Confirmation
and audit remain sealed.

See `docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md`, `docs/M13_14_DEVELOPMENT_PREFLIGHT.md`, `docs/M13_14_DEVELOPMENT_ABORT.md`, `docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md`, `docs/M13_14_R2_DEVELOPMENT_PREFLIGHT.md`, and `docs/M13_14_R2_DEVELOPMENT_ABORT.md`.

## Frozen Evidence State

- M13.14 manifest fingerprint:
  `5c215736ba2194f502f09600f6ce5f287543deb237f77546088eb7279490cfb8`.
- Development fit/check coverage passed every declared 20-seed row; the sealed report hash is
  `fcaa88d8c62c7730ae47c974b6809d22912cec2629deff84c0e59c866810c88f`.
- The M13.14 ledger contains only `development_preflight -> development_fit`.
- The 16,000-row teacher dataset contains every macro globally, but no learned policy artifact, development report, evaluation, replay, confirmation, or audit exists.
- M13.14-r2 manifest fingerprint:
  `47e6f7b4d7bd508407e43f901d22bd72906c7efbaf24f926dd6e4df2270fd5a6`.
- Every r2 development fit/check visibility row passed; sealed preflight hash:
  `98daea099f62f051c700933af46ffe028f034ba6a95056ffb56702de315a5928`.
- The r2 ledger contains `development_preflight -> development_fit ->
  development_check`.
- The r2 teacher dataset, 12 policy artifacts, 1,280 scored policy traces, 80
  scripted-ceiling traces, and one incomplete random diagnostic trace exist
  locally and are hashed in the abort record.
- No sealed r2 development report/gate, confirmation, or audit exists.

## Current Map

- `docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md`: frozen M13.14 protocol and abort boundary.
- `docs/M13_14_DEVELOPMENT_PREFLIGHT.md`: sealed M13.14 coverage preflight.
- `docs/M13_14_DEVELOPMENT_ABORT.md`: M13.14 warm-start abort record and preserved hashes.
- `docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md`: unopened M13.14-r2 protocol and no-run boundary.
- `docs/M13_14_R2_DEVELOPMENT_PREFLIGHT.md`: sealed passing r2 coverage preflight.
- `docs/M13_14_R2_DEVELOPMENT_ABORT.md`: diagnostic replay abort and preserved hashes.
- `ecosystem_gym/maintenance/policy_protocol_m1314.py`: frozen M13.14 manifest and splits.
- `ecosystem_gym/maintenance/policy_protocol_m1314r2.py`: additive M13.14-r2 manifest, fresh splits, and preflight order.
- `ecosystem_gym/maintenance/teacher_data_m1314r2.py`: public-only r2 teacher dataset helpers and minibatch validation split.
- `ecosystem_gym/maintenance/modular_q_m1314r2.py`: modular and monolithic r2 learners with teacher minibatches.
- `ecosystem_gym/maintenance/modular_q_artifacts_m1314r2.py`: r2 modular-Q artifact validation and legacy-path rejection.
- `ecosystem_gym/experiments/m1314r2.py`: unopened r2 runner and future-stage CLI wiring.
- `ecosystem_gym/experiments/m1314r2_support.py`: r2 ledger and preflight helpers.
- `tests/test_m1314r2_policy_protocol.py`: r2 manifest, CLI, and unopened-stage tests.
- `tests/test_maintenance_modular_q_m1314r2.py`: r2 teacher dataset and artifact tests.

## Verification

- Focused M13.14-r2 plus frozen M13.14 suite: 40 passed.
- Full repository suite: 264 collected tests passed.
- `python -m compileall -q ecosystem_gym tests`: passed.
- `git diff --check`: passed.
- Independent Luna-max integrated review: approved after candidate-only audit
  and runner source-hash fixes.
- The ignored r2 manifest, preflight, ledger, teacher data, policy artifacts,
  and partial development trace surface exist locally. The development report,
  confirmation, and audit remain absent.

## Next Step

Do not resume M13.14 or r2 under their frozen manifests, and do not reconstruct
an unofficial r2 result from traces. The next task is an additive successor
that fixes random diagnostic fingerprint binding and uses fresh splits. Any
corrected preflight or development stage requires new explicit authorization.
Confirmation and audit remain blocked.
