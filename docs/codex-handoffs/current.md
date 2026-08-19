# Codex Handoff: M13.14-r3 random-diagnostic fingerprint successor

Updated: 2026-08-19
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch: `codex/m1314r3-random-fingerprint` from r2 abort commit `4f4ac46`

## Outcome

M13.14 remains a frozen warm-start abort. Its exact development coverage preflight passed all fit/check visibility rows on `6600--6639`. The separately authorized development stage then opened fit and wrote a valid public teacher dataset, but aborted during the first warm-start minibatch because minibatches were incorrectly required to contain all eight macro classes. No learner fit or policy evaluation completed.

M13.14-r2 corrected the minibatch bug and advanced through fresh teacher
collection, all 12 fits, policy serialization, 1,280 scored policy replays, and
80 scripted-ceiling replays. It then aborted before report sealing because the
first mask-random diagnostic trace omitted the `pcg64-<seed>` fingerprint
required by strict replay. No official development gate exists. Confirmation
and audit remain sealed.

M13.14-r3 corrected the diagnostic fingerprint plumbing and completed all 12
fits, 1,280 scored policy evaluations, 160 shared diagnostics, and 1,440/1,440
strict replays. The official development gate failed: the modular anchored
candidate scored `258/320` versus `273/320` for the monolithic anchored control,
and no candidate replica passed every condition. Confirmation and audit remain
sealed.

See `docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md`, `docs/M13_14_DEVELOPMENT_PREFLIGHT.md`, `docs/M13_14_DEVELOPMENT_ABORT.md`, `docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md`, `docs/M13_14_R2_DEVELOPMENT_PREFLIGHT.md`, `docs/M13_14_R2_DEVELOPMENT_ABORT.md`, `docs/M13_14_R3_MODULAR_ANCHORED_PROTOCOL.md`, `docs/M13_14_R3_DEVELOPMENT_PREFLIGHT.md`, and `docs/M13_14_R3_DEVELOPMENT_RESULTS.md`.

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
- M13.14-r3 manifest fingerprint:
  `d4c309c0e9637727f555abc44064e206f6f11a29f58113da9884afd0cb3f38bb`.
- Every r3 development fit/check visibility row passed; sealed preflight hash:
  `d6878992c24855a9a5122dc770512b6d93f4ad05928cd6ce7983c9b628bbb237`.
- The r3 ledger is opened through development check; the fresh teacher dataset,
  12 policy artifacts, 1,440 traces, and sealed development report exist.
- R3 development gate: failed. Candidate totals were `49/80`, `67/80`,
  `73/80`, and `69/80`; no seed passed every condition.
- Confirmation and audit remain unopened.

## Current Map

- `docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md`: frozen M13.14 protocol and abort boundary.
- `docs/M13_14_DEVELOPMENT_PREFLIGHT.md`: sealed M13.14 coverage preflight.
- `docs/M13_14_DEVELOPMENT_ABORT.md`: M13.14 warm-start abort record and preserved hashes.
- `docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md`: unopened M13.14-r2 protocol and no-run boundary.
- `docs/M13_14_R2_DEVELOPMENT_PREFLIGHT.md`: sealed passing r2 coverage preflight.
- `docs/M13_14_R2_DEVELOPMENT_ABORT.md`: diagnostic replay abort and preserved hashes.
- `docs/M13_14_R3_MODULAR_ANCHORED_PROTOCOL.md`: unopened r3 protocol and no-run boundary.
- `docs/M13_14_R3_DEVELOPMENT_PREFLIGHT.md`: sealed passing r3 coverage preflight.
- `docs/M13_14_R3_DEVELOPMENT_RESULTS.md`: failed development gate and preserved hashes.
- `ecosystem_gym/maintenance/policy_protocol_m1314.py`: frozen M13.14 manifest and splits.
- `ecosystem_gym/maintenance/policy_protocol_m1314r2.py`: additive M13.14-r2 manifest, fresh splits, and preflight order.
- `ecosystem_gym/maintenance/policy_protocol_m1314r3.py`: additive M13.14-r3 manifest identity and fresh splits.
- `ecosystem_gym/maintenance/teacher_data_m1314r2.py`: public-only r2 teacher dataset helpers and minibatch validation split.
- `ecosystem_gym/maintenance/teacher_data_m1314r3.py`: public-only r3 teacher dataset helpers and validation split.
- `ecosystem_gym/maintenance/modular_q_m1314r2.py`: modular and monolithic r2 learners with teacher minibatches.
- `ecosystem_gym/maintenance/modular_q_artifacts_m1314r2.py`: r2 modular-Q artifact validation and legacy-path rejection.
- `ecosystem_gym/maintenance/modular_q_artifacts_m1314r3.py`: r3 modular-Q artifact validation and path rejection.
- `ecosystem_gym/experiments/m1314r2.py`: unopened r2 runner and future-stage CLI wiring.
- `ecosystem_gym/experiments/m1314r3.py`: unopened r3 runner and future-stage CLI wiring.
- `ecosystem_gym/experiments/m1314r2_support.py`: r2 ledger and preflight helpers.
- `ecosystem_gym/experiments/m1314r3_support.py`: r3 ledger and preflight helpers.
- `tests/test_m1314r2_policy_protocol.py`: r2 manifest, CLI, and unopened-stage tests.
- `tests/test_maintenance_modular_q_m1314r2.py`: r2 teacher dataset and artifact tests.
- `tests/test_m1314r3_policy_protocol.py`: r3 manifest, CLI, and unopened-stage tests.

## Verification

- Focused M13.14-r3/r2/legacy protocol suite: 51 passed.
- Full repository suite: 282 collected tests passed.
- `python -m compileall -q ecosystem_gym tests`: passed.
- `git diff --check`: passed.
- Independent Luna-max integrated review: approved after the r3-local learned
  replay verifier was added and the accidental shared M13.9 edit was removed.
- The r2 manifest, preflight, ledger, teacher data, policy artifacts, and
  partial development trace surface remain locally preserved. The r3 manifest,
  preflight, ledger, teacher dataset, 12 policies, 1,440 traces, and development
  report now exist locally. R3 confirmation and audit remain absent.

## Next Step

Do not resume M13.14/r2, tune on r3 check traces, select a favorable r3 seed, or
open confirmation/audit. The r3 modular candidate failed and the monolithic
anchored control is the stronger learned baseline. Any successor requires a new
predeclared hypothesis and fresh splits.
