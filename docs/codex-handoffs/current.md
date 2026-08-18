# Codex Handoff: M13.14 modular-anchored successor

Updated: 2026-08-18
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch: `codex/m1314-modular-anchored-q` from `main` `0f10002`

## Outcome

M13.14 is frozen and stopped. Its exact development coverage preflight passed
all fit/check visibility rows on `6600--6639`. The separately authorized
development stage then opened fit and wrote a valid public teacher dataset, but
aborted during the first warm-start minibatch because minibatches were
incorrectly required to contain all eight macro classes. No learner fit or
policy evaluation completed. See `docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md`,
`docs/M13_14_DEVELOPMENT_PREFLIGHT.md`, and
`docs/M13_14_DEVELOPMENT_ABORT.md`.

M13.13 remains the frozen abort record. Its authorized run stopped at public
coverage on fit `6200--6219` / check `6220--6239` before any policy evaluation,
trace, replay, confirmation, or audit. See `docs/M13_13_DEVELOPMENT_ABORT.md`.

M14 and M15 stay blocked pending separate authorization.

## Frozen Evidence State

- M13.14 manifest fingerprint:
  `5c215736ba2194f502f09600f6ce5f287543deb237f77546088eb7279490cfb8`.
- Development fit/check coverage passed every declared 20-seed row; the sealed
  report hash is
  `fcaa88d8c62c7730ae47c974b6809d22912cec2629deff84c0e59c866810c88f`.
- The M13.14 ledger contains only `development_preflight -> development_fit`.
- The 16,000-row teacher dataset contains every macro globally, but no learned
  policy artifact, development report, evaluation, replay, confirmation, or
  audit exists.
- M13.13 fit/check `6200--6239` are opened and retired. No reopened split is
  permitted in place.
- M14 and M15 remain blocked until a separately authorized successor run
  completes its own declared preflight and fit sequence.

## Current Map

- `ecosystem_gym/maintenance/policy_protocol_m1314.py`: frozen M13.14 manifest,
  fresh splits, preflight order, and no-run authorization.
- `ecosystem_gym/maintenance/modular_q.py`: modular and monolithic Q learners,
  teacher replay, shielded action selection, and artifact validation.
- `tests/test_m1314_policy_protocol.py`: fresh-split, preflight, ledger, CLI,
  and future-run short-circuit tests.

## Verification

- Final focused M13.14/maintenance suite: 28 passed.
- Final full repository suite: 244 collected tests passed.
- `python -m compileall -q ecosystem_gym tests`: passed.
- `git diff --check`: passed.
- Independent Luna-max integrated code review: approved with zero remaining
  findings after the replay, report-verification, gate, recovery, and ledger
  fixes.
- The ignored canonical manifest, preflight report, ledger, and teacher dataset
  exist locally and are hashed in the abort record. The development report,
  learned policies, evaluation traces, confirmation, and audit remain absent.

## Next Step

Do not resume or patch M13.14 under its frozen manifest. The next implementation
task is an additive corrected successor with fresh splits and separate complete
dataset versus minibatch validators. Any corrected preflight or fit requires
new explicit authorization. Confirmation and audit remain blocked.
