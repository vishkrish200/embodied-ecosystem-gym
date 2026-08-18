# Codex Handoff: M13.13 persistent-maintenance policy families

Updated: 2026-08-18
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch: `codex/m1313-coverage-abort` from merged `main` `957accd`

## Outcome

M13.13 development was authorized and stopped at its first failing hard gate.
Fit `6200--6219` passed public coverage and produced the four declared learned
actors. Check `6220--6239` then failed public scan coverage before any policy
evaluation. No development report, evaluation trace, replay result, family
gate, confirmation, or audit exists. This is an aborted run, not a policy
result. See `docs/M13_13_DEVELOPMENT_ABORT.md`.

Three causally different policy families now share the existing public
30-feature observation/eight-action contract:

1. `UrgencySchedulerPolicy`: deterministic deadline margins, hysteresis,
   bounded objective commitment, preemption, and continued maintenance after
   minimum quotas.
2. `ModelBasedSchedulerPolicy`: bounded receding-horizon search over an explicit
   optimistic model of public drive dynamics and macro durations, scored
   lexicographically for safety before quota progress.
3. `ShieldedLearnedPolicy`: a corrected-PPO-compatible MLP actor whose choices
   can be restricted by a deterministic safety/commitment supervisor. The
   candidate/control comparison uses exactly the same actor bytes.

A supervised-anchor/retention family is intentionally deferred. It remains a
separate possible hypothesis rather than being folded into the shielded-policy
comparison.

## Frozen Evidence State

- M13.10 imitation-only remained 32/32 per replica, while subsequent DQN
  fine-tuning collapsed both replicas to 0%.
- M13.11-r1 remains invalid. Corrected M13.11-r2 scored 33/320 versus DQN 9/320
  and failed its stability gate.
- M13.12 corrected PPO `3e-4` versus `1e-3` remains frozen at candidate
  `73,80,26,0` (`179/320`) versus baseline `80,4,0,26` (`110/320`). All 640
  traces replayed and the all-replica gate failed.
- No M13--M13.12 source, protocol, result, ledger, artifact, or trace was
  changed. Existing traces motivated policy families only; they were not used
  to tune M13.13 thresholds or select a policy.
- M13.13 fit/check `6200--6239` are opened and retired. Seeds `6140--6199` and
  `6240--6299` remain unused. Confirmation/audit `6300--6519` remain unopened
  but cannot proceed because no valid development report exists.

## Implementation Map

- `ecosystem_gym/maintenance/policy_state.py`: unencoded policy-owned memory,
  public-dynamics signature, drive projection, deadlines, margins, and semantic
  helpers. The public feature vector is still exactly length 30.
- `ecosystem_gym/maintenance/urgency.py`: deterministic urgency family and its
  commitment-disabled one-factor control.
- `ecosystem_gym/maintenance/model_based.py`: depth-four candidate, depth-one
  horizon control, public transition model, and deterministic search.
- `ecosystem_gym/maintenance/shielded.py`: inference actor, shielded candidate,
  unshielded geometry-only control, and actor-byte identity support.
- `ecosystem_gym/maintenance/policy_artifacts.py`: versioned, hashed,
  tamper-evident policy serialization with config/dynamics validation.
- `ecosystem_gym/maintenance/policy_protocol.py`: machine-readable frozen
  splits, arms, budgets, gates, source hashes, and exact commands.
- `ecosystem_gym/experiments/m1313_support.py`: ordered append-only split ledger.
- `ecosystem_gym/experiments/m1313.py`: future-only fit/development,
  confirmation, audit, replay, reporting, and gate implementation.
- `docs/M13_13_POLICY_FAMILIES_PROTOCOL.md`: human-readable frozen protocol,
  rationale, risks, gates, commands, and authorization boundary.
- `tests/test_maintenance_policies.py` and
  `tests/test_m1313_policy_protocol.py`: synthetic unit, numerical,
  serialization, tamper, gate, ledger, manifest, and CLI tests.

The CLI registrations are:

- `python -m ecosystem_gym maintenance-policy-manifest`
- `python -m ecosystem_gym.experiments.cli m1313-development`
- `python -m ecosystem_gym.experiments.cli m1313-confirmation`
- `python -m ecosystem_gym.experiments.cli m1313-audit`

The manifest and development commands were each invoked once under explicit
authorization. Do not invoke them again. Confirmation and audit are blocked.

## Frozen M13.13 Protocol

- Development fit: 6200--6219.
- Development check: 6220--6239.
- Confirmation fit: 6300--6339.
- Confirmation evaluation: 6400--6419.
- Audit: 6500--6519.
- Development budget ceiling: 1,120 evaluation traces.
- Later confirmation/audit ceiling: 1,760 evaluation traces.
- Every experimental episode must emit a trace and pass exact replay.
- Each family has its own one-factor control and passes independently; there is
  no cross-family winner selection.
- Core hard gate: at least 18/20 successes per condition, safety fraction at
  least 0.85, recovery at least 18/20, unsafe-wait rate at most 0.10, and zero
  contract violations. Predeclared candidate/control deltas and safety
  non-inferiority also apply.
- The shield candidate/control must use byte-identical learned actors.

Read `docs/M13_13_POLICY_FAMILIES_PROTOCOL.md` for the normative details and
exact future commands. The machine-readable constants in
`ecosystem_gym/maintenance/policy_protocol.py` are the executable counterpart.

## Verification

- Final focused suite: 15 passed.
- Final full repository suite: all 224 collected tests passed.
- `python -m compileall`: passed.
- `git diff --check`: passed.
- Ruff was not available in the environment (`Failed to spawn: ruff`), so no
  Ruff result is claimed.

The pre-run tests used synthetic fixtures and temporary directories. They did
not exercise canonical M13.13 layout coverage; the authorized runtime gate
caught that gap after fitting.

## What Remains Speculative

- None of the three families has empirical M13.13 performance evidence.
- The deterministic thresholds are mechanics-derived hypotheses, not tuned
  results.
- The model-based planner is intentionally optimistic about hidden travel and
  manipulation costs; horizon four may still be too short or model-mismatched.
- The shield may prevent known unsafe/semantically useless choices yet still
  leave too little useful authority to the learned actor, or the future actor
  fit may itself remain seed-unstable.
- Passing a synthetic or ordinary repository test is not evidence of policy
  success in environment episodes.

## Authorization Boundary And Next Step

M13.13 has stopped permanently. Do not rerun its fit/check, reconstruct missing
coverage counts by reopening `6220--6239`, edit its frozen sources or layouts,
fabricate a report, or invoke confirmation/audit.

The next permissible implementation task is an additive fresh-split successor
that preserves the same three hypotheses and independently persists public
coverage for every development partition before learner fitting. No successor
experiment is authorized yet. It requires a new frozen protocol, manifest,
ledger, split family, review, and explicit development authorization.

## Reactivation Prompt

Continue from
`/Users/vishnukrishnan/Developer/embodied-ecosystem-gym/docs/codex-handoffs/current.md`.
Read it, `docs/M13_13_POLICY_FAMILIES_PROTOCOL.md`, and
`docs/M13_13_DEVELOPMENT_ABORT.md` completely. Verify the local artifact hashes
and append-only ledger without executing an environment. Preserve M13.13 and
design an additive coverage-first successor only if the user asks. Do not run
new episodes or open any split without a new explicit authorization.
