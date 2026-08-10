# Codex Handoff: M13.13 persistent-maintenance policy families

Updated: 2026-08-10
Repo/path: `/Users/vishnukrishnan/.codex/worktrees/f890/embodied-ecosystem-gym`
Branch: `codex/m1313-policy-families`
Base: `eafeb8a` (the M13.12 durable-handoff commit)

## Outcome

M13.13 is implemented, documented, and verified, but deliberately unopened.
No M13.13 manifest, split ledger, training artifact, episode, trace, comparison,
confirmation, or audit was created or run.

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
- M13.13 reserves fresh 6200+ families. Seeds 6140--6199 and 6240--6299 remain
  deliberately unused.

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

These commands exist for a separately authorized future run. Do not invoke the
canonical commands merely to smoke-test them; registration is covered by tests.

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

The tests use synthetic fixtures and temporary directories only. No canonical
M13.13 experimental surface was touched.

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

The next permissible experimental action requires the user to explicitly say,
in substance: **Authorize the exact M13.13 manifest and development run frozen
in `docs/M13_13_POLICY_FAMILIES_PROTOCOL.md`.** That authorization would cover
only manifest creation plus development fit/check on 6200--6239 within the
declared budget and stop rules.

Confirmation and audit each require later, separate explicit authorization
after the preceding sealed report is reviewed. Development authorization does
not authorize confirmation or audit. Editing the frozen protocol after a split
is opened requires stopping and defining a new milestone/split family rather
than silently retuning M13.13.

## Reactivation Prompt

Continue from
`/Users/vishnukrishnan/.codex/worktrees/f890/embodied-ecosystem-gym/docs/codex-handoffs/current.md`.
Read it and `docs/M13_13_POLICY_FAMILIES_PROTOCOL.md` completely, inspect Git
state, and verify that M13.13 remains unopened. Do not create a manifest, ledger,
artifact, trace, training run, or environment episode unless the user gives the
specific development authorization above. Confirmation and audit remain
separate authorization gates.
