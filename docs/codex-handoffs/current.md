# Codex Handoff: Alternative persistent-maintenance policies

Updated: 2026-08-10
Repo/path: `/Users/vishnukrishnan/.codex/worktrees/dfab/embodied-ecosystem-gym`
Branch/worktree: `codex/m1312-ppo-lr`; implementation commit `a7ceb9b`, result commit `06937d4`

## Current Goal

- Plan and implement several genuinely different persistent-maintenance policy
  families that could succeed where random-initialized DQN/PPO was unstable.
- Implement code, unit/numerical tests, frozen protocol documents, CLIs, and
  future run manifests as appropriate.
- Do **not** run training, fit smokes, canonical development comparisons,
  screens, confirmations, audits, or new evaluation episodes in this task.

## Current State

- The policy boundary is sufficient: 30 public features, eight maintenance
  macros, complementary geometry mask, and policy-owned memory.
- The public scripted oracle in `ecosystem_gym/experiments/m13.py` uses a simple
  continuous drive scheduler: consume held food; rest at energy `<=.45`; feed
  when cooldown is ready and satiety `<=.55`; play at boredom `>=.60`; otherwise
  wait. It is a mechanics ceiling, not a learned result.
- M13.10 imitation-only `30 -> 64 -> 64 -> 8` policies generalized at 32/32 per
  seed using only the same public boundary. Subsequent Double-DQN fine-tuning
  destroyed the policy. Representation and expressivity are therefore not the
  main unresolved issue; discovery and retention are.
- M13.11-r1 was implementation-invalid. M13.11-r2 corrected the actor-gradient
  sign and multi-episode 2,048-row rollout, then failed stability at `33/320`
  versus DQN `9/320` on 6040--6079.
- M13.12 compared corrected PPO `3e-4` versus `1e-3` on fresh 6100--6139. The
  higher-rate candidate scored `73,80,26,0` (`179/320`) versus baseline
  `80,4,0,26` (`110/320`) and failed the predeclared all-replica gate. All 640
  traces replayed. No later gate was opened.

## Policy-Logic Findings

- Failed M13.12 candidate seed `20261324` reached feed quota in 80/80 episodes,
  play in 75/80, and rest in 70/80, but only 3/80 episodes met each safety
  fraction and none met the full objective.
- Successful candidate seeds continued restorative cycles after quotas:
  average feed/play/rest was `23.74/8.62/5.96` and `28.23/9.86/6.03`.
  Zero-success seed `20261324` averaged `21.25/3.55/2.36`: it treated minimum
  play/rest counts as if maintenance were finished.
- Failed policies entered `WAIT` attractors, sometimes 40--56 consecutive
  decisions while energy or boredom required intervention. WAIT frequency by
  itself is not the issue; successful policies wait only while drive margins
  remain safe and switch before danger.
- The mask enforces geometric eligibility, not semantic usefulness. A learned
  policy can legally play before boredom makes it productive or rest before
  energy makes it productive.
- Recovery passed 80/80 for every M13.12 candidate seed. The zero seed had no
  invalid or ordinary blocked-action loop. Its failure was strategic scheduling.
- Condition sensitivity remained large, suggesting the network entangles
  scheduling with geometry/dynamics instead of learning an invariant drive rule.

## Candidate Families To Consider

The new task should critically choose and implement at least three distinct,
testable hypotheses rather than cosmetic PPO variants. Strong candidates are:

1. A deterministic urgency scheduler with explicit safety margins, hysteresis,
   objective commitment, productive cooldown use, and continuous maintenance
   after quotas.
2. A short-horizon model-based scheduler using public drive dynamics, compiled
   macro durations, travel cost, and safety-margin prediction.
3. A hierarchical or shielded learned policy: deterministic safety/commitment
   supervisor plus a learned tie-breaker/residual that cannot select unsafe
   waiting or semantically useless interactions.
4. A supervised policy with a frozen teacher anchor or constrained retention
   objective, planned as a separate hypothesis from pure behavior cloning.

Do not assume all four belong in one experiment. Prefer a small number of
cleanly separated policy modules and future protocols with one causal change
per comparison.

## Files And Evidence

- `docs/M13_12_DEVELOPMENT_PROTOCOL.md`: frozen one-factor LR protocol.
- `docs/M13_12_DEVELOPMENT_RESULTS.md`: canonical M13.12 result and stop rule.
- `artifacts/reports/m1312-development.json`: ignored canonical report.
- `artifacts/reports/m1312-development-artifacts/traces/`: 640 replayed traces.
- `docs/M13_10_PROTOCOL.md` and `docs/M13_10_SCREEN_RESULTS.md`: imitation and
  catastrophic-retention evidence.
- `ecosystem_gym/maintenance/contract.py`: current public feature/memory/mask
  and macro compiler contract.
- `ecosystem_gym/maintenance/ppo_r2.py`: corrected frozen PPO implementation.
- `ecosystem_gym/experiments/m13.py`: scripted public scheduler ceiling.

## Decisions And Constraints

- Preserve every frozen M13--M13.12 source, protocol, report, ledger, artifact,
  trace, and split. New work must be additive.
- Do not tune on any opened evaluation/check data, including 5620--5627,
  6060--6079, or 6120--6139.
- `6080--6099` remain unused; do not opportunistically consume them. Reserve a
  clearly fresh future family, preferably 6200+, in separately frozen protocols.
- Planning and implementation are authorized. Unit tests, deterministic
  numerical tests, serialization tests, CLI-registration tests, compilation,
  full `pytest`, and `git diff --check` are authorized.
- No training-like smoke is authorized, even if called non-promotional. No new
  environment episode should be launched. Use fixtures/synthetic states for tests.
- Do not implement several policies and then select one using existing traces.
  Existing traces may motivate hypotheses, not tune thresholds or gates.
- A future run must require a separate explicit user authorization after code,
  tests, protocols, budgets, and stop rules are reviewed.

## Next Steps

1. Read this handoff and inspect current Git/repo state plus the named evidence.
2. Produce a skeptical design comparison of 3--4 policy families, including
   invariants, failure modes, public-input compliance, and what each isolates.
3. Choose a minimal implementable set of at least three distinct policies and
   freeze their future evaluation protocols before any run.
4. Implement additively with unit/numerical/serialization/CLI tests using only
   synthetic fixtures; do not execute environment episodes or training.
5. Run non-experimental verification only, document exact future commands but
   never execute them, and commit the scoped implementation.
6. Report what was implemented, what remains speculative, and the precise
   authorization needed before any future experiment.

## Reactivation Prompt

We are continuing from this handoff:
`/Users/vishnukrishnan/.codex/worktrees/dfab/embodied-ecosystem-gym/docs/codex-handoffs/current.md`.
Read it first, inspect the current repository and Git state, verify what still
applies, summarize the state in five bullets, then plan and implement several
distinct persistent-maintenance policy families. Do not run training, smokes,
canonical comparisons, screens, confirmations, audits, or new environment
episodes. Synthetic/unit/numerical tests and ordinary repository verification
are allowed.
