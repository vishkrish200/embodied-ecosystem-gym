# Codex Handoff: M13 persistent-maintenance RL reset

Updated: 2026-08-06
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch/worktree: `main` with committed M8–M12 history, original M13 failure evidence, and M13.1 in progress.

## Current Goal

Re-center the project on the v1 observation ladder: scripted controller,
state-oracle RL baseline, hybrid policy, then RGB policy. The immediate next
milestone is M13, a state-oracle RL baseline for M10's persistent-maintenance
task. Do not start another behavior-cloning or RGB policy variant first.

## Current State

- M0–M5 provide the deterministic MuJoCo Gym, typed skills and outcomes,
  state-oracle/hybrid/RGB observations, replayable JSONL traces, and a thin
  viewer using the same environment loop.
- M2 and M6 already demonstrate tabular state-oracle Q-learning, but only on
  short Find-and-eat or food-versus-play tasks; neither is a long-horizon
  persistent-maintenance RL result.
- M8–M8.9 provide coverage-gated RGB grounding/recovery evidence and sealed
  visual audits. They do not establish end-to-end visual control.
- M10 validates the persistent environment and oracle ceiling in all four
  conditions. M11 freezes M9 to expose its maintenance gap.
- M12 is a committed exploratory structured-RGB behavior-cloning diagnostic:
  it reaches 80/80 survival and full forced recovery, but only 3/20 compound
  maintenance completions. Its validation rows were inspected during
  diagnosis, so they are not a clean model-selection suite.

## Constraints

- Preserve the authoritative Gym, deterministic replay, bounded typed skills,
  and thin viewer boundary.
- A state-oracle RL policy may use only the documented state-oracle observation
  plus policy-owned memory. It must not receive task IDs, reset options,
  `info`, or private environment access.
- Keep M12 as an honest diagnostic. Do not merge it as the main project claim,
  retune it against the old validation rows, or open a sealed M12 audit.
- Original M13 validation is complete and failed: survival passed but no full
  policy episode completed the required cycles. Its 2200–2219 audit remains
  unopened. Do not reuse its validation rows as a development signal.
- `docs/M13_1_PROTOCOL.md` freezes M13.1: the same public-state tabular policy
  and 12,000-episode schedule, but guarded +0.25 reward only for authoritative
  feed/play/rest counter increments. Its development / validation / audit
  splits are new 2300–2339 / 2400–2419 / 2500–2519. M13.2 is conditional on a
  clean M13.1 validation failure and must use a new protocol/splits.

## Next Steps

1. Implement and test M13.1's zero-default, guarded cycle-transition reward;
   preserve all legacy/M13 reward behavior at default configuration.
2. Run `python -m ecosystem_gym m131-train --output ...` for the fixed,
   development-only 12,000-episode budget and inspect only its artifacts.
3. Run development-only performance checks without opening validation or audit;
   fix implementation defects only, not gates or protocol constants.
4. Freeze the code and run M13.1 validation once. It writes and replays every
   policy trace, then reports every condition,
   confidence intervals, completion cycles, safe-drive time, recovery chains,
   and ablations.
5. Score the M13.1 audit once only after validation passes. If validation
   fails, leave the audit unopened and write M13.2's distinct protocol.
6. Only then plan M14 hybrid RL and M15 RGB persistent maintenance with the
   same task and policy objective.

## Do Not Repeat

- Do not use a behavior-cloning score as evidence that RL has solved
  long-horizon maintenance.
- Do not use aggregate survival to hide a cycle-completion failure.
- Do not tune against a frozen validation or sealed-audit suite.
- Do not add frontend state, contact-physics claims, or VLM/VLA integration
  before the persistent-maintenance RL baseline is established.
