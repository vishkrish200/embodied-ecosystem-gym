# Codex Handoff: M13 persistent-maintenance RL reset

Updated: 2026-08-06
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch/worktree: `main` with committed M8–M12 history and M13 in progress.

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
- `docs/M13_PROTOCOL.md` freezes M13's reward, state adapter, action set,
  training budget, fresh 2000–2039 / 2100–2119 / 2200–2219 splits, and
  per-condition gates. The initial implementation adds `rest_xy` to the public
  state-oracle observation and a
  reward-led tabular macro learner. Its frozen validation has not been run.

## Next Steps

1. Finish M13 persistence artifacts: write/replay every frozen evaluation
   trace and serialize the learned policy plus its encoder/compiler hashes.
2. Run development-only performance checks without opening validation or audit;
   fix implementation defects only, not gates or protocol constants.
3. Freeze the code and run M13 validation once. Report every condition,
   confidence intervals, completion cycles, safe-drive time, recovery chains,
   and ablations.
4. Score the M13 audit once only after validation passes.
5. Only then plan M14 hybrid RL and M15 RGB persistent maintenance with the
   same task and policy objective.

## Do Not Repeat

- Do not use a behavior-cloning score as evidence that RL has solved
  long-horizon maintenance.
- Do not use aggregate survival to hide a cycle-completion failure.
- Do not tune against a frozen validation or sealed-audit suite.
- Do not add frontend state, contact-physics claims, or VLM/VLA integration
  before the persistent-maintenance RL baseline is established.
