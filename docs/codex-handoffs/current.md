# Codex Handoff: M13 persistent-maintenance RL reset

Updated: 2026-08-06
Repo/path: `/Users/vishnukrishnan/.codex/worktrees/ee80/embodied-ecosystem-gym`
Branch/worktree: `codex/m11-frozen-policy-baseline` at `f819d28`, with uncommitted M12 exploratory work.

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
- M12 is an uncommitted exploratory structured-RGB behavior-cloning result:
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
- Freeze M13's reward, observation encoding, action set, training budget,
  train/validation/audit layouts, and per-condition gates before training.

## Next Steps

1. Write `docs/M13_PROTOCOL.md` before training code. Define the state-oracle
   input boundary, macro/action representation, reward, persistent-cycle and
   recovery metrics, random/no-drive/no-memory ablations, and replay evidence.
2. Create new M13 development and validation layouts/seeds: M10's 1700–1819
   ranges have supported M12 fitting or iterative diagnosis. Keep 1900–1919
   untouched for the final audit only if the frozen audit layouts remain valid.
3. Implement an RL policy over the existing M10 typed skill interface. Start
   from the M2/M6 Q-learning patterns, but keep training genuinely reward-led
   rather than teacher-macro imitation.
4. Run the frozen validation once after development choices are fixed. Report
   every condition, confidence intervals, completion cycles, safe-drive time,
   recovery chains, ablations, and exact replay. Score the audit once only
   after validation passes.
5. Only then plan M14 hybrid RL and M15 RGB persistent maintenance with the
   same task and policy objective.

## Do Not Repeat

- Do not use a behavior-cloning score as evidence that RL has solved
  long-horizon maintenance.
- Do not use aggregate survival to hide a cycle-completion failure.
- Do not tune against a frozen validation or sealed-audit suite.
- Do not add frontend state, contact-physics claims, or VLM/VLA integration
  before the persistent-maintenance RL baseline is established.
