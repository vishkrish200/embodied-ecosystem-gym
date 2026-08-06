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
- M13.1 validation is complete and failed cleanly. Coverage and 400 traces
  passed/replayed, but full policy maintenance was 0/20 in every condition;
  its `2500–2519` audit is sealed. The guarded reward helped development but
  did not generalize. Do not rerun M13.1 validation or open its audit.
- `docs/M13_2_PROTOCOL.md` freezes M13.2: retain M13.1 reward, public-state
  boundary, macro compiler, memory and gates, but replace only tabular Q with
  a 30-feature 64x64 Double-DQN. Its new development / validation / audit
  splits are 2600–2639 / 2700–2719 / 2800–2819.

## Next Steps

1. Implement M13.2 only from `docs/M13_2_PROTOCOL.md`; retain the M13.1
   reward/config, public boundary, macro compiler, gates, and zero-default
   legacy reward behavior.
2. Add the fresh `m132_*` layouts and a deterministic 30-feature NumPy
   Double-DQN with strict model/feature/trace replay tests.
3. Run the fixed development-only M13.2 budget; freeze code before opening its
   new validation suite, then leave its audit sealed unless validation passes.
4. Only then plan M14 hybrid RL and M15 RGB persistent maintenance with the
   same task and policy objective.

## Do Not Repeat

- Do not use a behavior-cloning score as evidence that RL has solved
  long-horizon maintenance.
- Do not use aggregate survival to hide a cycle-completion failure.
- Do not tune against a frozen validation or sealed-audit suite.
- Do not add frontend state, contact-physics claims, or VLM/VLA integration
  before the persistent-maintenance RL baseline is established.
