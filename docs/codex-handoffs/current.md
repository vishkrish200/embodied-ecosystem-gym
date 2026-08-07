# Codex Handoff: M13 persistent-maintenance RL reset

Updated: 2026-08-07
Repo/path: `/Users/vishnukrishnan/Developer/embodied-ecosystem-gym`
Branch/worktree: `main` with committed M8–M13.8 history and a frozen M13.9 protocol; implementation has not started.

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
- M13.2 development is complete but failed before validation. Its compact
  Double-DQN completed no development maintenance episode in any condition;
  a diagnostic rollout selected `PICK_UP` for all 160 decisions and received
  160 cheap `blocked` outcomes. This exposes that the M13.1 reward/action
  semantics do not disincentivize blocked short actions under approximation.
  Its 2700–2819 validation/audit splits remain unopened. Do not introduce an
  action mask or blocked-action penalty without a newly frozen protocol.
- M13.3 development is complete and failed before validation. Its fresh
  2900–2939 coverage and scripted ceiling each pass 160/160, and strict replay
  passes all 960 development traces. The full DQN completes zero maintenance
  episodes in every condition; it removes the M13.2 repeated blocked-pickup
  loop but does not learn the required maintenance schedule. Its 3000–3019
  validation and 3100–3119 audit remain unopened. Do not retry M13.3
  development or add an action mask retroactively.

## Next Steps

1. Preserve `artifacts/reports/m133-development.json` and its 960 replayed
   development traces as a negative result; do not open M13.3 validation.
2. M13.4 development is complete and failed before validation. Its fresh
   3200–3239 coverage/ceiling checks pass 160/160 and all 3,040 traces replay,
   with zero public-mask violations; nevertheless none of its three full-policy
   replicas passes every survival/maintenance/safe-drive/recovery gate. Its
   3300–3419 validation/audit splits remain unopened. Do not retry M13.4 or
   silently widen the policy input.
3. M13.5 development is complete and failed before validation. Its six
   spawned training jobs used one numerical thread each; coverage and the
   scripted ceiling passed 160/160 and all 1,600 development traces strictly
   replayed. Update-every-4 substantially beat paired update-every-256 and
   random controls, and seed 20260812 passed every row, but seed 20260813
   collapsed in compound/persistent-reference and seed 20260814 missed the
   renewal/morphology safe-drive gate. The all-replicas rule therefore fails.
   Its 3600–3719 validation/audit splits remain unopened. Do not rerun M13.5,
   select a seed, or silently widen the public interface.
4. M13.6 development is complete and failed before validation. Its fresh
   eight-pair candidate-first matrix passed coverage/ceiling 160/160, strictly
   replayed all 4,000 development traces, and every dense candidate beat its
   sparse/null and random comparators. Only 20260820 and 20260823 passed every
   per-condition gate; six of eight candidates failed at least one
   maintenance/survival/recovery/safe-drive row. Dense update is therefore a
   strong but non-reproducible effect. Its 4200–4319 validation/audit splits
   remain unopened. Do not rerun M13.6, select a passing seed, or widen the
   public interface.
5. M13.7's fresh two-seed, 4,000-episode target-cadence screen rejected both
   faster hard-copy candidates (250 and 100 updates) against the 1,000-update
   control on its independent probe. It correctly stops here: no 12,000-episode
   confirmation, validation, or audit run is authorized; 4500–4719 remain
   unopened. Preserve the report as a fast negative result.
6. M13.8's bounded-reward screen is a clean negative result. It prevented
   uncapped feed farming but created quota-then-`WAIT` loitering: 1/64 full
   objective successes versus 40/64 for the paired legacy controls, despite
   all 512 replays passing. Its 4800–4827 fit/probe partitions are opened;
   4900–5119 remain sealed and must not be reused. Do not tweak M13.8
   coefficients. The next permitted experiment is M13.9's newly documented
   public-drive potential, using a new split family.
7. Only after a state-RL baseline passes should M14 hybrid RL and M15 RGB
   persistent maintenance proceed with the
   same task and policy objective.

## Do Not Repeat

- Do not use a behavior-cloning score as evidence that RL has solved
  long-horizon maintenance.
- Do not use aggregate survival to hide a cycle-completion failure.
- Do not tune against a frozen validation or sealed-audit suite.
- Do not add frontend state, contact-physics claims, or VLM/VLA integration
  before the persistent-maintenance RL baseline is established.
