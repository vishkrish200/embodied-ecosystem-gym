# M13.13 frozen protocol: distinct public maintenance-policy families

Status: **implemented, frozen, and unopened**. No fit, development check,
confirmation, audit, environment episode, canonical manifest, ledger marker, or
experimental trace has been produced under this protocol.

M13.13 does not retry M13.12 or select a favorable PPO seed. It implements
three causally different scheduling hypotheses and predeclares an independent
one-factor comparison for each. Existing M13--M13.12 source, protocols,
reports, ledgers, policies, traces, partitions, and conclusions remain frozen.

## Verified motivation, not tuning data

- M13.10's two imitation-only replicas generalized at 32/32 each with the same
  public 30-feature representation. The corresponding Double-DQN fine-tunes
  fell to 0/32. The representation can express a useful scheduler; retention
  and discovery remain unresolved.
- Corrected M13.11-r2 PPO reached 33/320 full objectives and failed its
  all-replica stability requirement.
- M13.12's `1e-3` PPO arm reached 179/320 but produced replica totals
  `73,80,26,0`, failing the declared gate. Its report and replay integrity are
  evidence that the failure is strategic and seed-sensitive, not permission
  to tune on 6120--6139.

No M13.10--M13.12 evaluation or check trace was used to choose an M13.13
threshold, policy, planning depth, or gate. The public mechanics, previously
published safe bands, and the existing scripted scheduler are the only
numerical priors.

## Implemented policy hypotheses

| Family | Causal hypothesis | Candidate/control | Enforced invariant | Main failure mode |
| --- | --- | --- | --- | --- |
| Urgency, hysteresis, and commitment | Long-horizon failure is primarily bad switching and abandonment of restorative chains | Identical deadline scheduler with commitment on versus off | Held food is consumed; a chosen goal continues through travel and interaction unless a materially more urgent safety deadline preempts it | Fixed margins may transfer poorly; stale commitment can still waste up to the six-step cap |
| Short-horizon public model | The policy needs to value a multi-macro chain rather than a single legal action | Identical public transition model and lexicographic score at depth 4 versus depth 1 | Every proposed interaction is productive in the public model; safety violations dominate quota progress and relief | `GO_*` duration is optimistic when a hidden dynamics variant slows real motion; food availability is inferred from public cooldown memory |
| Shielded learned actor | Learned preferences are useful inside a deterministic viability and objective envelope | Supervisor enabled versus disabled on byte-identical actor weights | The actor cannot choose geometrically illegal or semantically useless interactions, unsafe waiting, abandon required recovery, or abandon a supervisor commitment | A strong supervisor may dominate a weak actor, so any claim must remain about the hierarchy, not learned planning alone |

The implementations are additive:

- `ecosystem_gym/maintenance/urgency.py`;
- `ecosystem_gym/maintenance/model_based.py`;
- `ecosystem_gym/maintenance/shielded.py`;
- `ecosystem_gym/maintenance/policy_state.py` for shared public dynamics and
  policy-owned checkpoint state;
- `ecosystem_gym/maintenance/policy_artifacts.py` for tamper-evident policy
  serialization; and
- `ecosystem_gym/maintenance/policy_protocol.py` plus
  `ecosystem_gym/experiments/m1313.py` for the unopened manifest and future
  runner.

### Threshold provenance

- Safe boundaries remain satiety `>.15`, energy `>.15`, and boredom `<.90`.
- Feed activation remains `.55`, copied from the pre-existing public scripted
  ceiling rather than chosen from a learned-policy trace.
- Play and rest productivity use the environment's public `.60` boredom and
  `.45` energy interaction guards after compiled duration.
- A one-second WAIT reserve equals the compiled WAIT duration. The urgency
  scheduler's two-second preemption reserve is one interaction second plus one
  WAIT reserve. A one-second advantage prevents exact-margin chatter.
- Six committed decisions cover `GO -> PICK_UP -> CONSUME` plus a forced
  relocation retry with one spare decision. It is a loop bound, not a fitted
  performance coefficient.
- Model depth four covers the longest ordinary public chain plus one follow-up;
  depth one is the declared myopic control.

## Public inference boundary

Every family receives only the existing public observation keys
`agent_xy`, `food_xy`, `toy_xy`, `rest_xy`, `drives`, `holding_food`, and
`prior_outcome`, plus policy-owned memory. Every family emits one of the same
eight maintenance macros and uses the existing complementary geometry mask and
macro compiler.

`PolicyMemory.committed_goal`, `commitment_steps`, and
`supervisor_interventions` are additive policy-owned fields. They are
checkpointed and replay-deterministic but deliberately ignored by
`encode_features`; the learned actor still receives exactly 30 features. Task
IDs, reset controls, `info`, authoritative cycle counters, layouts, and private
environment state remain outside inference.

## Planner objective and declared approximation

The model-based family enumerates public semantic macro sequences and ranks
them lexicographically by:

1. lower predicted safety-violation cost;
2. higher minimum safety slack;
3. more capped `3/3/2` quota progress;
4. greater restorative effect;
5. higher terminal safety slack; and
6. lower elapsed public duration.

This avoids a fitted scalar reward mixture. The model applies public drive
rates and interaction effects exactly, assumes a public target is reached after
the compiler's nominal `distance / walk_speed` duration, and uses policy-owned
food cooldown as the availability estimate. In the compound grippy condition,
actual hidden movement scaling may invalidate that nominal-arrival assumption.
That discrepancy is a predeclared test of model robustness, not a reason to
repair the model after viewing check data.

## Fresh one-way partitions

The protocol leaves 6140--6199 and 6240--6299 deliberately unused.

| Partition | Seeds | Layout prefix | Access |
| --- | --- | --- | --- |
| `development_fit` | 6200--6219 | `m1313_fit_*` | corrected-PPO actor fitting only |
| `development_check` | 6220--6239 | `m1313_check_*` | one declared family comparison |
| `confirmation_fit` | 6300--6339 | `m1313_confirm_fit_*` | rebuild promoted learned actors only |
| `confirmation_evaluation` | 6400--6419 | `m1313_confirm_eval_*` | one confirmation after promotion |
| `audit` | 6500--6519 | `m1313_audit_*` | one no-refit audit |

All 20 layouts contain 100 unique four-decimal points, disjoint from every
registered earlier layout. The one-way ledger order is exactly
`development_fit -> development_check -> confirmation_fit ->
confirmation_evaluation -> audit`. Only a currently last fit partition may be
resumed idempotently.

## Development arms and budgets

The urgency pair changes only `commitment_enabled=True` versus `False`. The
model pair changes only `lookahead_depth=4` versus `1`.

The learned pair uses four fresh training seeds `20261331--20261334`. Each
fits the unchanged corrected M13.11-r2 `30 -> 64 -> 64 -> actor/critic` PPO at
`3e-4`, 2,048-row rollouts, four epochs, and 400,000 macro decisions on
development fit. This is the original corrected learning rate, not a choice of
M13.12's favorable aggregate. Candidate and control copy the same serialized
actor bytes; only `shield.enabled` changes. Critic and optimizer state do not
enter inference.

The development check contains:

- urgency candidate/control: 160 episodes;
- depth-four/myopic model: 160 episodes;
- four shielded/unshielded actor pairs: 640 episodes;
- one shared scripted ceiling: 80 episodes; and
- one shared mask-aware random diagnostic: 80 episodes.

All 1,120 traces must strictly replay. The ceiling and random rows are plumbing
and scale diagnostics, not selection arms.

## Frozen independent gates

Every candidate, and every shielded learned replica, must independently pass
all four 20-episode conditions:

- survival, maintenance, and full objective at least `18/20` each;
- mean decision-safe and duration-safe fractions at least `.85`;
- required recovery at least `18/20` in each relocation condition;
- aggregate unsafe-WAIT fraction at most `.10`; and
- zero mask, conformance, artifact, ledger, coverage, or replay violations.

The urgency and model candidates must each exceed their matched control by
`.10` pooled full-objective rate, lose no more than one success in any
condition, and remain within `.02` decision and duration safety. Every shielded
replica must exceed its byte-identical unshielded control by `.15` pooled full
objective and remain within `.02` safety. All four shield replicas must pass.

Families are not ranked. Each passes or fails its own gate. Check data may not
be used to choose one implemented family over another, change a threshold, or
create a hybrid. A passing family permits only separately authorized
confirmation. Confirmation rebuilds eight learned replicas from fresh seeds
`20261335--20261342` and repeats the same gates. Audit refits nothing and opens
6500--6519 once.

## Manifest and exact future commands

The canonical manifest is content- and source-hashed. The future runner rejects
a stale or edited manifest before ledger access. These commands are frozen but
have **not** been executed:

```text
uv run python -m ecosystem_gym maintenance-policy-manifest --output artifacts/manifests/m1313-policy-families.json

uv run python -m ecosystem_gym experiment m1313-development \
  --manifest artifacts/manifests/m1313-policy-families.json \
  --output artifacts/reports/m1313-development.json

uv run python -m ecosystem_gym experiment m1313-confirmation \
  --manifest artifacts/manifests/m1313-policy-families.json \
  --development-report artifacts/reports/m1313-development.json \
  --output artifacts/reports/m1313-confirmation.json

uv run python -m ecosystem_gym experiment m1313-audit \
  --manifest artifacts/manifests/m1313-policy-families.json \
  --confirmation-report artifacts/reports/m1313-confirmation.json \
  --output artifacts/reports/m1313-audit.json
```

Generating the canonical manifest does not authorize a fit or score. Before
the first command that opens `development_fit`, the user must explicitly
authorize the exact M13.13 development run after reviewing this protocol,
implementation diff, tests, budget, and destination paths. Confirmation and
audit each require new explicit authorization after verifying the preceding
sealed report and promotion gate.

## Deliberately deferred hypothesis

A supervised-anchor or teacher-retention learner remains causally distinct and
plausible after M13.10, but it is not smuggled into these three comparisons.
It would require its own fresh protocol specifying teacher rows, anchor loss or
constraint, update budget, retention metric, and byte-identical controls. No
such fit or claim is part of M13.13.
