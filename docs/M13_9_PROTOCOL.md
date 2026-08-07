# M13.9 protocol — semi-Markov public-drive potential screen

Protocol version: `m13.9-public-drive-potential-r1`

## Status, question, and decision rule

**Status: protocol only; no M13.9 split has been opened.** M13.9 is a new
two-stage experiment, not a retune of M13.8.

M13.8 established two facts on a fresh, replayed probe. First, uncapped cycle
bonuses can induce feed farming. Second, replacing them with capped milestones
plus a weak always-negative safety integral induced a worse policy: satisfy
the quota, then `WAIT` while drives decay. The capped design removed recurring
credit for repairing a public drive error, while its large terminal gate was
delayed and partly dependent on cumulative evaluator state outside the frozen
policy input.

M13.9 asks one narrow, falsifiable question:

> With the same public policy interface, macros, 200-decision mechanics, and
> capped milestones as M13.8, does a duration-aware potential difference over
> only the public drive vector supply useful immediate correction credit without
> making repeated restore/degrade loops profitable?

The potential is a **credit-assignment intervention**, not a claim that hidden
cumulative terminal gates have been made Markov. It must be rejected unless it
beats the M13.8-style static-cost control on fresh data and is at least
non-inferior to a contemporaneous legacy-reward comparator in the cheap screen.
If the short screen fails, M13 RL pauses for a broader redesign; no M13.9
coefficient, gate, schedule, or checkpoint variant is authorized.

## Frozen boundary and unchanged mechanics

The following are exactly the M13.8 200-decision task contract and remain
unchanged for every arm:

- `max_episode_steps=200`, four persistent-maintenance conditions, food renewal,
  event-triggered stale-pickup relocation, drive dynamics, typed-skill executor,
  public complementary macro mask, and eight macros;
- dense masked Double-DQN architecture, replay capacity/warmup/batch,
  optimizer, learning rate, target-copy interval, update every four decisions,
  epsilon schedule, final-checkpoint-only selection, and action compiler;
- the 30-dimensional public feature vector: relative food/toy/rest vectors,
  public drives, holding state, prior outcome, and declared policy-owned M13
  memory; and
- the public inference prohibition: no task/layout ID, seed, reset option,
  `info`, reward component, counter, recovery/resource flag, terminal cause,
  timer, availability flag, segmentation, renderer state, or private object.

The policy-owned capped cycle/cooldown/last-macro/recovery memory is also
unchanged. M13.9 may use public pre/post drives in the post-transition learning
reward only; it may not add a drive-error accumulator, a safety counter, or
any reward component to the inference feature, mask, or memory.

M13.8's screen-fit and probe partitions (4800–4827) are historical evidence,
not M13.9 training or selection data. M13.8 confirmation/audit partitions
(4900–5119) remain sealed and must not be touched or reassigned.

## Fresh one-way partitions

Every M13.9 layout must have a new identifier and a full coordinate tuple
distinct from every earlier M10–M13.8 layout. The four condition suffixes keep
the following M13.8 controls exactly; only the layouts and seeds are new.

| Condition | Suffix | Frozen controls beyond task/layout |
| --- | --- | --- |
| `persistent_reference` | `northeast` | orange food, ball toy, `scan_v2`, north initial scan |
| `renewal_and_morphology` | `southwest` | purple capsule food, cube toy, capsule agent, `scan_v2`, south initial scan |
| `event_relocation` | `northwest` | blue box food, capsule toy, dim lighting, `scan_v2`, east initial scan, relocation on first pickup |
| `compound` | `southeast` | red capsule food, cube toy, box agent, grippy dynamics, blocked distractor at `[-0.04, 0.04]`, `scan_v2`, west initial scan, relocation on first pickup |

| Partition | Environment seeds | Layout identifiers | Permitted use |
| --- | ---: | --- | --- |
| Screen fit | `5200–5219` | `m139_screen_{northeast,southwest,northwest,southeast}` | Preflight and fitting diagnostics only. |
| One-shot screen probe | `5220–5227` | `m139_probe_{northeast,southwest,northwest,southeast}` | Open once after every screen policy is serialized. Promotion only. |
| Confirmation fit | `5300–5339` | `m139_confirm_fit_{northeast,southwest,northwest,southeast}` | Reserved until screen promotion; fitting only. |
| Confirmation evaluation | `5400–5419` | `m139_confirm_eval_{northeast,southwest,northwest,southeast}` | Open once after every confirmation policy is serialized. |
| Sealed audit | `5500–5519` | `m139_audit_{northeast,southwest,northwest,southeast}` | Open once only after confirmation passes. |

A durable M13.9 split-open ledger records a partition before any coverage,
policy, trace, or aggregate result is examined. Opening a probe, confirmation
evaluation, or audit makes it permanently ineligible for any M13.9 fitting or
successor-reward selection. Before each split is scored, public scan coverage
and the balanced state-oracle ceiling must pass exactly as in M10/M13.8.

## Candidate reward: bounded milestones plus a public potential

Let `x = (satiety, energy, boredom)` be a permitted public drive vector. Define
the buffered normalized errors:

```text
u_s(x) = clip((0.30 - satiety) / 0.15, 0, 1)
u_e(x) = clip((0.30 - energy)  / 0.15, 0, 1)
u_b(x) = clip((boredom - 0.70) / 0.20, 0, 1)

E(x)   = max(u_s(x)^2, u_e(x)^2, u_b(x)^2)
Phi(x) = -1.00 * E(x)
Gamma(d) = 0.99^d
```

`E` is zero in the public comfort region, is one at or beyond a hard drive
limit, and uses the worst drive so healthy dimensions cannot cancel one
neglected need. For a compiled macro duration `d_t > 0`, the candidate's
potential component is:

```text
F_t = Gamma(d_t) * Phi(x_{t+1}) - Phi(x_t)
```

The terminal post-action public drive vector remains in `Phi(x_{t+1})` even
when the Q bootstrap is zero. Telescoping therefore leaves a bounded endpoint
preference for ending safer rather than depleted. It is deliberate: it is
public, directionally aligned with the evaluation metric, and at most one unit
of scaled drive error. Discounted sums of `F_t` telescope to that endpoint
term; a restore/degrade loop cannot generate unbounded discounted return. In
particular, a `WAIT` from comfort to worse drives has negative `F_t`; an
interaction that reduces public drive error has positive `F_t` of the
corresponding magnitude.

The complete candidate reward is:

```text
r_t = -0.01 * d_t
      -0.10 * I_invalid
      -0.10 * I_ordinary_blocked
      +0.25  * delta_feed_milestone
      +0.25  * delta_play_milestone
      +0.375 * delta_rest_milestone
      +0.25  * I_first_completed_required_recovery_chain
      +5.0   * I_full_terminal_success
      -2.0   * I_terminal_failure
      +F_t
```

The capped `3/3/2` milestone, recovery-chain, invalid/ordinary-blocked, and
terminal definitions are byte-for-byte the M13.8 definitions. The M13.8
trapezoidal `-0.02*C_t` safety integral is removed; it is not added to `F_t`.
No action, cycle, safe second, or `WAIT` receives an uncapped positive bonus.
All authoritative counters and recovery state are post-transition reward and
evaluation sidecars only.

For avoidance of doubt, the report's full-objective success is true only after
200 decisions when the agent survives, reaches at least `3/3/2`, has both
decision-safe and duration-safe fractions at least `.85` under M13.8's hard
safe predicate, and completes the stale-pickup-to-later-consume chain in each
relocation condition. This is an evaluator metric, not a policy feature.

## Semi-Markov Double-DQN backup

Because `F_t` uses a duration exponent, the learner must use the same
semi-Markov discount in every arm:

```text
y_t = r_t + I_not_done * Gamma(d_t) * Q_target(s_{t+1},
                                                argmax_masked Q_online(s_{t+1}))
```

`d_t` is the compiler-recorded positive duration already used for the action;
it is neither a policy input nor a free parameter. This is the sole learner
change from M13.8. Applying `Gamma(d_t)` to candidate reward but retaining a
per-decision `.99` bootstrap would not be a coherent semi-Markov experiment.
All arms share this backup, so the candidate-versus-static-cost comparison
isolates the replacement of M13.8's safety term with `F_t`.

This does change the effective planning horizon relative to M13.8's
per-decision backup. M13.9 therefore makes no causal score comparison to old
M13.6–M13.8 reports. Its contemporaneous static-cost control identifies the
potential's effect under the new coherent backup; its legacy comparator is a
guardrail against accepting a candidate that merely improves a weak control.

## Frozen screen arms

For each training seed, start all learned arms from byte-identical parameters
and use identical environment schedules and PCG64 derivations. They all use
the semi-Markov backup above.

| Arm | Scalar reward difference | Role |
| --- | --- | --- |
| `public_potential_candidate` | Exact candidate reward above. | Tests immediate public-drive correction credit. |
| `static_cost_control` | Exact M13.8 bounded reward: capped milestones plus `-0.02*C_t`, without `F_t`. | Causal control for the potential replacement. |
| `legacy_guardrail` | Legacy uncapped-cycle reward, without M13.8 caps/safety/terminal replacements. | Must not be materially outperformed by a weaker loitering solution. |
| `mask_aware_random` | Evaluation-only public-mask random policy. | Sanity floor; never fitted. |

The legacy guardrail is not used to select a reward coefficient or to make an
historical 160-step claim. It is trained from scratch on M13.9 fit data and
shares the 200-step mechanics and semi-Markov backup.

## Reward and replay preflight

Before fitting, tests and deterministic screen-fit diagnostics must prove all
of the following. Any failure rejects M13.9 before the probe opens.

1. Boundary/randomized-drive tests prove `0 <= E <= 1`, recomposition of every
   scalar reward, and the exact M13.8 one-time cap/recovery/blocked rules.
2. A scripted public transition suite proves a comfort-region `WAIT` has zero
   potential and negative total reward from time cost; a `WAIT` that increases
   `E` has negative potential; and a successful public-drive restoration has
   positive potential.
3. A two-transition restore/degrade loop and randomized multi-step trajectories
   prove the discounted sum of `F_t` equals
   `Gamma(total_duration)*Phi(final) - Phi(initial)` to numerical tolerance.
   No cycle of public drive states may create positive residual credit.
4. The exact compiled duration used in `F_t` is the duration used in the
   backup, trace, and replay. A modified duration, a per-decision bootstrap, or
   an absent terminal post-drive vector is a conformance failure.
5. Public scan coverage and the balanced public-state scripted ceiling pass all
   screen-fit condition/seed cells. The ceiling must also satisfy the declared
   safety and recovery gates.
6. On each screen-fit condition, the scripted balanced public oracle must beat
   fixed feed-only, play-only, rest/wait, and **quota-then-wait** diagnostics by
   at least `2.0` mean undiscounted candidate return. `quota-then-wait` uses
   only its own public memory to reach `3/3/2`, then selects `WAIT`; it is a
   reward-sanity diagnostic, not a teacher or a learned arm.

Every preflight trace is strictly replayed. Its reward-sidecar fields are
available only to reward/evaluation replay and are never supplied to inference.

## Stage 1 — short parallel fail-fast screen

Use paired training seeds `20260911` and `20260912`. For each seed, fit exactly
the three learned arms above for **2,000 episodes** on screen-fit seeds
5200–5219. The fixed condition/seed schedule visits every one of the 80 fit
cells exactly 25 times. This is 12,000 learned episodes total (two seeds ×
three arms × 2,000), deliberately smaller than M13.8's 16,000-episode screen.
The inherited epsilon schedule is not compressed and final weights are the
only eligible checkpoints.

Run the six fits as six isolated spawned processes in one submission wave;
each worker must set numerical-library threads to one. Process parallelism is
execution-only: random streams, initial weights, environment schedules,
artifacts, hashes, submission order, and worker/thread counts are recorded and
may not depend on completion order. This uses available cores without changing
the deterministic experiment.

Serialize and reload all six final policies before opening the one-shot probe.
Probe each policy greedily on four conditions × eight fresh seeds (32 episodes)
with strict policy, memory, reward, and environment replay. Report every
condition separately, pooled rates, duration-safe occupancy, total `WAIT`
occupancy, and `unsafe_wait_fraction` (the fraction of decisions selecting
`WAIT` whose public post-action `E(x)` is greater than zero).

For **each** candidate replica, promotion requires:

- every condition: survival at least `7/8`, `3/3/2` maintenance at least `6/8`,
  mean decision-safe fraction at least `.80`, and mean duration-safe fraction
  at least `.80`;
- each relocation condition: completed stale-pickup-to-later-consume recovery
  at least `7/8`;
- every condition: `unsafe_wait_fraction <= .15`; and
- zero feature-boundary, mask, redundant selected `GO_*`, reward recomposition,
  duration-discount, learned-action replay, memory replay, or environment
  replay violations.

Across its 32 probe episodes, each candidate must also have:

- survival at least `+0.20` over matched mask-aware random;
- decision-safe and duration-safe fractions at least `+0.10` over its paired
  `static_cost_control`;
- full-objective success at least `+0.15` over its paired
  `static_cost_control`;
- `unsafe_wait_fraction` at least `.15` lower than its paired
  `static_cost_control`; and
- full-objective success, decision-safe fraction, and duration-safe fraction
  each no worse than `.03` below its paired `legacy_guardrail`.

Both candidate replicas must meet every rule. At least one must additionally
exceed its static-cost control by `.25` in full-objective success. This is a
high bar intentionally: a merely less-bad bounded policy is not worth the
confirmation budget. Failure stops M13.9, publishes the complete screen, and
leaves 5300–5519 untouched. It does not authorize more short variants.

## Stage 2 — guarded confirmation and audit

Only an exact Stage-1 promotion authorizes confirmation. Train from scratch on
confirmation-fit seeds 5300–5339 with eight paired training seeds
`20260913–20260920`. Fit only the candidate and its `static_cost_control`,
12,000 episodes each: 192,000 learned episodes total, equal to M13.8's planned
candidate/control confirmation budget. The legacy guardrail is not refit at
this expensive stage because it is a screen guardrail, not the causal control.

Serialize every 16 final policy before opening confirmation evaluation. Score
each once on 5400–5419 (80 episodes per policy), with the same strict replay
and exact gates as M13.8 except for the added per-condition
`unsafe_wait_fraction <= .10` requirement. All eight candidate replicas must
pass every per-condition survival, maintenance, decision-safe, duration-safe,
recovery, integrity, and random-survival rule. A passing subset fails.

For the causal potential claim, calculate each policy's normalized gate margin
as M13.8 defines it, extended with `0.10 - unsafe_wait_fraction` in every
condition. The candidate margin must be strictly greater than its matched
static-cost-control margin in at least 7/8 pairs. A candidate 8/8 hard-gate
pass without 7/8 margin wins supports a robust policy result but not a causal
potential-improvement claim.

After both confirmation rules pass, freeze all policies, sources, report code,
and split ledger. Open 5500–5519 once, run coverage and the scripted ceiling,
then score the frozen candidates and static-cost controls exactly once. Apply
the same hard and paired-margin gates. No audit retry, replacement seed,
coefficient change, or retraining is permitted.

## Required evidence and limits

The protocol fingerprint includes this document, all reward/discount constants,
feature/memory/mask/compiler schemas, split and exact layout tuples, schedules,
source hashes, PCG64 derivation, and worker/thread configuration. Each policy,
trace, report, and split marker is content-hashed.

Every transition records the public pre/post drives, `E`, `Phi`, duration,
`Gamma(d)`, potential term, every base reward component, recomposition,
features, Q values, mask, selected macro, compiled action, memory transition,
counter/recovery sidecar, terminal state, and both safety occupancy measures.
Strict replay recomputes all of them from a clean reset and re-runs constrained
inference. A replay mismatch invalidates its replica and split.

M13.9 can still fail because the terminal quota/recovery/occupancy gate remains
partly history-dependent relative to the frozen 30-vector; a potential over
public drives cannot reveal that private evaluator history. It can also fail if
the macro abstraction makes corrective travel too delayed, if error reduction
near a threshold is too sparse, or if function approximation does not exploit
the shaped credit. A failure is evidence against this *specific* public-drive
potential formulation, not proof that no state-oracle maintenance learner can
work. It establishes neither learned perception nor broad embodied competence.
