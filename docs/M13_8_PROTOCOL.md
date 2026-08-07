# M13.8 protocol — bounded maintenance-aligned reward

Protocol version: `m13.8-bounded-maintenance-reward-r1`

## Status, diagnosis, and bounded question

M13.8 is a prospective two-stage reward experiment. It does not report an
implemented result, a passing screen, or a confirmed policy.

The primary diagnosis is reward misalignment. M13.6 showed that dense
Double-DQN consistently beats its sparse and random controls, but only 2/8
replicas passed every maintenance gate. Failed dense policies often survived
while performing roughly 29–30 feed cycles and neglecting play, rest, or safe
drive occupancy. Under the legacy reward, every feed pays `+0.25` forever, so
30 feeds can earn about `+7.5` while terminal maintenance success pays only
`+1.0`. That ranking makes cycle farming a rational learned solution to the
training objective.

This diagnosis is more likely than an insufficient public interface: the
scripted policy using the same public state and macros reaches the mechanics
ceiling, and some learned replicas pass. Seed-sensitive value learning remains
a live secondary cause. M13.8 therefore asks one narrow question: does a
bounded, safety-aware reward make the already competent dense learner reliably
optimize the declared long-horizon objective?

## Declared 160-to-200 correction and causal boundary

M10 and M13–M13.7 were implemented with `max_episode_steps=160`, but the stated
benchmark objective is survival over 200 macro decisions. M13.8 declares and
corrects that mismatch:

```text
max_episode_steps = 200
```

The candidate and its contemporaneous legacy-reward control both use 200.
Consequently, M13.8 is not a reward-only continuation of historical M13.6 and
must not compare its absolute scores causally against old 160-decision runs.
Within M13.8, the paired candidate and control differ only in the scalar reward
contract; both share the corrected horizon, mechanics, data, initialization,
and training settings.

Apart from the declared horizon correction, the four conditions, reset
controls, typed-skill mechanics, food renewal, event relocation, drive
dynamics, action compiler, public complementary mask, and eight macros remain
fixed. No observation, action, other mechanics, curriculum, network,
optimizer, replay, discount, or target-cadence change is part of this
experiment.

## Frozen public policy boundary

The complete decision-time source remains the M13.6 public state:

```text
agent_xy, food_xy, toy_xy, rest_xy
drives = [satiety, energy, boredom]
holding_food
prior_outcome
declared serializable policy-owned memory
```

The 30-vector remains exactly M13.2/M13.6's representation: six clipped and
normalized relative-target coordinates, three drives, holding state, the
six-way prior-outcome one-hot vector, five declared memory values, and the
nine-way `START`/last-macro one-hot vector. Policy memory remains the capped
feed/play/rest counts, inherited food-cooldown bucket, last macro, and
pending-recovery bit, updated only from permitted pre/post observations, the
selected macro and its duration, and the next public outcome. The cooldown
bucket is policy-owned action-duration memory: it resets after a publicly
observed successful `CONSUME`, accumulates the durations of the policy's own
subsequent actions up to its cap, and means only "respawn may now be
available." It is never read from an environment timer, resource flag,
availability field, or private resource timing.

The policy still selects one of:

```text
GO_FOOD, PICK_UP, CONSUME, GO_TOY, PLAY, GO_REST, REST, WAIT
```

The existing public complementary mask applies during epsilon exploration,
greedy choice, and the Double-DQN bootstrap. It may inspect only public
positions, public holding state, and fixed public radii. It does not inspect
drive thresholds or resource availability.

Task/layout IDs, seed, reset options, condition/split labels, `info`, timers,
food availability, event or resource flags, authoritative counters, terminal
causes, reward components, segmentation, renderer state, and private
environment objects are prohibited from features, mask computation, policy
memory, action selection, and inference. Authoritative counters and condition
requirements may be used only by the post-transition reward/evaluation layer
and must be kept out of the policy-visible transition.

## Fresh one-way partitions

No M10–M13.7 development, validation, audit, screen, probe, trace, policy, or
layout is reused.

| Partition | Environment seeds | Layout identifiers | Permitted use |
| --- | ---: | --- | --- |
| Screen fit | `4800–4819` | `m138_screen_{northeast,southwest,northwest,southeast}` | Contract preflight, fitting, and training diagnostics. |
| One-shot screen probe | `4820–4827` | `m138_probe_{northeast,southwest,northwest,southeast}` | Open once after both paired screen policies are serialized. Promotion decision only. |
| Confirmation fit | `4900–4939` | `m138_confirm_fit_{northeast,southwest,northwest,southeast}` | Reserved until screen promotion; fitting only. |
| Confirmation evaluation | `5000–5019` | `m138_confirm_eval_{northeast,southwest,northwest,southeast}` | Open once after all confirmation policies are serialized. |
| Sealed audit | `5100–5119` | `m138_audit_{northeast,southwest,northwest,southeast}` | Open once only after the confirmation hard gate passes. |

Every new `LayoutSpec` must use genuinely new coordinates, not an earlier
coordinate tuple under a new identifier. Layout specifications and coordinate
tuples are fingerprinted. The four suffixes retain the frozen condition
controls:

| Condition | Suffix | Controls beyond task and layout |
| --- | --- | --- |
| `persistent_reference` | `northeast` | orange food, ball toy, `scan_v2`, north initial scan |
| `renewal_and_morphology` | `southwest` | purple capsule food, cube toy, capsule agent, `scan_v2`, south initial scan |
| `event_relocation` | `northwest` | blue box food, capsule toy, dim lighting, `scan_v2`, east initial scan, relocation on first pickup |
| `compound` | `southeast` | red capsule food, cube toy, box agent, grippy dynamics, blocked distractor at `[-0.04, 0.04]`, `scan_v2`, west initial scan, relocation on first pickup |

A durable split-open ledger records when each partition is first touched. An
opened probe, confirmation-evaluation, or audit partition can never become fit
data for a revised reward.

## Exact candidate reward

Let the permitted public drives before and after macro decision `t` be

```text
x_t     = (satiety_t, energy_t, boredom_t)
x_{t+1} = (satiety_{t+1}, energy_{t+1}, boredom_{t+1})
```

and let `d_t > 0` be the duration compiled for that selected macro. Define the
buffered, normalized one-drive errors:

```text
u_s(x) = clip((0.30 - satiety) / 0.15, 0, 1)
u_e(x) = clip((0.30 - energy)  / 0.15, 0, 1)
u_b(x) = clip((boredom - 0.70) / 0.20, 0, 1)

E(x) = max(u_s(x)^2, u_e(x)^2, u_b(x)^2)
C_t  = d_t * (E(x_t) + E(x_{t+1})) / 2
```

The candidate reward replacing the complete legacy scalar is:

```text
r_t = -0.01 * d_t
      -0.02 * C_t
      -0.10 * I_invalid
      -0.10 * I_ordinary_blocked
      +0.25  * delta_feed_milestone
      +0.25  * delta_play_milestone
      +0.375 * delta_rest_milestone
      +0.25  * I_first_completed_required_recovery_chain
      +5.0   * I_full_terminal_success
      -2.0   * I_terminal_failure
```

The cycle terms use authoritative pre/post counters only after the transition:

```text
delta_feed_milestone = min(feed_cycles_{t+1}, 3) - min(feed_cycles_t, 3)
delta_play_milestone = min(play_cycles_{t+1}, 3) - min(play_cycles_t, 3)
delta_rest_milestone = min(rest_cycles_{t+1}, 2) - min(rest_cycles_t, 2)
```

Each delta is non-negative and a feed/play/rest cycle pays only until its
required quota is reached. Feed, play, and rest therefore each have the same
maximum milestone budget of `0.75`; all cycle milestones total at most `2.25`.
The first completed required recovery chain adds at most `0.25`, so the entire
nonterminal positive pool is bounded by `2.50`. The thirtieth feed pays zero.

`I_invalid` is one only for an outcome other than `SUCCESS` or `BLOCKED`.
`I_ordinary_blocked` is one for a blocked outcome except the single forced
stale pickup that triggers food relocation. That forced transition remains
unpenalized. The recovery-chain start requires the same transition to record
both `outcome == BLOCKED` and `disturbance == food_relocated`. Its completion
requires a strictly later selected `CONSUME` with `outcome == SUCCESS` after
that start. `I_first_completed_required_recovery_chain` is one exactly once,
on that completion transition, in an applicable relocation episode. A
`post_disturbance_completion` flag without the explicit stale-block start and
later successful consume is insufficient. The indicator is always zero in
conditions without a required relocation chain. Legacy bonuses for a recovery
walk and post-disturbance completion are removed; intermediate recovery
actions do not pay.

The hard safe-band predicate on a public post-action drive vector is:

```text
safe(x) = satiety > 0.15 and energy > 0.15 and boredom < 0.90
```

For an episode with `T` executed macros:

```text
decision_safe_fraction = sum_t I_safe(x_{t+1}) / T
duration_safe_fraction = sum_t d_t * I_safe(x_{t+1}) / sum_t d_t
```

The M13.8 evaluation maintenance metric is also explicit:

```text
maintenance_complete = survived_horizon_200
                       and feed_cycles >= 3
                       and play_cycles >= 3
                       and rest_cycles >= 2
```

Reaching `3/3/2` before an early depletion is not maintenance completion. This
metric is recomputed from the trace and does not trust a legacy environment
maintenance flag.

Full terminal success is true only on the episode-ending transition when all
of the following hold:

- the agent survived exactly 200 decisions;
- authoritative feed/play/rest counts are at least `3/3/2`;
- decision-safe fraction is at least `0.85`;
- duration-safe fraction is at least `0.85`;
- the stale-pickup-to-later-consume recovery chain completed when relocation
  was required.

The terminal indicators use the decision-safe and duration-safe accumulators,
cycle counts, and recovery accounting after incorporating the final
transition. They do not reuse the environment's legacy `task_success` or
cycle-only maintenance boolean. Those legacy fields may be recorded for
diagnostics but cannot trigger the candidate terminal bonus.

`I_terminal_failure` is one on any episode-ending transition that is not full
terminal success, including early drive depletion and a 200-decision horizon
reached with a missing cycle, safety, or required recovery gate. The success
and failure indicators are mutually exclusive and both are zero before the
episode-ending transition.

The safety error is zero in the buffered comfort region
`satiety>=.30`, `energy>=.30`, `boredom<=.70`; it rises quadratically toward
one at the hard `.15/.15/.90` limits. Taking the maximum prevents two healthy
drives from averaging away one neglected drive. The trapezoidal factor charges
the public before/after error for the full compiled duration. Since
`0 <= E <= 1`, `0 <= C_t <= d_t`, and the added safety charge is bounded by
`0.02*d_t`. There is no positive per-second safe reward that can make loitering
profitable.

## Paired legacy-reward control

The control uses the exact M13.6/M13.3 scalar reward at the corrected
200-decision horizon: duration cost, invalid penalty, recovery-exempt ordinary
blocked penalty, uncapped `+0.25` per authoritative feed/play/rest increment,
the legacy recovery action/completion bonus, legacy `+1.0` terminal
cycle-maintenance reward, and legacy depletion penalty. It does not receive
the candidate's buffered safety cost, capped milestones, one-time chain reward,
or full-gate terminal success/failure terms.

Historical M13.6 weights are forbidden. Every control is trained from scratch
on the same fresh fit partition as its candidate. Paired arms use identical
initial parameter seeds, environment schedules, and action/replay RNG seed
derivations. Divergence caused by their learned behavior is retained, not
corrected by selecting or rerunning a pair.

## Reward-contract preflight

The following cheap checks run before learned fitting. Any failure stops M13.8
without opening the probe.

1. Unit/replay checks prove that the fourth feed, fourth play, and third rest
   pay no milestone reward; recovery pays at most once; the forced stale pickup
   is blocked-penalty exempt; terminal success and failure are complementary;
   and `0 <= E(x) <= 1` over boundary and randomized public drive vectors.
2. Component recomposition must reproduce the candidate scalar exactly,
   including duration scaling and terminal terms.
3. Public scan coverage and the scripted balanced public-state ceiling must
   pass all 20 screen-fit seeds in every condition.
4. On the screen-fit matrix, the scripted balanced public oracle must exceed
   each frozen feed-only, play-only, and rest/wait specialist by at least `2.0`
   in paired mean undiscounted candidate return in every condition.

The specialists are fixed reward-sanity diagnostics, not training teachers,
demonstrations, policy inputs, or learned baselines. A redesign that rewards a
specialist as highly as balanced maintenance is rejected before consuming the
learned-policy budget.

Their exact public deterministic state machines are:

- **Feed-only:** if holding food, select `CONSUME`; otherwise, if the
  policy-owned cooldown bucket is below its cap, select `WAIT`; otherwise
  select `PICK_UP` inside the public pickup radius and `GO_FOOD` outside it.
  It ignores play and rest.
- **Play-only:** select `PLAY` inside the public toy radius and `GO_TOY`
  outside it. It ignores food and rest.
- **Rest/wait:** while public energy is at most `0.45`, select `REST` inside
  the public rest radius and `GO_REST` outside it; otherwise select `WAIT`. It
  ignores food and play.

Each choice must be eligible under the same complementary public mask, use the
same compiler, and update inherited policy memory through the same public
memory transition as M13.8. These definitions, their threshold/radius source,
policy class/source hashes, fixed condition order, ascending environment-seed
order, and return aggregation are part of the protocol fingerprint. They may
not be improvised after preflight results are observed.

## Frozen RNG derivation and evaluation order

For each declared training seed `s`, the candidate and paired legacy control
each instantiate the existing single NumPy PCG64 training stream with seed
exactly `s`; that stream drives He initialization, epsilon choices, and replay
sampling in the inherited order. The pair starts from byte-identical network
parameters. Its matched mask-aware-random policy uses a separate PCG64 stream
also seeded exactly `s`. The deterministic specialist policies consume no
policy RNG.

Training episode `e` uses condition index `e mod 4` in the frozen condition
table order and fit seed index `(e // 4) mod N` in ascending order, where
`N=20` for screen fit and `N=40` for confirmation fit. Scored evaluation uses
the condition table order and ascending seeds; a matched random stream is
created once per `(stage, s)` and consumed continuously in that order. No
Python hash, process ID, wall clock, OS entropy, worker slot, or completion
order may derive randomness. This mapping and the implementation that applies
it are included in the protocol fingerprint.

## Stage 1 — paired 4,000-episode fail-fast screen

Use paired PCG64 training seeds `20260901` and `20260902`. For each seed, train
exactly:

1. candidate reward, dense masked Double-DQN;
2. paired legacy reward, otherwise identical dense masked Double-DQN.

Mask-aware macro random is evaluated with each paired seed but is not fitted.
Each learned fit is exactly 4,000 episodes on the four conditions and screen-fit
seeds `4800–4819`, visiting every condition/seed cell exactly 50 times. This is
16,000 learned episodes total, one twelfth of the 192,000 learned episodes in
the paired eight-seed confirmation.

All four learned final weight files are serialized before the probe is opened.
Only final weights are eligible; no checkpoint, seed, arm, or episode count is
selected. Probe evaluation must reload those files into fresh policy objects
and verify their fingerprints; it may not score the mutable in-memory training
objects. The probe is then scored once over four conditions x eight seeds, 32
episodes per policy, with greedy masked inference and complete strict replay.

For one policy over the 32 probe episodes, a pooled binary rate is the count
divided by 32. A pooled safe fraction is the unweighted mean of the 32
episode-level fractions. For each paired seed, candidate promotion requires:

- in every condition, survival `8/8`;
- in every condition, cycle-complete maintenance at least `7/8`, where an
  episode must survive 200 and reach `3/3/2`;
- in every condition, mean decision-safe fraction at least `0.85`;
- in every condition, mean duration-safe fraction at least `0.85`;
- in `event_relocation` and `compound`, completed recovery `8/8`;
- zero feature-boundary, mask, redundant selected `GO_*`, reward-recomposition,
  learned-action replay, memory replay, and environment replay violations;
- candidate pooled survival minus matched mask-aware-random pooled survival at
  least `0.20`;
- candidate minus paired-control pooled survival at least `0.00` and pooled
  cycle-maintenance at least `0.00`;
- candidate minus paired-control pooled decision-safe and duration-safe
  fractions each at least `-0.02`.

Both candidate replicas must satisfy every requirement. In addition, at least
one of the two pairs must show either candidate-minus-control pooled
cycle-maintenance of at least `0.10` or candidate-minus-control pooled
decision-safe fraction of at least `0.05`.

If any promotion requirement fails, stop. Publish the complete screen and leave
confirmation fit/evaluation and audit unopened. Do not change a coefficient,
gate, training setting, or implementation against the opened probe. A revised
reward is a new protocol and requires an entirely new split family.

Passing the screen only promotes the frozen reward design. It is not evidence
of a robust baseline and does not permit a claim that reward misalignment has
been confirmed.

## Frozen learner settings in both stages

The following remain identical across reward arms and fixed from M13.6:

```text
network                         30 -> 64 -> 64 -> 8, ReLU
algorithm                       Double-DQN with public next-state mask
replay capacity                 100,000 FIFO transitions
replay warmup                   1,000 transitions
batch size                      128, uniform without replacement
optimizer                       Adam, lr=3e-4, beta1=.9, beta2=.999, eps=1e-8
loss                            mean Huber, delta=1
discount                        gamma=.99 per macro, not duration exponentiated
optimizer cadence               one update every 4 macro decisions
hard target copy                every 1,000 optimizer updates
epsilon                         1.00 to .05 inclusive over episodes 0..9000,
                                then .05
checkpoint                      serialized final weights only
```

The 4,000-episode screen does not compress the epsilon schedule. No early
stopping, checkpoint selection, prioritized replay, demonstrations,
curriculum, extra fitting, or target-cadence change is allowed. Process count
is execution-only; each isolated spawned worker must cap numerical-library
threads at one, and submission/start/finish order and worker count are
recorded.

## Stage 2 — guarded paired eight-seed confirmation

This stage exists only after an exact screen promotion. Train from scratch on
confirmation-fit seeds `4900–4939` with paired PCG64 seeds `20260903` through
`20260910` inclusive. For each of the eight seeds, train:

1. the promoted candidate reward;
2. the paired legacy-reward control.

Each learned fit is exactly 12,000 episodes: four conditions x 40 fit seeds x
75 visits. Matched mask-aware random is evaluation-only. All 16 final learned
policies are serialized and hashed before confirmation evaluation is opened.
No screen weights, checkpoint, fit seed, training replica, or arm may be
selected or dropped.

Score every serialized policy exactly once on confirmation-evaluation seeds
`5000–5019`, giving 20 episodes per condition and 80 per policy. Every one of
the eight candidate replicas must independently satisfy:

- per condition, survival at least `18/20`;
- per condition, cycle-complete maintenance at least `18/20`;
- per condition, mean decision-safe fraction at least `0.85`;
- per condition, mean duration-safe fraction at least `0.85`;
- in both relocation conditions, completed recovery at least `18/20`;
- zero policy-boundary, mask, redundant-macro, reward-recomposition,
  learned-action, memory, and environment replay violations;
- pooled candidate survival minus its matched random survival at least `0.20`.

A passing subset is failure; the robust candidate gate is 8/8 replicas.

For the paired reward-improvement claim, define each evaluated policy's
normalized gate margin as the minimum of:

```text
survival_rate(condition)         - 0.90, for all four conditions
maintenance_rate(condition)      - 0.90, for all four conditions
mean_decision_safe(condition)    - 0.85, for all four conditions
mean_duration_safe(condition)    - 0.85, for all four conditions
recovery_rate(condition)         - 0.90, for the two relocation conditions
pooled_survival - matched_random_pooled_survival - 0.20
```

Any conformance or replay violation sets that policy's margin to negative
infinity. The candidate margin must be strictly greater than its paired
legacy-control margin in at least 7/8 training-seed pairs.

If the candidate hard gate fails for any replica, M13.8 confirmation fails and
the audit remains closed. If the candidate passes 8/8 but the 7/8 paired-margin
test fails, the result may support a robust state-oracle baseline under the new
reward, but it does not confirm reward misalignment as the causal reason for
the improvement.

## Sealed audit

After an 8/8 candidate hard-gate pass, freeze the exact eight candidate and
eight control policies, source/config fingerprints, report code, and matched
random seeds. Open `5100–5119` once. Run coverage/ceiling first, then evaluate
all frozen pairs and matched random policies without retraining. Apply the same
per-replica hard gates and 7/8 paired-margin test using audit rows. Any failure
is published unchanged. No audit rerun, replacement seed, coefficient update,
or retraining on confirmation/audit evidence is permitted.

## Required artifacts, replay, and leakage evidence

The protocol fingerprint covers this document/version, the 200-decision
configuration, reward constants and formula version, feature schema, memory
update, mask, macro compiler and durations, learner constants, condition and
split tables, layout coordinates, training schedules, source hashes, and
one-way split ledger. Every final policy, trace, and report has its own content
hash.

Every training/evaluation transition needed for reward verification records:

- pre/post permitted observations and public drive vectors;
- exact 30-feature vector and policy memory before/after;
- raw eight-Q vector, public mask, eligible macros, and selected macro;
- compiled typed action and exact duration;
- outcome and ordinary versus forced-exempt blocked classification;
- authoritative pre/post cycle counts in a privileged reward/evaluation
  sidecar, never in the inference record;
- safety errors `E(x_t)`, `E(x_{t+1})`, `C_t`, both episode safety
  accumulators, capped milestone deltas, one-time recovery state, every scalar
  reward component, and recomposed reward;
- termination/truncation, full-success/failure indicators, terminal gates,
  and resource/recovery events;
- policy, protocol, config, layout, source, RNG, model, trace, split-marker,
  update-cadence, and target-copy fingerprints.

Strict learned replay starts from a clean reset and must recompute the public
observation, 30-vector, mask, constrained action, compiled action/duration,
memory update, environment transition, reward components and scalar,
authoritative counters, safety accumulators, and terminal gates. Random replay
restores the recorded pre-choice PCG64 state and recomputes the eligible-set
draw. The privileged sidecar is available to reward/evaluation replay but is
never passed to policy inference. Any mismatch invalidates the affected
replica and split.

## Stop rules, failure modes, and claim limits

M13.8 can still fail or mislead in the following ways:

- An arm accidentally left at 160 decisions would invalidate the paired study;
  both arm configs and traces must prove 200.
- `gamma=.99` remains per macro despite variable real-time duration. The
  duration-scaled reward limits one inconsistency but does not make the learner
  a formally duration-correct semi-Markov DQN.
- The terminal `+5.0` is not dominant from the start of a 200-decision episode
  under the frozen discount: `5.0 * .99^199` is only about `0.68`. Bounded
  undiscounted reward accounting therefore does not guarantee that the DQN's
  discounted objective ranks every successful trajectory above every failed
  one.
- The candidate reward is history-dependent relative to the frozen 30
  features. Cumulative decision-safe and duration-safe occupancy, whether the
  authoritative recovery chain already completed/paid, and the resulting
  terminal gate are not fully encoded in policy state. Identical 30-vectors can
  therefore have different reward-to-go, violating the Markov assumption used
  by ordinary DQN. This protocol measures the empirical consequence; it does
  not claim to have removed that state aliasing.
- The trapezoidal public-drive cost can undercharge an interaction whose
  restorative effect occurs at the endpoint. The exact pre/post values and
  durations must therefore remain visible in traces.
- The policy can chatter around the `.30/.30/.70` comfort boundary or exploit
  one-step endpoint sampling. Report action occupancy and inspect the worst
  traces.
- A better-aligned reward may still expose seed-unstable Double-DQN. If the
  screen promotes but the eight-seed hard gate fails, do not resume reward
  tuning on confirmation data; revisit the learning algorithm on new data.
- Reward/evaluation legitimately uses counters and required-recovery state,
  but any path from those fields into features, mask, memory, Q selection, or
  inference is leakage and invalidates the result.
- Per-condition means can hide a bad episode tail. Reports must include
  episode-level metrics, quantiles, and the five worst traces per condition;
  these diagnostics do not replace or retroactively change the frozen gates.
- Extra post-quota feed/play/rest cycles are not automatically reward hacking:
  they may be needed to restore public drives. They pay no cycle bonus, and
  traces must show whether they improve safety rather than merely inflate a
  count.
- Fresh layouts are exact-tuple fresh: their full coordinate tuples are unique
  against prior and other M13.8 layouts. They are not guaranteed to be
  distribution-support-disjoint or geometrically out of distribution; nearby
  coordinates and the same bounded room semantics remain possible. Results
  therefore support fresh seeded-layout generalization, not a strong OOD
  claim.

The screen can only reject or promote a reward design. An 8/8 confirmation
hard-gate pass supports a robust state-oracle maintenance baseline on the
fresh confirmation split. The 7/8 paired-margin requirement is additionally
needed to say the contemporaneous experiment supports reward alignment over
the legacy reward. A passing sealed audit extends that bounded claim to the
unopened audit suite. No M13.8 result establishes learned perception, raw
motor control, physical manipulation, broad embodiment, or generalization
beyond these frozen mechanics, public state, macros, and split families.
