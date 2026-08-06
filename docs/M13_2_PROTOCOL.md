# M13.2 protocol — compact-state function-approximation revision

Protocol version: `m13.2-compact-q-r1`

## Decision and causal scope

M13.1 is a completed negative result. Its fresh validation had passing public
coverage, a passing scripted mechanics ceiling, 400 written/replayed traces,
and a failed learned-policy gate: zero maintenance completions in every
condition. Its `2500–2519` audit is sealed. The cycle-transition reward is
therefore retained exactly; it is not retuned.

M13.2 tests the next isolated hypothesis: M13/M13.1's tabular encoder
fragments nearby continuous states too severely to generalize between the
development and validation layouts. The only learned-policy change is the
value-function representation and its fitting machinery: a compact,
continuous-feature double-DQN replaces the table. The M10 environment,
M13.1 reward version, state boundary, macro set/compiler, policy memory,
conditions, gates, and ablation intent are unchanged. M12, original M13, and
M13.1 trajectories, tables, validation scores, and labels are prohibited
training inputs.

## New split family

M13.2 never reuses seeds/layouts from M10–M13.1, including the opened M13 and
M13.1 validation suites. Each row evaluates every listed seed.

| Split | Seeds | Layout identifiers | Permitted use |
| --- | --- | --- | --- |
| Development | 2600–2639 | `m132_dev_northeast`, `m132_dev_southwest`, `m132_dev_northwest`, `m132_dev_southeast` | Fixed training and development diagnostics only. |
| Validation | 2700–2719 | `m132_validation_northeast`, `m132_validation_southwest`, `m132_validation_northwest`, `m132_validation_southeast` | One frozen final-policy evaluation after code and development choices are frozen. |
| Sealed audit | 2800–2819 | `m132_audit_northeast`, `m132_audit_southwest`, `m132_audit_northwest`, `m132_audit_southeast` | Open once only after validation passes. |

The four condition names and reset controls are exactly M13.1's
`persistent_reference`, `renewal_and_morphology`, `event_relocation`, and
`compound`; only their named layouts change. The M10 coverage procedure and
the public-state scripted ceiling must pass every split before learned scores
are accepted. Coverage/ceiling are offline mechanics evidence, never model
features.

## Frozen inference boundary and feature encoder

The policy receives exactly the public M13 adapter at decision time:
`agent_xy`, `food_xy`, `toy_xy`, `rest_xy`, drives, `holding_food`,
`prior_outcome`, and the same serializable policy-owned memory. It may not
receive a task/layout ID, seed, reset option, condition, `info`, reward
components, cycle counter, resource timer/availability flag, segmentation,
private environment object, or external trace.

Before the network, the adapter and memory are deterministically encoded into
this fixed `float32[30]` vector:

```text
6  relative target coordinates: (food, toy, rest) - agent, clipped [-1.8, 1.8] / 1.8
3  drives: [satiety, energy, boredom]
1  holding_food: 0 or 1
6  prior_outcome one-hot in enum order:
   [success, blocked, target_not_visible, not_holding_object, not_edible, timeout]
3  feed_count/3, play_count/3, rest_count/2
1  food_cooldown_bucket/5
1  pending_recovery: 0 or 1
9  last_macro one-hot in [GO_FOOD, PICK_UP, CONSUME, GO_TOY, PLAY, GO_REST,
   REST, WAIT, START] order
```

The network is a fully deterministic NumPy MLP: `30 → 64 → 64 → 8`, ReLU
hidden layers, one scalar Q value per existing M13 macro. It has no recurrence
beyond the declared memory, no action mask, no expert branch, no handcrafted
macro preference, and no world handle. `no_drive_q` replaces only the three
drive features by zero during fitting and inference; `no_memory_q` replaces
only the declared memory features with their canonical empty representation;
`macro_random` samples the same eight macros uniformly with no model/memory.

## Reward, action, and training specification

The environment configuration is exactly `m13.1-cycle-reward-r1`: M10's time,
invalid-action, terminal, and recovery rewards plus `+0.25` only after each
authoritative persistent feed/play/rest counter increments. The M13.1
eight-macro compiler and fixed action durations are unchanged. A reward
component is never exposed as an inference feature.

Training uses exactly 12,000 episodes and the 2600–2639 / four-condition
round-robin schedule used by M13.1: each seed/condition pair occurs 75 times.
The base PCG64 stream is `20260807`, with `+1`, `+2`, and `+3` for no-drive,
no-memory, and random. Epsilon-greedy exploration is 1.00 linearly to 0.05
over the first 9,000 episodes then 0.05. The replay buffer has capacity
100,000 FIFO transitions; updates begin after 1,000 transitions, sample 128
uniformly without replacement, occur once every 16 environment decisions,
and use Double-DQN targets with `gamma=0.99`. Online parameters use Adam
(`lr=3e-4`, `beta1=.9`, `beta2=.999`, `epsilon=1e-8`) and mean Huber loss
(`delta=1.0`); target parameters copy the online network every 1,000 updates.
Weights use PCG64 He-normal initialization from the policy stream. No early
stopping, validation selection, reward update, or architecture update is
allowed.

## Gates, replay, and audit

M13.2 uses M13.1's unchanged condition gates: for each 20-seed condition,
at least 18/20 survival and maintenance completion, mean safe-drive fraction
at least .85, and at least 18/20 stale-pickup→later-consume recoveries in the
two relocation conditions. Matched 80-episode full-policy maintenance must
exceed each ablation by .10; full-policy survival must exceed random by .20.

Every validation/audit trace includes reset/config/protocol/model hashes, the
30-vector, memory before/after, macro, compiled action, scalar reward,
outcome, resource/cycle fields, and terminal flags. Generic replay must match
the environment exactly; strict replay must regenerate every full/no-drive/
no-memory macro/action from the recorded permitted observation/memory and the
frozen weights. Model parameters, optimizer-free final inference weights,
network/feature specification, source revision, training schedule, and every
trace receive content hashes. The audit opens once only after a passing
validation and never reruns ablations; any validation failure leaves it sealed.

## Acceptance and limits

M13.2 passes only when validation and its one-shot audit pass with complete
replay evidence. A pass supports only learned state-oracle macro maintenance
under the declared M10 mechanics. It does not establish learned vision, raw
motor control, physical manipulation, or broad embodied competence.
