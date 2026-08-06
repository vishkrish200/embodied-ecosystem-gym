# M10 protocol — persistent-maintenance mechanics

## Purpose and bounded claim

M10 tests whether the environment can support repeated maintenance rather than
terminating after one food and one play event. It adds renewable food,
repeatable boredom relief, explicit energy restoration, a fixed survival
horizon, and event-triggered food relocation. M10 scores only a privileged
oracle ceiling and public RGB observability; it does not claim that an RGB
policy can maintain the agent.

The simulator remains a typed, kinematic-skill benchmark. `rest`, resource
renewal, and maintenance counters are inspectable task mechanics, not
contact-rich manipulation or a biological model.

## Frozen task contract

`persistent_maintenance` never terminates because one intermediate need was
met. Food consumption increments `feed_cycles`, hides the consumed item, and
replenishes it after a deterministic cooldown at a new seeded location. Play
increments `play_cycles` only when boredom has returned to the frozen trigger.
`rest` succeeds only near the visible rest target and below the frozen energy
threshold, then increments `rest_cycles`.

The episode ends only through drive depletion or the 160-step horizon. Horizon
success requires survival plus at least three feed cycles, three play cycles,
and two rest cycles. Every step reports the three counters, food availability,
resource events, explicit action outcome, and maintenance completion state.

When `event_relocation_on_first_pickup` is enabled, the first pickup attempt
moves food before the guarded pickup executes. The stale attempt must therefore
return `blocked` in every applicable episode; recovery is complete only after a
later successful consumption. This replaces M9.1's fixed-step trigger.

## Splits and conditions

| Split | Seeds | Role |
| --- | --- | --- |
| Development | 1700–1719 | Future M11/M12 diagnostics and fitting only. |
| Validation | 1800–1819 | Frozen M10 mechanics and observability gate. |
| Sealed audit | 1900–1919 | Reserved for M13; M10 does not score it. |

Validation has four disjoint layouts: persistent reference, renewal plus visual
morphology, event-triggered relocation, and a compound row with relocation and
a visible blocked distractor. Development and audit layouts are frozen in
`ecosystem_gym/m10.py` and share no layout ids with validation.

## Coverage and oracle gate

Before oracle scoring, an offline MuJoCo-segmentation audit requires every
four-sector public scan to expose initial food, replenished food, toy, and rest
targets for all 20 seeds in every row. Relocated food and blocked distractors
must also be visible in every applicable episode. Segmentation is unavailable
to policy or oracle action methods.

The privileged oracle must reach horizon success in 20/20 seeds for every row,
with the required minimum cycle counts and exact replay of one event-relocation
trace. Reports include Wilson intervals, safe-drive time, replenishment counts,
forced relocation failures, recovered relocations, split constants, and a
protocol fingerprint. Failure blocks M11 policy interpretation; it must not be
hidden by an aggregate score.

## Acceptance criteria

- Legacy M0–M9 tasks retain their terminal semantics and integer action values.
- `rest` is a new appended action, succeeds only under its public guarded
  contract, and restores energy without changing satiety or boredom.
- Consumed food is unavailable during cooldown, visibly reappears at a seeded
  new location, and produces a replay-validated `food_replenished` event.
- Event relocation produces a stale blocked pickup in 20/20 applicable
  episodes and never relies on a fixed action index.
- M10 validation rejects custom seeds and never opens the reserved audit.
- M10's report explicitly says that no RGB policy was scored.
