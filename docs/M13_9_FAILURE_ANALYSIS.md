# M13.9 post-screen failure analysis

Status: **read-only diagnosis complete; no new evaluation split opened**

This analysis uses only the frozen M13.9 screen report and its already-open
screen-fit/probe traces. It does not change the M13.9 result, fit a successor on
probe data, or authorize confirmation.

## What the failed policies did

Unsafe `WAIT` was usually a learned long-horizon preference rather than a
directly rewarded action:

| Candidate seed | Unsafe WAIT decisions | After `3/3/2` quota | Median first unsafe WAIT | Median WAIT Q-margin over best available alternative |
| --- | ---: | ---: | ---: | ---: |
| 20260911 | 1,739 | 63.8% | decision 4 | +0.086 |
| 20260912 | 1,196 | 90.3% | decision 32 | +0.117 |

The immediate candidate reward on unsafe waits was negative on average
(`-0.062` and `-0.066`), including negative potential terms (`-0.050` and
`-0.057`). Available non-WAIT alternatives included consume, rest, play, and
navigation actions. Nevertheless, the learned Q function ranked `WAIT` above
the best eligible alternative, often for long consecutive runs. This is a
value-estimation/optimization failure, not positive immediate reward farming or
a missing-action-mask problem.

Seed `20260911` had a second failure mode in compound: none of its 317 unsafe
waits occurred after completing `3/3/2`. It averaged 17.6 feeds but only 2.1
plays, so it over-specialized before completing the quota. The other candidate
cells were predominantly post-quota loitering.

## Training-regime mismatch

M13.9 inherited the 9,000-episode epsilon decay while each screen fit lasted
only 2,000 episodes. Exploration therefore started at 100%, ended at 78.9%, and
averaged 89.4% by episode. The final greedy network was trained mostly from
random exploratory actions. This schedule was shared by every arm, so it does
not invalidate the frozen comparison, but it helps explain unstable greedy
policies and why changing the reward alone did not solve the behavior.

## Can the public network represent the strategy?

A non-promotional supervised diagnostic trained the unchanged 30→64→64→8
network on the 16,000 balanced-oracle decisions already present in screen-fit
preflight traces. No probe action was used for fitting.

Across five fixed classifier initializations, action accuracy on the already
opened 6,400 probe-oracle decisions was 88.5–91.4%; non-`WAIT` accuracy was
93.6–97.4%. An always-`WAIT` baseline was only 70.8% accurate.

The same five fitted classifiers were then rolled out only on the 80 screen-fit
cells. Every classifier achieved:

- 80/80 survival;
- 80/80 maintenance completion;
- 80/80 full-objective success;
- 99.6–100% decision-safe and 99.3–100% duration-safe occupancy; and
- 0–0.08% unsafe-WAIT fraction.

These are development-data diagnostics, not generalization results. They do
show that the frozen public features, policy memory, macro surface, and compact
network can express and execute the required maintenance strategy without
private inputs.

## Conclusion and successor constraint

M13.9 failed because from-scratch off-policy learning did not reliably discover
or retain a good policy under its screen regime. The next experiment should not
change M13.9 coefficients or add a drive-based action shield. It should freeze
fresh splits and compare:

1. balanced-oracle imitation initialization followed by RL fine-tuning;
2. matched random initialization with the identical RL schedule; and
3. imitation-only retention as a guardrail.

All three arms must retain the same 30 public features, policy-owned memory,
mask, compiler, semi-Markov backup, and M13.9 reward. A screen-aligned low
exploration schedule must be common to both RL arms so the causal comparison is
imitation initialization, not reward or environment changes. No M13.9 probe
trace may enter successor fitting or model selection.
