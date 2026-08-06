# M9.1 protocol — distractor choice and forced recovery

M9 completion passed while its required blocked-recovery gate failed, because
safe avoidance of a blocked distractor was counted as missing recovery. M9.1
freezes the M9 policy and separates those two behaviours on fresh layouts and
seeds; it does not retrain or score M9's sealed audit.

`distractor_choice` asks whether the policy can complete food and play with a
visible food-like blocked distractor while avoiding invalid pickup. Success is
at least 15/20 completed episodes and no more than 5/20 episodes with a
blocked pickup. It does not require the agent to make a deliberately bad
choice merely to demonstrate recovery.

`forced_recovery` moves the food immediately after the policy's first
target-directed approach. A stale pickup must fail, then the policy must make
another four-sector public scan and complete food and play. Success requires
at least 15/20 task completions and 15/20 episodes with the full
failure-rescan-completion chain. Both rows use seeds 1600–1619, complete
offline public-visibility coverage checks, and an explicitly separate
state-oracle ceiling.

The initial frozen-policy result is mixed: distractor choice passes 20/20 with
zero invalid pickups, while forced recovery reaches 13/20 completion and
11/20 full recovery chains despite 20/20 post-relocation rescans. That is a
real sequential recovery limit, so M9.1 stops before retraining or opening a
sealed audit.
