# M13.11-r2 development protocol: corrected masked semi-Markov PPO

Status: **development-only, non-promotional; frozen before execution**.

M13.11-r2 is an additive correctness rerun of M13.11.  The original
M13.11 code, protocol, report, ledger, policies, traces, and its 6000--6039
partitions remain immutable implementation-failure evidence, not a PPO result.

## Frozen correction and unchanged boundary

The only PPO changes are: (1) the actor loss derivative is
`+advantage * ratio * (p - one_hot(action))` in the active PPO region, and
(2) terminal rows are retained in a multi-episode 2,048-transition rollout;
their `done` flag resets GAE continuation, rather than flushing the batch.

The 30 public features, eight macros, explicit memory, complementary mask,
M13.9 potential reward, macro durations, environment, reward recomposition,
PPO topology and hyperparameters, four paired training seeds, DQN control, and
400,000-decision target are unchanged.  No labels, warm start, oracle loss,
recurrence, reward change, environment change, or tuning is permitted.

## Fresh partitions and access

| Partition | Seeds | Layouts | Access |
| --- | --- | --- | --- |
| `development_fit` | 6040--6059 | `m1311r2_fit_*` | training and capped fit smoke |
| `development_check` | 6060--6079 | `m1311r2_check_*` | one post-fit development evaluation |

The r2 ledger is separate and ordered `fit -> check`; it fingerprints this
protocol, sources, layouts, and seeds.  6020--6039 and all prior probe/sealed
partitions are not referenced.  6080--6099 remain unused.  A future screen
would need a separately frozen 6100+ split and explicit authorization.

## Training, diagnostics, and gates

Each of the four paired seeds `20261311`--`20261314` runs the corrected PPO and
unchanged random-init duration-aware Double-DQN control.  All eight jobs are
submitted before awaiting results through spawned workers, each numeric thread
environment variable is set to one, and jobs finish a started episode after
the 400,000-decision target.  PPO updates at every full 2,048-row rollout and
once for a final residual, using four epochs each.

Every PPO update records entropy, approximate KL (`old_logp - new_logp`), clip
fraction, action histogram, actor/critic/total loss, and pre-clipping gradient
norm. Reports record PIDs, thread settings, decisions/overshoot, episodes,
update batches/epochs, artifact hashes, ledger state, coverage, and strict
serialized-policy replay counts.

Before training, finite-difference actor/entropy/critic/trunk gradients,
probability direction, clipping, masks, terminal-boundary GAE, rollout
batching, artifact rejection, fresh layouts/ledger, CLI, deterministic
artifacts/replay, prior milestones, the full suite, and `git diff --check`
must pass. The smoke uses only original M13.11 fit layouts and seeds 6000--6019
with no canonical r2-ledger access and at most 32 episodes per learned job. It is followed by a
scoped implementation commit before the one canonical r2 comparison.

No screen, confirmation, audit, M14, or M15 run is authorized. A clean PPO
negative stops the line of work; a pass only proposes a separate screen.
