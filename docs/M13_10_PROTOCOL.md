# M13.10 frozen protocol: imitation-initialized RL retention

Status: **frozen before implementation; no M13.10 split has been opened**

M13.9 showed that the existing public representation can execute the
maintenance strategy but that from-scratch Double-DQN does not discover it
reliably. M13.10 changes initialization and the exploration schedule, not the
environment, public boundary, reward, macro surface, or semi-Markov backup.

## Claim and boundaries

The candidate claim is narrow: balanced-public-oracle imitation initialization
allows subsequent RL to retain and generalize a safe maintenance policy more
reliably than matched random initialization under the same RL schedule.

This is not a from-scratch RL or RGB claim. M13.9 probe traces `5220–5227` are
forbidden for fitting, early stopping, selection, or coefficient choice.

Keep byte-for-byte M13.9 mechanics for the 200-decision environment, 30 public
features, `M13Memory`, complementary mask, eight macros, macro compiler,
public-drive-potential reward, duration-bearing replay, `0.99**duration`
Double-DQN backup, and strict replay. The teacher receives only the same public
observation and policy-owned memory. Reset controls, task IDs, `info`, reward
state, private counters, and environment internals may not enter fitting or
inference.

## Fresh one-way splits

- screen fit `5600–5619` and probe `5620–5627`;
- confirmation fit `5700–5739` and evaluation `5800–5819`; and
- audit `5900–5919`.

Use a fresh `m13.10-split-ledger-r1` dependency chain. Verify prerequisite
reports, policies, teacher dataset, sources, hashes, and scored traces before
opening the next partition. Only screen fit may be idempotently resumed; all
evaluation partitions are one-shot.

## Frozen layouts

The following 20 complete tuples have 20 unique geometries, 100 unique points,
no point reuse with M10–M13.9, maximum absolute coordinate `.831`, and complete
public scan coverage on the exact partitions above.

```python
"m1310_screen_northeast": LayoutSpec("m1310_screen_northeast", (-0.095, -0.211), (0.335, 0.189), (0.635, 0.539), (0.565, -0.461), (-0.465, 0.359)),
"m1310_screen_southwest": LayoutSpec("m1310_screen_southwest", (0.145, 0.169), (-0.545, -0.701), (-0.205, -0.321), (-0.475, 0.379), (0.575, -0.481)),
"m1310_screen_northwest": LayoutSpec("m1310_screen_northwest", (0.275, -0.221), (-0.655, 0.149), (-0.315, 0.489), (0.455, 0.399), (-0.495, -0.401)),
"m1310_screen_southeast": LayoutSpec("m1310_screen_southeast", (-0.225, 0.179), (0.205, -0.711), (0.595, -0.371), (-0.505, -0.391), (0.465, 0.389)),
"m1310_probe_northeast": LayoutSpec("m1310_probe_northeast", (-0.391, -0.271), (0.379, 0.299), (0.699, 0.639), (0.649, -0.431), (-0.681, 0.499)),
"m1310_probe_southwest": LayoutSpec("m1310_probe_southwest", (0.349, 0.329), (-0.771, -0.631), (-0.421, -0.251), (-0.691, 0.509), (0.659, -0.441)),
"m1310_probe_northwest": LayoutSpec("m1310_probe_northwest", (0.509, -0.291), (-0.791, 0.179), (-0.441, 0.629), (0.609, 0.519), (-0.651, -0.451)),
"m1310_probe_southeast": LayoutSpec("m1310_probe_southeast", (-0.571, 0.349), (0.309, -0.751), (0.669, -0.391), (-0.661, -0.461), (0.619, 0.529)),
"m1310_confirm_fit_northeast": LayoutSpec("m1310_confirm_fit_northeast", (-0.055, -0.207), (0.315, 0.253), (0.615, 0.613), (0.555, -0.457), (-0.435, 0.433)),
"m1310_confirm_fit_southwest": LayoutSpec("m1310_confirm_fit_southwest", (0.125, 0.253), (-0.555, -0.687), (-0.215, -0.307), (-0.445, 0.453), (0.565, -0.467)),
"m1310_confirm_fit_northwest": LayoutSpec("m1310_confirm_fit_northwest", (0.305, -0.207), (-0.675, 0.213), (-0.335, 0.563), (0.485, 0.473), (-0.505, -0.387)),
"m1310_confirm_fit_southeast": LayoutSpec("m1310_confirm_fit_southeast", (-0.235, 0.253), (0.245, -0.697), (0.625, -0.347), (-0.515, -0.377), (0.495, 0.463)),
"m1310_confirm_eval_northeast": LayoutSpec("m1310_confirm_eval_northeast", (-0.461, -0.371), (0.399, 0.299), (0.729, 0.639), (0.679, -0.521), (-0.711, 0.499)),
"m1310_confirm_eval_southwest": LayoutSpec("m1310_confirm_eval_southwest", (0.419, 0.329), (-0.801, -0.721), (-0.451, -0.341), (-0.721, 0.509), (0.689, -0.531)),
"m1310_confirm_eval_northwest": LayoutSpec("m1310_confirm_eval_northwest", (0.559, -0.381), (-0.821, 0.119), (-0.471, 0.619), (0.639, 0.519), (-0.681, -0.541)),
"m1310_confirm_eval_southeast": LayoutSpec("m1310_confirm_eval_southeast", (-0.621, 0.339), (0.339, -0.831), (0.699, -0.471), (-0.691, -0.551), (0.649, 0.529)),
"m1310_audit_northeast": LayoutSpec("m1310_audit_northeast", (-0.405, -0.335), (0.455, 0.335), (0.785, 0.675), (0.735, -0.485), (-0.655, 0.535)),
"m1310_audit_southwest": LayoutSpec("m1310_audit_southwest", (0.475, 0.365), (-0.745, -0.685), (-0.395, -0.305), (-0.665, 0.545), (0.745, -0.495)),
"m1310_audit_northwest": LayoutSpec("m1310_audit_northwest", (0.615, -0.345), (-0.763, 0.157), (-0.415, 0.655), (0.695, 0.555), (-0.625, -0.505)),
"m1310_audit_southeast": LayoutSpec("m1310_audit_southeast", (-0.565, 0.375), (0.395, -0.795), (0.755, -0.435), (-0.635, -0.515), (0.705, 0.565)),
```

## Teacher dataset and supervised fit

Run coverage and the balanced oracle once over every fit cell, strictly replay
each trace, then serialize only public features, eligible-action masks, and
oracle macro labels.

- screen dataset: four conditions × 20 seeds × 200 decisions = 16,000 rows;
- confirmation dataset: four × 40 × 200 = 32,000 rows;
- unchanged `30→64→64→8` ReLU MLP;
- class-balanced masked cross-entropy;
- Adam (`lr=.001`, betas `.9/.999`, epsilon `1e-8`), batch 256, 80 epochs;
- deterministic PCG64 shuffle from the training seed; and
- class weight `N / (8 * class_count)`.

Before probe access, fit accuracy must be at least `.88`, non-WAIT accuracy at
least `.90`, and every macro recall at least `.75`. These are training-integrity
checks, not generalization evidence.

## Three arms and screen fit

Screen training seeds are `20260921` and `20260922`. For each seed create one
base initialization and derive:

1. `imitation_warmstart_rl`: supervised fit followed by RL;
2. `random_init_rl_control`: byte-identical pre-supervision base weights,
   followed by identical RL; and
3. `imitation_only_guardrail`: the same supervised fit as the candidate, with
   no RL update.

Candidate and imitation-only weights must be byte-identical after supervision.
The two RL arms share environment schedule, reward, replay, update/target
cadence, optimizer, and epsilon schedule; only initialization differs.
Before RL, zero Adam moments and the optimizer step for both RL arms, then copy
the online weights into the target network. Use a separate RL PCG64 stream
seeded by `training_seed xor 0x4D31333130` so supervised shuffling cannot alter
the candidate's exploration stream.

Each RL arm receives 2,000 episodes, 25 visits per fit cell. Epsilon is `.20`
at episode 0 and decreases linearly to `.05` at episode 1,999. Use the exact
M13.9 potential reward and duration-aware backup. Run all six seed/arm jobs in
one spawned six-process wave with one numerical-library thread per worker.
Serialize and reload all six policies before opening the probe.

## Screen gate

Score the six policies once over four conditions × eight probe seeds, plus
matched mask-aware random, with strict replay.

For every candidate condition require survival `>=7/8`, maintenance `>=6/8`,
decision/duration safety `>=.80`, required recovery `>=7/8`, unsafe-WAIT
`<=.15`, and zero integrity violations. Its imitation-only guardrail must pass
the same gates.

Across 32 episodes, each candidate must:

- exceed matched random survival by `.20`;
- exceed random-init RL by `.15` full-objective success, `.05` decision-safe,
  `.05` duration-safe, and `.10` lower unsafe-WAIT; and
- remain within `.05` of imitation-only full-objective and safety, with
  unsafe-WAIT no more than `.05` worse.

Both replicas must pass, and at least one must exceed random-init RL by `.25`
full-objective success. Rejection stops M13.10 and leaves 5700–5919 sealed.

## Confirmation and audit

Only promotion authorizes confirmation. Use training seeds
`20260923–20260930`, rebuild the teacher dataset only on confirmation-fit, and
fit all arms from scratch. Each RL arm receives 12,000 episodes; epsilon decays
`.20→.05` over the first 2,000 and remains `.05` thereafter.

On evaluation every candidate replica must pass each condition at `18/20`
survival, maintenance, and required recovery, decision/duration safety `>=.85`,
unsafe-WAIT `<=.10`, and zero integrity violations. Imitation-only must pass the
same hard gate. Candidate normalized gate margin must beat matched random-init
RL in at least 7/8 pairs and remain within `.03` of imitation-only in at least
7/8. Only both paired rules plus the 8/8 hard gate support the narrow claim and
authorize audit.

Audit freezes all sources, datasets, policies, reports, and the ledger, opens
`5900–5919` once, scores without retraining, and applies identical gates. No
retry, replacement seed, epoch change, exploration change, or probe-driven
adjustment is permitted.
