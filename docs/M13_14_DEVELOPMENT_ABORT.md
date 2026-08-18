# M13.14 development abort: teacher minibatch validation

Date: 2026-08-18

Status: **implementation abort before any learner fit or policy evaluation**.

## Authorized command

After the development coverage preflight passed, the user separately authorized
the exact frozen development stage:

```text
uv run python -m ecosystem_gym experiment m1314-development \
  --manifest artifacts/manifests/m1314-modular-anchored.json \
  --preflight-report artifacts/reports/m1314-development-preflight.json \
  --output artifacts/reports/m1314-development.json
```

Protocol fingerprint:
`5c215736ba2194f502f09600f6ce5f287543deb237f77546088eb7279490cfb8`.

## What happened

1. The exact manifest and sealed passing preflight were verified.
2. The append-only ledger opened `development_preflight`, then
   `development_fit`.
3. The public oracle teacher dataset was collected and serialized successfully.
4. The first learner warm-start sampled a random 32-row teacher minibatch.
5. `TeacherBatch.__post_init__` reconstructed that minibatch as a full
   `TeacherDataset`, whose whole-dataset validator requires all eight macros.
6. The sampled minibatch happened not to contain `CONSUME` or `GO_REST`, so the
   worker raised:

```text
ValueError: teacher dataset must cover every macro; missing ['CONSUME', 'GO_REST']
```

The requirement is valid for the complete teacher dataset but invalid for a
random training minibatch. This is an implementation error, not evidence about
the modular learner, teacher anchoring, shield, or task.

## Preserved evidence

The complete public teacher dataset is valid:

- features: `16000 x 30`
- masks: `16000 x 8`
- labels: `16000`
- macro counts in enum order:
  `[789, 760, 720, 743, 752, 400, 400, 11436]`

Every macro is represented globally. No learned policy artifact or development
report exists.

SHA-256 values:

- manifest: `d61d767268f3b2f788cba97adc689ac891ab5900e8877876d07ae3f1c30811ce`
- preflight report: `ef474ba032615d3c05a16d6f63982fd5168bf291a4ccd9216fa8d9a29e27e20c`
- stage ledger: `6eea6cef2e8e70b4e9575f81df6e63fad2c5c65604e23e9dfb4911010ade7978`
- teacher dataset: `be6093d55b1aed9acee432829b7eded289d20c84a648fa8f8fe700bb5a18eb69`

The ledger contains exactly:

```text
development_preflight -> development_fit
```

`development_check`, confirmation, and audit remain unopened.

## Boundary and required correction

Do not resume M13.14, alter its frozen source under the existing manifest, or
reuse its teacher data as successor training data. A corrected additive
successor must:

1. use a new protocol fingerprint, manifest, ledger, milestone identity, and
   fresh fit/check split family;
2. preserve this abort, the original manifest, ledger, and teacher dataset;
3. separate whole-dataset validation from minibatch validation;
4. require all eight macros only for the complete serialized teacher dataset;
5. require minibatches only to have valid shapes, finite public features,
   legal masks, and mask-legal labels;
6. add regression tests with minibatches that legitimately omit one or more
   macro classes; and
7. require new explicit authorization before any corrected preflight or fit.

