# M13.9 implementation and fit-only smoke

Status: **implementation complete; screen rejected; confirmation not launched**

The frozen `m13.9-public-drive-potential-r1` protocol now has additive code,
layouts, tests, and CLI entry points. M13.8 remains unchanged as a negative
result. Its confirmation/audit ranges `4900–5119` were not accessed or
repurposed.

## Implemented contract

- Twenty fresh layouts cover screen fit/probe, confirmation fit/evaluation,
  and audit at seeds `5200–5519` exactly as frozen.
- `public_potential_candidate`, `static_cost_control`, and
  `legacy_guardrail` start byte-identically within each training seed. Every
  learned arm stores the compiler duration in replay and uses
  `0.99**duration` in its Double-DQN bootstrap.
- The candidate uses
  `0.99**duration * Phi(after) - Phi(before)` over public drives, the static
  control preserves the exact M13.8 bounded reward, and the legacy guardrail
  preserves the inherited uncapped scalar.
- Preflight includes coverage, the balanced public oracle, feed/play/rest
  specialists, quota-then-wait, potential signs and telescoping, reward
  recomposition, and strict replay.
- The screen submits exactly six spawned jobs in one wave with every numerical
  thread cap set to one. Confirmation fits only candidate/static pairs. Probe,
  confirmation evaluation, and audit each have explicit verify-before-open and
  coverage/ceiling-before-score boundaries.
- The durable protocol ledger hashes each ordered split marker and keeps all
  held-out partitions one-shot. The fit-only smoke uses a separate ledger
  inside its non-protocol artifact directory, so the canonical protocol ledger
  remains pristine until the exact screen starts.
- Screen and confirmation authorization re-hash every scored trace, strictly
  replay it, and re-derive compact gate evidence before any later split opens.
  Audit also revalidates the originating promoted screen report.

## CLI

```bash
# Fixed non-protocol plumbing check. No budget, seed, layout, worker, or probe flags.
uv run python -m ecosystem_gym experiment m139-fit-smoke --output PATH.json

# Full frozen protocol stages; do not run without explicit authorization.
uv run python -m ecosystem_gym experiment m139-screen --output PATH.json
uv run python -m ecosystem_gym experiment m139-confirm \
  --screen-report SCREEN.json --output PATH.json
uv run python -m ecosystem_gym experiment m139-audit \
  --confirmation-report CONFIRMATION.json --output PATH.json
```

## Non-protocol smoke evidence

The one permitted fit-only smoke wrote
`artifacts/m139-smoke-non-protocol/fit-only.json` and a separate sibling
artifact directory. It is explicitly labeled
`NON-PROTOCOL / NON-PROMOTIONAL FIT-ONLY SMOKE`; its policy files are marked
ineligible for protocol use.

- Internal elapsed time: `1.5284552090015495` seconds; external wall time:
  `1.75` seconds.
- Training budget: six fits x 16 episodes = `96` screen-fit episodes.
- Evaluation plumbing: four conditions x six reloaded policies = `24`
  screen-fit episodes, all strictly replayed.
- CPU setup: six spawned processes, one numerical-library thread per worker.
- Potential diagnostics: 130 telescoping cases, maximum residual
  `2.220446049250313e-16`; sign checks passed.
- Split access: only `screen_fit` was marker-hashed in the smoke-only
  non-protocol ledger. The canonical M13.9 ledger was not accessed and has no
  opened partitions. `screen_probe`, `confirmation_fit`,
  `confirmation_evaluation`, and `audit` all remained closed.
- This result validates execution/reward/replay plumbing only. It makes no
  score, promotion, or policy-quality claim.

The subsequently authorized full screen rejected both replicas under the
frozen all-replicas rule. See [`M13_9_SCREEN_RESULTS.md`](M13_9_SCREEN_RESULTS.md).
Confirmation and audit remain sealed; do not rerun or tune against the probe.
