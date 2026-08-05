"""M8.5 sealed external audit of the frozen M8.4 RGB grounder.

The audit defines fresh scan_v2 layouts, seeds, and compound conditions.  It
may inspect target visibility with offline segmentation before a rollout, but
it never supplies those labels, reset metadata, or simulator coordinates to
the learned grounder or public recovery shell.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .env import EcosystemEnv
from .m8 import M8_TASK_ID
from .m84 import (
    M84_PROTOCOL_VERSION,
    LearnedHeatmapRecoveryPolicy,
    _aggregate,
    _run_episode,
    _run_oracle,
    fit_m84_grounder,
    m84_config,
    m84_policy_fingerprint,
    m84_protocol_fingerprint,
    m84_scan_coverage,
)
from .trajectory import ReplayResult
from .viewer import ViewerSession


M85_PROTOCOL_VERSION = "m85-sealed-scan-v2-external-audit-v1"
M85_TEST_SEEDS = tuple(range(600, 620))
M85_MIN_SUCCESS_RATE = 0.75

# This post-result suite is disjoint from M8.4 train/validation layouts and
# seeds. Its fields were defined before obtaining any M8.5 policy outcomes.
M85_CONDITIONS: dict[str, dict[str, Any]] = {
    "northeast_blue_grippy": {
        "layout_id": "m85_northeast",
        "food_variant": "blue",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "dynamics_variant": "grippy",
    },
    "southwest_orange_landmark": {
        "layout_id": "m85_southwest",
        "food_variant": "orange",
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "geometry_variant": "m81_landmark",
    },
    "northwest_purple_blocked": {
        "layout_id": "m85_northwest",
        "food_variant": "purple",
        "lighting_variant": "dim",
        "camera_control": "scan_v2",
        "initial_scan_sector": "south",
        "blocked_distractor": True,
        "distractor_xy": [0.20, -0.10],
    },
    "southeast_red_relocation_slippery": {
        "layout_id": "m85_southeast",
        "food_variant": "red",
        "camera_control": "scan_v2",
        "initial_scan_sector": "north",
        "dynamics_variant": "slippery",
        "disturbance_step": 1,
    },
}

# Filled from the deterministic M8.4 trainer after this audit protocol was
# written.  Any M8.4 training or fitted-weight change rejects this audit.
M84_FROZEN_PROTOCOL_FINGERPRINT = "00d11d3b65a9707d1cff61796210d67c58fcf0281325746fc5d87e86eda606b6"
M84_FROZEN_POLICY_FINGERPRINT = "e09ddbddcc8eef3c1057d4a25d95c9416d96565a3fbb8e8954ac9e539b85e7e3"
M85_FROZEN_PROTOCOL_FINGERPRINT = "a9a3940e013fca5cd1aa49673c419b6dfe5352e6718c71be52875835e959e0b6"


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def m85_protocol_fingerprint() -> str:
    payload = {
        "protocol_version": M85_PROTOCOL_VERSION,
        "test_seeds": M85_TEST_SEEDS,
        "conditions": M85_CONDITIONS,
        "minimum_success_rate": M85_MIN_SUCCESS_RATE,
        "m84_protocol_version": M84_PROTOCOL_VERSION,
        "m84_protocol_fingerprint": M84_FROZEN_PROTOCOL_FINGERPRINT,
        "m84_policy_fingerprint": M84_FROZEN_POLICY_FINGERPRINT,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_coverage() -> dict[str, dict[str, object]]:
    coverage = m84_scan_coverage(M85_CONDITIONS, seeds=M85_TEST_SEEDS)
    failed = [condition for condition, result in coverage.items() if not result["passes"]]
    if failed:
        raise RuntimeError(f"M8.5 has unobservable scan_v2 targets: {', '.join(failed)}")
    return coverage


def m85_benchmark(*, test_seeds: tuple[int, ...] = M85_TEST_SEEDS) -> dict[str, object]:
    """Run the one-time external audit; custom seeds are forbidden."""

    if test_seeds != M85_TEST_SEEDS:
        raise ValueError("M8.5 test seeds are frozen; use M85_TEST_SEEDS")
    coverage = _require_coverage()
    grounder = fit_m84_grounder()
    observed_m84_protocol = m84_protocol_fingerprint()
    observed_m84_policy = m84_policy_fingerprint(grounder)
    observed_m85_protocol = m85_protocol_fingerprint()
    if observed_m84_protocol != M84_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.4 training protocol drifted; M8.5 must not evaluate an edited trainer")
    if observed_m84_policy != M84_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.4 fitted grounder drifted; M8.5 must not evaluate changed weights")
    if observed_m85_protocol != M85_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.5 protocol drifted; this audit must remain sealed")

    policy = LearnedHeatmapRecoveryPolicy(grounder)
    results: dict[str, dict[str, object]] = {}
    for condition, controls in M85_CONDITIONS.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        results[condition] = {
            "frozen_m84_rgb_heatmap": _aggregate(learned),
            "state_oracle_ceiling": _aggregate(oracle),
        }

    condition_rates = {condition: result["frozen_m84_rgb_heatmap"]["task_success_rate"] for condition, result in results.items()}
    successes = sum(result["frozen_m84_rgb_heatmap"]["successes"] for result in results.values())
    episodes = len(results) * len(test_seeds)
    oracle_successes = sum(result["state_oracle_ceiling"]["successes"] for result in results.values())
    passes = (
        oracle_successes == episodes
        and successes / episodes >= M85_MIN_SUCCESS_RATE
        and all(rate >= M85_MIN_SUCCESS_RATE for rate in condition_rates.values())
    )
    return {
        "schema_version": "0.85",
        "protocol_version": M85_PROTOCOL_VERSION,
        "test_seeds": list(test_seeds),
        "conditions": M85_CONDITIONS,
        "scan_coverage": coverage,
        "m84_freeze": {
            "m84_protocol_version": M84_PROTOCOL_VERSION,
            "frozen_protocol_fingerprint": M84_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_protocol_fingerprint": observed_m84_protocol,
            "frozen_policy_fingerprint": M84_FROZEN_POLICY_FINGERPRINT,
            "observed_policy_fingerprint": observed_m84_policy,
            "m85_training_episodes": 0,
        },
        "m85_protocol": {
            "frozen_fingerprint": M85_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_fingerprint": observed_m85_protocol,
        },
        "policy_boundary": ["rgb", "holding_food", "prior_outcome", "policy_owned_memory"],
        "results": results,
        "external_validity_gate": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "condition_success_rates": condition_rates,
            "oracle_successes": oracle_successes,
            "predeclared_minimum_success_rate": M85_MIN_SUCCESS_RATE,
            "passes": passes,
            "next_step": "report this sealed audit; do not tune M8.4 on M8.5" if passes else "stop: record failed external audit without tuning M8.4",
        },
        "limits": [
            "M8.5 evaluates a deterministic frozen M8.4 grounder. It does not train, select, or tune a policy on its held-out episodes.",
            "The scan-coverage audit uses MuJoCo segmentation offline only; the deployed policy receives RGB, holding state, prior outcome, and its own memory.",
            "The state oracle is a privileged ceiling, not an equal-input baseline.",
            "This tests transfer across the declared layouts and condition combinations, not end-to-end learned control, real-world sensing, or contact-rich physics.",
        ],
    }


def write_m85_report(path: str | Path) -> dict[str, object]:
    report = m85_benchmark()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m85_viewer_demo(trace_path: str | Path, *, seed: int = 607) -> ReplayResult:
    """Record and replay one sealed M8.5 relocation rollout."""

    coverage = _require_coverage()
    if not all(result["passes"] for result in coverage.values()):
        raise AssertionError("M8.5 demo requires a visible-target protocol")
    if m85_protocol_fingerprint() != M85_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.5 protocol drifted")
    grounder = fit_m84_grounder()
    if m84_protocol_fingerprint() != M84_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.4 training protocol drifted")
    if m84_policy_fingerprint(grounder) != M84_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.4 fitted grounder drifted")
    policy = LearnedHeatmapRecoveryPolicy(grounder)
    controls = M85_CONDITIONS["southeast_red_relocation_slippery"]
    with ViewerSession(EcosystemEnv(m84_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m85-frozen-heatmap") as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
