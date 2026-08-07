"""M8.6 development-only diagnosis of the frozen M8.4 transfer failure.

This module never reads M8.5 episode outcomes or reuses its layouts or seeds.
It changes blue appearance and grippy dynamics independently on a new spatial
relation, then compares blue dynamics on a seen development layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..env import EcosystemEnv
from .m8 import M8_TASK_ID
from .m84 import (
    LearnedHeatmapRecoveryPolicy,
    _aggregate,
    _grounding_metrics,
    _run_episode,
    _run_oracle,
    fit_m84_grounder,
    m84_config,
    m84_policy_fingerprint,
    m84_protocol_fingerprint,
    m84_scan_coverage,
)
from .m85 import M84_FROZEN_POLICY_FINGERPRINT, M84_FROZEN_PROTOCOL_FINGERPRINT, M85_TEST_SEEDS
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


M86_PROTOCOL_VERSION = "m86-development-factorial-diagnosis-v1"
M86_DIAGNOSTIC_SEEDS = tuple(range(700, 720))

# `m84_train_southeast` is a seen-layout control. `m86_northeast` is a new
# development layout and does not share M8.5's held-out coordinates.
M86_CONDITIONS: dict[str, dict[str, Any]] = {
    "seen_blue_nominal": {
        "layout_id": "m84_train_southeast",
        "food_variant": "blue",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "seen_blue_grippy": {
        "layout_id": "m84_train_southeast",
        "food_variant": "blue",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "dynamics_variant": "grippy",
    },
    "new_northeast_red_nominal": {
        "layout_id": "m86_northeast",
        "food_variant": "red",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "new_northeast_blue_nominal": {
        "layout_id": "m86_northeast",
        "food_variant": "blue",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "new_northeast_red_grippy": {
        "layout_id": "m86_northeast",
        "food_variant": "red",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "dynamics_variant": "grippy",
    },
    "new_northeast_blue_grippy": {
        "layout_id": "m86_northeast",
        "food_variant": "blue",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "dynamics_variant": "grippy",
    },
}


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def _require_coverage() -> dict[str, dict[str, object]]:
    coverage = m84_scan_coverage(M86_CONDITIONS, seeds=M86_DIAGNOSTIC_SEEDS)
    failed = [condition for condition, result in coverage.items() if not result["passes"]]
    if failed:
        raise RuntimeError(f"M8.6 has unobservable scan_v2 targets: {', '.join(failed)}")
    return coverage


def m86_diagnostics(*, seeds: tuple[int, ...] = M86_DIAGNOSTIC_SEEDS) -> dict[str, object]:
    """Diagnose frozen M8.4 behavior without changing a single learned weight."""

    if seeds != M86_DIAGNOSTIC_SEEDS:
        raise ValueError("M8.6 diagnostic seeds are frozen; use M86_DIAGNOSTIC_SEEDS")
    coverage = _require_coverage()
    grounder = fit_m84_grounder()
    observed_protocol = m84_protocol_fingerprint()
    observed_policy = m84_policy_fingerprint(grounder)
    if observed_protocol != M84_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.4 training protocol drifted")
    if observed_policy != M84_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.4 fitted grounder drifted")

    policy = LearnedHeatmapRecoveryPolicy(grounder)
    results: dict[str, dict[str, object]] = {}
    grounding: dict[str, dict[str, object]] = {}
    for condition, controls in M86_CONDITIONS.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in seeds]
        results[condition] = {
            "frozen_m84_rgb_heatmap": _aggregate(learned),
            "state_oracle_ceiling": _aggregate(oracle),
        }
        grounding[condition] = _grounding_metrics(grounder, controls, seeds=seeds)

    return {
        "schema_version": "0.86",
        "protocol_version": M86_PROTOCOL_VERSION,
        "diagnostic_seeds": list(seeds),
        "conditions": M86_CONDITIONS,
        "scan_coverage": coverage,
        "m84_freeze": {
            "frozen_protocol_fingerprint": M84_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_protocol_fingerprint": observed_protocol,
            "frozen_policy_fingerprint": M84_FROZEN_POLICY_FINGERPRINT,
            "observed_policy_fingerprint": observed_policy,
            "m86_training_episodes": 0,
        },
        "m85_exclusion": {
            "test_seeds": list(M85_TEST_SEEDS),
            "reused_m85_seeds": 0,
            "reused_m85_layouts": 0,
        },
        "policy_boundary": ["rgb", "holding_food", "prior_outcome", "policy_owned_memory"],
        "results": results,
        "grounding_metrics": grounding,
        "limits": [
            "M8.6 is a diagnosis of the frozen M8.4 grounder, not a new policy result or a pass gate.",
            "The new northeast rows vary blue appearance and grippy dynamics factorially, but they do not isolate every spatial-calibration factor present in M8.5.",
            "MuJoCo segmentation is used only for offline coverage and grounding measurements, never by the deployed policy.",
            "M8.5 remains sealed and none of its layouts, seeds, or episode outcomes are used here.",
        ],
    }


def write_m86_report(path: str | Path) -> dict[str, object]:
    report = m86_diagnostics()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m86_viewer_demo(trace_path: str | Path, *, seed: int = 707) -> ReplayResult:
    """Record/replay the predeclared new-layout blue/grippy diagnostic."""

    _require_coverage()
    grounder = fit_m84_grounder()
    if m84_protocol_fingerprint() != M84_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.4 training protocol drifted")
    if m84_policy_fingerprint(grounder) != M84_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.4 fitted grounder drifted")
    policy = LearnedHeatmapRecoveryPolicy(grounder)
    with ViewerSession(
        EcosystemEnv(m84_config(), render_mode="rgb_array"),
        trace_path=trace_path,
        episode_id="m86-blue-grippy-diagnosis",
    ) as session:
        observation, _ = session.reset(seed=seed, options=_options(M86_CONDITIONS["new_northeast_blue_grippy"]))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
