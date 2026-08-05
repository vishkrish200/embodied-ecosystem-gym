"""M8.9 sealed visual-morphology audit of the frozen M8.7 RGB grounder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .env import EcosystemEnv
from .m8 import M8_TASK_ID
from .m84 import LearnedHeatmapRecoveryPolicy, _aggregate, _run_episode, _run_oracle, m84_config, m84_scan_coverage
from .m87 import M87_PROTOCOL_VERSION, fit_m87_grounder, m87_policy_fingerprint, m87_protocol_fingerprint
from .trajectory import ReplayResult
from .viewer import ViewerSession


M89_PROTOCOL_VERSION = "m89-sealed-visual-morphology-audit-v1"
M89_TEST_SEEDS = tuple(range(1_200, 1_220))
M89_MIN_SUCCESS_RATE = 0.75

# Defined before running M8.9. These are visual-only changes: the original
# sphere collision geoms remain active, while each alternative geom is
# non-colliding. Layouts and seeds are disjoint from all M8.7 and M8.8 suites.
M89_CONDITIONS: dict[str, dict[str, Any]] = {
    "northeast_blue_food_capsule": {
        "layout_id": "m89_northeast",
        "food_variant": "blue",
        "food_shape_variant": "capsule",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "southwest_red_food_box_relocation": {
        "layout_id": "m89_southwest",
        "food_variant": "red",
        "food_shape_variant": "box",
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "disturbance_step": 1,
    },
    "northwest_purple_agent_capsule": {
        "layout_id": "m89_northwest",
        "food_variant": "purple",
        "agent_shape_variant": "capsule",
        "camera_control": "scan_v2",
        "initial_scan_sector": "south",
        "geometry_variant": "m81_landmark",
    },
    "southeast_blue_agent_box_blocked": {
        "layout_id": "m89_southeast",
        "food_variant": "blue",
        "agent_shape_variant": "box",
        "camera_control": "scan_v2",
        "initial_scan_sector": "north",
        "blocked_distractor": True,
        "distractor_xy": [0.10, -0.04],
    },
}

M87_FROZEN_PROTOCOL_FINGERPRINT = "183fbc840103443bc0257926a05eab9f1427baf8cf9966504cd3ea8da0b284ec"
M87_FROZEN_POLICY_FINGERPRINT = "f0bce81d4767beb9b3279f66577f6f2da4a9a16827c1886113f43c83ed36a551"
M89_FROZEN_PROTOCOL_FINGERPRINT = "cb8bed83bafc6655753c8d80a1e1737cd76691c943cc2c206e47a8419ce44dce"


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def m89_protocol_fingerprint() -> str:
    payload = {
        "protocol_version": M89_PROTOCOL_VERSION,
        "test_seeds": M89_TEST_SEEDS,
        "conditions": M89_CONDITIONS,
        "minimum_success_rate": M89_MIN_SUCCESS_RATE,
        "m87_protocol_version": M87_PROTOCOL_VERSION,
        "m87_protocol_fingerprint": M87_FROZEN_PROTOCOL_FINGERPRINT,
        "m87_policy_fingerprint": M87_FROZEN_POLICY_FINGERPRINT,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_coverage() -> dict[str, dict[str, object]]:
    coverage = m84_scan_coverage(M89_CONDITIONS, seeds=M89_TEST_SEEDS)
    failed = [condition for condition, result in coverage.items() if not result["passes"]]
    if failed:
        raise RuntimeError(f"M8.9 has unobservable scan_v2 targets: {', '.join(failed)}")
    return coverage


def m89_benchmark(*, test_seeds: tuple[int, ...] = M89_TEST_SEEDS) -> dict[str, object]:
    if test_seeds != M89_TEST_SEEDS:
        raise ValueError("M8.9 test seeds are frozen; use M89_TEST_SEEDS")
    coverage = _require_coverage()
    grounder = fit_m87_grounder()
    observed_m87_protocol = m87_protocol_fingerprint()
    observed_m87_policy = m87_policy_fingerprint(grounder)
    observed_m89_protocol = m89_protocol_fingerprint()
    if observed_m87_protocol != M87_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.7 training protocol drifted; M8.9 must not evaluate an edited trainer")
    if observed_m87_policy != M87_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.7 fitted grounder drifted; M8.9 must not evaluate changed weights")
    if observed_m89_protocol != M89_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.9 protocol drifted; this audit must remain sealed")

    policy = LearnedHeatmapRecoveryPolicy(grounder)  # type: ignore[arg-type]
    results: dict[str, dict[str, object]] = {}
    for condition, controls in M89_CONDITIONS.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        results[condition] = {"frozen_m87_rgb_grounder": _aggregate(learned), "state_oracle_ceiling": _aggregate(oracle)}
    condition_rates = {condition: result["frozen_m87_rgb_grounder"]["task_success_rate"] for condition, result in results.items()}
    successes = sum(result["frozen_m87_rgb_grounder"]["successes"] for result in results.values())
    episodes = len(results) * len(test_seeds)
    oracle_successes = sum(result["state_oracle_ceiling"]["successes"] for result in results.values())
    passes = oracle_successes == episodes and successes / episodes >= M89_MIN_SUCCESS_RATE and all(
        rate >= M89_MIN_SUCCESS_RATE for rate in condition_rates.values()
    )
    return {
        "schema_version": "0.89",
        "protocol_version": M89_PROTOCOL_VERSION,
        "test_seeds": list(test_seeds),
        "conditions": M89_CONDITIONS,
        "scan_coverage": coverage,
        "m87_freeze": {
            "m87_protocol_version": M87_PROTOCOL_VERSION,
            "frozen_protocol_fingerprint": M87_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_protocol_fingerprint": observed_m87_protocol,
            "frozen_policy_fingerprint": M87_FROZEN_POLICY_FINGERPRINT,
            "observed_policy_fingerprint": observed_m87_policy,
            "m89_training_episodes": 0,
        },
        "m89_protocol": {"frozen_fingerprint": M89_FROZEN_PROTOCOL_FINGERPRINT, "observed_fingerprint": observed_m89_protocol},
        "policy_boundary": ["rgb", "holding_food", "prior_outcome", "policy_owned_memory"],
        "results": results,
        "external_validity_gate": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "condition_success_rates": condition_rates,
            "oracle_successes": oracle_successes,
            "predeclared_minimum_success_rate": M89_MIN_SUCCESS_RATE,
            "passes": passes,
            "next_step": "report this sealed audit; do not tune M8.7 on M8.9" if passes else "stop: record failed morphology audit without tuning M8.7",
        },
        "limits": [
            "M8.9 evaluates a deterministic frozen M8.7 grounder and does not train, select, or tune on audit episodes.",
            "Capsule and box variants alter rendered morphology only; original sphere collision geometry preserves task mechanics.",
            "Segmentation is restricted to the offline scan-coverage audit; the deployed policy receives only RGB, holding state, prior outcome, and its own memory.",
            "The state oracle is a privileged ceiling, not an equal-input baseline.",
            "This tests declared simulator visual shifts, not real-world sensing, end-to-end learned control, or contact-rich physics.",
        ],
    }


def write_m89_report(path: str | Path) -> dict[str, object]:
    report = m89_benchmark()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m89_viewer_demo(trace_path: str | Path, *, seed: int = 1_207) -> ReplayResult:
    _require_coverage()
    if m89_protocol_fingerprint() != M89_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.9 protocol drifted")
    grounder = fit_m87_grounder()
    if m87_protocol_fingerprint() != M87_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.7 training protocol drifted")
    if m87_policy_fingerprint(grounder) != M87_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.7 fitted grounder drifted")
    policy = LearnedHeatmapRecoveryPolicy(grounder)  # type: ignore[arg-type]
    with ViewerSession(EcosystemEnv(m84_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m89-frozen-morphology-audit") as session:
        observation, _ = session.reset(seed=seed, options=_options(M89_CONDITIONS["southwest_red_food_box_relocation"]))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
