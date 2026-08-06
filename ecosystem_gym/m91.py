"""M9.1 protocol repair: separate blocked-item choice from forced recovery.

M9's integrated policy and its fitted parameters are frozen here.  M9.1 does
not select, train, or tune a successor; it tests whether the failed M9 gate
mistook safe distractor avoidance for failed recovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import EcosystemEnv
from .m8 import wilson_interval
from .m9 import (
    M9_AUDIT_SEEDS,
    M9_TRAIN_SEEDS,
    M9_VALIDATION_SEEDS,
    M9Episode,
    _options,
    _run_episode,
    _run_oracle,
    fit_m9_policy,
    m9_config,
    m9_policy_fingerprint,
    m9_scan_coverage,
)
from .trajectory import ReplayResult
from .viewer import ViewerSession


M91_PROTOCOL_VERSION = "m91-distractor-choice-forced-recovery-v1"
M91_SEEDS = tuple(range(1_600, 1_620))
M91_MIN_SUCCESS_RATE = 0.75
M91_MAX_INVALID_PICKUP_RATE = 0.25
M91_MIN_FORCED_RECOVERY_RATE = 0.75

M91_CONDITIONS: dict[str, dict[str, Any]] = {
    "distractor_choice": {
        "layout_id": "m91_choice_northwest",
        "food_variant": "purple",
        "food_shape_variant": "capsule",
        "toy_variant": "cube",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "initial_drives": [0.80, 1.0, 0.88],
        "blocked_distractor": True,
        "distractor_xy": [0.04, -0.04],
        "agent_shape_variant": "capsule",
    },
    "forced_recovery": {
        "layout_id": "m91_recovery_southeast",
        "food_variant": "blue",
        "food_shape_variant": "box",
        "toy_variant": "capsule",
        "camera_control": "scan_v2",
        "initial_scan_sector": "south",
        "initial_drives": [0.20, 1.0, 0.84],
        # The M9 policy completes its initial four-view scan then approaches
        # food. Relocation at that fifth action forces an observable failed
        # pickup before a fresh policy-owned scan cycle can finish the task.
        "disturbance_step": 5,
        "lighting_variant": "dim",
        "agent_shape_variant": "box",
    },
}


def m91_protocol_fingerprint() -> str:
    payload = {
        "protocol_version": M91_PROTOCOL_VERSION,
        "seeds": M91_SEEDS,
        "conditions": M91_CONDITIONS,
        "minimum_success_rate": M91_MIN_SUCCESS_RATE,
        "maximum_invalid_pickup_rate": M91_MAX_INVALID_PICKUP_RATE,
        "minimum_forced_recovery_rate": M91_MIN_FORCED_RECOVERY_RATE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _aggregate_choice(episodes: list[M9Episode]) -> dict[str, object]:
    successes = sum(item.success for item in episodes)
    invalid_pickups = sum(item.blocked_outcomes for item in episodes)
    count = len(episodes)
    success_rate = successes / count
    invalid_rate = invalid_pickups / count
    return {
        "episodes": count,
        "successes": successes,
        "completion_rate": success_rate,
        "completion_wilson_95": list(wilson_interval(successes, count)),
        "feed_rate": sum(item.food_done for item in episodes) / count,
        "play_rate": sum(item.toy_done for item in episodes) / count,
        "invalid_pickup_episodes": sum(item.blocked_outcomes > 0 for item in episodes),
        "invalid_pickup_rate": invalid_rate,
        "passes": success_rate >= M91_MIN_SUCCESS_RATE and invalid_rate <= M91_MAX_INVALID_PICKUP_RATE,
    }


def _aggregate_recovery(episodes: list[M9Episode]) -> dict[str, object]:
    successes = sum(item.success for item in episodes)
    count = len(episodes)
    forced_outcomes = sum(item.blocked_outcomes > 0 for item in episodes)
    recovered = sum(item.success and item.blocked_outcomes > 0 and item.scans_after_relocation >= 4 for item in episodes)
    success_rate = successes / count
    recovery_rate = recovered / count
    return {
        "episodes": count,
        "successes": successes,
        "completion_rate": success_rate,
        "completion_wilson_95": list(wilson_interval(successes, count)),
        "forced_failure_rate": forced_outcomes / count,
        "post_relocation_rescan_rate": sum(item.scans_after_relocation >= 4 for item in episodes) / count,
        "forced_recovery_rate": recovery_rate,
        "forced_recovery_wilson_95": list(wilson_interval(recovered, count)),
        "passes": success_rate >= M91_MIN_SUCCESS_RATE and recovery_rate >= M91_MIN_FORCED_RECOVERY_RATE,
    }


def m91_diagnosis(*, seeds: tuple[int, ...] = M91_SEEDS) -> dict[str, object]:
    if seeds != M91_SEEDS:
        raise ValueError("M9.1 diagnosis seeds are frozen; use M91_SEEDS")
    if set(seeds) & (set(M9_TRAIN_SEEDS) | set(M9_VALIDATION_SEEDS) | set(M9_AUDIT_SEEDS)):
        raise AssertionError("M9.1 must not reuse M9 train, validation, or audit seeds")
    coverage = m9_scan_coverage(M91_CONDITIONS, seeds=seeds)
    missing = [name for name, row in coverage.items() if not row["passes"]]
    if missing:
        raise RuntimeError(f"M9.1 has unobservable public scan targets: {', '.join(missing)}")
    policy = fit_m9_policy()
    before = m9_policy_fingerprint(policy)
    choice = [_run_episode(policy, seed=seed, condition="distractor_choice", controls=M91_CONDITIONS["distractor_choice"]) for seed in seeds]
    recovery = [_run_episode(policy, seed=seed, condition="forced_recovery", controls=M91_CONDITIONS["forced_recovery"]) for seed in seeds]
    choice_oracle = [_run_oracle(seed=seed, condition="distractor_choice", controls=M91_CONDITIONS["distractor_choice"]) for seed in seeds]
    recovery_oracle = [_run_oracle(seed=seed, condition="forced_recovery", controls=M91_CONDITIONS["forced_recovery"]) for seed in seeds]
    after = m9_policy_fingerprint(policy)
    choice_result = _aggregate_choice(choice)
    recovery_result = _aggregate_recovery(recovery)
    oracle_complete = all(item.success for item in (*choice_oracle, *recovery_oracle))
    return {
        "schema_version": "0.91",
        "protocol_version": M91_PROTOCOL_VERSION,
        "seeds": list(seeds),
        "conditions": M91_CONDITIONS,
        "coverage": coverage,
        "policy_boundary": ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"],
        "frozen_m9_policy_fingerprint": before,
        "policy_fingerprint_after_diagnosis": after,
        "policy_unchanged": before == after,
        "results": {"distractor_choice": choice_result, "forced_recovery": recovery_result},
        "state_oracle_ceiling": {"episodes": 2 * len(seeds), "successes": sum(item.success for item in (*choice_oracle, *recovery_oracle)), "complete": oracle_complete},
        "protocol_fingerprint": m91_protocol_fingerprint(),
        "verdict": {
            "passes": bool(choice_result["passes"] and recovery_result["passes"] and oracle_complete and before == after),
            "distractor_choice_is_success_without_invalid_pickup": bool(choice_result["passes"]),
            "forced_recovery_is_separately_demonstrated": bool(recovery_result["passes"]),
            "next_step": "define a fresh sealed M9.1 audit only if both distinct behaviours pass" if choice_result["passes"] and recovery_result["passes"] else "keep the M9 policy frozen and diagnose the failing behaviour before any retraining",
        },
        "limits": [
            "This repair separates safe avoidance from recovery; it does not prove semantic affordance understanding.",
            "The policy is frozen M9 structured behaviour cloning over typed kinematic skills, not end-to-end learned control.",
            "M9's sealed audit remains unscored and is not used here.",
        ],
    }


def write_m91_report(path: str | Path) -> dict[str, object]:
    report = m91_diagnosis()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m91_viewer_demo(trace_path: str | Path, *, seed: int = 1_607) -> ReplayResult:
    policy = fit_m9_policy()
    with ViewerSession(EcosystemEnv(m9_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m91-forced-recovery") as session:
        observation, _ = session.reset(seed=seed, options=_options(M91_CONDITIONS["forced_recovery"]))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
