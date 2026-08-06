"""M11 frozen-M9 baseline and per-step persistent-maintenance failure atlas."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionKind, ActionOutcome
from .env import EcosystemEnv
from .m10 import (
    M10_VALIDATION_CONDITIONS,
    M10_VALIDATION_SEEDS,
    _options as m10_options,
    m10_config,
    m10_protocol_fingerprint,
    m10_validation,
)
from .m8 import wilson_interval
from .m9 import IntegratedRgbDrivePolicy, M9Macro, fit_m9_policy, m9_policy_fingerprint


M11_PROTOCOL_VERSION = "m11-frozen-m9-persistent-failure-atlas-v1"
M11_SAFE_DRIVE_BANDS = {"satiety_min": 0.15, "energy_min": 0.15, "boredom_max": 0.90}
# This is the deterministic M9 fit at the M10 checkpoint. M11 must fail if a
# future edit silently changes the supposedly frozen baseline.
M11_FROZEN_M9_POLICY_FINGERPRINT = "af31319a24ba092a99368f469c2ca3dcb992220ceb67070ba1b52ece88c4ff36"


@dataclass(frozen=True, slots=True)
class M11Episode:
    seed: int
    condition: str
    survived: bool
    maintenance_complete: bool
    terminal_cause: str
    feed_cycles: int
    play_cycles: int
    rest_cycles: int
    safe_drive_fraction: float
    forced_recovery: dict[str, object]
    interventions: dict[str, int]
    first_failure: dict[str, object] | None
    steps: tuple[dict[str, object], ...]


def _drive_record(drives: np.ndarray) -> dict[str, float]:
    values = np.asarray(drives, dtype=np.float32)
    return {"satiety": float(values[0]), "energy": float(values[1]), "boredom": float(values[2])}


def _inside_safe_band(drives: np.ndarray) -> bool:
    values = _drive_record(drives)
    return (
        values["satiety"] > M11_SAFE_DRIVE_BANDS["satiety_min"]
        and values["energy"] > M11_SAFE_DRIVE_BANDS["energy_min"]
        and values["boredom"] < M11_SAFE_DRIVE_BANDS["boredom_max"]
    )


def _candidate_snapshot(policy: IntegratedRgbDrivePolicy, observation: dict[str, Any]) -> dict[str, object]:
    """Record the frozen policy's public RGB evidence without changing its action."""

    detections = policy.grounder.detect(np.asarray(observation["rgb"], dtype=np.uint8))
    return {
        "food_offsets": [np.asarray(offset, dtype=np.float32).tolist() for offset in detections.food],
        "toy_offsets": [np.asarray(offset, dtype=np.float32).tolist() for offset in detections.toy],
        "food_detected": bool(detections.food),
        "toy_detected": bool(detections.toy),
    }


def _first_failure(
    steps: list[dict[str, object]], *, survived: bool, maintenance_complete: bool
) -> dict[str, object] | None:
    if survived and maintenance_complete:
        return None
    relocation = next((row for row in steps if row["disturbance"] == "food_relocated"), None)
    if relocation is not None:
        later = [row for row in steps if int(row["step"]) > int(relocation["step"])]
        if not any(row["action_kind"] == ActionKind.SCAN.name for row in later):
            return {"stage": "recovery", "classification": "no_rescan_after_relocation", "step": relocation["step"]}
        if not any(row["action_kind"] == ActionKind.CONSUME.name and row["outcome"] == "success" for row in later):
            return {"stage": "recovery", "classification": "relocated_food_not_consumed", "step": relocation["step"]}
    energy_breach = next((row for row in steps if float(row["drives_after"]["energy"]) <= M11_SAFE_DRIVE_BANDS["energy_min"]), None)
    if energy_breach is not None and not any(row["action_kind"] == ActionKind.REST.name for row in steps):
        return {"stage": "need_selection", "classification": "energy_need_has_no_rest_action", "step": energy_breach["step"]}
    if not maintenance_complete and not any(row["action_kind"] == ActionKind.REST.name for row in steps):
        return {"stage": "need_selection", "classification": "persistent_rest_capability_absent", "step": steps[-1]["step"]}
    blocked = next(
        (row for row in steps if row["action_kind"] == ActionKind.PICK_UP_RELATIVE.name and row["outcome"] == "blocked"),
        None,
    )
    if blocked is not None:
        return {"stage": "candidate_pickup", "classification": "blocked_pickup_not_recovered", "step": blocked["step"]}
    missing = next(
        (row for row in steps if row["action_kind"] == ActionKind.SCAN.name and not row["target_observation"]["food_detected"]),
        None,
    )
    if missing is not None:
        return {"stage": "grounding", "classification": "food_not_observed_in_policy_scan", "step": missing["step"]}
    return {"stage": "maintenance", "classification": "cycle_requirements_not_met", "step": steps[-1]["step"]}


def run_m11_episode(
    policy: IntegratedRgbDrivePolicy, *, seed: int, condition: str, controls: dict[str, Any]
) -> M11Episode:
    """Run M9 unchanged and retain enough public evidence to diagnose each failure."""

    env = EcosystemEnv(m10_config())
    try:
        observation, _ = env.reset(seed=seed, options=m10_options(controls))
        memory = policy.reset()
        records: list[dict[str, object]] = []
        safe_steps = 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        scans_after_relocation = 0
        interventions: Counter[str] = Counter()
        for step in range(1, env.config.max_episode_steps + 1):
            candidates = _candidate_snapshot(policy, observation)
            drives_before = _drive_record(np.asarray(observation["drives"], dtype=np.float32))
            action = policy.act(observation, memory)
            macro = memory.last_macro.name if memory.last_macro is not None else None
            action_kind = ActionKind(int(action["kind"])).name
            target = np.asarray(action["target"], dtype=np.float32).tolist()
            if action_kind == ActionKind.WALK_RELATIVE.name and macro == M9Macro.WALK_FOOD.name:
                candidate_rank: int | None = min(memory.blocked_attempts, max(0, len(candidates["food_offsets"]) - 1))
            elif action_kind == ActionKind.WALK_RELATIVE.name and macro == M9Macro.WALK_TOY.name:
                candidate_rank = 0
            else:
                candidate_rank = None
            observation, _, terminated, truncated, info = env.step(action)
            if relocation_seen and action_kind == ActionKind.SCAN.name:
                scans_after_relocation += 1
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen = relocation_seen or relocation_now
            stale_pickup = stale_pickup or (relocation_now and info["outcome"] == ActionOutcome.BLOCKED.value)
            consumed_after_relocation = consumed_after_relocation or (
                relocation_seen and action_kind == ActionKind.CONSUME.name and info["outcome"] == ActionOutcome.SUCCESS.value
            )
            if relocation_now:
                interventions["food_relocated"] += 1
            if info["resource_event"] is not None:
                interventions[str(info["resource_event"])] += 1
            if info["outcome"] == ActionOutcome.BLOCKED.value:
                interventions["blocked_action"] += 1
            safe_steps += int(_inside_safe_band(np.asarray(observation["drives"], dtype=np.float32)))
            records.append(
                {
                    "step": step,
                    "drives_before": drives_before,
                    "drives_after": _drive_record(np.asarray(observation["drives"], dtype=np.float32)),
                    "inside_safe_drive_band": _inside_safe_band(np.asarray(observation["drives"], dtype=np.float32)),
                    "target_observation": candidates,
                    "macro": macro,
                    "action_kind": action_kind,
                    "candidate_rank": candidate_rank,
                    "local_offset": target if action_kind == ActionKind.WALK_RELATIVE.name else None,
                    "outcome": info["outcome"],
                    "disturbance": info["disturbance"],
                    "resource_event": info["resource_event"],
                    "feed_cycles": info["feed_cycles"],
                    "play_cycles": info["play_cycles"],
                    "rest_cycles": info["rest_cycles"],
                    "food_available": info["food_available"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                state = env._require_state()
                survived = bool(info["survived"])
                maintenance_complete = bool(info["maintenance_complete"])
                terminal_cause = "survived_horizon" if truncated and survived else "energy_depleted" if state.drives.energy <= 0.0 else "satiety_depleted" if state.drives.satiety <= 0.0 else "terminated"
                recovery = {
                    "required": bool(controls.get("event_relocation_on_first_pickup")),
                    "stale_pickup": stale_pickup,
                    "scans_after_relocation": scans_after_relocation,
                    "consumed_after_relocation": consumed_after_relocation,
                    "complete": stale_pickup and scans_after_relocation >= 4 and consumed_after_relocation,
                }
                return M11Episode(
                    seed=seed,
                    condition=condition,
                    survived=survived,
                    maintenance_complete=maintenance_complete,
                    terminal_cause=terminal_cause,
                    feed_cycles=state.feed_cycles,
                    play_cycles=state.play_cycles,
                    rest_cycles=state.rest_cycles,
                    safe_drive_fraction=safe_steps / step,
                    forced_recovery=recovery,
                    interventions=dict(interventions),
                    first_failure=_first_failure(records, survived=survived, maintenance_complete=maintenance_complete),
                    steps=tuple(records),
                )
        raise AssertionError("M11 episode did not terminate")
    finally:
        env.close()


def _episode_record(episode: M11Episode) -> dict[str, object]:
    return {
        "seed": episode.seed,
        "survived": episode.survived,
        "maintenance_complete": episode.maintenance_complete,
        "terminal_cause": episode.terminal_cause,
        "completed_cycles": {"feed": episode.feed_cycles, "play": episode.play_cycles, "rest": episode.rest_cycles},
        "time_inside_safe_drive_bands": episode.safe_drive_fraction,
        "forced_recovery": episode.forced_recovery,
        "interventions": episode.interventions,
        "first_failure": episode.first_failure,
        "steps": list(episode.steps),
    }


def _aggregate(episodes: list[M11Episode]) -> dict[str, object]:
    count = len(episodes)
    curve = [
        {"step": step, "surviving_episodes": sum(item.survived or len(item.steps) >= step for item in episodes)}
        for step in range(1, max(len(item.steps) for item in episodes) + 1)
    ]
    failure_classes = Counter(
        str(item.first_failure["classification"]) for item in episodes if item.first_failure is not None
    )
    intervention_counts: Counter[str] = Counter()
    for item in episodes:
        intervention_counts.update(item.interventions)
    required = [item for item in episodes if bool(item.forced_recovery["required"])]
    return {
        "episodes": count,
        "survivals": sum(item.survived for item in episodes),
        "survival_rate": sum(item.survived for item in episodes) / count,
        "survival_wilson_95": list(wilson_interval(sum(item.survived for item in episodes), count)),
        "survival_curve": curve,
        "completed_maintenance_episodes": sum(item.maintenance_complete for item in episodes),
        "completed_cycles": {
            "minimum_feed": min(item.feed_cycles for item in episodes),
            "minimum_play": min(item.play_cycles for item in episodes),
            "minimum_rest": min(item.rest_cycles for item in episodes),
            "mean_feed": float(np.mean([item.feed_cycles for item in episodes])),
            "mean_play": float(np.mean([item.play_cycles for item in episodes])),
            "mean_rest": float(np.mean([item.rest_cycles for item in episodes])),
        },
        "mean_time_inside_safe_drive_bands": float(np.mean([item.safe_drive_fraction for item in episodes])),
        "forced_recovery_chains": {
            "required_episodes": len(required),
            "stale_pickups": sum(bool(item.forced_recovery["stale_pickup"]) for item in required),
            "full_rescans": sum(int(item.forced_recovery["scans_after_relocation"]) >= 4 for item in required),
            "completed": sum(bool(item.forced_recovery["complete"]) for item in required),
        },
        "interventions": dict(sorted(intervention_counts.items())),
        "terminal_causes": dict(sorted(Counter(item.terminal_cause for item in episodes).items())),
        "first_failure_classifications": dict(sorted(failure_classes.items())),
        "failures_are_inspectable": all(item.first_failure is not None for item in episodes if not (item.survived and item.maintenance_complete)),
        "episodes_detail": [_episode_record(item) for item in episodes],
    }


def m11_protocol_fingerprint() -> str:
    payload = {
        "version": M11_PROTOCOL_VERSION,
        "m10_protocol_fingerprint": m10_protocol_fingerprint(),
        "seeds": M10_VALIDATION_SEEDS,
        "conditions": M10_VALIDATION_CONDITIONS,
        "safe_drive_bands": M11_SAFE_DRIVE_BANDS,
        "frozen_m9_policy_fingerprint": M11_FROZEN_M9_POLICY_FINGERPRINT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def m11_baseline(*, seeds: tuple[int, ...] = M10_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M10_VALIDATION_SEEDS:
        raise ValueError("M11 baseline seeds are frozen; use M10_VALIDATION_SEEDS")
    mechanics = m10_validation()
    if not bool(mechanics["gate"]["passes"]):
        raise RuntimeError("M10 mechanics gate failed; M11 policy interpretation is invalid")
    policy = fit_m9_policy()
    before = m9_policy_fingerprint(policy)
    if before != M11_FROZEN_M9_POLICY_FINGERPRINT:
        raise RuntimeError("the M9 fit changed; update neither policy nor M11 baseline without a new protocol")
    results = {
        name: _aggregate([run_m11_episode(policy, seed=seed, condition=name, controls=controls) for seed in seeds])
        for name, controls in M10_VALIDATION_CONDITIONS.items()
    }
    after = m9_policy_fingerprint(policy)
    return {
        "schema_version": "0.11",
        "protocol_version": M11_PROTOCOL_VERSION,
        "protocol_fingerprint": m11_protocol_fingerprint(),
        "m10_protocol_fingerprint": m10_protocol_fingerprint(),
        "split": "m10_validation_frozen_baseline",
        "seeds": list(seeds),
        "conditions": M10_VALIDATION_CONDITIONS,
        "safe_drive_bands": M11_SAFE_DRIVE_BANDS,
        "policy_boundary": ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"],
        "frozen_m9_policy_fingerprint": before,
        "policy_fingerprint_after_baseline": after,
        "policy_unchanged": before == after,
        "m10_mechanics_gate": mechanics["gate"],
        "m10_coverage": mechanics["coverage"],
        "results": results,
        "atlas_index": [
            {"condition": name, "seed": item["seed"], "first_failure": item["first_failure"], "terminal_cause": item["terminal_cause"]}
            for name, result in results.items()
            for item in result["episodes_detail"]
        ],
        "diagnostic_complete": before == after and all(
            row["failures_are_inspectable"] for row in results.values()
        ),
        "limits": [
            "M11 is a frozen-policy diagnostic with no policy pass threshold or policy update.",
            "M9 has no rest macro or rest target grounder, so the atlas exposes that interface limit rather than treating it as an RGB failure.",
            "The environment remains a typed kinematic-skill benchmark rather than contact-rich embodied control.",
        ],
    }


def write_m11_report(path: str | Path) -> dict[str, object]:
    report = m11_baseline()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
