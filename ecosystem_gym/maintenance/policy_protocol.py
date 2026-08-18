"""Machine-readable, unopened protocol for M13.13 policy-family tests.

Importing or writing this manifest performs no fitting, reset, environment
step, evaluation, ledger mutation, or trace generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .model_based import ModelBasedSchedulerConfig
from .shielded import ShieldConfig
from .urgency import UrgencySchedulerConfig


M1313_PROTOCOL_VERSION = "m13.13-distinct-public-policy-families-r1"
M1313_DEVELOPMENT_FIT_SEEDS = tuple(range(6200, 6220))
M1313_DEVELOPMENT_CHECK_SEEDS = tuple(range(6220, 6240))
M1313_CONFIRMATION_FIT_SEEDS = tuple(range(6300, 6340))
M1313_CONFIRMATION_EVALUATION_SEEDS = tuple(range(6400, 6420))
M1313_AUDIT_SEEDS = tuple(range(6500, 6520))
M1313_DEVELOPMENT_TRAINING_SEEDS = (20261331, 20261332, 20261333, 20261334)
M1313_CONFIRMATION_TRAINING_SEEDS = tuple(range(20261335, 20261343))
M1313_PPO_DECISION_BUDGET = 400_000


def _conditions(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        "persistent_reference": {
            "layout_id": f"{prefix}_northeast",
            "food_variant": "orange",
            "toy_variant": "ball",
            "camera_control": "scan_v2",
            "initial_scan_sector": "north",
        },
        "renewal_and_morphology": {
            "layout_id": f"{prefix}_southwest",
            "food_variant": "purple",
            "food_shape_variant": "capsule",
            "toy_variant": "cube",
            "agent_shape_variant": "capsule",
            "camera_control": "scan_v2",
            "initial_scan_sector": "south",
        },
        "event_relocation": {
            "layout_id": f"{prefix}_northwest",
            "food_variant": "blue",
            "food_shape_variant": "box",
            "toy_variant": "capsule",
            "lighting_variant": "dim",
            "camera_control": "scan_v2",
            "initial_scan_sector": "east",
            "event_relocation_on_first_pickup": True,
        },
        "compound": {
            "layout_id": f"{prefix}_southeast",
            "food_variant": "red",
            "food_shape_variant": "capsule",
            "toy_variant": "cube",
            "agent_shape_variant": "box",
            "dynamics_variant": "grippy",
            "blocked_distractor": True,
            "distractor_xy": [-0.04, 0.04],
            "camera_control": "scan_v2",
            "initial_scan_sector": "west",
            "event_relocation_on_first_pickup": True,
        },
    }


M1313_CONDITIONS = {
    "development_fit": _conditions("m1313_fit"),
    "development_check": _conditions("m1313_check"),
    "confirmation_fit": _conditions("m1313_confirm_fit"),
    "confirmation_evaluation": _conditions("m1313_confirm_eval"),
    "audit": _conditions("m1313_audit"),
}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        root / "ecosystem_gym/__main__.py",
        root / "ecosystem_gym/actions.py",
        root / "ecosystem_gym/config.py",
        root / "ecosystem_gym/drives.py",
        root / "ecosystem_gym/env.py",
        root / "ecosystem_gym/trajectory.py",
        root / "ecosystem_gym/maintenance/contract.py",
        root / "ecosystem_gym/maintenance/reward.py",
        root / "ecosystem_gym/maintenance/policy_state.py",
        root / "ecosystem_gym/maintenance/urgency.py",
        root / "ecosystem_gym/maintenance/model_based.py",
        root / "ecosystem_gym/maintenance/shielded.py",
        root / "ecosystem_gym/maintenance/policy_artifacts.py",
        root / "ecosystem_gym/maintenance/ppo_r2.py",
        root / "ecosystem_gym/experiments/cli.py",
        root / "ecosystem_gym/experiments/m10.py",
        root / "ecosystem_gym/experiments/m13.py",
        root / "ecosystem_gym/experiments/m139.py",
        root / "ecosystem_gym/experiments/m1313.py",
        root / "ecosystem_gym/experiments/m1313_support.py",
        root / "ecosystem_gym/tasks.py",
        root / "docs/M13_13_POLICY_FAMILIES_PROTOCOL.md",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.13 fingerprint sources are missing: {missing}")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _protocol_core(*, include_sources: bool) -> dict[str, Any]:
    urgency = UrgencySchedulerConfig()
    planner = ModelBasedSchedulerConfig()
    shield = ShieldConfig()
    result: dict[str, Any] = {
        "schema_version": "m13.13-policy-family-manifest-v1",
        "protocol_version": M1313_PROTOCOL_VERSION,
        "status": "frozen-and-unopened",
        "public_boundary": {
            "features": 30,
            "macros": [
                "GO_FOOD",
                "PICK_UP",
                "CONSUME",
                "GO_TOY",
                "PLAY",
                "GO_REST",
                "REST",
                "WAIT",
            ],
            "inputs": [
                "agent_xy",
                "food_xy",
                "toy_xy",
                "rest_xy",
                "drives",
                "holding_food",
                "prior_outcome",
                "policy-owned memory",
            ],
            "forbidden": ["task_id", "reset options", "info", "private counters", "environment internals"],
            "extra_commitment_fields_are_unencoded": True,
        },
        "partitions": {
            "development_fit": list(M1313_DEVELOPMENT_FIT_SEEDS),
            "development_check": list(M1313_DEVELOPMENT_CHECK_SEEDS),
            "confirmation_fit": list(M1313_CONFIRMATION_FIT_SEEDS),
            "confirmation_evaluation": list(M1313_CONFIRMATION_EVALUATION_SEEDS),
            "audit": list(M1313_AUDIT_SEEDS),
            "deliberately_unused": ["6140--6199", "6240--6299"],
        },
        "conditions": M1313_CONDITIONS,
        "families": {
            "urgency_commitment": {
                "candidate": asdict(urgency),
                "control": asdict(replace(urgency, commitment_enabled=False)),
                "only_changed_factor": "goal commitment and hysteretic continuation",
                "fit_budget": 0,
            },
            "short_horizon_model": {
                "candidate": asdict(planner),
                "control": asdict(replace(planner, lookahead_depth=1)),
                "only_changed_factor": "lookahead depth 4 versus depth 1",
                "fit_budget": 0,
                "declared_model_limit": "GO duration is optimistic when a hidden dynamics variant slows actual motion",
            },
            "shielded_learned": {
                "candidate": asdict(shield),
                "control": asdict(replace(shield, enabled=False)),
                "only_changed_factor": "deterministic supervisor enabled versus disabled on identical actor bytes",
                "actor": {
                    "topology": [30, 64, 64, 8],
                    "learner": "corrected M13.11-r2 masked semi-Markov PPO",
                    "learning_rate": 3e-4,
                    "rollout_rows": 2048,
                    "epochs": 4,
                    "decision_budget_per_replica": M1313_PPO_DECISION_BUDGET,
                    "development_training_seeds": list(M1313_DEVELOPMENT_TRAINING_SEEDS),
                    "confirmation_training_seeds": list(M1313_CONFIRMATION_TRAINING_SEEDS),
                    "candidate_and_control_actor_bytes_identical": True,
                },
            },
        },
        "development_episode_budget": {
            "urgency_pair": 160,
            "model_pair": 160,
            "shield_pair": 640,
            "shared_scripted_ceiling": 80,
            "shared_mask_random_diagnostic": 80,
            "total_check_episodes": 1120,
            "strict_replay_required": 1120,
        },
        "later_episode_budgets": {
            "per_promoted_deterministic_family_pair": 160,
            "shield_pair_eight_replicas": 1280,
            "shared_diagnostics_per_stage": 160,
            "maximum_confirmation_episodes": 1760,
            "maximum_audit_episodes": 1760,
            "strict_replay": "every episode",
        },
        "hard_gate_per_candidate_condition": {
            "episodes": 20,
            "survival_at_least": 18,
            "maintenance_at_least": 18,
            "full_objective_at_least": 18,
            "mean_decision_safe_at_least": 0.85,
            "mean_duration_safe_at_least": 0.85,
            "required_recovery_at_least": 18,
            "unsafe_wait_at_most": 0.10,
            "integrity_violations": 0,
        },
        "paired_family_gate": {
            "deterministic_candidates": {
                "pooled_full_objective_advantage": 0.10,
                "maximum_per_condition_success_regression_episodes": 1,
                "maximum_decision_or_duration_safety_regression": 0.02,
            },
            "shielded_candidate_each_replica": {
                "pooled_full_objective_advantage": 0.15,
                "maximum_decision_or_duration_safety_regression": 0.02,
                "all_four_development_replicas_must_pass": True,
            },
            "cross_family_selection": "forbidden; each family passes or fails independently",
        },
        "confirmation_and_audit": {
            "confirmation_rebuilds_learned_actors_from_scratch": True,
            "confirmation_requires_every_promoted_family_and_learned_replica_to_repeat_the_same_hard_gate": True,
            "audit_refits_nothing_and_applies_the_same_gate_once": True,
            "failed_family_stops_without_replacement_seeds_or_threshold_changes": True,
        },
        "future_commands": [
            "uv run python -m ecosystem_gym maintenance-policy-manifest --output artifacts/manifests/m1313-policy-families.json",
            "uv run python -m ecosystem_gym experiment m1313-development --manifest artifacts/manifests/m1313-policy-families.json --output artifacts/reports/m1313-development.json",
            "uv run python -m ecosystem_gym experiment m1313-confirmation --manifest artifacts/manifests/m1313-policy-families.json --development-report artifacts/reports/m1313-development.json --output artifacts/reports/m1313-confirmation.json",
            "uv run python -m ecosystem_gym experiment m1313-audit --manifest artifacts/manifests/m1313-policy-families.json --confirmation-report artifacts/reports/m1313-confirmation.json --output artifacts/reports/m1313-audit.json",
        ],
        "authorization": "No fit, development check, confirmation, or audit command is authorized by this manifest.",
    }
    if include_sources:
        result["source_hashes"] = _source_hashes()
    return result


def m1313_protocol_fingerprint() -> str:
    core = _protocol_core(include_sources=True)
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def policy_family_manifest() -> dict[str, Any]:
    payload = _protocol_core(include_sources=True)
    payload["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def write_policy_family_manifest(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"policy-family manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = policy_family_manifest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def verify_policy_family_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("policy-family manifest is unreadable") from exc
    expected = policy_family_manifest()
    if payload != expected:
        raise ValueError("policy-family manifest does not match frozen sources and protocol")
    return payload
