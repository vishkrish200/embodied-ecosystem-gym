"""Machine-readable, unopened protocol for the additive M13.14-r3 lane.

The r3 correction is deliberately operational: r2's learner classes remain
frozen, while random diagnostic traces must carry their exact ``pcg64-<seed>``
fingerprint before strict replay.  Importing this module is pure; no manifest,
ledger, environment, fit, evaluation, or replay work occurs here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


M1314R3_PROTOCOL_VERSION = "m13.14-r3-random-diagnostic-fingerprint-successor-r1"
M1314R3_MANIFEST_SCHEMA_VERSION = "m13.14-r3-modular-anchored-manifest-v1"

M1314R3_DEVELOPMENT_FIT_SEEDS = tuple(range(6840, 6860))
M1314R3_DEVELOPMENT_CHECK_SEEDS = tuple(range(6860, 6880))
M1314R3_CONFIRMATION_FIT_SEEDS = tuple(range(6880, 6920))
M1314R3_CONFIRMATION_EVALUATION_SEEDS = tuple(range(6920, 6940))
M1314R3_AUDIT_SEEDS = tuple(range(6940, 6960))

M1314R3_DEVELOPMENT_TRAINING_SEEDS = (20261435, 20261436, 20261437, 20261438)
M1314R3_CONFIRMATION_TRAINING_SEEDS = (
    20261439,
    20261440,
    20261441,
    20261442,
    20261443,
    20261444,
    20261445,
    20261446,
)

M1314R3_ACTION_SURFACE = (
    "GO_FOOD",
    "PICK_UP",
    "CONSUME",
    "GO_TOY",
    "PLAY",
    "GO_REST",
    "REST",
    "WAIT",
)
M1314R3_ARM_NAMES = (
    "modular_anchor_shield_candidate",
    "monolithic_anchor_shield_control",
    "modular_no_anchor_shield_control",
    "modular_anchor_no_shield_control",
)
M1314R3_COMMAND_NAMES = (
    "maintenance-policy-manifest-m1314r3",
    "m1314r3-development-preflight",
    "m1314r3-development",
    "m1314r3-confirmation-preflight",
    "m1314r3-confirmation",
    "m1314r3-audit-preflight",
    "m1314r3-audit",
    "m1314r3-replay",
)
M1314R3_DECISION_BUDGET_PER_FIT = 400_000
M1314R3_PREFLIGHT_STAGE_ORDER = (
    "development_preflight",
    "confirmation_preflight",
    "audit_preflight",
)
M1314R3_REPLAY_REQUIREMENT = "strict replay for every scored and diagnostic episode"
M1314R3_NO_RUN_AUTHORIZATION = (
    "No M13.14-r3 preflight, fit, evaluation, replay, confirmation, or audit command "
    "is authorized by this manifest alone."
)


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


M1314R3_CONDITIONS = {
    "development_fit": _conditions("m1314r3_fit"),
    "development_check": _conditions("m1314r3_check"),
    "confirmation_fit": _conditions("m1314r3_confirm_fit"),
    "confirmation_evaluation": _conditions("m1314r3_confirm_eval"),
    "audit": _conditions("m1314r3_audit"),
}
M1314R3_PREFLIGHT_TARGETS = {
    "development_preflight": {
        "scan_partitions": ("development_fit", "development_check"),
        "authorizes_command": "m1314r3-development",
    },
    "confirmation_preflight": {
        "scan_partitions": ("confirmation_fit", "confirmation_evaluation"),
        "authorizes_command": "m1314r3-confirmation",
    },
    "audit_preflight": {
        "scan_partitions": ("audit",),
        "authorizes_command": "m1314r3-audit",
    },
}

M1314R3_CAUSAL_COMPARISONS = {
    "vs_monolithic_anchor_shield_control": {
        "candidate": "modular_anchor_shield_candidate",
        "control": "monolithic_anchor_shield_control",
        "isolated_factor": "drive decomposition with matched shield and persistent teacher margin",
        "minimum_pooled_full_objective_advantage": 0.10,
        "maximum_per_condition_success_regression_episodes": 1,
        "maximum_decision_or_duration_safety_regression": 0.02,
    },
    "vs_modular_no_anchor_shield_control": {
        "candidate": "modular_anchor_shield_candidate",
        "control": "modular_no_anchor_shield_control",
        "isolated_factor": "persistent teacher margin after byte-identical supervised initialization",
        "minimum_pooled_full_objective_advantage": 0.10,
        "maximum_per_condition_success_regression_episodes": 1,
        "maximum_decision_or_duration_safety_regression": 0.02,
    },
    "vs_modular_anchor_no_shield_control": {
        "candidate": "modular_anchor_shield_candidate",
        "control": "modular_anchor_no_shield_control",
        "isolated_factor": "deterministic acting-loop shield on byte-identical modular anchored Q weights",
        "minimum_pooled_full_objective_advantage": 0.15,
        "maximum_per_condition_success_regression_episodes": 1,
        "maximum_decision_or_duration_safety_regression": 0.02,
        "parameter_fingerprint_must_match": True,
    },
}


def _fingerprint_sources() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "ecosystem_gym/__main__.py",
        root / "ecosystem_gym/actions.py",
        root / "ecosystem_gym/config.py",
        root / "ecosystem_gym/drives.py",
        root / "ecosystem_gym/env.py",
        root / "ecosystem_gym/trajectory.py",
        root / "ecosystem_gym/maintenance/__init__.py",
        root / "ecosystem_gym/maintenance/contract.py",
        root / "ecosystem_gym/maintenance/policy_state.py",
        root / "ecosystem_gym/maintenance/reward.py",
        root / "ecosystem_gym/maintenance/shielded.py",
        root / "ecosystem_gym/maintenance/modular_q_m1314r2.py",
        root / "ecosystem_gym/maintenance/teacher_data_m1314r3.py",
        root / "ecosystem_gym/maintenance/modular_q_artifacts_m1314r3.py",
        root / "ecosystem_gym/maintenance/policy_protocol_m1314r3.py",
        root / "ecosystem_gym/experiments/cli.py",
        root / "ecosystem_gym/experiments/m10.py",
        root / "ecosystem_gym/experiments/m13.py",
        root / "ecosystem_gym/experiments/m139.py",
        root / "ecosystem_gym/experiments/m1314r3.py",
        root / "ecosystem_gym/experiments/m1314r3_support.py",
        root / "ecosystem_gym/tasks.py",
        root / "docs/M13_14_MODULAR_ANCHORED_PROTOCOL.md",
        root / "docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md",
        root / "docs/M13_14_R3_MODULAR_ANCHORED_PROTOCOL.md",
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = _fingerprint_sources()
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.14-r3 runtime fingerprint sources are missing: {missing}")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _protocol_core(*, include_sources: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": M1314R3_MANIFEST_SCHEMA_VERSION,
        "protocol_version": M1314R3_PROTOCOL_VERSION,
        "status": "implemented-frozen-and-unopened",
        "supersession": {
            "predecessor": "M13.14-r2 frozen protocol",
            "predecessor_doc": "docs/M13_14_R2_MODULAR_ANCHORED_PROTOCOL.md",
            "supersedes_default_successor_request": True,
            "rationale": (
                "M13.14-r3 preserves the frozen r2 learner and modular-Q hypothesis while "
                "binding the exact PCG64 random-policy fingerprint before strict replay, "
                "with a fresh protocol, source-hash set, data lane, and split family."
            ),
            "frozen_boundaries_preserved": [
                "M13.14 and M13.14-r2 source, manifests, ledgers, data, artifacts, traces, and conclusions remain frozen",
                "no M13.14, M13.14-r2, or M13.13 teacher dataset, policy artifact, check data, or report path may be loaded",
                "all M13.14-r3 partitions, layouts, and training seeds are fresh",
            ],
        },
        "public_boundary": {
            "features": 30,
            "macros": list(M1314R3_ACTION_SURFACE),
            "inputs": [
                "agent_xy", "food_xy", "toy_xy", "rest_xy", "drives",
                "holding_food", "prior_outcome", "policy-owned memory",
            ],
            "forbidden": [
                "task_id", "reset options", "info", "authoritative counters",
                "environment internals", "M13.14 teacher datasets and policy artifacts",
                "M13.14-r2 teacher datasets and policy artifacts", "M13.13 policy bytes",
                "opened M13.14-r2 check data",
            ],
            "action_value_boundary": {
                "modular_head_count": 3,
                "macros_per_head": 8,
                "aggregated_macros": 8,
                "private_observation_keys_rejected": True,
            },
        },
        "partitions": {
            "fresh_seed_range": "6840--6959",
            "development_fit": list(M1314R3_DEVELOPMENT_FIT_SEEDS),
            "development_check": list(M1314R3_DEVELOPMENT_CHECK_SEEDS),
            "confirmation_fit": list(M1314R3_CONFIRMATION_FIT_SEEDS),
            "confirmation_evaluation": list(M1314R3_CONFIRMATION_EVALUATION_SEEDS),
            "audit": list(M1314R3_AUDIT_SEEDS),
        },
        "conditions": M1314R3_CONDITIONS,
        "preflight": {
            "stage_order": list(M1314R3_PREFLIGHT_STAGE_ORDER),
            "targets": M1314R3_PREFLIGHT_TARGETS,
            "requirements": {
                "persist_every_count_before_fit": True,
                "failing_row_writes_sealed_report": True,
                "failing_report_blocks_fit_and_evaluation": True,
                "runner_requires_matching_manifest_fingerprint": True,
                "source_hashes_bind_runner_cli_and_core": True,
            },
        },
        "teacher_dataset": {
            "source_partition": "fresh fit partition only",
            "policy_source": "public oracle",
            "schema_owner": "ecosystem_gym/maintenance/teacher_data_m1314r3.py",
            "schema_version": "m1314r3-teacher-dataset-v1",
            "public_only_features": True,
            "legal_masks_required": True,
            "all_eight_macros_must_appear": True,
            "sampled_minibatches_may_be_partial": True,
            "dedicated_immutable_replay_surface": True,
            "retained_throughout_training": True,
            "legacy_teacher_reuse_forbidden": True,
        },
        "arms": {
            "modular_anchor_shield_candidate": {
                "q_structure": "shared torso plus satiety/feed, energy/rest, and boredom/play heads",
                "teacher_margin_schedule": "persistent for the full off-policy run",
                "shield_enabled": True,
                "fit_artifact_reused_by": ["modular_anchor_no_shield_control"],
            },
            "monolithic_anchor_shield_control": {
                "q_structure": "single combined-reward head matched to the modular parameter budget",
                "teacher_margin_schedule": "persistent for the full off-policy run",
                "shield_enabled": True,
                "isolates": "modular decomposition",
            },
            "modular_no_anchor_shield_control": {
                "q_structure": "same modular torso and three drive heads as the candidate",
                "teacher_margin_schedule": "zero after byte-identical supervised initialization",
                "shield_enabled": True,
                "isolates": "persistent retention anchor",
            },
            "modular_anchor_no_shield_control": {
                "q_structure": "same modular torso and three drive heads as the candidate",
                "teacher_margin_schedule": "persistent for the full off-policy run",
                "shield_enabled": False,
                "parameter_fingerprint_must_match_candidate": True,
                "isolates": "deterministic acting-loop shield",
            },
        },
        "training_budget": {
            "decision_budget_per_fit": M1314R3_DECISION_BUDGET_PER_FIT,
            "development_training_seeds": list(M1314R3_DEVELOPMENT_TRAINING_SEEDS),
            "confirmation_training_seeds": list(M1314R3_CONFIRMATION_TRAINING_SEEDS),
            "fresh_initializations_required": True,
            "candidate_fit_reused_byte_for_byte_for_no_shield": True,
            "separate_fits_only_for": [
                "monolithic_anchor_shield_control", "modular_no_anchor_shield_control",
            ],
            "training_time_action_selection_for_shielded_arms_uses_evaluation_policy_class": True,
        },
        "evaluation_budget": {
            "episodes_per_replica_per_arm": 80,
            "episodes_per_condition_per_replica": 20,
            "development_total_scored_episodes": 1280,
            "confirmation_total_scored_episodes": 2560,
            "audit_total_scored_episodes": 640,
            "audit_scoring_scope": "candidate-only",
            "audit_scored_training_seeds": list(M1314R3_CONFIRMATION_TRAINING_SEEDS),
            "audit_scored_policy_episodes": 640,
            "audit_shared_diagnostics_are_separate_from_scored_policy_episodes": True,
            "replay_requirement": M1314R3_REPLAY_REQUIREMENT,
            "cross_arm_winner_selection": "forbidden",
            "random_diagnostic_policy_fingerprint": "pcg64-<seed>",
            "random_diagnostic_trace_replay_required": True,
        },
        "hard_gate_per_arm_condition": {
            "episodes": 20,
            "survival_at_least": 18,
            "maintenance_at_least": 18,
            "full_objective_at_least": 18,
            "mean_decision_safe_at_least": 0.85,
            "mean_duration_safe_at_least": 0.85,
            "required_recovery_at_least": 18,
            "unsafe_wait_at_most": 0.10,
            "integrity_violations": 0,
            "coverage_failures": 0,
            "replay_violations": 0,
        },
        "causal_comparison_gate": {
            "candidate_must_pass_all_hard_gates": True,
            "candidate_must_beat_each_matched_control": M1314R3_CAUSAL_COMPARISONS,
            "no_cross_arm_selection": True,
            "no_replacement_seeds_after_failure": True,
        },
        "serialization_and_reuse": {
            "artifact_schema_owner": "ecosystem_gym/maintenance/modular_q_artifacts_m1314r3.py",
            "artifact_schema_version": "m1314r3-modular-q-artifact-v1",
            "tamper_evident": True,
            "candidate_and_no_shield_parameter_identity_required": True,
            "m1314_r2_and_older_paths_rejected": True,
            "legacy_teacher_or_artifact_reuse_forbidden": True,
            "r2_learner_source_is_frozen_but_r2_artifacts_are_forbidden": True,
        },
        "future_command_names": list(M1314R3_COMMAND_NAMES),
        "future_commands": [
            "uv run python -m ecosystem_gym maintenance-policy-manifest-m1314r3 --output artifacts/manifests/m1314r3-modular-anchored.json",
            "uv run python -m ecosystem_gym experiment m1314r3-development-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --output artifacts/reports/m1314r3-development-preflight.json",
            "uv run python -m ecosystem_gym experiment m1314r3-development --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-development-preflight.json --output artifacts/reports/m1314r3-development.json",
            "uv run python -m ecosystem_gym experiment m1314r3-confirmation-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --development-report artifacts/reports/m1314r3-development.json --output artifacts/reports/m1314r3-confirmation-preflight.json",
            "uv run python -m ecosystem_gym experiment m1314r3-confirmation --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-confirmation-preflight.json --development-report artifacts/reports/m1314r3-development.json --output artifacts/reports/m1314r3-confirmation.json",
            "uv run python -m ecosystem_gym experiment m1314r3-audit-preflight --manifest artifacts/manifests/m1314r3-modular-anchored.json --confirmation-report artifacts/reports/m1314r3-confirmation.json --output artifacts/reports/m1314r3-audit-preflight.json",
            "uv run python -m ecosystem_gym experiment m1314r3-audit --manifest artifacts/manifests/m1314r3-modular-anchored.json --preflight-report artifacts/reports/m1314r3-audit-preflight.json --confirmation-report artifacts/reports/m1314r3-confirmation.json --output artifacts/reports/m1314r3-audit.json",
            "uv run python -m ecosystem_gym experiment m1314r3-replay --manifest artifacts/manifests/m1314r3-modular-anchored.json --trace <trace.json> --policy <artifact.json>",
        ],
        "authorization": M1314R3_NO_RUN_AUTHORIZATION,
    }
    if include_sources:
        result["source_hashes"] = _source_hashes()
    return _json_native(result)


def m1314r3_protocol_fingerprint() -> str:
    core = _protocol_core(include_sources=True)
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def m1314r3_policy_manifest() -> dict[str, Any]:
    payload = _protocol_core(include_sources=True)
    payload["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def write_m1314r3_policy_manifest(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"M13.14-r3 manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = m1314r3_policy_manifest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def verify_m1314r3_policy_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("M13.14-r3 manifest is unreadable") from exc
    expected = m1314r3_policy_manifest()
    if payload != expected:
        raise ValueError("M13.14-r3 manifest does not match frozen sources and protocol")
    return payload


__all__ = [
    "M1314R3_ACTION_SURFACE", "M1314R3_ARM_NAMES", "M1314R3_AUDIT_SEEDS",
    "M1314R3_CAUSAL_COMPARISONS", "M1314R3_COMMAND_NAMES", "M1314R3_CONDITIONS",
    "M1314R3_CONFIRMATION_EVALUATION_SEEDS", "M1314R3_CONFIRMATION_FIT_SEEDS",
    "M1314R3_CONFIRMATION_TRAINING_SEEDS", "M1314R3_DECISION_BUDGET_PER_FIT",
    "M1314R3_DEVELOPMENT_CHECK_SEEDS", "M1314R3_DEVELOPMENT_FIT_SEEDS",
    "M1314R3_DEVELOPMENT_TRAINING_SEEDS", "M1314R3_MANIFEST_SCHEMA_VERSION",
    "M1314R3_NO_RUN_AUTHORIZATION", "M1314R3_PREFLIGHT_STAGE_ORDER",
    "M1314R3_PREFLIGHT_TARGETS", "M1314R3_PROTOCOL_VERSION", "M1314R3_REPLAY_REQUIREMENT",
    "m1314r3_policy_manifest", "m1314r3_protocol_fingerprint",
    "verify_m1314r3_policy_manifest", "write_m1314r3_policy_manifest",
]
