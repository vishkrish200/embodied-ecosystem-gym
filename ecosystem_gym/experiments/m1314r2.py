"""M13.14-r2: modular anchored homeostatic-Q successor runner.

This module is future-only. Importing it performs no manifest generation,
ledger mutation, environment reset/step, training, evaluation, or replay.
Every public entry point validates the frozen manifest and uses exclusive
output paths before it can perform any side-effectful stage work.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..env import EcosystemEnv
from ..maintenance import (
    MaintenanceRewardState,
    ModularQArmM1314R2,
    ModularQPolicyM1314R2,
    PolicyMemory,
    ShieldConfig,
    ShieldedModularQPolicyM1314R2,
    TeacherDatasetM1314R2,
    compile_macro,
    learning_reward,
    load_modular_q_policy_m1314r2,
    load_teacher_dataset_m1314r2,
    modular_q_evaluation_spec_m1314r2,
    save_teacher_dataset_m1314r2,
    teacher_example_m1314r2,
)
from ..maintenance.policy_protocol_m1314r2 import (
    M1314R2_AUDIT_SEEDS,
    M1314R2_CAUSAL_COMPARISONS,
    M1314R2_CONDITIONS,
    M1314R2_CONFIRMATION_EVALUATION_SEEDS,
    M1314R2_CONFIRMATION_FIT_SEEDS,
    M1314R2_CONFIRMATION_TRAINING_SEEDS,
    M1314R2_DEVELOPMENT_CHECK_SEEDS,
    M1314R2_DEVELOPMENT_FIT_SEEDS,
    M1314R2_DEVELOPMENT_TRAINING_SEEDS,
    m1314r2_protocol_fingerprint,
    verify_m1314r2_policy_manifest,
)
from ..tasks import LAYOUTS
from ..trajectory import ReplayResult
from .m10 import m10_scan_coverage
from .m13 import _options
from .m133 import m133_epsilon
from .m139 import (
    BalancedM139Oracle,
    SeededRandomM139Policy,
    m139_config,
    replay_m139_trace,
    run_m139_episode,
)
from .m1314r2_support import (
    M1314R2_PREFLIGHT_STAGE_TARGETS,
    M1314R2StageLedger,
    m1314r2_content_hash,
    seal_m1314r2_preflight_report,
    verify_m1314r2_preflight_report,
)

ModularQArm = ModularQArmM1314R2
ModularQPolicy = ModularQPolicyM1314R2
ShieldedModularQPolicy = ShieldedModularQPolicyM1314R2
TeacherDataset = TeacherDatasetM1314R2
load_modular_q_policy = load_modular_q_policy_m1314r2
load_teacher_dataset = load_teacher_dataset_m1314r2
modular_q_evaluation_spec = modular_q_evaluation_spec_m1314r2
save_teacher_dataset = save_teacher_dataset_m1314r2
teacher_example = teacher_example_m1314r2


for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"


M1314R2_DEVELOPMENT_REPORT_SCHEMA_VERSION = "m13.14-r2-development-report-v1"
M1314R2_CONFIRMATION_REPORT_SCHEMA_VERSION = "m13.14-r2-confirmation-report-v1"
M1314R2_AUDIT_REPORT_SCHEMA_VERSION = "m13.14-r2-audit-report-v1"
M1314R2_CANONICAL_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "m1314r2" / "stage-ledger.json"
)
M1314R2_DECISION_BUDGET = 400_000
M1314R2_UPDATE_EVERY = 4
M1314R2_RANDOM_SEEDS = {
    "development": 20_261_431,
    "confirmation": 20_261_432,
    "audit": 20_261_433,
}
M1314R2_REAL_FIT_ARMS = (
    ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
    ModularQArm.MONOLITHIC_ANCHOR_SHIELD_CONTROL,
    ModularQArm.MODULAR_NO_ANCHOR_SHIELD_CONTROL,
)
M1314R2_ALL_ARMS = tuple(ModularQArm(name) for name in (
    "modular_anchor_shield_candidate",
    "monolithic_anchor_shield_control",
    "modular_no_anchor_shield_control",
    "modular_anchor_no_shield_control",
))
M1314R2_ARM_ROLES = {
    ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE: "candidate",
    ModularQArm.MONOLITHIC_ANCHOR_SHIELD_CONTROL: "control",
    ModularQArm.MODULAR_NO_ANCHOR_SHIELD_CONTROL: "control",
    ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL: "control",
}
M1314R2_REPORT_SCHEMAS = {
    "development": M1314R2_DEVELOPMENT_REPORT_SCHEMA_VERSION,
    "confirmation": M1314R2_CONFIRMATION_REPORT_SCHEMA_VERSION,
    "audit": M1314R2_AUDIT_REPORT_SCHEMA_VERSION,
}
M1314R2_RECOVERY_REQUIRED_CONDITIONS = frozenset({"event_relocation", "compound"})


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _report_hash(report: Mapping[str, Any]) -> str:
    return m1314r2_content_hash({key: value for key, value in report.items() if key != "content_hash"})


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"M13.14 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _seal_stage_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result["content_hash"] = _report_hash(result)
    return result


def _stage_gate_inputs(
    stage: str,
) -> tuple[
    tuple[int, ...],
    Mapping[str, Mapping[str, Any]],
    tuple[int, ...],
    tuple[ModularQArm, ...],
]:
    if stage == "development":
        return (
            M1314R2_DEVELOPMENT_TRAINING_SEEDS,
            M1314R2_CONDITIONS["development_check"],
            M1314R2_DEVELOPMENT_CHECK_SEEDS,
            M1314R2_ALL_ARMS,
        )
    if stage == "confirmation":
        return (
            M1314R2_CONFIRMATION_TRAINING_SEEDS,
            M1314R2_CONDITIONS["confirmation_evaluation"],
            M1314R2_CONFIRMATION_EVALUATION_SEEDS,
            M1314R2_ALL_ARMS,
        )
    if stage == "audit":
        return (
            M1314R2_CONFIRMATION_TRAINING_SEEDS,
            M1314R2_CONDITIONS["audit"],
            M1314R2_AUDIT_SEEDS,
            (ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,),
        )
    raise ValueError(f"unsupported M13.14 stage {stage!r}")


def _expected_stage_row_count(stage: str) -> int:
    training_seeds, conditions, env_seeds, arms = _stage_gate_inputs(stage)
    per_matrix = len(conditions) * len(env_seeds)
    return len(arms) * len(training_seeds) * per_matrix + 2 * per_matrix


def _canonical_equal(left: object, right: object) -> bool:
    return m1314r2_content_hash(left) == m1314r2_content_hash(right)


def _normalized_policy_artifacts(
    rows: object,
    *,
    training_seeds: tuple[int, ...],
    expected_arms: tuple[ModularQArm, ...],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("M13.14 policy_artifacts must be a list")
    protocol = m1314r2_protocol_fingerprint()
    config = m139_config()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("M13.14 policy_artifact rows must be objects")
        arm = ModularQArm(str(row.get("arm")))
        if arm not in expected_arms:
            raise ValueError("M13.14 policy_artifact references an undeclared arm for this stage")
        training_seed = int(row.get("training_seed"))
        if training_seed not in training_seeds:
            raise ValueError("M13.14 policy_artifact references an undeclared training seed")
        if str(row.get("label")) != _stage_label(arm, training_seed):
            raise ValueError("M13.14 policy_artifact label does not match the frozen naming scheme")
        if str(row.get("role")) != M1314R2_ARM_ROLES[arm]:
            raise ValueError("M13.14 policy_artifact role does not match the frozen arm role")
        artifact = row.get("artifact")
        evaluation_spec = row.get("evaluation_spec")
        if not isinstance(artifact, dict) or not isinstance(evaluation_spec, dict):
            raise ValueError("M13.14 policy_artifact rows must carry artifact and evaluation_spec objects")
        artifact_path = Path(str(artifact.get("path")))
        if _file_sha256(artifact_path) != str(artifact.get("sha256")):
            raise ValueError("M13.14 policy_artifact sha256 does not match the referenced artifact")
        learner = load_modular_q_policy(
            artifact_path,
            config=config,
            expected_protocol_fingerprint=protocol,
        )
        shield_payload = evaluation_spec.get("shield")
        if not isinstance(shield_payload, dict):
            raise ValueError("M13.14 evaluation_spec shield payload must be an object")
        canonical_spec = modular_q_evaluation_spec(
            learner,
            shield=ShieldConfig(**shield_payload),
        )
        if not _canonical_equal(canonical_spec, evaluation_spec):
            raise ValueError("M13.14 evaluation_spec does not match the referenced modular-Q artifact")
        normalized.append(
            {
                "label": _stage_label(arm, training_seed),
                "arm": arm.value,
                "role": M1314R2_ARM_ROLES[arm],
                "training_seed": training_seed,
                "artifact": {
                    "path": str(artifact_path.resolve()),
                    "sha256": str(artifact["sha256"]),
                    "policy_fingerprint": str(artifact["policy_fingerprint"]),
                    "weight_fingerprint": str(artifact["weight_fingerprint"]),
                    "parameter_bytes_sha256": str(artifact["parameter_bytes_sha256"]),
                    "weight_bytes_sha256": str(artifact["weight_bytes_sha256"]),
                },
                "evaluation_spec": canonical_spec,
            }
        )
    normalized.sort(key=lambda entry: (int(entry["training_seed"]), str(entry["arm"])))
    expected_count = len(training_seeds) * len(expected_arms)
    if len(normalized) != expected_count:
        raise ValueError("M13.14 policy_artifacts count does not match the frozen stage matrix")
    expected_pairs = {
        (arm.value, int(training_seed))
        for training_seed in training_seeds
        for arm in expected_arms
    }
    observed_pairs = {(str(row["arm"]), int(row["training_seed"])) for row in normalized}
    if observed_pairs != expected_pairs:
        raise ValueError("M13.14 policy_artifacts do not cover the frozen arm/seed matrix")
    if ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL in expected_arms:
        by_seed = {
            training_seed: {
                row["arm"]: row
                for row in normalized
                if int(row["training_seed"]) == int(training_seed)
            }
            for training_seed in training_seeds
        }
        for training_seed, seed_rows in by_seed.items():
            candidate = seed_rows[ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value]
            no_shield = seed_rows[ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL.value]
            if not _canonical_equal(candidate["artifact"], no_shield["artifact"]):
                raise ValueError("M13.14 no-shield control must reuse the exact candidate artifact")
            for key in (
                "policy_fingerprint",
                "weight_fingerprint",
                "parameter_bytes_sha256",
                "weight_bytes_sha256",
            ):
                if candidate["evaluation_spec"][key] != no_shield["evaluation_spec"][key]:
                    raise ValueError("M13.14 candidate/no-shield specs must share identical learned weights")
            if bool(candidate["evaluation_spec"]["shield"]["enabled"]) is not True:
                raise ValueError("M13.14 candidate evaluation spec must be shield-enabled")
            if bool(no_shield["evaluation_spec"]["shield"]["enabled"]) is not False:
                raise ValueError("M13.14 no-shield control must disable only the shield flag")
    return normalized


def _validate_stage_episode_evidence(
    evidence: object,
    *,
    stage: str,
    training_seeds: tuple[int, ...],
    conditions: Mapping[str, Mapping[str, Any]],
    expected_env_seeds: tuple[int, ...],
    expected_arms: tuple[ModularQArm, ...],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        raise ValueError(f"M13.14 {stage} episode_evidence must be a list")
    rows: list[dict[str, Any]] = []
    for row in evidence:
        if not isinstance(row, dict):
            raise ValueError(f"M13.14 {stage} episode_evidence rows must be objects")
        rows.append(dict(row))
    expected_total = _expected_stage_row_count(stage)
    if len(rows) != expected_total:
        raise ValueError(f"M13.14 {stage} episode_evidence row count mismatch")

    expected_policy_cells = {
        (arm.value, int(training_seed), str(condition), int(env_seed))
        for arm in expected_arms
        for training_seed in training_seeds
        for condition in conditions
        for env_seed in expected_env_seeds
    }
    observed_policy_cells = {
        (
            str(row["arm"]),
            int(row["training_seed"]),
            str(row["condition"]),
            int(row["env_seed"]),
        )
        for row in rows
        if str(row.get("arm")) in {arm.value for arm in expected_arms}
    }
    if observed_policy_cells != expected_policy_cells:
        raise ValueError(f"M13.14 {stage} policy evidence does not cover the frozen arm/seed/condition matrix")

    expected_diagnostic_cells = {
        ("scripted_ceiling", 0, f"{stage}_scripted_ceiling", "scripted_ceiling"),
        ("mask_random", M1314R2_RANDOM_SEEDS[stage], f"{stage}_mask_random", "mask_random"),
    }
    for arm, training_seed, label, role in expected_diagnostic_cells:
        observed = [
            row
            for row in rows
            if str(row.get("arm")) == arm
            and int(row.get("training_seed")) == int(training_seed)
            and str(row.get("label")) == label
            and str(row.get("role")) == role
        ]
        if len(observed) != len(conditions) * len(expected_env_seeds):
            raise ValueError(f"M13.14 {stage} diagnostic evidence for {arm} is incomplete")
        observed_cells = {(str(row["condition"]), int(row["env_seed"])) for row in observed}
        expected_cells = {
            (str(condition), int(env_seed))
            for condition in conditions
            for env_seed in expected_env_seeds
        }
        if observed_cells != expected_cells:
            raise ValueError(f"M13.14 {stage} diagnostic evidence for {arm} does not match the frozen matrix")
    return rows


def _verify_stage_report(
    path: str | Path,
    *,
    stage: str,
    manifest_fingerprint: str | None = None,
) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"M13.14 {stage} report is unreadable") from exc
    if not isinstance(report, dict):
        raise ValueError(f"M13.14 {stage} report root must be an object")
    if report.get("schema_version") != M1314R2_REPORT_SCHEMAS[stage]:
        raise ValueError(f"M13.14 {stage} report schema mismatch")
    if report.get("stage") != stage:
        raise ValueError(f"M13.14 {stage} report stage mismatch")
    if report.get("protocol_fingerprint") != m1314r2_protocol_fingerprint():
        raise ValueError(f"M13.14 {stage} report protocol fingerprint mismatch")
    if manifest_fingerprint is not None and report.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError(f"M13.14 {stage} report manifest fingerprint mismatch")
    if report.get("content_hash") != _report_hash(report):
        raise ValueError(f"M13.14 {stage} report content hash mismatch")
    training_seeds, conditions, expected_env_seeds, expected_arms = _stage_gate_inputs(stage)
    normalized_artifacts = _normalized_policy_artifacts(
        report.get("policy_artifacts"),
        training_seeds=training_seeds,
        expected_arms=expected_arms,
    )
    if not _canonical_equal(normalized_artifacts, report.get("policy_artifacts")):
        raise ValueError(f"M13.14 {stage} report policy_artifacts do not canonically match the referenced artifacts")
    evidence = _validate_stage_episode_evidence(
        report.get("episode_evidence"),
        stage=stage,
        training_seeds=training_seeds,
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
        expected_arms=expected_arms,
    )
    expected_replay_count = len(evidence)
    if int(report.get("strict_replay_count", -1)) != expected_replay_count:
        raise ValueError(f"M13.14 {stage} report strict_replay_count mismatch")
    gate = m1314r2_gate(
        evidence,
        training_seeds=training_seeds,
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
        expected_arms=expected_arms,
    )
    if not _canonical_equal(gate, report.get("gate")):
        raise ValueError(f"M13.14 {stage} report gate does not match the stored episode evidence")
    return report


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _assert_layouts() -> None:
    names = {name for name in LAYOUTS if name.startswith("m1314r2_")}
    expected = {
        str(row["layout_id"])
        for partitions in M1314R2_CONDITIONS.values()
        for row in partitions.values()
    }
    if names != expected or len(names) != 20:
        raise ValueError("M13.14 layouts are incomplete")

    def _points(name: str) -> tuple[tuple[float, float], ...]:
        layout = LAYOUTS[name]
        return (
            layout.agent_xy,
            layout.food_low,
            layout.food_high,
            layout.toy_xy,
            layout.rest_xy,
        )

    current = {point for name in names for point in _points(name)}
    previous = {
        point
        for name in LAYOUTS
        if not name.startswith("m1314r2_")
        for point in _points(name)
    }
    if len(current) != 100 or current & previous:
        raise ValueError("M13.14 geometry is not fresh")


def _coverage_expectations(stage: str) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
    target = M1314R2_PREFLIGHT_STAGE_TARGETS[stage]
    partitions = tuple(str(name) for name in target["scan_partitions"])
    conditions = tuple(str(name) for name in M1314R2_CONDITIONS[partitions[0]])
    episode_counts = {
        "development_fit": len(M1314R2_DEVELOPMENT_FIT_SEEDS),
        "development_check": len(M1314R2_DEVELOPMENT_CHECK_SEEDS),
        "confirmation_fit": len(M1314R2_CONFIRMATION_FIT_SEEDS),
        "confirmation_evaluation": len(M1314R2_CONFIRMATION_EVALUATION_SEEDS),
        "audit": len(M1314R2_AUDIT_SEEDS),
    }
    return partitions, conditions, {partition: episode_counts[partition] for partition in partitions}


def _scan_preflight_coverage(stage: str) -> dict[str, dict[str, dict[str, object]]]:
    partitions, _, _ = _coverage_expectations(stage)
    seeds_by_partition = {
        "development_fit": M1314R2_DEVELOPMENT_FIT_SEEDS,
        "development_check": M1314R2_DEVELOPMENT_CHECK_SEEDS,
        "confirmation_fit": M1314R2_CONFIRMATION_FIT_SEEDS,
        "confirmation_evaluation": M1314R2_CONFIRMATION_EVALUATION_SEEDS,
        "audit": M1314R2_AUDIT_SEEDS,
    }
    return {
        partition: m10_scan_coverage(M1314R2_CONDITIONS[partition], seeds=seeds_by_partition[partition])
        for partition in partitions
    }


def _preflight_report(
    *,
    stage: str,
    manifest_fingerprint: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = _scan_preflight_coverage(stage)
    _, conditions, episode_counts = _coverage_expectations(stage)
    payload: dict[str, Any] = {} if extra is None else dict(extra)
    payload["coverage"] = coverage
    return seal_m1314r2_preflight_report(
        payload,
        expected_protocol_fingerprint=m1314r2_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=conditions,
        expected_episode_counts=episode_counts,
    )


def m1314r2_development_preflight(*, manifest_path: str | Path) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    _assert_layouts()
    return _preflight_report(
        stage="development_preflight",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        extra={
            "schema_note": "future-only sealed coverage authorization for development fit/check",
        },
    )


def m1314r2_confirmation_preflight(
    *,
    manifest_path: str | Path,
    development_report_path: str | Path,
) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    development = _verify_stage_report(
        development_report_path,
        stage="development",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
    )
    if not bool(development.get("gate", {}).get("passes", False)):
        raise ValueError("M13.14 confirmation preflight requires a passing development report")
    _assert_layouts()
    return _preflight_report(
        stage="confirmation_preflight",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        extra={
            "development_report_sha256": _file_sha256(development_report_path),
        },
    )


def m1314r2_audit_preflight(
    *,
    manifest_path: str | Path,
    confirmation_report_path: str | Path,
) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    confirmation = _verify_stage_report(
        confirmation_report_path,
        stage="confirmation",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
    )
    if not bool(confirmation.get("gate", {}).get("passes", False)):
        raise ValueError("M13.14 audit preflight requires a passing confirmation report")
    _assert_layouts()
    return _preflight_report(
        stage="audit_preflight",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        extra={
            "confirmation_report_sha256": _file_sha256(confirmation_report_path),
        },
    )


def _collect_teacher_dataset(
    *,
    conditions: Mapping[str, Mapping[str, Any]],
    env_seeds: tuple[int, ...],
    dataset_path: Path,
) -> tuple[dict[str, Any], TeacherDataset]:
    config = m139_config()
    env = EcosystemEnv(config)
    oracle = BalancedM139Oracle()
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for condition, controls in conditions.items():
            for seed in env_seeds:
                observation, _ = env.reset(seed=seed, options=_options(dict(controls)))
                memory = oracle.reset()
                while True:
                    macro = oracle.choose(observation, memory)
                    feature_row, mask_row, label = teacher_example(
                        observation,
                        memory,
                        label=macro,
                        config=config,
                    )
                    features.append(feature_row)
                    masks.append(mask_row)
                    labels.append(label)
                    action = compile_macro(macro, observation, config)
                    next_observation, _, terminated, truncated, _ = env.step(action)
                    oracle.observe(
                        memory,
                        observation_before=observation,
                        macro=macro,
                        action=action,
                        observation_after=next_observation,
                    )
                    observation = next_observation
                    if terminated or truncated:
                        break
    finally:
        env.close()
    dataset = TeacherDataset(
        features=np.asarray(features, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.bool_),
        labels=np.asarray(labels, dtype=np.int64),
    )
    artifact = save_teacher_dataset(dataset_path, dataset)
    return {
        "artifact": artifact,
        "rows": dataset.rows,
        "feature_dim": dataset.feature_dim,
        "class_counts": dataset.class_counts,
        "source_partition": sorted(int(seed) for seed in env_seeds),
        "source_conditions": list(conditions),
        "public_fields_only": True,
    }, dataset


def _fit_policy(
    training_seed: int,
    *,
    arm: ModularQArm,
    conditions: Mapping[str, Mapping[str, Any]],
    env_seeds: tuple[int, ...],
    dataset_path: str,
    artifact_dir: str,
    decision_budget: int = M1314R2_DECISION_BUDGET,
) -> dict[str, Any]:
    config = m139_config()
    learner = ModularQPolicy(
        arm=arm,
        seed=training_seed,
        config=config,
        protocol_fingerprint=m1314r2_protocol_fingerprint(),
    )
    dataset = load_teacher_dataset(dataset_path)
    learner.attach_teacher_dataset(dataset)
    warmstart = learner.warmstart_from_teacher(dataset)
    acting_policy = ShieldedModularQPolicy(
        learner=learner,
        shield=ShieldConfig(enabled=bool(arm != ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL)),
    )
    rng = np.random.default_rng(training_seed ^ 0x4D31333134)
    env = EcosystemEnv(config)
    decisions = 0
    episodes = 0
    losses: list[dict[str, float]] = []
    condition_items = tuple((str(name), dict(values)) for name, values in conditions.items())
    started = time.perf_counter()
    try:
        while decisions < decision_budget:
            condition, controls = condition_items[episodes % len(condition_items)]
            env_seed = env_seeds[(episodes // len(condition_items)) % len(env_seeds)]
            observation, _ = env.reset(seed=env_seed, options=_options(controls))
            memory: PolicyMemory = acting_policy.reset()
            reward_state = MaintenanceRewardState(
                required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
            )
            epsilon = m133_epsilon(episodes)
            episodes += 1
            while True:
                features = learner.features(observation, memory)
                macro = acting_policy.select_action(observation, memory, rng=rng, epsilon=epsilon)
                action = compile_macro(macro, observation, config)
                next_observation, environment_reward, terminated, truncated, info = env.step(action)
                reward, _ = learning_reward(
                    environment_reward=environment_reward,
                    state=reward_state,
                    observation_before=observation,
                    observation_after=next_observation,
                    action=action,
                    info=info,
                    terminated=terminated,
                    truncated=truncated,
                )
                acting_policy.observe(
                    memory,
                    observation_before=observation,
                    macro=macro,
                    action=action,
                    observation_after=next_observation,
                )
                next_features = learner.features(next_observation, memory)
                next_mask = learner.mask(next_observation)
                learner.replay.add(
                    features=features,
                    action=macro,
                    reward=reward,
                    next_features=next_features,
                    next_mask=next_mask,
                    done=terminated or truncated,
                    duration=float(np.asarray(action["duration"], dtype=np.float32).item()),
                )
                decisions += 1
                if (
                    learner.teacher_replay is not None
                    and learner.replay.size >= learner.learner.td_batch_size
                    and decisions % M1314R2_UPDATE_EVERY == 0
                ):
                    losses.append(learner.sample_and_update(rng=rng))
                observation = next_observation
                if terminated or truncated or decisions >= decision_budget:
                    break
    finally:
        env.close()
    artifact = learner.save(Path(artifact_dir) / f"{arm.value}-seed-{training_seed}.json")
    td_losses = [float(row["td_loss"]) for row in losses]
    teacher_losses = [float(row["teacher_margin_loss"]) for row in losses]
    return {
        "arm": arm.value,
        "training_seed": int(training_seed),
        "elapsed_seconds": time.perf_counter() - started,
        "warmstart": warmstart,
        "training": {
            "decision_budget": int(decision_budget),
            "decisions": int(decisions),
            "overshoot": max(0, int(decisions - decision_budget)),
            "episodes": int(episodes),
            "updates": len(losses),
            "mean_td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
            "mean_teacher_margin_loss": float(np.mean(teacher_losses)) if teacher_losses else 0.0,
            "teacher_margin_weight": float(learner.teacher_margin_weight),
        },
        "artifact": artifact,
    }


def _fit_wave(
    *,
    training_seeds: tuple[int, ...],
    conditions: Mapping[str, Mapping[str, Any]],
    env_seeds: tuple[int, ...],
    dataset_path: Path,
    artifact_dir: Path,
    workers: int | None,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    jobs = [
        (training_seed, arm)
        for training_seed in training_seeds
        for arm in M1314R2_REAL_FIT_ARMS
    ]
    worker_count = min(
        workers or min(len(jobs), max(1, os.cpu_count() or 1)),
        len(jobs),
    )
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(
                _fit_policy,
                training_seed,
                arm=arm,
                conditions=conditions,
                env_seeds=env_seeds,
                dataset_path=str(dataset_path),
                artifact_dir=str(artifact_dir),
            )
            for training_seed, arm in jobs
        ]
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda row: (int(row["training_seed"]), str(row["arm"])))


def _stage_label(arm: ModularQArm, training_seed: int) -> str:
    return f"{arm.value}_seed_{training_seed}"


def _evaluation_specs_from_fit_jobs(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    protocol = m1314r2_protocol_fingerprint()
    config = m139_config()
    specs: list[dict[str, Any]] = []
    for job in jobs:
        training_seed = int(job["training_seed"])
        arm = ModularQArm(str(job["arm"]))
        artifact = dict(job["artifact"])
        learner = load_modular_q_policy(
            artifact["path"],
            config=config,
            expected_protocol_fingerprint=protocol,
        )
        if _file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("M13.14 artifact sha256 mismatch after reload")
        if arm is ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE:
            for eval_arm, shield_enabled, role in (
                (ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE, True, "candidate"),
                (ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL, False, "control"),
            ):
                shield = ShieldConfig(enabled=shield_enabled)
                specs.append(
                    {
                        "label": _stage_label(eval_arm, training_seed),
                        "arm": eval_arm.value,
                        "role": role,
                        "training_seed": training_seed,
                        "artifact": artifact,
                        "evaluation_spec": modular_q_evaluation_spec(learner, shield=shield),
                        "policy": ShieldedModularQPolicy(learner=learner, shield=shield),
                    }
                )
            continue
        shield = ShieldConfig(enabled=True)
        specs.append(
            {
                "label": _stage_label(arm, training_seed),
                "arm": arm.value,
                "role": "control",
                "training_seed": training_seed,
                "artifact": artifact,
                "evaluation_spec": modular_q_evaluation_spec(learner, shield=shield),
                "policy": ShieldedModularQPolicy(learner=learner, shield=shield),
            }
        )
    return sorted(specs, key=lambda row: (int(row["training_seed"]), str(row["arm"])))


def _public_specs(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for spec in specs:
        public_rows.append(
            {
                "label": spec["label"],
                "arm": spec["arm"],
                "role": spec["role"],
                "training_seed": spec["training_seed"],
                "artifact": spec["artifact"],
                "evaluation_spec": spec["evaluation_spec"],
            }
        )
    return public_rows


def _load_specs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    protocol = m1314r2_protocol_fingerprint()
    config = m139_config()
    specs: list[dict[str, Any]] = []
    for row in rows:
        evaluation_spec = dict(row["evaluation_spec"])
        shield_payload = dict(evaluation_spec["shield"])
        shield = ShieldConfig(**shield_payload)
        learner = load_modular_q_policy(
            row["artifact"]["path"],
            config=config,
            expected_protocol_fingerprint=protocol,
        )
        if evaluation_spec["weight_fingerprint"] != learner.weight_fingerprint():
            raise ValueError("M13.14 loaded artifact disagrees with stored evaluation spec")
        specs.append(
            {
                "label": row["label"],
                "arm": row["arm"],
                "role": row["role"],
                "training_seed": int(row["training_seed"]),
                "artifact": row["artifact"],
                "evaluation_spec": evaluation_spec,
                "policy": ShieldedModularQPolicy(learner=learner, shield=shield),
            }
        )
    return sorted(specs, key=lambda row: (int(row["training_seed"]), str(row["arm"])))


def _evaluate_one(
    spec: Mapping[str, Any],
    *,
    conditions: Mapping[str, Mapping[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    label = str(spec["label"])
    policy = spec["policy"]
    training_seed = int(spec["training_seed"])
    arm = str(spec["arm"])
    evaluation_spec = dict(spec["evaluation_spec"])
    rows: list[dict[str, Any]] = []
    for condition, controls in conditions.items():
        for env_seed in seeds:
            trace = trace_dir / label / condition / f"seed-{env_seed}.jsonl"
            episode = run_m139_episode(
                policy,
                arm="public_potential_candidate",
                training_seed=training_seed,
                seed=env_seed,
                condition=condition,
                controls=dict(controls),
                trace_path=trace,
                policy_label=f"m1314r2/{label}",
                policy_fingerprint=None,
            )
            replay_m139_trace(trace, policy)
            rows.append(
                {
                    "label": label,
                    "arm": arm,
                    "role": str(spec["role"]),
                    "training_seed": training_seed,
                    "condition": condition,
                    "env_seed": int(env_seed),
                    "survived": bool(episode.survived),
                    "maintenance_complete": bool(episode.maintenance_complete),
                    "full_objective_success": bool(episode.full_gate_success),
                    "decision_safe_fraction": float(episode.decision_safe_fraction),
                    "duration_safe_fraction": float(episode.duration_safe_fraction),
                    "recovery_required": bool(episode.recovery_required),
                    "recovery_complete": bool(episode.recovery_complete),
                    "unsafe_wait_decisions": int(episode.unsafe_wait_decisions),
                    "wait_decisions": int(episode.wait_decisions),
                    "conformance_violations": int(sum(episode.interventions.values())),
                    "replay_violations": 0,
                    "shield_enabled": bool(evaluation_spec["shield"]["enabled"]),
                    "policy_fingerprint": str(evaluation_spec["policy_fingerprint"]),
                    "weight_fingerprint": str(evaluation_spec["weight_fingerprint"]),
                    "parameter_bytes_sha256": str(evaluation_spec["parameter_bytes_sha256"]),
                    "weight_bytes_sha256": str(evaluation_spec["weight_bytes_sha256"]),
                    "artifact_sha256": str(spec["artifact"]["sha256"]),
                    "trace_sha256": _file_sha256(trace),
                }
            )
    return rows, len(rows)


def _evaluate_specs(
    specs: Iterable[dict[str, Any]],
    *,
    conditions: Mapping[str, Mapping[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    evidence: list[dict[str, Any]] = []
    replay_count = 0
    for spec in specs:
        rows, count = _evaluate_one(spec, conditions=conditions, seeds=seeds, trace_dir=trace_dir)
        evidence.extend(rows)
        replay_count += count
    return evidence, replay_count


def _hard_condition_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("M13.14-r2 hard-condition gate requires at least one row")
    wait = sum(int(row["wait_decisions"]) for row in rows)
    unsafe = sum(int(row["unsafe_wait_decisions"]) for row in rows)
    recovery = [row for row in rows if bool(row["recovery_required"])]
    condition_names = {str(row["condition"]) for row in rows}
    if len(condition_names) != 1:
        raise ValueError("M13.14-r2 hard-condition gate rows must belong to exactly one condition")
    condition_name = next(iter(condition_names))
    recovery_expected = condition_name in M1314R2_RECOVERY_REQUIRED_CONDITIONS
    recovery_requirement_satisfied = (
        len(recovery) == len(rows) and sum(bool(row["recovery_complete"]) for row in recovery) >= 18
        if recovery_expected
        else len(recovery) == 0
    )
    result = {
        "episodes": len(rows),
        "condition": condition_name,
        "survival": sum(bool(row["survived"]) for row in rows),
        "maintenance": sum(bool(row["maintenance_complete"]) for row in rows),
        "full_objective": sum(bool(row["full_objective_success"]) for row in rows),
        "mean_decision_safe": float(np.mean([float(row["decision_safe_fraction"]) for row in rows])),
        "mean_duration_safe": float(np.mean([float(row["duration_safe_fraction"]) for row in rows])),
        "recovery_complete": sum(bool(row["recovery_complete"]) for row in recovery),
        "recovery_required": len(recovery),
        "recovery_required_expected": recovery_expected,
        "recovery_requirement_satisfied": recovery_requirement_satisfied,
        "unsafe_wait_fraction": unsafe / wait if wait else 0.0,
        "integrity_violations": sum(int(row["conformance_violations"]) for row in rows),
        "replay_violations": sum(int(row["replay_violations"]) for row in rows),
    }
    result["passes"] = bool(
        result["episodes"] == 20
        and result["survival"] >= 18
        and result["maintenance"] >= 18
        and result["full_objective"] >= 18
        and result["mean_decision_safe"] >= 0.85
        and result["mean_duration_safe"] >= 0.85
        and result["recovery_requirement_satisfied"]
        and result["unsafe_wait_fraction"] <= 0.10
        and result["integrity_violations"] == 0
        and result["replay_violations"] == 0
    )
    return result


def _legacy_path_detected(path: object) -> bool:
    if not isinstance(path, (str, Path)):
        return False
    resolved = Path(path)
    return any(
        "m1313" in part.lower() or ("m1314" in part.lower() and "r2" not in part.lower())
        for part in resolved.parts
    )


def _legacy_teacher_or_artifact_reuse_detected(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for key in ("teacher_dataset_path", "artifact_path"):
            if _legacy_path_detected(row.get(key)):
                return True
    return False


def _arm_replica_gate(
    evidence: list[dict[str, Any]],
    *,
    arm: str,
    training_seed: int,
    conditions: Mapping[str, Mapping[str, Any]],
    expected_env_seeds: tuple[int, ...],
) -> dict[str, Any]:
    rows = [
        row
        for row in evidence
        if row.get("arm") == arm and int(row["training_seed"]) == training_seed
    ]
    if len(rows) != len(conditions) * len(expected_env_seeds):
        raise ValueError("M13.14 arm replica gate requires the full declared evaluation matrix")
    expected_cells = {
        (condition, env_seed)
        for condition in conditions
        for env_seed in expected_env_seeds
    }
    observed_cells = {(str(row["condition"]), int(row["env_seed"])) for row in rows}
    if observed_cells != expected_cells:
        raise ValueError("M13.14 arm replica evidence does not cover the declared matrix")
    by_condition = {
        condition: _hard_condition_gate(
            [row for row in rows if str(row["condition"]) == condition]
        )
        for condition in conditions
    }
    weight_fingerprints = {str(row["weight_fingerprint"]) for row in rows}
    artifact_hashes = {str(row["artifact_sha256"]) for row in rows}
    policy_fingerprints = {str(row["policy_fingerprint"]) for row in rows}
    legacy_reuse = _legacy_teacher_or_artifact_reuse_detected(rows)
    return {
        "arm": arm,
        "training_seed": training_seed,
        "role": M1314R2_ARM_ROLES[ModularQArm(arm)],
        "conditions": by_condition,
        "full_objective_rate": sum(bool(row["full_objective_success"]) for row in rows) / len(rows),
        "mean_decision_safe": float(np.mean([float(row["decision_safe_fraction"]) for row in rows])),
        "mean_duration_safe": float(np.mean([float(row["duration_safe_fraction"]) for row in rows])),
        "weight_fingerprints": sorted(weight_fingerprints),
        "policy_fingerprints": sorted(policy_fingerprints),
        "artifact_hashes": sorted(artifact_hashes),
        "legacy_teacher_or_artifact_reuse_detected": legacy_reuse,
        "hard_gate_passes": (not legacy_reuse) and all(bool(summary["passes"]) for summary in by_condition.values()),
    }


def _hard_gate_for_arm(
    evidence: list[dict[str, Any]],
    *,
    arm: str,
    training_seed: int,
    conditions: Mapping[str, Mapping[str, Any]],
    expected_env_seeds: tuple[int, ...],
) -> dict[str, Any]:
    gate = _arm_replica_gate(
        evidence,
        arm=arm,
        training_seed=training_seed,
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
    )
    return {**gate, "passes": bool(gate["hard_gate_passes"])}


def _legacy_condition_passes(rows: list[dict[str, Any]]) -> bool:
    wait = sum(int(row["wait_decisions"]) for row in rows)
    unsafe = sum(int(row["unsafe_wait_decisions"]) for row in rows)
    recovery = [row for row in rows if bool(row["recovery_required"])]
    return bool(
        len(rows) == 20
        and sum(bool(row["survived"]) for row in rows) >= 18
        and sum(bool(row["maintenance_complete"]) for row in rows) >= 18
        and sum(bool(row["full_objective_success"]) for row in rows) >= 18
        and float(np.mean([float(row["decision_safe_fraction"]) for row in rows])) >= 0.85
        and float(np.mean([float(row["duration_safe_fraction"]) for row in rows])) >= 0.85
        and (len(recovery) == 0 or sum(bool(row["recovery_complete"]) for row in recovery) >= 18)
        and (unsafe / wait if wait else 0.0) <= 0.10
        and sum(int(row["conformance_violations"]) for row in rows) == 0
        and sum(int(row["replay_violations"]) for row in rows) == 0
    )


def _legacy_arm_comparison_passes(
    evidence: list[dict[str, Any]],
    *,
    arm: str,
    training_seed: int,
    conditions: Mapping[str, Mapping[str, Any]],
) -> bool:
    return all(
        _legacy_condition_passes(
            [
                row
                for row in evidence
                if row.get("arm") == arm
                and int(row["training_seed"]) == training_seed
                and str(row["condition"]) == condition
            ]
        )
        for condition in conditions
    )


def _comparison_gate(
    evidence: list[dict[str, Any]],
    *,
    training_seed: int,
    comparison_key: str,
    conditions: Mapping[str, Mapping[str, Any]],
    expected_env_seeds: tuple[int, ...],
) -> dict[str, Any]:
    spec = dict(M1314R2_CAUSAL_COMPARISONS[comparison_key])
    candidate_arm = str(spec["candidate"])
    control_arm = str(spec["control"])
    candidate = _arm_replica_gate(
        evidence,
        arm=candidate_arm,
        training_seed=training_seed,
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
    )
    control = _arm_replica_gate(
        evidence,
        arm=control_arm,
        training_seed=training_seed,
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
    )
    candidate_rows = [
        row
        for row in evidence
        if row.get("arm") == candidate_arm and int(row["training_seed"]) == training_seed
    ]
    control_rows = [
        row
        for row in evidence
        if row.get("arm") == control_arm and int(row["training_seed"]) == training_seed
    ]
    candidate_legacy_passes = _legacy_arm_comparison_passes(
        evidence,
        arm=candidate_arm,
        training_seed=training_seed,
        conditions=conditions,
    )
    control_legacy_passes = _legacy_arm_comparison_passes(
        evidence,
        arm=control_arm,
        training_seed=training_seed,
        conditions=conditions,
    )
    per_condition_non_regression = all(
        sum(bool(row["full_objective_success"]) for row in candidate_rows if row["condition"] == condition)
        >= sum(bool(row["full_objective_success"]) for row in control_rows if row["condition"] == condition)
        - int(spec["maximum_per_condition_success_regression_episodes"])
        for condition in conditions
    )
    candidate_decision = candidate["mean_decision_safe"]
    control_decision = control["mean_decision_safe"]
    candidate_duration = candidate["mean_duration_safe"]
    control_duration = control["mean_duration_safe"]
    parameter_identity = True
    if bool(spec.get("parameter_fingerprint_must_match", False)):
        parameter_identity = (
            len(candidate["weight_fingerprints"]) == 1
            and candidate["weight_fingerprints"] == control["weight_fingerprints"]
            and candidate["artifact_hashes"] == control["artifact_hashes"]
        )
    passes = bool(
        candidate_legacy_passes
        and candidate["full_objective_rate"] - control["full_objective_rate"]
        >= float(spec["minimum_pooled_full_objective_advantage"])
        and candidate_decision >= control_decision - float(spec["maximum_decision_or_duration_safety_regression"])
        and candidate_duration >= control_duration - float(spec["maximum_decision_or_duration_safety_regression"])
        and per_condition_non_regression
        and parameter_identity
    )
    promotion_passes = bool(
        candidate["hard_gate_passes"]
        and candidate["full_objective_rate"] - control["full_objective_rate"]
        >= float(spec["minimum_pooled_full_objective_advantage"])
        and candidate_decision >= control_decision - float(spec["maximum_decision_or_duration_safety_regression"])
        and candidate_duration >= control_duration - float(spec["maximum_decision_or_duration_safety_regression"])
        and per_condition_non_regression
        and parameter_identity
    )
    return {
        "comparison": comparison_key,
        "training_seed": training_seed,
        "candidate": candidate_arm,
        "control": control_arm,
        "candidate_hard_gate_passes": bool(candidate["hard_gate_passes"]),
        "control_hard_gate_passes": bool(control["hard_gate_passes"]),
        "candidate_passes": candidate_legacy_passes,
        "control_passes": control_legacy_passes,
        "candidate_full_objective_rate": candidate["full_objective_rate"],
        "control_full_objective_rate": control["full_objective_rate"],
        "full_objective_advantage": candidate["full_objective_rate"] - control["full_objective_rate"],
        "required_advantage": float(spec["minimum_pooled_full_objective_advantage"]),
        "decision_safe_delta": candidate_decision - control_decision,
        "duration_safe_delta": candidate_duration - control_duration,
        "maximum_safety_regression": float(spec["maximum_decision_or_duration_safety_regression"]),
        "per_condition_non_regression": per_condition_non_regression,
        "parameter_identity_required": bool(spec.get("parameter_fingerprint_must_match", False)),
        "parameter_identity": parameter_identity,
        "passes": passes,
        "promotion_passes": promotion_passes,
    }


def m1314r2_gate(
    evidence: list[dict[str, Any]],
    *,
    training_seeds: tuple[int, ...],
    conditions: Mapping[str, Mapping[str, Any]],
    expected_env_seeds: tuple[int, ...],
    expected_arms: tuple[ModularQArm, ...] | None = None,
) -> dict[str, Any]:
    if expected_arms is None:
        is_audit = (
            tuple(int(seed) for seed in training_seeds) == tuple(int(seed) for seed in M1314R2_CONFIRMATION_TRAINING_SEEDS)
            and tuple(str(name) for name in conditions) == tuple(str(name) for name in M1314R2_CONDITIONS["audit"])
            and tuple(int(seed) for seed in expected_env_seeds) == tuple(int(seed) for seed in M1314R2_AUDIT_SEEDS)
        )
        expected_arms = (
            (ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,)
            if is_audit
            else M1314R2_ALL_ARMS
        )
    replicas: list[dict[str, Any]] = []
    for training_seed in training_seeds:
        arm_gates = {
            arm.value: _arm_replica_gate(
                evidence,
                arm=arm.value,
                training_seed=training_seed,
                conditions=conditions,
                expected_env_seeds=expected_env_seeds,
            )
            for arm in expected_arms
        }
        comparisons = (
            {
                key: _comparison_gate(
                    evidence,
                    training_seed=training_seed,
                    comparison_key=key,
                    conditions=conditions,
                    expected_env_seeds=expected_env_seeds,
                )
                for key in M1314R2_CAUSAL_COMPARISONS
            }
            if ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL in expected_arms
            else {}
        )
        replicas.append(
            {
                "training_seed": training_seed,
                "arms": arm_gates,
                "comparisons": comparisons,
                "candidate_hard_gate_passes": bool(
                    arm_gates[ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value]["hard_gate_passes"]
                ),
                "passes": bool(
                    arm_gates[ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value]["hard_gate_passes"]
                    and all(bool(row["promotion_passes"]) for row in comparisons.values())
                ),
            }
        )
    return {
        "replicas": replicas,
        "cross_arm_selection_performed": False,
        "passes": len(replicas) == len(training_seeds) and all(bool(row["passes"]) for row in replicas),
    }


def _run_evaluation_stage(
    *,
    stage: str,
    specs: list[dict[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    evidence, replay_count = _evaluate_specs(specs, conditions=conditions, seeds=seeds, trace_dir=trace_dir)
    oracle_rows, oracle_count = _evaluate_one(
        {
            "label": f"{stage}_scripted_ceiling",
            "arm": "scripted_ceiling",
            "role": "scripted_ceiling",
            "training_seed": 0,
            "artifact": {"sha256": "none"},
            "evaluation_spec": {
                "shield": {"enabled": True},
                "policy_fingerprint": "scripted",
                "weight_fingerprint": "scripted",
                "parameter_bytes_sha256": "scripted",
                "weight_bytes_sha256": "scripted",
            },
            "policy": BalancedM139Oracle(),
        },
        conditions=conditions,
        seeds=seeds,
        trace_dir=trace_dir,
    )
    random_seed = M1314R2_RANDOM_SEEDS[stage]
    random_rows, random_count = _evaluate_one(
        {
            "label": f"{stage}_mask_random",
            "arm": "mask_random",
            "role": "mask_random",
            "training_seed": random_seed,
            "artifact": {"sha256": "pcg64"},
            "evaluation_spec": {
                "shield": {"enabled": False},
                "policy_fingerprint": f"pcg64-{random_seed}",
                "weight_fingerprint": f"pcg64-{random_seed}",
                "parameter_bytes_sha256": f"pcg64-{random_seed}",
                "weight_bytes_sha256": f"pcg64-{random_seed}",
            },
            "policy": SeededRandomM139Policy(random_seed),
        },
        conditions=conditions,
        seeds=seeds,
        trace_dir=trace_dir,
    )
    diagnostics = {
        "scripted_ceiling_full_objective": sum(bool(row["full_objective_success"]) for row in oracle_rows),
        "mask_random_full_objective": sum(bool(row["full_objective_success"]) for row in random_rows),
        "episodes_each": len(oracle_rows),
    }
    return (
        evidence + oracle_rows + random_rows,
        replay_count + oracle_count + random_count,
        diagnostics,
    )


def m1314r2_development(
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1314R2_CANONICAL_LEDGER_PATH,
    workers: int | None = None,
) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    _assert_layouts()
    verify_m1314r2_preflight_report(
        _read_json_object(preflight_report_path, label="M13.14 preflight report"),
        expected_protocol_fingerprint=m1314r2_protocol_fingerprint(),
        expected_manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        expected_stage="development_preflight",
        expected_conditions=tuple(M1314R2_CONDITIONS["development_fit"]),
        expected_episode_counts={
            "development_fit": len(M1314R2_DEVELOPMENT_FIT_SEEDS),
            "development_check": len(M1314R2_DEVELOPMENT_CHECK_SEEDS),
        },
        require_pass=True,
    )
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.14 development artifact directory already exists: {root}")
    protocol = m1314r2_protocol_fingerprint()
    ledger = M1314R2StageLedger(ledger_path, protocol_fingerprint=protocol)
    started = time.perf_counter()
    ledger.open("development_preflight")
    ledger.open("development_fit", resumable_fit=True)
    root.mkdir(parents=True)
    teacher_summary, _ = _collect_teacher_dataset(
        conditions=M1314R2_CONDITIONS["development_fit"],
        env_seeds=M1314R2_DEVELOPMENT_FIT_SEEDS,
        dataset_path=root / "teacher" / "development-fit-teacher.npz",
    )
    jobs = _fit_wave(
        training_seeds=M1314R2_DEVELOPMENT_TRAINING_SEEDS,
        conditions=M1314R2_CONDITIONS["development_fit"],
        env_seeds=M1314R2_DEVELOPMENT_FIT_SEEDS,
        dataset_path=Path(teacher_summary["artifact"]["path"]),
        artifact_dir=root / "policies",
        workers=workers,
    )
    specs = _evaluation_specs_from_fit_jobs(jobs)
    ledger.open("development_check")
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="development",
        specs=specs,
        conditions=M1314R2_CONDITIONS["development_check"],
        seeds=M1314R2_DEVELOPMENT_CHECK_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1314r2_gate(
        evidence,
        training_seeds=M1314R2_DEVELOPMENT_TRAINING_SEEDS,
        conditions=M1314R2_CONDITIONS["development_check"],
        expected_env_seeds=M1314R2_DEVELOPMENT_CHECK_SEEDS,
    )
    return _seal_stage_report(
        {
            "schema_version": M1314R2_DEVELOPMENT_REPORT_SCHEMA_VERSION,
            "stage": "development",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": str(manifest["protocol_fingerprint"]),
            "preflight_report_sha256": _file_sha256(preflight_report_path),
            "elapsed_seconds": time.perf_counter() - started,
            "teacher_dataset": teacher_summary,
            "fit_jobs": jobs,
            "policy_artifacts": _public_specs(specs),
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": ["confirmation_preflight", "confirmation_fit", "confirmation_evaluation", "audit_preflight", "audit"],
            "limits": [
                "No cross-arm winner selection is performed.",
                "A passing development gate only authorizes a separately frozen confirmation stage.",
            ],
        }
    )


def m1314r2_confirmation(
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
    development_report_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1314R2_CANONICAL_LEDGER_PATH,
    workers: int | None = None,
) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    development = _verify_stage_report(
        development_report_path,
        stage="development",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
    )
    if not bool(development.get("gate", {}).get("passes", False)):
        raise ValueError("M13.14 confirmation requires a passing development report")
    _assert_layouts()
    verify_m1314r2_preflight_report(
        _read_json_object(preflight_report_path, label="M13.14 preflight report"),
        expected_protocol_fingerprint=m1314r2_protocol_fingerprint(),
        expected_manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        expected_stage="confirmation_preflight",
        expected_conditions=tuple(M1314R2_CONDITIONS["confirmation_fit"]),
        expected_episode_counts={
            "confirmation_fit": len(M1314R2_CONFIRMATION_FIT_SEEDS),
            "confirmation_evaluation": len(M1314R2_CONFIRMATION_EVALUATION_SEEDS),
        },
        require_pass=True,
    )
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.14 confirmation artifact directory already exists: {root}")
    protocol = m1314r2_protocol_fingerprint()
    ledger = M1314R2StageLedger(ledger_path, protocol_fingerprint=protocol)
    started = time.perf_counter()
    ledger.open("confirmation_preflight")
    ledger.open("confirmation_fit", resumable_fit=True)
    root.mkdir(parents=True)
    teacher_summary, _ = _collect_teacher_dataset(
        conditions=M1314R2_CONDITIONS["confirmation_fit"],
        env_seeds=M1314R2_CONFIRMATION_FIT_SEEDS,
        dataset_path=root / "teacher" / "confirmation-fit-teacher.npz",
    )
    jobs = _fit_wave(
        training_seeds=M1314R2_CONFIRMATION_TRAINING_SEEDS,
        conditions=M1314R2_CONDITIONS["confirmation_fit"],
        env_seeds=M1314R2_CONFIRMATION_FIT_SEEDS,
        dataset_path=Path(teacher_summary["artifact"]["path"]),
        artifact_dir=root / "policies",
        workers=workers,
    )
    specs = _evaluation_specs_from_fit_jobs(jobs)
    ledger.open("confirmation_evaluation")
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="confirmation",
        specs=specs,
        conditions=M1314R2_CONDITIONS["confirmation_evaluation"],
        seeds=M1314R2_CONFIRMATION_EVALUATION_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1314r2_gate(
        evidence,
        training_seeds=M1314R2_CONFIRMATION_TRAINING_SEEDS,
        conditions=M1314R2_CONDITIONS["confirmation_evaluation"],
        expected_env_seeds=M1314R2_CONFIRMATION_EVALUATION_SEEDS,
    )
    return _seal_stage_report(
        {
            "schema_version": M1314R2_CONFIRMATION_REPORT_SCHEMA_VERSION,
            "stage": "confirmation",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": str(manifest["protocol_fingerprint"]),
            "development_report_sha256": _file_sha256(development_report_path),
            "preflight_report_sha256": _file_sha256(preflight_report_path),
            "elapsed_seconds": time.perf_counter() - started,
            "teacher_dataset": teacher_summary,
            "fit_jobs": jobs,
            "policy_artifacts": _public_specs(specs),
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": ["audit_preflight", "audit"],
            "limits": [
                "No cross-arm winner selection is performed.",
                "Confirmation is one separately frozen stage with no replacement seeds.",
            ],
        }
    )


def m1314r2_audit(
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
    confirmation_report_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1314R2_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    confirmation = _verify_stage_report(
        confirmation_report_path,
        stage="confirmation",
        manifest_fingerprint=str(manifest["protocol_fingerprint"]),
    )
    if not bool(confirmation.get("gate", {}).get("passes", False)):
        raise ValueError("M13.14 audit requires a passing confirmation report")
    _assert_layouts()
    verify_m1314r2_preflight_report(
        _read_json_object(preflight_report_path, label="M13.14 preflight report"),
        expected_protocol_fingerprint=m1314r2_protocol_fingerprint(),
        expected_manifest_fingerprint=str(manifest["protocol_fingerprint"]),
        expected_stage="audit_preflight",
        expected_conditions=tuple(M1314R2_CONDITIONS["audit"]),
        expected_episode_counts={"audit": len(M1314R2_AUDIT_SEEDS)},
        require_pass=True,
    )
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.14 audit artifact directory already exists: {root}")
    protocol = m1314r2_protocol_fingerprint()
    ledger = M1314R2StageLedger(ledger_path, protocol_fingerprint=protocol)
    started = time.perf_counter()
    ledger.open("audit_preflight")
    ledger.open("audit")
    root.mkdir(parents=True)
    specs = _load_specs(confirmation["policy_artifacts"])
    specs = [
        spec
        for spec in specs
        if str(spec["arm"]) == ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value
    ]
    audit_policy_artifacts = [
        row
        for row in confirmation["policy_artifacts"]
        if str(row["arm"]) == ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value
    ]
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="audit",
        specs=specs,
        conditions=M1314R2_CONDITIONS["audit"],
        seeds=M1314R2_AUDIT_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1314r2_gate(
        evidence,
        training_seeds=tuple(int(seed) for seed in M1314R2_CONFIRMATION_TRAINING_SEEDS),
        conditions=M1314R2_CONDITIONS["audit"],
        expected_env_seeds=M1314R2_AUDIT_SEEDS,
        expected_arms=(ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,),
    )
    return _seal_stage_report(
        {
            "schema_version": M1314R2_AUDIT_REPORT_SCHEMA_VERSION,
            "stage": "audit",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": str(manifest["protocol_fingerprint"]),
            "confirmation_report_sha256": _file_sha256(confirmation_report_path),
            "preflight_report_sha256": _file_sha256(preflight_report_path),
            "elapsed_seconds": time.perf_counter() - started,
            "policy_artifacts": audit_policy_artifacts,
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": [],
            "limits": ["Audit is one-shot. No retry, replacement seed, or post-audit tuning is allowed."],
        }
    )


def replay_m1314r2_trace(
    *,
    manifest_path: str | Path,
    trace_path: str | Path,
    policy_path: str | Path,
) -> ReplayResult:
    manifest = verify_m1314r2_policy_manifest(manifest_path)
    del manifest
    trace = Path(trace_path)
    try:
        first_line = trace.read_text(encoding="utf-8").splitlines()[0]
        header = json.loads(first_line)
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("M13.14 replay trace header is unreadable") from exc
    label = str(header.get("m1314r2_policy_label", header.get("m139_policy_label", "")))
    shield_enabled = "modular_anchor_no_shield_control" not in label
    learner = load_modular_q_policy_m1314r2(
        policy_path,
        config=m139_config(),
        expected_protocol_fingerprint=m1314r2_protocol_fingerprint(),
    )
    policy = ShieldedModularQPolicyM1314R2(
        learner=learner,
        shield=ShieldConfig(enabled=shield_enabled),
    )
    return replay_m139_trace(trace_path, policy)


def write_m1314r2_development_preflight_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_development_preflight(manifest_path=manifest_path)
    _write_json_exclusive(destination, report)
    return report


def write_m1314r2_confirmation_preflight_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    development_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_confirmation_preflight(
        manifest_path=manifest_path,
        development_report_path=development_report_path,
    )
    _write_json_exclusive(destination, report)
    return report


def write_m1314r2_audit_preflight_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    confirmation_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_audit_preflight(
        manifest_path=manifest_path,
        confirmation_report_path=confirmation_report_path,
    )
    _write_json_exclusive(destination, report)
    return report


def write_m1314r2_development_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_development(
        manifest_path=manifest_path,
        preflight_report_path=preflight_report_path,
        artifact_dir=destination.parent / f"{destination.stem}-artifacts",
    )
    _write_json_exclusive(destination, report)
    return report


def write_m1314r2_confirmation_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
    development_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_confirmation(
        manifest_path=manifest_path,
        preflight_report_path=preflight_report_path,
        development_report_path=development_report_path,
        artifact_dir=destination.parent / f"{destination.stem}-artifacts",
    )
    _write_json_exclusive(destination, report)
    return report


def write_m1314r2_audit_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    preflight_report_path: str | Path,
    confirmation_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.14 output already exists: {destination}")
    report = m1314r2_audit(
        manifest_path=manifest_path,
        preflight_report_path=preflight_report_path,
        confirmation_report_path=confirmation_report_path,
        artifact_dir=destination.parent / f"{destination.stem}-artifacts",
    )
    _write_json_exclusive(destination, report)
    return report


__all__ = [
    "M1314R2_AUDIT_REPORT_SCHEMA_VERSION",
    "M1314R2_CANONICAL_LEDGER_PATH",
    "M1314R2_CONFIRMATION_REPORT_SCHEMA_VERSION",
    "M1314R2_DEVELOPMENT_REPORT_SCHEMA_VERSION",
    "m1314r2_audit",
    "m1314r2_audit_preflight",
    "m1314r2_confirmation",
    "m1314r2_confirmation_preflight",
    "m1314r2_development",
    "m1314r2_development_preflight",
    "m1314r2_gate",
    "replay_m1314r2_trace",
    "write_m1314r2_audit_preflight_report",
    "write_m1314r2_audit_report",
    "write_m1314r2_confirmation_preflight_report",
    "write_m1314r2_confirmation_report",
    "write_m1314r2_development_preflight_report",
    "write_m1314r2_development_report",
]
