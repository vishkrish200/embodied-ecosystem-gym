"""M13.13: unopened comparison of three distinct public policy families.

This module implements the future runner but performs no work at import time.
Every public entry point requires the byte-exact manifest and a one-way ledger.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..env import EcosystemEnv
from ..maintenance import (
    MaintenanceMacro,
    MaintenanceRewardState,
    ModelBasedSchedulerConfig,
    ModelBasedSchedulerPolicy,
    PublicMLPActor,
    ShieldConfig,
    ShieldedLearnedPolicy,
    UrgencySchedulerConfig,
    UrgencySchedulerPolicy,
    advance_memory,
    compile_macro,
    eligible_macro_mask,
    encode_features,
    learning_reward,
    load_policy,
    save_policy,
)
from ..maintenance.policy_protocol import (
    M1313_AUDIT_SEEDS,
    M1313_CONDITIONS,
    M1313_CONFIRMATION_EVALUATION_SEEDS,
    M1313_CONFIRMATION_FIT_SEEDS,
    M1313_CONFIRMATION_TRAINING_SEEDS,
    M1313_DEVELOPMENT_CHECK_SEEDS,
    M1313_DEVELOPMENT_FIT_SEEDS,
    M1313_DEVELOPMENT_TRAINING_SEEDS,
    M1313_PPO_DECISION_BUDGET,
    m1313_protocol_fingerprint,
    verify_policy_family_manifest,
)
from ..maintenance.ppo_r2 import CorrectedMaskedPPOPolicy, Rollout
from ..tasks import LAYOUTS
from .m10 import m10_scan_coverage
from .m13 import _options
from .m139 import (
    BalancedM139Oracle,
    SeededRandomM139Policy,
    m139_config,
    replay_m139_trace,
    run_m139_episode,
)
from .m1313_support import M1313SplitLedger, content_hash


for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"


M1313_CANONICAL_LEDGER_PATH = Path(__file__).resolve().parents[2] / "artifacts/m1313/split-open-ledger.json"
M1313_PPO_ROLLOUT = 2048
M1313_PPO_EPOCHS = 4
M1313_PPO_LEARNING_RATE = 3e-4
M1313_RANDOM_SEEDS = {"development": 20261399, "confirmation": 20261400, "audit": 20261401}


def _assert_layouts() -> None:
    names = {name for name in LAYOUTS if name.startswith("m1313_")}
    expected = {
        str(row["layout_id"])
        for conditions in M1313_CONDITIONS.values()
        for row in conditions.values()
    }
    if names != expected or len(names) != 20:
        raise ValueError("M13.13 layouts are incomplete")
    points = lambda layout: (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)
    current = {point for name in names for point in points(LAYOUTS[name])}
    previous = {
        point
        for name, layout in LAYOUTS.items()
        if not name.startswith("m1313_")
        for point in points(layout)
    }
    if len(current) != 100 or current & previous:
        raise ValueError("M13.13 geometry is not fresh")


def _rollout(rows: list[tuple[Any, ...]]) -> Rollout:
    dtypes = (np.float32, np.int64, np.bool_, np.float32, np.float32, np.float32, np.bool_, np.float32, np.float32)
    return Rollout(
        *(np.asarray([row[index] for row in rows], dtype=dtype) for index, dtype in enumerate(dtypes))
    )


def _summarize_updates(updates: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
    )
    summary = {
        key: float(np.mean([float(row[key]) for row in updates])) if updates else 0.0
        for key in scalar_keys
    }
    histogram = {str(index): 0 for index in range(len(MaintenanceMacro))}
    for row in updates:
        for index, value in row.get("action_histogram", {}).items():
            histogram[str(index)] += int(value)
    return {**summary, "action_histogram": histogram}


def _train_actor(
    seed: int,
    *,
    conditions: dict[str, dict[str, Any]],
    env_seeds: tuple[int, ...],
    decision_budget: int = M1313_PPO_DECISION_BUDGET,
) -> tuple[CorrectedMaskedPPOPolicy, dict[str, Any]]:
    """The unchanged corrected 3e-4 PPO actor on a declared M13.13 fit split."""

    policy = CorrectedMaskedPPOPolicy(
        seed=seed,
        config=m139_config(),
        protocol_fingerprint=m1313_protocol_fingerprint(),
    )
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x4D31333133))
    env = EcosystemEnv(policy.config)
    rows: list[tuple[Any, ...]] = []
    updates: list[dict[str, Any]] = []
    decisions = episodes = batches = 0
    condition_items = tuple(conditions.items())
    try:
        while decisions < decision_budget:
            _, controls = condition_items[episodes % len(condition_items)]
            env_seed = env_seeds[(episodes // len(condition_items)) % len(env_seeds)]
            observation, _ = env.reset(seed=env_seed, options=_options(controls))
            memory = policy.reset()
            reward_state = MaintenanceRewardState(
                required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
            )
            episodes += 1
            while True:
                features = encode_features(observation, memory)
                mask = eligible_macro_mask(observation, policy.config)
                action_index, log_probability, value = policy.sample(features, mask, rng)
                macro = MaintenanceMacro(action_index)
                action = compile_macro(macro, observation, policy.config)
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
                advance_memory(
                    memory,
                    observation_before=observation,
                    macro=macro,
                    action=action,
                    observation_after=next_observation,
                    config=policy.config,
                )
                next_features = encode_features(next_observation, memory)
                rows.append(
                    (
                        features,
                        action_index,
                        mask,
                        log_probability,
                        reward,
                        float(np.asarray(action["duration"]).item()),
                        terminated or truncated,
                        value,
                        policy.value(next_features),
                    )
                )
                decisions += 1
                observation = next_observation
                if len(rows) == M1313_PPO_ROLLOUT:
                    updates.append(
                        policy.update(
                            _rollout(rows),
                            epochs=M1313_PPO_EPOCHS,
                            learning_rate=M1313_PPO_LEARNING_RATE,
                        )
                    )
                    batches += 1
                    rows.clear()
                if terminated or truncated:
                    break
    finally:
        env.close()
    if rows:
        updates.append(
            policy.update(
                _rollout(rows),
                epochs=M1313_PPO_EPOCHS,
                learning_rate=M1313_PPO_LEARNING_RATE,
            )
        )
        batches += 1
    return policy, {
        "learning_rate": M1313_PPO_LEARNING_RATE,
        "decision_budget": decision_budget,
        "decisions": decisions,
        "overshoot": max(0, decisions - decision_budget),
        "episodes": episodes,
        "update_batches": batches,
        "update_epochs": batches * M1313_PPO_EPOCHS,
        "diagnostics": _summarize_updates(updates),
    }


def _fit_worker(
    seed: int,
    conditions: dict[str, dict[str, Any]],
    env_seeds: tuple[int, ...],
    artifact_dir: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    policy, training = _train_actor(seed, conditions=conditions, env_seeds=env_seeds)
    artifact = policy.save(Path(artifact_dir) / f"ppo-source-seed-{seed}.json")
    return {
        "training_seed": seed,
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
        "training": training,
        "source_actor_artifact": artifact,
    }


def _fit_wave(
    *,
    training_seeds: tuple[int, ...],
    conditions: dict[str, dict[str, Any]],
    env_seeds: tuple[int, ...],
    artifact_dir: Path,
    workers: int | None,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    worker_count = min(workers or min(len(training_seeds), max(1, os.cpu_count() or 1)), len(training_seeds))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(_fit_worker, seed, conditions, env_seeds, str(artifact_dir))
            for seed in training_seeds
        ]
        return sorted((future.result() for future in as_completed(futures)), key=lambda row: int(row["training_seed"]))


def _policy_spec(
    *,
    label: str,
    family: str,
    role: str,
    training_seed: int,
    policy: Any,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact = save_policy(artifact_dir / f"{label}.json", policy)
    return {
        "label": label,
        "family": family,
        "role": role,
        "training_seed": training_seed,
        "policy": policy,
        "artifact": artifact,
        "actor_fingerprint": getattr(getattr(policy, "actor", None), "fingerprint", lambda: None)(),
    }


def _deterministic_specs(artifact_dir: Path, families: set[str]) -> list[dict[str, Any]]:
    protocol = m1313_protocol_fingerprint()
    config = m139_config()
    specs: list[dict[str, Any]] = []
    if "urgency_commitment" in families:
        base = UrgencySchedulerConfig()
        specs.extend(
            (
                _policy_spec(
                    label="urgency_commitment_candidate",
                    family="urgency_commitment",
                    role="candidate",
                    training_seed=0,
                    policy=UrgencySchedulerPolicy(config=config, scheduler=base, protocol_fingerprint=protocol),
                    artifact_dir=artifact_dir,
                ),
                _policy_spec(
                    label="urgency_reactive_control",
                    family="urgency_commitment",
                    role="control",
                    training_seed=0,
                    policy=UrgencySchedulerPolicy(
                        config=config,
                        scheduler=replace(base, commitment_enabled=False),
                        protocol_fingerprint=protocol,
                    ),
                    artifact_dir=artifact_dir,
                ),
            )
        )
    if "short_horizon_model" in families:
        base = ModelBasedSchedulerConfig()
        specs.extend(
            (
                _policy_spec(
                    label="model_horizon4_candidate",
                    family="short_horizon_model",
                    role="candidate",
                    training_seed=0,
                    policy=ModelBasedSchedulerPolicy(config=config, planner=base, protocol_fingerprint=protocol),
                    artifact_dir=artifact_dir,
                ),
                _policy_spec(
                    label="model_myopic_control",
                    family="short_horizon_model",
                    role="control",
                    training_seed=0,
                    policy=ModelBasedSchedulerPolicy(
                        config=config,
                        planner=replace(base, lookahead_depth=1),
                        protocol_fingerprint=protocol,
                    ),
                    artifact_dir=artifact_dir,
                ),
            )
        )
    return specs


def _learned_specs(
    jobs: Iterable[dict[str, Any]],
    *,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    protocol = m1313_protocol_fingerprint()
    config = m139_config()
    specs: list[dict[str, Any]] = []
    for job in jobs:
        seed = int(job["training_seed"])
        source = CorrectedMaskedPPOPolicy.load(
            job["source_actor_artifact"]["path"],
            config=config,
            protocol_fingerprint=protocol,
        )
        actor = PublicMLPActor.from_corrected_ppo(source)
        specs.extend(
            (
                _policy_spec(
                    label=f"shielded_candidate_seed_{seed}",
                    family="shielded_learned",
                    role="candidate",
                    training_seed=seed,
                    policy=ShieldedLearnedPolicy(
                        actor=actor,
                        config=config,
                        shield=ShieldConfig(),
                        protocol_fingerprint=protocol,
                    ),
                    artifact_dir=artifact_dir,
                ),
                _policy_spec(
                    label=f"unshielded_control_seed_{seed}",
                    family="shielded_learned",
                    role="control",
                    training_seed=seed,
                    policy=ShieldedLearnedPolicy(
                        actor=actor,
                        config=config,
                        shield=ShieldConfig(enabled=False),
                        protocol_fingerprint=protocol,
                    ),
                    artifact_dir=artifact_dir,
                ),
            )
        )
    return specs


def _load_specs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        policy = load_policy(
            row["artifact"]["path"],
            config=m139_config(),
            expected_protocol_fingerprint=m1313_protocol_fingerprint(),
        )
        if hashlib.sha256(Path(row["artifact"]["path"]).read_bytes()).hexdigest() != row["artifact"]["sha256"]:
            raise ValueError("promoted M13.13 policy artifact hash mismatch")
        result.append({**row, "policy": policy})
    return result


def _evaluate_one(
    policy: Any,
    *,
    label: str,
    family: str,
    role: str,
    training_seed: int,
    conditions: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
    policy_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
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
                controls=controls,
                trace_path=trace,
                policy_label=f"m1313/{label}",
                policy_fingerprint=policy_fingerprint,
            )
            replay_m139_trace(trace, policy)
            rows.append(
                {
                    "label": label,
                    "family": family,
                    "role": role,
                    "training_seed": training_seed,
                    "condition": condition,
                    "env_seed": env_seed,
                    "survived": episode.survived,
                    "maintenance_complete": episode.maintenance_complete,
                    "full_objective_success": episode.full_gate_success,
                    "decision_safe_fraction": episode.decision_safe_fraction,
                    "duration_safe_fraction": episode.duration_safe_fraction,
                    "recovery_required": episode.recovery_required,
                    "recovery_complete": episode.recovery_complete,
                    "unsafe_wait_decisions": episode.unsafe_wait_decisions,
                    "wait_decisions": episode.wait_decisions,
                    "conformance_violations": int(sum(episode.interventions.values())),
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
            )
    return rows, len(rows)


def _evaluate_specs(
    specs: Iterable[dict[str, Any]],
    *,
    conditions: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    evidence: list[dict[str, Any]] = []
    replay_count = 0
    for spec in specs:
        rows, count = _evaluate_one(
            spec["policy"],
            label=spec["label"],
            family=spec["family"],
            role=spec["role"],
            training_seed=int(spec["training_seed"]),
            conditions=conditions,
            seeds=seeds,
            trace_dir=trace_dir,
        )
        evidence.extend(rows)
        replay_count += count
    return evidence, replay_count


def _hard_condition_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wait = sum(int(row["wait_decisions"]) for row in rows)
    unsafe = sum(int(row["unsafe_wait_decisions"]) for row in rows)
    recovery = [row for row in rows if bool(row["recovery_required"])]
    result = {
        "episodes": len(rows),
        "survival": sum(bool(row["survived"]) for row in rows),
        "maintenance": sum(bool(row["maintenance_complete"]) for row in rows),
        "full_objective": sum(bool(row["full_objective_success"]) for row in rows),
        "mean_decision_safe": float(np.mean([float(row["decision_safe_fraction"]) for row in rows])),
        "mean_duration_safe": float(np.mean([float(row["duration_safe_fraction"]) for row in rows])),
        "recovery_complete": sum(bool(row["recovery_complete"]) for row in recovery),
        "recovery_required": len(recovery),
        "unsafe_wait_fraction": unsafe / wait if wait else 0.0,
        "integrity_violations": sum(int(row["conformance_violations"]) for row in rows),
    }
    result["passes"] = bool(
        result["episodes"] == 20
        and result["survival"] >= 18
        and result["maintenance"] >= 18
        and result["full_objective"] >= 18
        and result["mean_decision_safe"] >= 0.85
        and result["mean_duration_safe"] >= 0.85
        and (result["recovery_required"] == 0 or result["recovery_complete"] >= 18)
        and result["unsafe_wait_fraction"] <= 0.10
        and result["integrity_violations"] == 0
    )
    return result


def _pair_gate(
    evidence: list[dict[str, Any]],
    *,
    family: str,
    training_seed: int,
) -> dict[str, Any]:
    candidate = [
        row
        for row in evidence
        if row["family"] == family and row["role"] == "candidate" and row["training_seed"] == training_seed
    ]
    control = [
        row
        for row in evidence
        if row["family"] == family and row["role"] == "control" and row["training_seed"] == training_seed
    ]
    if len(candidate) != 80 or len(control) != 80:
        raise ValueError("M13.13 paired gate requires 80 candidate and 80 control episodes")
    observed_seeds = tuple(sorted({int(row["env_seed"]) for row in candidate}))
    if observed_seeds not in (
        M1313_DEVELOPMENT_CHECK_SEEDS,
        M1313_CONFIRMATION_EVALUATION_SEEDS,
        M1313_AUDIT_SEEDS,
    ):
        raise ValueError("M13.13 paired gate evidence uses an undeclared evaluation split")
    expected_cells = {
        (condition, env_seed)
        for condition in M1313_CONDITIONS["development_check"]
        for env_seed in observed_seeds
    }
    candidate_cells = {(str(row["condition"]), int(row["env_seed"])) for row in candidate}
    control_cells = {(str(row["condition"]), int(row["env_seed"])) for row in control}
    if candidate_cells != expected_cells or control_cells != expected_cells:
        raise ValueError("M13.13 paired gate evidence does not cover the frozen 80-cell matrix")
    by_condition = {
        condition: _hard_condition_gate([row for row in candidate if row["condition"] == condition])
        for condition in M1313_CONDITIONS["development_check"]
    }
    candidate_full = sum(bool(row["full_objective_success"]) for row in candidate) / len(candidate)
    control_full = sum(bool(row["full_objective_success"]) for row in control) / len(control)
    candidate_decision = float(np.mean([float(row["decision_safe_fraction"]) for row in candidate]))
    control_decision = float(np.mean([float(row["decision_safe_fraction"]) for row in control]))
    candidate_duration = float(np.mean([float(row["duration_safe_fraction"]) for row in candidate]))
    control_duration = float(np.mean([float(row["duration_safe_fraction"]) for row in control]))
    required_advantage = 0.15 if family == "shielded_learned" else 0.10
    per_condition_non_regression = all(
        sum(bool(row["full_objective_success"]) for row in candidate if row["condition"] == condition)
        >= sum(bool(row["full_objective_success"]) for row in control if row["condition"] == condition) - 1
        for condition in by_condition
    )
    passes = bool(
        all(row["passes"] for row in by_condition.values())
        and candidate_full - control_full >= required_advantage
        and candidate_decision >= control_decision - 0.02
        and candidate_duration >= control_duration - 0.02
        and (family == "shielded_learned" or per_condition_non_regression)
    )
    return {
        "family": family,
        "training_seed": training_seed,
        "candidate_conditions": by_condition,
        "candidate_full_objective_rate": candidate_full,
        "control_full_objective_rate": control_full,
        "full_objective_advantage": candidate_full - control_full,
        "required_advantage": required_advantage,
        "decision_safe_delta": candidate_decision - control_decision,
        "duration_safe_delta": candidate_duration - control_duration,
        "per_condition_non_regression": per_condition_non_regression,
        "passes": passes,
    }


def m1313_gate(evidence: list[dict[str, Any]], families: set[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if "urgency_commitment" in families:
        results["urgency_commitment"] = _pair_gate(
            evidence,
            family="urgency_commitment",
            training_seed=0,
        )
    if "short_horizon_model" in families:
        results["short_horizon_model"] = _pair_gate(
            evidence,
            family="short_horizon_model",
            training_seed=0,
        )
    if "shielded_learned" in families:
        seeds = sorted(
            {
                int(row["training_seed"])
                for row in evidence
                if row["family"] == "shielded_learned"
            }
        )
        replicas = [_pair_gate(evidence, family="shielded_learned", training_seed=seed) for seed in seeds]
        for replica in replicas:
            seed = int(replica["training_seed"])
            by_role = {
                role: {
                    row.get("actor_fingerprint")
                    for row in evidence
                    if row["family"] == "shielded_learned"
                    and row["training_seed"] == seed
                    and row["role"] == role
                }
                for role in ("candidate", "control")
            }
            identical = (
                len(by_role["candidate"]) == 1
                and by_role["candidate"] == by_role["control"]
                and None not in by_role["candidate"]
            )
            replica["actor_bytes_identical"] = identical
            replica["passes"] = bool(replica["passes"] and identical)
        results["shielded_learned"] = {
            "replicas": replicas,
            "all_replicas_pass": len(replicas) in {4, 8} and all(row["passes"] for row in replicas),
            "passes": len(replicas) in {4, 8} and all(row["passes"] for row in replicas),
        }
    promoted = sorted(family for family, row in results.items() if bool(row["passes"]))
    return {
        "families": results,
        "promoted_families": promoted,
        "cross_family_selection_performed": False,
    }


def _public_specs(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in spec.items()
            if key != "policy"
        }
        for spec in specs
    ]


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result["content_sha256"] = content_hash(result)
    return result


def _verify_report(path: str | Path, *, stage: str) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"M13.13 {stage} report is unreadable") from exc
    supplied = report.pop("content_sha256", None)
    if supplied != content_hash(report):
        raise ValueError(f"M13.13 {stage} report content hash mismatch")
    report["content_sha256"] = supplied
    if report.get("protocol_fingerprint") != m1313_protocol_fingerprint() or report.get("stage") != stage:
        raise ValueError(f"M13.13 {stage} report protocol or stage mismatch")
    return report


def _run_evaluation_stage(
    *,
    stage: str,
    specs: list[dict[str, Any]],
    conditions: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    evidence, replay_count = _evaluate_specs(specs, conditions=conditions, seeds=seeds, trace_dir=trace_dir)
    actor_by_label = {spec["label"]: spec.get("actor_fingerprint") for spec in specs}
    for row in evidence:
        row["actor_fingerprint"] = actor_by_label.get(str(row["label"]))

    oracle = BalancedM139Oracle()
    oracle_rows, oracle_count = _evaluate_one(
        oracle,
        label=f"{stage}_scripted_ceiling",
        family="shared_diagnostics",
        role="scripted_ceiling",
        training_seed=0,
        conditions=conditions,
        seeds=seeds,
        trace_dir=trace_dir,
    )
    random_seed = M1313_RANDOM_SEEDS[stage]
    random = SeededRandomM139Policy(random_seed)
    random_rows, random_count = _evaluate_one(
        random,
        label=f"{stage}_mask_random",
        family="shared_diagnostics",
        role="mask_random",
        training_seed=random_seed,
        conditions=conditions,
        seeds=seeds,
        trace_dir=trace_dir,
        policy_fingerprint=f"pcg64-{random_seed}",
    )
    diagnostics = {
        "scripted_ceiling_full_objective": sum(bool(row["full_objective_success"]) for row in oracle_rows),
        "mask_random_full_objective": sum(bool(row["full_objective_success"]) for row in random_rows),
        "episodes_each": len(oracle_rows),
    }
    return evidence + oracle_rows + random_rows, replay_count + oracle_count + random_count, diagnostics


def m1313_development(
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1313_CANONICAL_LEDGER_PATH,
    workers: int | None = None,
) -> dict[str, Any]:
    manifest = verify_policy_family_manifest(manifest_path)
    _assert_layouts()
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.13 development artifact directory already exists: {root}")
    protocol = m1313_protocol_fingerprint()
    ledger = M1313SplitLedger(ledger_path, protocol_fingerprint=protocol)
    ledger.open("development_fit", resumable_fit=True)
    root.mkdir(parents=True)
    coverage = {
        "development_fit": m10_scan_coverage(
            M1313_CONDITIONS["development_fit"],
            seeds=M1313_DEVELOPMENT_FIT_SEEDS,
        )
    }
    if not all(bool(row["passes"]) for row in coverage["development_fit"].values()):
        raise RuntimeError("M13.13 development-fit public coverage failed before fitting")
    started = time.perf_counter()
    jobs = _fit_wave(
        training_seeds=M1313_DEVELOPMENT_TRAINING_SEEDS,
        conditions=M1313_CONDITIONS["development_fit"],
        env_seeds=M1313_DEVELOPMENT_FIT_SEEDS,
        artifact_dir=root / "ppo-source-policies",
        workers=workers,
    )
    specs = _deterministic_specs(root / "policies", {"urgency_commitment", "short_horizon_model"})
    specs.extend(_learned_specs(jobs, artifact_dir=root / "policies"))
    ledger.open("development_check")
    coverage["development_check"] = m10_scan_coverage(
        M1313_CONDITIONS["development_check"],
        seeds=M1313_DEVELOPMENT_CHECK_SEEDS,
    )
    if not all(bool(row["passes"]) for row in coverage["development_check"].values()):
        raise RuntimeError("M13.13 development-check public coverage failed")
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="development",
        specs=specs,
        conditions=M1313_CONDITIONS["development_check"],
        seeds=M1313_DEVELOPMENT_CHECK_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1313_gate(evidence, {"urgency_commitment", "short_horizon_model", "shielded_learned"})
    return _seal_report(
        {
            "schema_version": "m13.13-development-report-v1",
            "stage": "development",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": manifest["protocol_fingerprint"],
            "elapsed_seconds": time.perf_counter() - started,
            "coverage": coverage,
            "fit_jobs": jobs,
            "policy_artifacts": _public_specs(specs),
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": ["confirmation_fit", "confirmation_evaluation", "audit"],
            "limits": [
                "No family is selected against another family.",
                "A passing family only permits separately authorized confirmation under the frozen manifest.",
            ],
        }
    )


def m1313_confirmation(
    *,
    manifest_path: str | Path,
    development_report_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1313_CANONICAL_LEDGER_PATH,
    workers: int | None = None,
) -> dict[str, Any]:
    manifest = verify_policy_family_manifest(manifest_path)
    development = _verify_report(development_report_path, stage="development")
    if development.get("manifest_fingerprint") != manifest["protocol_fingerprint"]:
        raise ValueError("M13.13 development report manifest fingerprint mismatch")
    families = set(development["gate"]["promoted_families"])
    if not families:
        raise ValueError("M13.13 confirmation requires at least one independently promoted family")
    _assert_layouts()
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.13 confirmation artifact directory already exists: {root}")
    protocol = m1313_protocol_fingerprint()
    ledger = M1313SplitLedger(ledger_path, protocol_fingerprint=protocol)
    ledger.open("confirmation_fit", resumable_fit=True)
    root.mkdir(parents=True)
    coverage = {
        "confirmation_fit": m10_scan_coverage(
            M1313_CONDITIONS["confirmation_fit"],
            seeds=M1313_CONFIRMATION_FIT_SEEDS,
        )
    }
    if not all(bool(row["passes"]) for row in coverage["confirmation_fit"].values()):
        raise RuntimeError("M13.13 confirmation-fit public coverage failed before fitting")
    started = time.perf_counter()
    jobs: list[dict[str, Any]] = []
    if "shielded_learned" in families:
        jobs = _fit_wave(
            training_seeds=M1313_CONFIRMATION_TRAINING_SEEDS,
            conditions=M1313_CONDITIONS["confirmation_fit"],
            env_seeds=M1313_CONFIRMATION_FIT_SEEDS,
            artifact_dir=root / "ppo-source-policies",
            workers=workers,
        )
    specs = _deterministic_specs(root / "policies", families)
    if jobs:
        specs.extend(_learned_specs(jobs, artifact_dir=root / "policies"))
    ledger.open("confirmation_evaluation")
    coverage["confirmation_evaluation"] = m10_scan_coverage(
        M1313_CONDITIONS["confirmation_evaluation"],
        seeds=M1313_CONFIRMATION_EVALUATION_SEEDS,
    )
    if not all(bool(row["passes"]) for row in coverage["confirmation_evaluation"].values()):
        raise RuntimeError("M13.13 confirmation-evaluation public coverage failed")
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="confirmation",
        specs=specs,
        conditions=M1313_CONDITIONS["confirmation_evaluation"],
        seeds=M1313_CONFIRMATION_EVALUATION_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1313_gate(evidence, families)
    confirmed = sorted(family for family in families if family in gate["promoted_families"])
    gate["confirmed_families"] = confirmed
    return _seal_report(
        {
            "schema_version": "m13.13-confirmation-report-v1",
            "stage": "confirmation",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": manifest["protocol_fingerprint"],
            "development_report_sha256": hashlib.sha256(Path(development_report_path).read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "coverage": coverage,
            "fit_jobs": jobs,
            "policy_artifacts": _public_specs(specs),
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": ["audit"],
            "limits": ["Only confirmed families may enter one separately authorized audit; no refit is allowed."],
        }
    )


def m1313_audit(
    *,
    manifest_path: str | Path,
    confirmation_report_path: str | Path,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1313_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    manifest = verify_policy_family_manifest(manifest_path)
    confirmation = _verify_report(confirmation_report_path, stage="confirmation")
    if confirmation.get("manifest_fingerprint") != manifest["protocol_fingerprint"]:
        raise ValueError("M13.13 confirmation report manifest fingerprint mismatch")
    families = set(confirmation["gate"].get("confirmed_families", []))
    if not families:
        raise ValueError("M13.13 audit requires at least one independently confirmed family")
    root = Path(artifact_dir)
    if root.exists():
        raise FileExistsError(f"M13.13 audit artifact directory already exists: {root}")
    protocol = m1313_protocol_fingerprint()
    ledger = M1313SplitLedger(ledger_path, protocol_fingerprint=protocol)
    ledger.open("audit")
    root.mkdir(parents=True)
    selected_rows = [row for row in confirmation["policy_artifacts"] if row["family"] in families]
    specs = _load_specs(selected_rows)
    coverage = m10_scan_coverage(M1313_CONDITIONS["audit"], seeds=M1313_AUDIT_SEEDS)
    if not all(bool(row["passes"]) for row in coverage.values()):
        raise RuntimeError("M13.13 audit public coverage failed")
    started = time.perf_counter()
    evidence, replay_count, diagnostics = _run_evaluation_stage(
        stage="audit",
        specs=specs,
        conditions=M1313_CONDITIONS["audit"],
        seeds=M1313_AUDIT_SEEDS,
        trace_dir=root / "traces",
    )
    gate = m1313_gate(evidence, families)
    audited = sorted(family for family in families if family in gate["promoted_families"])
    gate["audited_families"] = audited
    return _seal_report(
        {
            "schema_version": "m13.13-audit-report-v1",
            "stage": "audit",
            "protocol_fingerprint": protocol,
            "manifest_fingerprint": manifest["protocol_fingerprint"],
            "confirmation_report_sha256": hashlib.sha256(Path(confirmation_report_path).read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "coverage": coverage,
            "policy_artifacts": selected_rows,
            "strict_replay_count": replay_count,
            "episode_evidence": evidence,
            "shared_diagnostics": diagnostics,
            "gate": gate,
            "split_ledger": ledger.snapshot(),
            "unopened_partitions": [],
            "limits": ["Audit is one-shot. No retry, replacement seed, threshold change, or post-audit tuning is allowed."],
        }
    )


def _write_report(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"M13.13 report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def write_m1313_development_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.13 report already exists: {destination}")
    return _write_report(
        destination,
        m1313_development(
            manifest_path=manifest_path,
            artifact_dir=destination.parent / f"{destination.stem}-artifacts",
        ),
    )


def write_m1313_confirmation_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    development_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.13 report already exists: {destination}")
    return _write_report(
        destination,
        m1313_confirmation(
            manifest_path=manifest_path,
            development_report_path=development_report_path,
            artifact_dir=destination.parent / f"{destination.stem}-artifacts",
        ),
    )


def write_m1313_audit_report(
    output: str | Path,
    *,
    manifest_path: str | Path,
    confirmation_report_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"M13.13 report already exists: {destination}")
    return _write_report(
        destination,
        m1313_audit(
            manifest_path=manifest_path,
            confirmation_report_path=confirmation_report_path,
            artifact_dir=destination.parent / f"{destination.stem}-artifacts",
        ),
    )
