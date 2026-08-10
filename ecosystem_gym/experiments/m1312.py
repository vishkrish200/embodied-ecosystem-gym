"""M13.12 one-factor PPO learning-rate development comparison.

The experiment owns fresh 6100--6139 development data. It compares the
unchanged corrected M13.11-r2 PPO at 3e-4 with one predeclared 1e-3 candidate.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..env import EcosystemEnv
from ..maintenance import MaintenanceMacro, MaintenanceRewardState, advance_memory, compile_macro, eligible_macro_mask, encode_features, learning_reward
from ..maintenance.ppo_r2 import CorrectedMaskedPPOPolicy, Rollout
from ..tasks import LAYOUTS
from .m10 import m10_scan_coverage
from .m13 import _options
from .m139 import m139_config, replay_m139_trace, run_m139_episode
from .m1311_support import content_hash
from .m1311r2 import M1311R2_FIT_CONDITIONS, M1311R2_FIT_SEEDS
from .m1312_support import M1312DevelopmentLedger

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

M1312_PROTOCOL_VERSION = "m13.12-development-ppo-learning-rate"
M1312_FIT_SEEDS = tuple(range(6100, 6120))
M1312_CHECK_SEEDS = tuple(range(6120, 6140))
M1312_TRAINING_SEEDS = (20261321, 20261322, 20261323, 20261324)
M1312_DECISION_BUDGET, M1312_PPO_ROLLOUT, M1312_PPO_EPOCHS = 400_000, 2048, 4
M1312_LEARNING_RATES = {"ppo_lr_3e4_baseline": 3e-4, "ppo_lr_1e3_candidate": 1e-3}
M1312_CANONICAL_LEDGER_PATH = Path(__file__).resolve().parents[2] / "artifacts/m1312/split-open-ledger.json"
Arm = Literal["ppo_lr_3e4_baseline", "ppo_lr_1e3_candidate"]


def _controls(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        "persistent_reference": {"layout_id": f"{prefix}_northeast", "food_variant": "orange", "toy_variant": "ball", "camera_control": "scan_v2", "initial_scan_sector": "north"},
        "renewal_and_morphology": {"layout_id": f"{prefix}_southwest", "food_variant": "purple", "food_shape_variant": "capsule", "toy_variant": "cube", "agent_shape_variant": "capsule", "camera_control": "scan_v2", "initial_scan_sector": "south"},
        "event_relocation": {"layout_id": f"{prefix}_northwest", "food_variant": "blue", "food_shape_variant": "box", "toy_variant": "capsule", "lighting_variant": "dim", "camera_control": "scan_v2", "initial_scan_sector": "east", "event_relocation_on_first_pickup": True},
        "compound": {"layout_id": f"{prefix}_southeast", "food_variant": "red", "food_shape_variant": "capsule", "toy_variant": "cube", "agent_shape_variant": "box", "dynamics_variant": "grippy", "blocked_distractor": True, "distractor_xy": [-0.04, 0.04], "camera_control": "scan_v2", "initial_scan_sector": "west", "event_relocation_on_first_pickup": True},
    }


M1312_FIT_CONDITIONS = _controls("m1312_fit")
M1312_CHECK_CONDITIONS = _controls("m1312_check")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    package = root / "ecosystem_gym"
    paths = (Path(__file__), package / "maintenance/contract.py", package / "maintenance/reward.py", package / "maintenance/ppo_r2.py", package / "tasks.py")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def m1312_protocol_fingerprint() -> str:
    return content_hash({
        "version": M1312_PROTOCOL_VERSION,
        "hypothesis": "M13.11-r2 PPO updates were too conservative at 3e-4",
        "splits": {"development_fit": M1312_FIT_SEEDS, "development_check": M1312_CHECK_SEEDS},
        "conditions": {"fit": M1312_FIT_CONDITIONS, "check": M1312_CHECK_CONDITIONS},
        "training_seeds": M1312_TRAINING_SEEDS,
        "decision_budget": M1312_DECISION_BUDGET,
        "arms": M1312_LEARNING_RATES,
        "ppo": {"network": [30, 64, 64, 8, 1], "rollout": M1312_PPO_ROLLOUT, "epochs": M1312_PPO_EPOCHS, "clip": .2, "value_coef": .5, "entropy": .01, "gae_lambda": .95, "gamma": .99, "max_gradient_norm": .5},
        "primary_gate": "candidate has at least one full-objective success in every training-seed replica",
        "workers": "spawn one numeric thread",
        "sources": _source_hashes(),
    })


def _assert_layouts() -> None:
    names = [name for name in LAYOUTS if name.startswith("m1312_")]
    expected = [str(value["layout_id"]) for conditions in (M1312_FIT_CONDITIONS, M1312_CHECK_CONDITIONS) for value in conditions.values()]
    if len(names) != 8 or set(names) != set(expected):
        raise ValueError("M13.12 layouts are incomplete")
    points = lambda layout: (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)
    current = {point for name in names for point in points(LAYOUTS[name])}
    previous = {point for name, layout in LAYOUTS.items() if not name.startswith("m1312_") for point in points(layout)}
    if len(current) != 40 or current & previous:
        raise ValueError("M13.12 geometry is not fresh")


def _rollout(rows: list[tuple[Any, ...]]) -> Rollout:
    dtypes = (np.float32, np.int64, np.bool_, np.float32, np.float32, np.float32, np.bool_, np.float32, np.float32)
    return Rollout(*(np.asarray([row[index] for row in rows], dtype=dtype) for index, dtype in enumerate(dtypes)))


def _summarize_updates(updates: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = ("total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm")
    summary = {key: float(np.mean([float(row[key]) for row in updates])) if updates else 0.0 for key in scalar_keys}
    histogram = {str(index): 0 for index in range(8)}
    for row in updates:
        for index, value in row.get("action_histogram", {}).items():
            histogram[index] += int(value)
    return {**summary, "action_histogram": histogram}


def _train_ppo(arm: Arm, seed: int, *, decision_budget: int, max_episodes: int | None = None, fit_conditions: dict[str, dict[str, Any]] = M1312_FIT_CONDITIONS, fit_seeds: tuple[int, ...] = M1312_FIT_SEEDS) -> tuple[CorrectedMaskedPPOPolicy, dict[str, Any]]:
    learning_rate = M1312_LEARNING_RATES[arm]
    policy = CorrectedMaskedPPOPolicy(seed=seed, config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x50504F))
    env = EcosystemEnv(policy.config)
    decisions = episodes = update_batches = 0
    updates: list[dict[str, Any]] = []
    rows: list[tuple[Any, ...]] = []
    items = tuple(fit_conditions.items())
    try:
        while decisions < decision_budget and (max_episodes is None or episodes < max_episodes):
            _, controls = items[episodes % len(items)]
            env_seed = fit_seeds[(episodes // len(items)) % len(fit_seeds)]
            observation, _ = env.reset(seed=env_seed, options=_options(controls))
            memory = policy.reset()
            reward_state = MaintenanceRewardState(required_recovery=bool(controls.get("event_relocation_on_first_pickup")))
            episodes += 1
            while True:
                features = encode_features(observation, memory)
                mask = eligible_macro_mask(observation, policy.config)
                action_index, logp, value = policy.sample(features, mask, rng)
                macro = MaintenanceMacro(action_index)
                action = compile_macro(macro, observation, policy.config)
                next_observation, environment_reward, terminated, truncated, info = env.step(action)
                reward, _ = learning_reward(environment_reward=environment_reward, state=reward_state, observation_before=observation, observation_after=next_observation, action=action, info=info, terminated=terminated, truncated=truncated)
                advance_memory(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation, config=policy.config)
                next_features = encode_features(next_observation, memory)
                rows.append((features, action_index, mask, logp, reward, float(np.asarray(action["duration"]).item()), terminated or truncated, value, policy.value(next_features)))
                decisions += 1
                observation = next_observation
                if len(rows) == M1312_PPO_ROLLOUT:
                    updates.append(policy.update(_rollout(rows), epochs=M1312_PPO_EPOCHS, learning_rate=learning_rate))
                    update_batches += 1
                    rows.clear()
                if terminated or truncated:
                    break
    finally:
        env.close()
    if rows:
        updates.append(policy.update(_rollout(rows), epochs=M1312_PPO_EPOCHS, learning_rate=learning_rate))
        update_batches += 1
    return policy, {"learning_rate": learning_rate, "decisions": decisions, "overshoot": max(0, decisions - decision_budget), "episodes": episodes, "update_batches": update_batches, "update_epochs": update_batches * M1312_PPO_EPOCHS, "diagnostics": _summarize_updates(updates)}


def _worker(arm: Arm, seed: int, budget: int, max_episodes: int | None, artifact_dir: str, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    train_kwargs = {"fit_conditions": M1311R2_FIT_CONDITIONS, "fit_seeds": M1311R2_FIT_SEEDS} if smoke else {}
    policy, training = _train_ppo(arm, seed, decision_budget=budget, max_episodes=max_episodes, **train_kwargs)
    artifact = policy.save(Path(artifact_dir) / f"seed-{seed}" / f"{arm}.json")
    return {"arm": arm, "seed": seed, "pid": os.getpid(), "elapsed_seconds": time.perf_counter() - started, "training": training, "artifact": artifact}


def _wave(*, artifact_dir: Path, budget: int, max_episodes: int | None, workers: int | None, smoke: bool = False) -> list[dict[str, Any]]:
    jobs = [(arm, seed) for arm in M1312_LEARNING_RATES for seed in M1312_TRAINING_SEEDS]
    context = multiprocessing.get_context("spawn")
    worker_count = min(workers or min(8, max(1, os.cpu_count() or 1)), len(jobs))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [executor.submit(_worker, arm, seed, budget, max_episodes, str(artifact_dir), smoke) for arm, seed in jobs]
        return [future.result() for future in as_completed(futures)]


def _evaluate(policy: CorrectedMaskedPPOPolicy, *, arm: Arm, seed: int, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], trace_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for condition, controls in conditions.items():
        for env_seed in seeds:
            trace = trace_dir / f"seed-{seed}" / arm / condition / f"seed-{env_seed}.jsonl"
            episode = run_m139_episode(policy, arm="public_potential_candidate", training_seed=seed, seed=env_seed, condition=condition, controls=controls, trace_path=trace, policy_label=f"m1312/{arm}/{seed}", policy_fingerprint=None)
            replay_m139_trace(trace, policy)
            rows.append({"training_seed": seed, "arm": arm, "condition": condition, "env_seed": env_seed, "survived": episode.survived, "maintenance_complete": episode.maintenance_complete, "full_objective_success": episode.full_gate_success, "decision_safe_fraction": episode.decision_safe_fraction, "duration_safe_fraction": episode.duration_safe_fraction, "recovery_required": episode.recovery_required, "recovery_complete": episode.recovery_complete, "unsafe_wait_fraction": episode.unsafe_wait_fraction, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
    return rows, len(rows)


def _gate(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, int]] = {}
    for arm in M1312_LEARNING_RATES:
        by_seed = {}
        for seed in M1312_TRAINING_SEEDS:
            by_seed[str(seed)] = sum(bool(row["full_objective_success"]) for row in evidence if row["arm"] == arm and row["training_seed"] == seed)
        by_arm[arm] = by_seed
    candidate = by_arm["ppo_lr_1e3_candidate"]
    return {
        "definition": "candidate has at least one full-objective success in every training-seed replica",
        "full_objective_successes_by_arm_and_seed": by_arm,
        "candidate_all_replicas_positive": all(value > 0 for value in candidate.values()),
        "passes": all(value > 0 for value in candidate.values()),
    }


def m1312_fit_smoke(*, artifact_dir: str | Path, workers: int | None = None) -> dict[str, Any]:
    root = Path(artifact_dir)
    ledger = M1312DevelopmentLedger(root / "split-open-ledger.non-protocol.json", protocol_fingerprint=m1312_protocol_fingerprint())
    ledger.open("development_fit", resumable_fit=True)
    started = time.perf_counter()
    worker_count = min(workers or min(8, max(1, os.cpu_count() or 1)), 8)
    jobs = _wave(artifact_dir=root / "policies", budget=10**9, max_episodes=32, workers=worker_count, smoke=True)
    evidence: list[dict[str, Any]] = []
    replay_count = 0
    for job in jobs:
        policy = CorrectedMaskedPPOPolicy.load(job["artifact"]["path"], config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
        rows, count = _evaluate(policy, arm=job["arm"], seed=int(job["seed"]), conditions={"persistent_reference": M1311R2_FIT_CONDITIONS["persistent_reference"]}, seeds=(M1311R2_FIT_SEEDS[0],), trace_dir=root / "traces")
        evidence.extend(rows)
        replay_count += count
    return {"label": "M13.12 fit-only plumbing smoke; non-promotional", "elapsed_seconds": time.perf_counter() - started, "workers": worker_count, "jobs": jobs, "strict_replay_count": replay_count, "evaluation": evidence, "split_ledger": ledger.snapshot(), "canonical_protocol_ledger_accessed": False}


def m1312_development(*, artifact_dir: str | Path, ledger_path: str | Path = M1312_CANONICAL_LEDGER_PATH, workers: int | None = None) -> dict[str, Any]:
    _assert_layouts()
    root = Path(artifact_dir)
    ledger = M1312DevelopmentLedger(ledger_path, protocol_fingerprint=m1312_protocol_fingerprint())
    ledger.open("development_fit", resumable_fit=True)
    coverage = m10_scan_coverage(M1312_FIT_CONDITIONS, seeds=M1312_FIT_SEEDS)
    started = time.perf_counter()
    jobs = _wave(artifact_dir=root / "policies", budget=M1312_DECISION_BUDGET, max_episodes=None, workers=workers)
    ledger.open("development_check")
    evidence: list[dict[str, Any]] = []
    replay_count = 0
    for job in jobs:
        policy = CorrectedMaskedPPOPolicy.load(job["artifact"]["path"], config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
        rows, count = _evaluate(policy, arm=job["arm"], seed=int(job["seed"]), conditions=M1312_CHECK_CONDITIONS, seeds=M1312_CHECK_SEEDS, trace_dir=root / "traces")
        evidence.extend(rows)
        replay_count += count
    return {"label": "M13.12 development-only one-factor PPO learning-rate comparison", "elapsed_seconds": time.perf_counter() - started, "protocol_fingerprint": m1312_protocol_fingerprint(), "coverage": coverage, "workers": min(workers or min(8, max(1, os.cpu_count() or 1)), 8), "thread_settings": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, "jobs": jobs, "strict_replay_count": replay_count, "episode_evidence": evidence, "gate": _gate(evidence), "split_ledger": ledger.snapshot(), "unopened_partitions": ["screen", "confirmation", "audit"], "limits": ["A pass only permits proposing a separately frozen screen; no screen, confirmation, audit, M14, or M15 run is authorized."]}


def _write_report(output: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"M13.12 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def write_m1312_fit_smoke_report(output: str | Path) -> dict[str, Any]:
    destination = Path(output)
    return _write_report(destination, m1312_fit_smoke(artifact_dir=destination.parent / f"{destination.stem}-artifacts"))


def write_m1312_development_report(output: str | Path) -> dict[str, Any]:
    destination = Path(output)
    return _write_report(destination, m1312_development(artifact_dir=destination.parent / f"{destination.stem}-artifacts"))
