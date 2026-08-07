"""M13.1: the frozen cycle-transition reward-only revision of M13."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..env import EcosystemEnv
from .m10 import m10_scan_coverage
from .m13 import (
    M13_ALPHA,
    M13_EPSILON_DECAY_EPISODES,
    M13_EPSILON_FINAL,
    M13_GAMMA,
    M13_SAFE_DRIVE_BANDS,
    M13_TRAINING_EPISODES,
    M13_TRAINING_SEED,
    M13Macro,
    M13Memory,
    RandomM13Policy,
    ScriptedM13Oracle,
    TabularM13QPolicy,
    _aggregate,
    _file_sha256,
    _memory_snapshot,
    _safe,
    _options,
    compile_macro,
    encode_state,
    run_m13_episode,
)
from .m8 import wilson_interval
from ..trajectory import ReplayResult, SCHEMA_VERSION, _json_value, replay_and_validate


M131_PROTOCOL_VERSION = "m13.1-cycle-reward-r1"
M131_DEVELOPMENT_SEEDS = tuple(range(2_300, 2_340))
M131_VALIDATION_SEEDS = tuple(range(2_400, 2_420))
M131_AUDIT_SEEDS = tuple(range(2_500, 2_520))


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M131_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m131_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m131_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m131_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m131_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M131_VALIDATION_CONDITIONS = {name: {**controls, "layout_id": f"m131_validation_{controls['layout_id'].removeprefix('m131_dev_')}"} for name, controls in M131_DEVELOPMENT_CONDITIONS.items()}
M131_AUDIT_CONDITIONS = {name: {**controls, "layout_id": f"m131_audit_{controls['layout_id'].removeprefix('m131_dev_')}"} for name, controls in M131_DEVELOPMENT_CONDITIONS.items()}


def m131_config():
    """M13 mechanics plus the only M13.1 change: guarded cycle rewards."""

    from .m13 import m13_config

    return replace(
        m13_config(),
        persistent_feed_cycle_reward=0.25,
        persistent_play_cycle_reward=0.25,
        persistent_rest_cycle_reward=0.25,
    )


class TabularM131QPolicy(TabularM13QPolicy):
    """Same M13 encoder and Q update, trained only against M13.1 reward."""

    def train(self, *, episodes: int = M13_TRAINING_EPISODES, seed: int = M13_TRAINING_SEED) -> None:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13.1 training budget is frozen")
        rng = np.random.default_rng(seed)
        items = tuple(M131_DEVELOPMENT_CONDITIONS.items())
        env = EcosystemEnv(m131_config())
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                episode_seed = M131_DEVELOPMENT_SEEDS[(episode // len(items)) % len(M131_DEVELOPMENT_SEEDS)]
                observation, _ = env.reset(seed=episode_seed, options=_options(controls))
                memory = self.reset()
                epsilon = max(M13_EPSILON_FINAL, 1.0 - episode / M13_EPSILON_DECAY_EPISODES)
                for _ in range(env.config.max_episode_steps):
                    state = self._state(observation, memory)
                    macro = M13Macro(int(rng.integers(len(M13Macro)))) if rng.random() < epsilon else self.choose(observation, memory)
                    action = compile_macro(macro, observation, env.config)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    self.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
                    next_state = self._state(next_observation, memory)
                    target = reward if terminated or truncated else reward + M13_GAMMA * float(np.max(self._q(next_state)))
                    self._q(state)[int(macro)] += M13_ALPHA * (target - self._q(state)[int(macro)])
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()


def m131_policy_fingerprint(policy: TabularM13QPolicy) -> str:
    digest = hashlib.sha256(M131_PROTOCOL_VERSION.encode())
    digest.update(json.dumps({"include_drives": policy.include_drives, "use_memory": policy.use_memory}, sort_keys=True).encode())
    for state, values in sorted(policy.q_values.items()):
        digest.update(np.asarray(state, dtype=np.int16).tobytes())
        digest.update(np.asarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def m131_protocol_fingerprint() -> str:
    payload = {
        "version": M131_PROTOCOL_VERSION,
        "config": asdict(m131_config()),
        "development_seeds": M131_DEVELOPMENT_SEEDS,
        "validation_seeds": M131_VALIDATION_SEEDS,
        "audit_seeds": M131_AUDIT_SEEDS,
        "development_conditions": M131_DEVELOPMENT_CONDITIONS,
        "validation_conditions": M131_VALIDATION_CONDITIONS,
        "audit_conditions": M131_AUDIT_CONDITIONS,
        "training": {"episodes": M13_TRAINING_EPISODES, "seed": M13_TRAINING_SEED, "alpha": M13_ALPHA, "gamma": M13_GAMMA, "epsilon_final": M13_EPSILON_FINAL, "epsilon_decay_episodes": M13_EPSILON_DECAY_EPISODES},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_m131_policy(path: str | Path, policy: TabularM131QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "m131-policy-v1", "protocol_fingerprint": m131_protocol_fingerprint(), "include_drives": policy.include_drives, "use_memory": policy.use_memory, "q_values": [{"state": list(state), "values": np.asarray(values, dtype=np.float64).tolist()} for state, values in sorted(policy.q_values.items())]}
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": _file_sha256(output), "policy_fingerprint": m131_policy_fingerprint(policy)}


def load_m131_policy(path: str | Path) -> TabularM131QPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m131-policy-v1" or payload.get("protocol_fingerprint") != m131_protocol_fingerprint():
        raise ValueError("M13.1 policy artifact does not match the frozen protocol")
    policy = TabularM131QPolicy(include_drives=bool(payload["include_drives"]), use_memory=bool(payload["use_memory"]))
    for row in payload["q_values"]:
        values = np.asarray(row["values"], dtype=np.float64)
        if values.shape != (len(M13Macro),):
            raise ValueError("M13.1 policy artifact has an invalid action-value row")
        policy.q_values[tuple(int(value) for value in row["state"])] = values
    return policy


def _run(policy: Any, *, seed: int, condition: str, controls: dict[str, Any], trace_path: Path | None = None, policy_fingerprint: str | None = None):
    env = EcosystemEnv(m131_config())
    try:
        episode = run_m13_episode(policy, seed=seed, condition=condition, controls=controls, env=env)
    finally:
        env.close()
    if trace_path is not None:
        _write_trace(trace_path, episode, controls, policy_fingerprint)
    return episode


def _write_trace(path: Path, episode: Any, controls: dict[str, Any], policy_fingerprint: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "record_type": "episode_metadata", "episode_id": f"m131-{episode.condition}-seed-{episode.seed}", "seed": episode.seed, "reset_options": _json_value(_options(controls)), "observation_mode": "state_oracle", "config": asdict(m131_config()), "m131_protocol_fingerprint": m131_protocol_fingerprint(), "m131_policy_fingerprint": policy_fingerprint}
    rows = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for record in episode.steps:
        row = {"schema_version": SCHEMA_VERSION, "episode_id": header["episode_id"], "step": record["step"], "seed": episode.seed, "observation_mode": "state_oracle", "observation": record["observation_after"], "action": record["action"], "outcome": record["outcome"], "reward": record["reward"], "task_success": record["task_success"], "terminated": record["terminated"], "truncated": record["truncated"], "environment_version": record["environment_version"], "disturbance": record["disturbance"], "post_disturbance_completion": record["post_disturbance_completion"], "resource_event": record["resource_event"], "feed_cycles": record["feed_cycles"], "play_cycles": record["play_cycles"], "rest_cycles": record["rest_cycles"], "food_available": record["food_available"], "camera_sector": record["camera_sector"], "m131_policy_observation": record["policy_observation"], "m131_encoded_state": record["encoded_state"], "m131_memory_before": record["memory_before"], "m131_memory_after": record["memory_after"], "m131_macro": record["macro"]}
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def replay_m131_trace(path: str | Path, policy: TabularM131QPolicy) -> ReplayResult:
    generic = replay_and_validate(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("m131_protocol_fingerprint") != m131_protocol_fingerprint() or header.get("m131_policy_fingerprint") != m131_policy_fingerprint(policy):
        raise ValueError("M13.1 trace fingerprint mismatch")
    env = EcosystemEnv(m131_config())
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            if _json_value(observation) != expected["m131_policy_observation"]:
                raise ValueError(f"M13.1 policy observation mismatch at step {expected['step']}")
            state = policy._state(observation, memory)
            if list(state) != expected["m131_encoded_state"] or _memory_snapshot(memory) != expected["m131_memory_before"]:
                raise ValueError(f"M13.1 policy state mismatch at step {expected['step']}")
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if macro.name != expected["m131_macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.1 policy action mismatch at step {expected['step']}")
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            if _memory_snapshot(memory) != expected["m131_memory_after"]:
                raise ValueError(f"M13.1 policy memory mismatch at step {expected['step']}")
            observation = next_observation
    finally:
        env.close()
    return generic


def _evaluate(policy: Any, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], *, trace_dir: Path | None = None, label: str = "", fingerprint: str | None = None):
    results, manifest = {}, []
    for name, controls in conditions.items():
        episodes = []
        for seed in seeds:
            trace = None if trace_dir is None else trace_dir / label / name / f"seed-{seed}.jsonl"
            episodes.append(_run(policy, seed=seed, condition=name, controls=controls, trace_path=trace, policy_fingerprint=fingerprint))
            if trace is not None:
                manifest.append({"policy": label, "condition": name, "seed": str(seed), "path": str(trace.resolve()), "sha256": _file_sha256(trace)})
        results[name] = _aggregate(episodes)
    return results, manifest


def _ceiling_passes(rows: dict[str, dict[str, object]], seeds: tuple[int, ...]) -> bool:
    return all(int(row["survivals"]) == len(seeds) and int(row["completed_maintenance_episodes"]) == len(seeds) and int(row["completed_cycles"]["minimum_feed"]) >= 3 and int(row["completed_cycles"]["minimum_play"]) >= 3 and int(row["completed_cycles"]["minimum_rest"]) >= 2 for row in rows.values())  # type: ignore[index]


def _gates(rows: dict[str, dict[str, object]]) -> dict[str, dict[str, bool]]:
    return {name: {"survival_pass": int(row["survivals"]) >= 18, "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 18, "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= 0.85, "recovery_pass": not bool(M131_VALIDATION_CONDITIONS[name].get("event_relocation_on_first_pickup")) or int(row["forced_recovery_chains"]["completed"]) >= 18} for name, row in rows.items()}  # type: ignore[index]


def _flat(rows: dict[str, dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(bool(item[field])) for condition in M131_VALIDATION_CONDITIONS for item in rows[condition]["episodes_detail"]], dtype=np.float64)  # type: ignore[index]


def m131_train(*, artifact_dir: str | Path) -> dict[str, object]:
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.1 development artifact directory already exists")
    artifacts.mkdir(parents=True)
    policies = {"full_state_oracle_q": TabularM131QPolicy(), "no_drive_q": TabularM131QPolicy(include_drives=False), "no_memory_q": TabularM131QPolicy(use_memory=False)}
    for offset, policy in enumerate(policies.values()):
        policy.train(seed=M13_TRAINING_SEED + offset)
    serialized = {name: write_m131_policy(artifacts / "policies" / f"{name}.json", policy) for name, policy in policies.items()}
    return {"schema_version": "0.13.1", "protocol_fingerprint": m131_protocol_fingerprint(), "split": "development_only", "training": {"episodes": M13_TRAINING_EPISODES, "seed": M13_TRAINING_SEED}, "policy_artifacts": serialized, "limits": ["No validation or audit score is present.", "The only M13.1 policy difference is the documented scalar cycle-transition reward."]}


def write_m131_training_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError("M13.1 development report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m131_train(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def m131_validation(*, artifact_dir: str | Path) -> dict[str, object]:
    coverage = m10_scan_coverage(M131_VALIDATION_CONDITIONS, seeds=M131_VALIDATION_SEEDS)
    if not all(bool(row["passes"]) for row in coverage.values()):
        raise RuntimeError("M13.1 validation has unobservable public targets")
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.1 validation cannot be overwritten or rerun")
    artifacts.mkdir(parents=True)
    ceiling, ceiling_manifest = _evaluate(ScriptedM13Oracle(), M131_VALIDATION_CONDITIONS, M131_VALIDATION_SEEDS, trace_dir=artifacts / "traces", label="state_oracle_scripted_ceiling")
    if not _ceiling_passes(ceiling, M131_VALIDATION_SEEDS):
        raise RuntimeError("M13.1 mechanics ceiling failed")
    policies = {"full_state_oracle_q": TabularM131QPolicy(), "no_drive_q": TabularM131QPolicy(include_drives=False), "no_memory_q": TabularM131QPolicy(use_memory=False)}
    for offset, policy in enumerate(policies.values()):
        policy.train(seed=M13_TRAINING_SEED + offset)
    artifacts_by_policy = {name: write_m131_policy(artifacts / "policies" / f"{name}.json", policy) for name, policy in policies.items()}
    results, manifests = {}, [*ceiling_manifest]
    for name, policy in policies.items():
        rows, manifest = _evaluate(policy, M131_VALIDATION_CONDITIONS, M131_VALIDATION_SEEDS, trace_dir=artifacts / "traces", label=name, fingerprint=m131_policy_fingerprint(policy))
        results[name] = rows
        manifests.extend(manifest)
        for item in manifest:
            replay_m131_trace(item["path"], load_m131_policy(artifacts_by_policy[name]["path"]))
    random_rows, random_manifest = _evaluate(RandomM13Policy(M13_TRAINING_SEED + 3), M131_VALIDATION_CONDITIONS, M131_VALIDATION_SEEDS, trace_dir=artifacts / "traces", label="macro_random")
    results["macro_random"] = random_rows
    manifests.extend(random_manifest)
    for item in [*ceiling_manifest, *random_manifest]:
        replay_and_validate(item["path"])
    gates = _gates(results["full_state_oracle_q"])
    full_maintenance, full_survival = _flat(results["full_state_oracle_q"], "maintenance_complete"), _flat(results["full_state_oracle_q"], "survived")
    comparisons = {name: {"maintenance_delta": float(np.mean(full_maintenance - _flat(rows, "maintenance_complete"))), "survival_delta": float(np.mean(full_survival - _flat(rows, "survived")))} for name, rows in results.items() if name != "full_state_oracle_q"}
    passes = all(all(row.values()) for row in gates.values()) and all(float(row["maintenance_delta"]) >= 0.10 for row in comparisons.values()) and float(comparisons["macro_random"]["survival_delta"]) >= 0.20
    return {"schema_version": "0.13.1", "protocol_version": M131_PROTOCOL_VERSION, "protocol_fingerprint": m131_protocol_fingerprint(), "policy_boundary": ["agent_xy", "food_xy", "toy_xy", "rest_xy", "drives", "holding_food", "prior_outcome", "policy_owned_memory"], "reward_version": M131_PROTOCOL_VERSION, "training": {"episodes": M13_TRAINING_EPISODES, "seed": M13_TRAINING_SEED, "full_policy_fingerprint": m131_policy_fingerprint(policies["full_state_oracle_q"])}, "policy_artifacts": artifacts_by_policy, "coverage": coverage, "state_oracle_scripted_ceiling": ceiling, "results": results, "condition_gates": gates, "comparisons": comparisons, "trace_manifest": manifests, "gate": {"passes": passes}}


def write_m131_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError("M13.1 validation report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m131_validation(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
