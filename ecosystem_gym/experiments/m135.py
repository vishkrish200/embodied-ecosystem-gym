"""M13.5 parallel update-ratio diagnostic under the frozen public boundary."""

from __future__ import annotations

import os

# Set before NumPy/MuJoCo load in spawned workers: one numerical thread per job.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..actions import ActionOutcome
from ..env import EcosystemEnv
from .m10 import m10_scan_coverage
from .m13 import (
    M13_EPSILON_DECAY_EPISODES, M13_EPSILON_FINAL, M13_GAMMA,
    M13_TRAINING_EPISODES, M13Episode, M13Macro, M13Memory,
    ScriptedM13Oracle, _aggregate, _clear_memory, _file_sha256,
    _memory_snapshot, _options, _safe, compile_macro,
)
from .m132 import (
    M132_BATCH_SIZE, M132_FEATURE_DIM, M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP, M132_TARGET_UPDATE_EVERY, _MLP,
    encode_features,
)
from .m133 import advance_m133_memory, m133_config, m133_epsilon
from .m134 import _MaskedReplay, _masked_argmax, m134_mask
from ..trajectory import ReplayResult, SCHEMA_VERSION, _json_value, replay_and_validate


M135_PROTOCOL_VERSION = "m13.5-update-ratio-r1"
M135_DEVELOPMENT_SEEDS = tuple(range(3_500, 3_540))
M135_VALIDATION_SEEDS = tuple(range(3_600, 3_620))
M135_AUDIT_SEEDS = tuple(range(3_700, 3_720))
M135_TRAINING_SEEDS = (20_260_812, 20_260_813, 20_260_814)
M135_CANDIDATE_UPDATE_EVERY = 4
M135_NULL_UPDATE_EVERY = 256
M135_WORKERS = 8
Arm = Literal["update_ratio", "cadence_null"]


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M135_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m135_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m135_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m135_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m135_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M135_VALIDATION_CONDITIONS = {name: {**controls, "layout_id": f"m135_validation_{controls['layout_id'].removeprefix('m135_dev_')}"} for name, controls in M135_DEVELOPMENT_CONDITIONS.items()}
M135_AUDIT_CONDITIONS = {name: {**controls, "layout_id": f"m135_audit_{controls['layout_id'].removeprefix('m135_dev_')}"} for name, controls in M135_DEVELOPMENT_CONDITIONS.items()}


def m135_update_every(arm: Arm) -> int:
    if arm == "update_ratio":
        return M135_CANDIDATE_UPDATE_EVERY
    if arm == "cadence_null":
        return M135_NULL_UPDATE_EVERY
    raise ValueError(f"unknown M13.5 arm {arm!r}")


class CompactM135QPolicy:
    """M13.4 public-mask DQN with only its declared update cadence varied."""

    include_drives = use_memory = True
    mask_mode = "complementary"

    def __init__(self, *, arm: Arm, seed: int) -> None:
        self.arm, self.seed, self.update_every = arm, seed, m135_update_every(arm)
        self.config = m133_config()
        self.online = _MLP(np.random.default_rng(seed))
        self.target = self.online.copy()
        self.update_count = 0

    def reset(self) -> M13Memory:
        return M13Memory()

    def features(self, observation: dict[str, Any], memory: M13Memory) -> np.ndarray:
        return encode_features(observation, memory, include_drives=True, use_memory=True)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(_masked_argmax(self.online.predict(self.features(observation, memory)), self.mask(observation)))

    def observe(self, memory: M13Memory, *, observation_before: dict[str, Any], macro: M13Macro, action: dict[str, np.ndarray | int], observation_after: dict[str, Any]) -> None:
        advance_m133_memory(memory, observation_before=observation_before, macro=macro, action=action, observation_after=observation_after, config=self.config)

    def _update(self, replay: _MaskedReplay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, next_masks, done = replay.sample(rng)
        selected = np.argmax(np.where(next_masks, self.online.predict(next_features), -np.inf), axis=1)
        targets = rewards + M13_GAMMA * (~done) * self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % M132_TARGET_UPDATE_EVERY == 0:
            self.target = self.online.copy()
        return loss

    def train(self, *, episodes: int = M13_TRAINING_EPISODES) -> dict[str, float]:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13.5 training budget is frozen")
        rng, replay, decisions, losses = np.random.default_rng(self.seed), _MaskedReplay(), 0, []
        self.online, self.target, self.update_count = _MLP(rng), None, 0
        self.target = self.online.copy()
        env, items = EcosystemEnv(self.config), tuple(M135_DEVELOPMENT_CONDITIONS.items())
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                seed = M135_DEVELOPMENT_SEEDS[(episode // len(items)) % len(M135_DEVELOPMENT_SEEDS)]
                observation, _ = env.reset(seed=seed, options=_options(controls))
                memory, epsilon = self.reset(), m133_epsilon(episode)
                for _ in range(env.config.max_episode_steps):
                    features, mask = self.features(observation, memory), self.mask(observation)
                    macro = M13Macro(int(rng.choice(np.flatnonzero(mask)))) if rng.random() < epsilon else self.choose(observation, memory)
                    action = compile_macro(macro, observation, env.config)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    self.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
                    replay.add(features, int(macro), reward, self.features(next_observation, memory), self.mask(next_observation), terminated or truncated)
                    decisions += 1
                    if replay.size >= M132_REPLAY_WARMUP and decisions % self.update_every == 0:
                        losses.append(self._update(replay, rng))
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()
        return {"decisions": float(decisions), "updates": float(self.update_count), "target_copies": float(self.update_count // M132_TARGET_UPDATE_EVERY), "mean_huber_loss": float(np.mean(losses)) if losses else 0.0}


class SeededRandomM135Policy:
    include_drives, use_memory, mask_mode = True, False, "complementary"

    def __init__(self, seed: int) -> None:
        self.seed, self.config, self.rng = seed, m133_config(), np.random.default_rng(seed)

    def reset(self) -> M13Memory:
        return M13Memory()

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        del memory
        return M13Macro(int(self.rng.choice(np.flatnonzero(self.mask(observation)))))

    def observe(self, memory: M13Memory, **_: Any) -> None:
        _clear_memory(memory)


def m135_policy_fingerprint(policy: CompactM135QPolicy) -> str:
    digest = hashlib.sha256(M135_PROTOCOL_VERSION.encode())
    digest.update(json.dumps({"arm": policy.arm, "seed": policy.seed, "update_every": policy.update_every}, sort_keys=True).encode())
    for key in sorted(policy.online.params):
        digest.update(policy.online.params[key].tobytes())
    return digest.hexdigest()


def m135_protocol_fingerprint() -> str:
    source_dir = Path(__file__).parent
    sources = ("m135.py", "m134.py", "m133.py", "m132.py", "m13.py", "env.py", "config.py")
    source_paths = {
        name: source_dir.parent / name if name in {"env.py", "config.py"} else source_dir / name
        for name in sources
    }
    payload = {"version": M135_PROTOCOL_VERSION, "config": asdict(m133_config()), "splits": {"development": M135_DEVELOPMENT_SEEDS, "validation": M135_VALIDATION_SEEDS, "audit": M135_AUDIT_SEEDS}, "conditions": {"development": M135_DEVELOPMENT_CONDITIONS, "validation": M135_VALIDATION_CONDITIONS, "audit": M135_AUDIT_CONDITIONS}, "training_seeds": M135_TRAINING_SEEDS, "candidate_update_every": M135_CANDIDATE_UPDATE_EVERY, "null_update_every": M135_NULL_UPDATE_EVERY, "workers": M135_WORKERS, "training": {"episodes": M13_TRAINING_EPISODES, "epsilon_final": M13_EPSILON_FINAL, "epsilon_decay": M13_EPSILON_DECAY_EPISODES, "gamma": M13_GAMMA, "replay_capacity": M132_REPLAY_CAPACITY, "warmup": M132_REPLAY_WARMUP, "batch": M132_BATCH_SIZE, "target_update_every": M132_TARGET_UPDATE_EVERY}, "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_m135_policy(path: str | Path, policy: CompactM135QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "m135-policy-v1", "protocol_fingerprint": m135_protocol_fingerprint(), "arm": policy.arm, "seed": policy.seed, "update_every": policy.update_every, "network": {key: value.tolist() for key, value in policy.online.params.items()}}
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": _file_sha256(output), "policy_fingerprint": m135_policy_fingerprint(policy)}


def load_m135_policy(path: str | Path) -> CompactM135QPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m135-policy-v1" or payload.get("protocol_fingerprint") != m135_protocol_fingerprint():
        raise ValueError("M13.5 policy artifact does not match the frozen protocol")
    policy = CompactM135QPolicy(arm=payload["arm"], seed=int(payload["seed"]))
    if int(payload["update_every"]) != policy.update_every:
        raise ValueError("M13.5 policy artifact has inconsistent update cadence")
    for key, values in payload["network"].items():
        if key not in policy.online.params or np.asarray(values).shape != policy.online.params[key].shape:
            raise ValueError("M13.5 policy artifact has invalid parameters")
        policy.online.params[key] = np.asarray(values, dtype=np.float32)
    policy.target = policy.online.copy()
    return policy


def _train_worker(arm: Arm, seed: int) -> tuple[Arm, int, CompactM135QPolicy, dict[str, float]]:
    """Spawn-safe independent fit; no artifacts or shared mutable state are touched."""

    policy = CompactM135QPolicy(arm=arm, seed=seed)
    return arm, seed, policy, policy.train()


def _blocked(info: dict[str, Any]) -> tuple[bool, bool]:
    blocked = info["outcome"] == ActionOutcome.BLOCKED.value
    exempt = blocked and info["disturbance"] == "food_relocated"
    return blocked and not exempt, exempt


def run_m135_episode(policy: Any, *, seed: int, condition: str, controls: dict[str, Any], trace_path: Path | None = None, policy_fingerprint: str | None = None) -> M13Episode:
    env = EcosystemEnv(policy.config if hasattr(policy, "config") else m133_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory, records, safe_steps = policy.reset(), [], 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        interventions: dict[str, int] = {"mask_violation": 0, "redundant_go": 0}
        for step in range(1, env.config.max_episode_steps + 1):
            before = _memory_snapshot(memory)
            mask = policy.mask(observation) if hasattr(policy, "mask") else np.ones(len(M13Macro), dtype=np.bool_)
            features = policy.features(observation, memory) if isinstance(policy, CompactM135QPolicy) else None
            values = policy.online.predict(features) if isinstance(policy, CompactM135QPolicy) else None
            rng_state = _json_value(policy.rng.bit_generator.state) if isinstance(policy, SeededRandomM135Policy) else None
            macro = policy.choose(observation, memory)
            if not bool(mask[int(macro)]): interventions["mask_violation"] += 1
            action = compile_macro(macro, observation, env.config)
            next_observation, reward, terminated, truncated, info = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            ordinary, exempt = _blocked(info)
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen, stale_pickup = relocation_seen or relocation_now, stale_pickup or exempt
            consumed_after_relocation = consumed_after_relocation or (relocation_seen and macro is M13Macro.CONSUME and info["outcome"] == ActionOutcome.SUCCESS.value)
            if ordinary: interventions["blocked_action"] = interventions.get("blocked_action", 0) + 1
            if relocation_now: interventions["food_relocated"] = interventions.get("food_relocated", 0) + 1
            if info["resource_event"] is not None:
                key = str(info["resource_event"]); interventions[key] = interventions.get(key, 0) + 1
            if macro in {M13Macro.GO_FOOD, M13Macro.GO_TOY, M13Macro.GO_REST} and not bool(mask[int(macro)]): interventions["redundant_go"] += 1
            safe_steps += int(_safe(np.asarray(next_observation["drives"], dtype=np.float32)))
            records.append({"step": step, "policy_observation": _json_value(observation), "features": None if features is None else features.tolist(), "q_values": None if values is None else np.asarray(values, dtype=np.float32).tolist(), "mask": mask.astype(bool).tolist(), "eligible_macros": [item.name for item in M13Macro if mask[int(item)]], "memory_before": before, "memory_after": _memory_snapshot(memory), "macro": macro.name, "action": _json_value(action), "random_rng_state_before": rng_state, "reward": float(reward), "outcome": info["outcome"], "ordinary_blocked": ordinary, "exempt_forced_relocation_blocked": exempt, "disturbance": info["disturbance"], "post_disturbance_completion": bool(info["post_disturbance_completion"]), "resource_event": info["resource_event"], "feed_cycles": int(info["feed_cycles"]), "play_cycles": int(info["play_cycles"]), "rest_cycles": int(info["rest_cycles"]), "food_available": bool(info["food_available"]), "camera_sector": info["camera_sector"], "task_success": bool(info["task_success"]), "environment_version": str(info["environment_version"]), "observation_after": _json_value(next_observation), "terminated": bool(terminated), "truncated": bool(truncated)})
            observation = next_observation
            if terminated or truncated:
                drives = np.asarray(observation["drives"], dtype=np.float32)
                terminal = "survived_horizon" if truncated and bool(info["survived"]) else "energy_depleted" if drives[1] <= 0 else "satiety_depleted" if drives[0] <= 0 else "terminated"
                result = M13Episode(seed=seed, condition=condition, survived=bool(info["survived"]), maintenance_complete=bool(info["maintenance_complete"]), terminal_cause=terminal, feed_cycles=int(info["feed_cycles"]), play_cycles=int(info["play_cycles"]), rest_cycles=int(info["rest_cycles"]), safe_drive_fraction=safe_steps / step, forced_recovery={"required": bool(controls.get("event_relocation_on_first_pickup")), "stale_pickup": stale_pickup, "consumed_after_relocation": consumed_after_relocation, "complete": stale_pickup and consumed_after_relocation}, interventions=interventions, steps=tuple(records))
                if trace_path is not None: _write_trace(trace_path, result, controls, env.config, policy_fingerprint)
                return result
        raise AssertionError("M13.5 episode did not terminate")
    finally:
        env.close()


def _write_trace(path: Path, episode: M13Episode, controls: dict[str, Any], config: Any, policy_fingerprint: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "record_type": "episode_metadata", "episode_id": f"m135-{episode.condition}-seed-{episode.seed}", "seed": episode.seed, "reset_options": _json_value(_options(controls)), "observation_mode": "state_oracle", "config": asdict(config), "m135_protocol_fingerprint": m135_protocol_fingerprint(), "m135_policy_fingerprint": policy_fingerprint}
    rows = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for record in episode.steps:
        rows.append(json.dumps({"schema_version": SCHEMA_VERSION, "episode_id": header["episode_id"], "seed": episode.seed, "observation_mode": "state_oracle", "observation": record["observation_after"], **record}, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def replay_m135_trace(path: str | Path, policy: CompactM135QPolicy) -> ReplayResult:
    generic = replay_and_validate(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("m135_protocol_fingerprint") != m135_protocol_fingerprint() or header.get("m135_policy_fingerprint") != m135_policy_fingerprint(policy):
        raise ValueError("M13.5 trace fingerprint mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            features, mask = policy.features(observation, memory), policy.mask(observation)
            values = policy.online.predict(features)
            if _json_value(observation) != expected["policy_observation"] or features.tolist() != expected["features"] or np.asarray(values, dtype=np.float32).tolist() != expected["q_values"] or mask.astype(bool).tolist() != expected["mask"] or _memory_snapshot(memory) != expected["memory_before"]:
                raise ValueError(f"M13.5 inference mismatch at step {expected['step']}")
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if not bool(mask[int(macro)]) or macro.name != expected["macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.5 constrained action mismatch at step {expected['step']}")
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            if _memory_snapshot(memory) != expected["memory_after"]:
                raise ValueError(f"M13.5 memory mismatch at step {expected['step']}")
            observation = next_observation
    finally:
        env.close()
    return generic


def replay_m135_random_trace(path: str | Path, policy: SeededRandomM135Policy) -> ReplayResult:
    generic = replay_and_validate(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("m135_protocol_fingerprint") != m135_protocol_fingerprint():
        raise ValueError("M13.5 random trace protocol fingerprint mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            state = expected.get("random_rng_state_before")
            if state is None: raise ValueError("M13.5 random trace lacks pre-choice RNG state")
            policy.rng.bit_generator.state = state
            mask = policy.mask(observation)
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if mask.astype(bool).tolist() != expected["mask"] or macro.name != expected["macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.5 random replay mismatch at step {expected['step']}")
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            observation = next_observation
    finally:
        env.close()
    return generic


def _evaluate(policy: Any, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], *, trace_dir: Path, label: str, fingerprint: str | None) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    results, manifest = {}, []
    for name, controls in conditions.items():
        episodes = []
        for seed in seeds:
            trace = trace_dir / label / name / f"seed-{seed}.jsonl"
            episodes.append(run_m135_episode(policy, seed=seed, condition=name, controls=controls, trace_path=trace, policy_fingerprint=fingerprint))
            manifest.append({"policy": label, "condition": name, "seed": str(seed), "path": str(trace.resolve()), "sha256": _file_sha256(trace)})
        results[name] = _aggregate(episodes)
    return results, manifest


def _ceiling_passes(rows: dict[str, dict[str, object]]) -> bool:
    return all(int(row["survivals"]) == 40 and int(row["completed_maintenance_episodes"]) == 40 and int(row["completed_cycles"]["minimum_feed"]) >= 3 and int(row["completed_cycles"]["minimum_play"]) >= 3 and int(row["completed_cycles"]["minimum_rest"]) >= 2 for row in rows.values())


def _flat(rows: dict[str, dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(bool(episode[field])) for condition in M135_DEVELOPMENT_CONDITIONS for episode in rows[condition]["episodes_detail"]], dtype=np.float64)


def _gates(candidate: dict[str, dict[str, object]], control: dict[str, dict[str, object]], random: dict[str, dict[str, object]]) -> dict[str, object]:
    conditions = {name: {"survival_pass": int(row["survivals"]) >= 36, "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 36, "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= .85, "recovery_pass": not bool(M135_DEVELOPMENT_CONDITIONS[name].get("event_relocation_on_first_pickup")) or int(row["forced_recovery_chains"]["completed"]) >= 36, "mask_pass": all(int(e["interventions"].get("mask_violation", 0)) == 0 and int(e["interventions"].get("redundant_go", 0)) == 0 for e in row["episodes_detail"])} for name, row in candidate.items()}
    maintenance_delta = float(np.mean(_flat(candidate, "maintenance_complete") - _flat(control, "maintenance_complete")))
    survival_delta = float(np.mean(_flat(candidate, "survived") - _flat(control, "survived")))
    random_delta = float(np.mean(_flat(candidate, "survived") - _flat(random, "survived")))
    return {"condition_gates": conditions, "maintenance_delta_vs_cadence_null": maintenance_delta, "survival_delta_vs_cadence_null": survival_delta, "survival_delta_vs_random": random_delta, "passes": all(all(item.values()) for item in conditions.values()) and maintenance_delta >= .10 and survival_delta >= .10 and random_delta >= .20}


def m135_train(*, artifact_dir: str | Path, workers: int = M135_WORKERS) -> dict[str, object]:
    """Run six independent fits in spawned workers, then deterministic parent evaluation."""

    if workers != M135_WORKERS:
        raise ValueError(f"M13.5 worker count is frozen at {M135_WORKERS}")
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.5 development artifact directory already exists")
    artifacts.mkdir(parents=True)
    coverage = m10_scan_coverage(M135_DEVELOPMENT_CONDITIONS, seeds=M135_DEVELOPMENT_SEEDS)
    ceiling, manifest = _evaluate(ScriptedM13Oracle(), M135_DEVELOPMENT_CONDITIONS, M135_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label="state_oracle_scripted_ceiling", fingerprint=None)
    jobs = [(arm, seed) for seed in M135_TRAINING_SEEDS for arm in ("update_ratio", "cadence_null")]
    completed: dict[tuple[str, int], tuple[CompactM135QPolicy, dict[str, float]]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=M135_WORKERS, mp_context=context) as executor:
        futures = {executor.submit(_train_worker, arm, seed): (arm, seed) for arm, seed in jobs}
        for future in as_completed(futures):
            arm, seed, policy, training = future.result()
            completed[(arm, seed)] = (policy, training)
    replicates: dict[str, object] = {}
    for seed in M135_TRAINING_SEEDS:
        candidate, candidate_training = completed[("update_ratio", seed)]
        control, control_training = completed[("cadence_null", seed)]
        random = SeededRandomM135Policy(seed)
        policies = {"update_ratio": candidate, "cadence_null": control}
        serial = {name: write_m135_policy(artifacts / "policies" / f"seed-{seed}" / f"{name}.json", policy) for name, policy in policies.items()}
        results: dict[str, dict[str, dict[str, object]]] = {}
        for name, policy in policies.items():
            rows, traces = _evaluate(policy, M135_DEVELOPMENT_CONDITIONS, M135_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label=f"seed-{seed}/{name}", fingerprint=m135_policy_fingerprint(policy))
            results[name] = rows; manifest.extend(traces)
            restored = load_m135_policy(serial[name]["path"])
            for trace in traces: replay_m135_trace(trace["path"], restored)
        random_rows, random_traces = _evaluate(random, M135_DEVELOPMENT_CONDITIONS, M135_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label=f"seed-{seed}/macro_random", fingerprint=f"pcg64-{seed}")
        results["macro_random"] = random_rows; manifest.extend(random_traces)
        for trace in random_traces: replay_m135_random_trace(trace["path"], SeededRandomM135Policy(seed))
        replicates[str(seed)] = {"training": {"update_ratio": candidate_training, "cadence_null": control_training}, "policy_artifacts": serial, "results": results, "gates": _gates(results["update_ratio"], results["cadence_null"], results["macro_random"])}
    passes = all(bool(row["passes"]) for row in coverage.values()) and _ceiling_passes(ceiling) and all(bool(value["gates"]["passes"]) for value in replicates.values())
    return {"schema_version": "0.13.5", "protocol_version": M135_PROTOCOL_VERSION, "protocol_fingerprint": m135_protocol_fingerprint(), "split": "development_only", "execution": {"workers": M135_WORKERS, "start_method": "spawn", "worker_threads": 1}, "coverage": coverage, "state_oracle_scripted_ceiling": ceiling, "replicates": replicates, "trace_manifest": manifest, "gate": {"passes": passes}, "limits": ["No validation or audit score is present.", "M13.5 changes training dynamics, not the public policy interface."]}


def write_m135_training_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists(): raise FileExistsError("M13.5 development report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m135_train(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
