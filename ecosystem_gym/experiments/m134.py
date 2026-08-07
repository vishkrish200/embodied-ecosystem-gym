"""M13.4 public macro-precondition Double-DQN baseline.

The mask is a deterministic action-selection constraint over the public M13
state.  It is deliberately not a feature and never receives resource/event
state which the policy boundary excludes.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..actions import ActionOutcome
from ..env import EcosystemEnv
from .m10 import m10_scan_coverage
from .m13 import (
    M13_EPSILON_DECAY_EPISODES,
    M13_EPSILON_FINAL,
    M13_GAMMA,
    M13_TRAINING_EPISODES,
    M13Episode,
    M13Macro,
    M13Memory,
    ScriptedM13Oracle,
    _aggregate,
    _clear_memory,
    _file_sha256,
    _memory_snapshot,
    _options,
    _safe,
    compile_macro,
)
from .m132 import (
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP,
    M132_TARGET_UPDATE_EVERY,
    M132_UPDATE_EVERY,
    _MLP,
    encode_features,
)
from .m133 import advance_m133_memory, m133_config, m133_epsilon
from ..trajectory import ReplayResult, SCHEMA_VERSION, _json_value, replay_and_validate


M134_PROTOCOL_VERSION = "m13.4-public-macro-preconditions-r1"
M134_DEVELOPMENT_SEEDS = tuple(range(3_200, 3_240))
M134_VALIDATION_SEEDS = tuple(range(3_300, 3_320))
M134_AUDIT_SEEDS = tuple(range(3_400, 3_420))
M134_TRAINING_SEEDS = (20_260_809, 20_260_810, 20_260_811)
MaskMode = Literal["complementary", "interaction_only", "none"]


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M134_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m134_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m134_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m134_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m134_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M134_VALIDATION_CONDITIONS = {name: {**controls, "layout_id": f"m134_validation_{controls['layout_id'].removeprefix('m134_dev_')}"} for name, controls in M134_DEVELOPMENT_CONDITIONS.items()}
M134_AUDIT_CONDITIONS = {name: {**controls, "layout_id": f"m134_audit_{controls['layout_id'].removeprefix('m134_dev_')}"} for name, controls in M134_DEVELOPMENT_CONDITIONS.items()}


def m134_mask(observation: dict[str, Any], config: Any, mode: MaskMode = "complementary") -> np.ndarray:
    """Return the exact public macro eligibility vector in M13.4 protocol order."""

    if mode not in {"complementary", "interaction_only", "none"}:
        raise ValueError(f"unknown M13.4 mask mode {mode!r}")
    if mode == "none":
        return np.ones(len(M13Macro), dtype=np.bool_)
    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    food_near = bool(np.linalg.norm(np.asarray(observation["food_xy"], dtype=np.float32) - agent) <= config.pickup_radius)
    toy_near = bool(np.linalg.norm(np.asarray(observation["toy_xy"], dtype=np.float32) - agent) <= config.toy_interaction_radius)
    rest_near = bool(np.linalg.norm(np.asarray(observation["rest_xy"], dtype=np.float32) - agent) <= config.rest_interaction_radius)
    mask = np.zeros(len(M13Macro), dtype=np.bool_)
    mask[M13Macro.PICK_UP] = food_near and not bool(observation["holding_food"])
    mask[M13Macro.CONSUME] = bool(observation["holding_food"])
    mask[M13Macro.PLAY] = toy_near
    mask[M13Macro.REST] = rest_near
    mask[M13Macro.WAIT] = True
    if mode == "interaction_only":
        mask[M13Macro.GO_FOOD] = mask[M13Macro.GO_TOY] = mask[M13Macro.GO_REST] = True
    else:
        mask[M13Macro.GO_FOOD] = not food_near
        mask[M13Macro.GO_TOY] = not toy_near
        mask[M13Macro.GO_REST] = not rest_near
    return mask


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    if values.shape != (len(M13Macro),) or mask.shape != (len(M13Macro),) or not bool(mask.any()):
        raise ValueError("invalid M13.4 action values or empty eligibility mask")
    return int(np.argmax(np.where(mask, values, -np.inf)))


class _MaskedReplay:
    def __init__(self) -> None:
        self.features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.actions = np.empty(M132_REPLAY_CAPACITY, dtype=np.int64)
        self.rewards = np.empty(M132_REPLAY_CAPACITY, dtype=np.float32)
        self.next_features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.next_masks = np.empty((M132_REPLAY_CAPACITY, len(M13Macro)), dtype=np.bool_)
        self.done = np.empty(M132_REPLAY_CAPACITY, dtype=np.bool_)
        self.size = self.cursor = 0

    def add(self, features: np.ndarray, action: int, reward: float, next_features: np.ndarray, next_mask: np.ndarray, done: bool) -> None:
        index = self.cursor
        self.features[index], self.actions[index], self.rewards[index], self.next_features[index], self.next_masks[index], self.done[index] = features, action, reward, next_features, next_mask, done
        self.cursor, self.size = (index + 1) % M132_REPLAY_CAPACITY, min(self.size + 1, M132_REPLAY_CAPACITY)

    def sample(self, rng: np.random.Generator):
        indices = rng.choice(self.size, size=M132_BATCH_SIZE, replace=False)
        return self.features[indices], self.actions[indices], self.rewards[indices], self.next_features[indices], self.next_masks[indices], self.done[indices]


class CompactM134QPolicy:
    """M13.3 DQN with a selectable public M13.4 macro-precondition mask."""

    def __init__(self, *, mask_mode: MaskMode = "complementary", include_drives: bool = True, use_memory: bool = True, seed: int = M134_TRAINING_SEEDS[0]) -> None:
        self.mask_mode, self.include_drives, self.use_memory, self.seed = mask_mode, include_drives, use_memory, seed
        self.config = m133_config()
        self.online = _MLP(np.random.default_rng(seed))
        self.target = self.online.copy()
        self.update_count = 0

    def reset(self) -> M13Memory:
        return M13Memory()

    def features(self, observation: dict[str, Any], memory: M13Memory) -> np.ndarray:
        return encode_features(observation, memory, include_drives=self.include_drives, use_memory=self.use_memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, self.mask_mode)

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(_masked_argmax(self.online.predict(self.features(observation, memory)), self.mask(observation)))

    def observe(self, memory: M13Memory, *, observation_before: dict[str, Any], macro: M13Macro, action: dict[str, np.ndarray | int], observation_after: dict[str, Any]) -> None:
        advance_m133_memory(memory, observation_before=observation_before, macro=macro, action=action, observation_after=observation_after, config=self.config)
        if not self.use_memory:
            _clear_memory(memory)

    def _update(self, replay: _MaskedReplay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, next_masks, done = replay.sample(rng)
        online_values = self.online.predict(next_features)
        selected = np.argmax(np.where(next_masks, online_values, -np.inf), axis=1)
        bootstrap = self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        targets = rewards + M13_GAMMA * (~done) * bootstrap
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % M132_TARGET_UPDATE_EVERY == 0:
            self.target = self.online.copy()
        return loss

    def train(self, *, episodes: int = M13_TRAINING_EPISODES) -> dict[str, float]:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13.4 training budget is frozen")
        rng, replay, decisions, losses = np.random.default_rng(self.seed), _MaskedReplay(), 0, []
        self.online, self.target, self.update_count = _MLP(rng), None, 0
        self.target = self.online.copy()
        env, items = EcosystemEnv(self.config), tuple(M134_DEVELOPMENT_CONDITIONS.items())
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                seed = M134_DEVELOPMENT_SEEDS[(episode // len(items)) % len(M134_DEVELOPMENT_SEEDS)]
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
                    if replay.size >= M132_REPLAY_WARMUP and decisions % M132_UPDATE_EVERY == 0:
                        losses.append(self._update(replay, rng))
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()
        return {"decisions": float(decisions), "updates": float(self.update_count), "mean_huber_loss": float(np.mean(losses)) if losses else 0.0}


class SeededRandomM134Policy:
    include_drives, use_memory = True, False

    def __init__(self, *, mask_mode: MaskMode = "complementary", seed: int = M134_TRAINING_SEEDS[0]) -> None:
        self.mask_mode, self.seed, self.config, self.rng = mask_mode, seed, m133_config(), np.random.default_rng(seed)

    def reset(self) -> M13Memory:
        return M13Memory()

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, self.mask_mode)

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        del memory
        return M13Macro(int(self.rng.choice(np.flatnonzero(self.mask(observation)))))

    def observe(self, memory: M13Memory, **_: Any) -> None:
        _clear_memory(memory)


def m134_policy_fingerprint(policy: CompactM134QPolicy) -> str:
    digest = hashlib.sha256(M134_PROTOCOL_VERSION.encode())
    digest.update(json.dumps({"mask_mode": policy.mask_mode, "include_drives": policy.include_drives, "use_memory": policy.use_memory, "seed": policy.seed}, sort_keys=True).encode())
    for key in sorted(policy.online.params):
        digest.update(policy.online.params[key].tobytes())
    return digest.hexdigest()


def m134_protocol_fingerprint() -> str:
    source_dir = Path(__file__).parent
    sources = ("m134.py", "m133.py", "m132.py", "m13.py", "env.py", "config.py")
    payload = {
        "version": M134_PROTOCOL_VERSION,
        "config": asdict(m133_config()),
        "splits": {"development": M134_DEVELOPMENT_SEEDS, "validation": M134_VALIDATION_SEEDS, "audit": M134_AUDIT_SEEDS},
        "conditions": {"development": M134_DEVELOPMENT_CONDITIONS, "validation": M134_VALIDATION_CONDITIONS, "audit": M134_AUDIT_CONDITIONS},
        "training_seeds": M134_TRAINING_SEEDS,
        "mask_modes": ("complementary", "interaction_only", "none"),
        "training": {"episodes": M13_TRAINING_EPISODES, "epsilon_final": M13_EPSILON_FINAL, "epsilon_decay": M13_EPSILON_DECAY_EPISODES, "gamma": M13_GAMMA, "replay_capacity": M132_REPLAY_CAPACITY, "warmup": M132_REPLAY_WARMUP, "batch": M132_BATCH_SIZE, "update_every": M132_UPDATE_EVERY, "target_update_every": M132_TARGET_UPDATE_EVERY},
        "source_hashes": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in sources},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_m134_policy(path: str | Path, policy: CompactM134QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "m134-policy-v1", "protocol_fingerprint": m134_protocol_fingerprint(), "mask_mode": policy.mask_mode, "include_drives": policy.include_drives, "use_memory": policy.use_memory, "seed": policy.seed, "network": {key: value.tolist() for key, value in policy.online.params.items()}}
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": _file_sha256(output), "policy_fingerprint": m134_policy_fingerprint(policy)}


def load_m134_policy(path: str | Path) -> CompactM134QPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m134-policy-v1" or payload.get("protocol_fingerprint") != m134_protocol_fingerprint():
        raise ValueError("M13.4 policy artifact does not match the frozen protocol")
    policy = CompactM134QPolicy(mask_mode=payload["mask_mode"], include_drives=bool(payload["include_drives"]), use_memory=bool(payload["use_memory"]), seed=int(payload["seed"]))
    for key, values in payload["network"].items():
        if key not in policy.online.params or np.asarray(values).shape != policy.online.params[key].shape:
            raise ValueError("M13.4 policy artifact has an invalid parameter")
        policy.online.params[key] = np.asarray(values, dtype=np.float32)
    policy.target = policy.online.copy()
    return policy


def _blocked(info: dict[str, Any]) -> tuple[bool, bool]:
    blocked = info["outcome"] == ActionOutcome.BLOCKED.value
    exempt = blocked and info["disturbance"] == "food_relocated"
    return blocked and not exempt, exempt


def run_m134_episode(policy: Any, *, seed: int, condition: str, controls: dict[str, Any], trace_path: Path | None = None, policy_fingerprint: str | None = None) -> M13Episode:
    env = EcosystemEnv(policy.config if hasattr(policy, "config") else m133_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory, records, safe_steps = policy.reset(), [], 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        interventions: dict[str, int] = {"mask_violation": 0, "redundant_go": 0}
        for step in range(1, env.config.max_episode_steps + 1):
            memory_before = _memory_snapshot(memory)
            mask = policy.mask(observation) if hasattr(policy, "mask") else np.ones(len(M13Macro), dtype=np.bool_)
            features = policy.features(observation, memory) if isinstance(policy, CompactM134QPolicy) else None
            values = policy.online.predict(features) if isinstance(policy, CompactM134QPolicy) else None
            rng_state = _json_value(policy.rng.bit_generator.state) if isinstance(policy, SeededRandomM134Policy) else None
            macro = policy.choose(observation, memory)
            if not bool(mask[int(macro)]):
                interventions["mask_violation"] += 1
            action = compile_macro(macro, observation, env.config)
            next_observation, reward, terminated, truncated, info = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            ordinary, exempt = _blocked(info)
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen, stale_pickup = relocation_seen or relocation_now, stale_pickup or exempt
            consumed_after_relocation = consumed_after_relocation or (relocation_seen and macro is M13Macro.CONSUME and info["outcome"] == ActionOutcome.SUCCESS.value)
            if ordinary:
                interventions["blocked_action"] = interventions.get("blocked_action", 0) + 1
            if relocation_now:
                interventions["food_relocated"] = interventions.get("food_relocated", 0) + 1
            if info["resource_event"] is not None:
                key = str(info["resource_event"]); interventions[key] = interventions.get(key, 0) + 1
            if macro in {M13Macro.GO_FOOD, M13Macro.GO_TOY, M13Macro.GO_REST} and not bool(mask[int(macro)]):
                interventions["redundant_go"] += 1
            safe_steps += int(_safe(np.asarray(next_observation["drives"], dtype=np.float32)))
            records.append({"step": step, "policy_observation": _json_value(observation), "features": None if features is None else features.tolist(), "q_values": None if values is None else np.asarray(values, dtype=np.float32).tolist(), "mask": mask.astype(bool).tolist(), "eligible_macros": [item.name for item in M13Macro if mask[int(item)]], "memory_before": memory_before, "memory_after": _memory_snapshot(memory), "macro": macro.name, "action": _json_value(action), "random_rng_state_before": rng_state, "reward": float(reward), "outcome": info["outcome"], "ordinary_blocked": ordinary, "exempt_forced_relocation_blocked": exempt, "disturbance": info["disturbance"], "post_disturbance_completion": bool(info["post_disturbance_completion"]), "resource_event": info["resource_event"], "feed_cycles": int(info["feed_cycles"]), "play_cycles": int(info["play_cycles"]), "rest_cycles": int(info["rest_cycles"]), "food_available": bool(info["food_available"]), "camera_sector": info["camera_sector"], "task_success": bool(info["task_success"]), "environment_version": str(info["environment_version"]), "observation_after": _json_value(next_observation), "terminated": bool(terminated), "truncated": bool(truncated)})
            observation = next_observation
            if terminated or truncated:
                drives = np.asarray(observation["drives"], dtype=np.float32)
                terminal = "survived_horizon" if truncated and bool(info["survived"]) else "energy_depleted" if drives[1] <= 0 else "satiety_depleted" if drives[0] <= 0 else "terminated"
                episode = M13Episode(seed=seed, condition=condition, survived=bool(info["survived"]), maintenance_complete=bool(info["maintenance_complete"]), terminal_cause=terminal, feed_cycles=int(info["feed_cycles"]), play_cycles=int(info["play_cycles"]), rest_cycles=int(info["rest_cycles"]), safe_drive_fraction=safe_steps / step, forced_recovery={"required": bool(controls.get("event_relocation_on_first_pickup")), "stale_pickup": stale_pickup, "consumed_after_relocation": consumed_after_relocation, "complete": stale_pickup and consumed_after_relocation}, interventions=interventions, steps=tuple(records))
                if trace_path is not None:
                    _write_trace(trace_path, episode, controls, env.config, policy_fingerprint)
                return episode
        raise AssertionError("M13.4 episode did not terminate")
    finally:
        env.close()


def _write_trace(path: Path, episode: M13Episode, controls: dict[str, Any], config: Any, policy_fingerprint: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "record_type": "episode_metadata", "episode_id": f"m134-{episode.condition}-seed-{episode.seed}", "seed": episode.seed, "reset_options": _json_value(_options(controls)), "observation_mode": "state_oracle", "config": asdict(config), "m134_protocol_fingerprint": m134_protocol_fingerprint(), "m134_policy_fingerprint": policy_fingerprint}
    rows = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for record in episode.steps:
        rows.append(json.dumps({"schema_version": SCHEMA_VERSION, "episode_id": header["episode_id"], "seed": episode.seed, "observation_mode": "state_oracle", "observation": record["observation_after"], **record}, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def replay_m134_trace(path: str | Path, policy: CompactM134QPolicy) -> ReplayResult:
    generic = replay_and_validate(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("m134_protocol_fingerprint") != m134_protocol_fingerprint() or header.get("m134_policy_fingerprint") != m134_policy_fingerprint(policy):
        raise ValueError("M13.4 trace fingerprint mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            features, mask = policy.features(observation, memory), policy.mask(observation)
            values = policy.online.predict(features)
            if _json_value(observation) != expected["policy_observation"] or features.tolist() != expected["features"] or np.asarray(values, dtype=np.float32).tolist() != expected["q_values"] or mask.astype(bool).tolist() != expected["mask"] or _memory_snapshot(memory) != expected["memory_before"]:
                raise ValueError(f"M13.4 public inference mismatch at step {expected['step']}")
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if not bool(mask[int(macro)]) or macro.name != expected["macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.4 constrained action mismatch at step {expected['step']}")
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            if _memory_snapshot(memory) != expected["memory_after"]:
                raise ValueError(f"M13.4 memory mismatch at step {expected['step']}")
            observation = next_observation
    finally:
        env.close()
    return generic


def replay_m134_random_trace(path: str | Path, policy: SeededRandomM134Policy) -> ReplayResult:
    """Replay random choices from their recorded pre-choice PCG64 states."""

    generic = replay_and_validate(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("m134_protocol_fingerprint") != m134_protocol_fingerprint():
        raise ValueError("M13.4 random trace protocol fingerprint mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            state = expected.get("random_rng_state_before")
            if state is None:
                raise ValueError("M13.4 random trace lacks a pre-choice RNG state")
            policy.rng.bit_generator.state = state
            mask = policy.mask(observation)
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if mask.astype(bool).tolist() != expected["mask"] or macro.name != expected["macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.4 random replay mismatch at step {expected['step']}")
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            observation = next_observation
    finally:
        env.close()
    return generic


def _evaluate(policy: Any, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], *, trace_dir: Path, label: str, fingerprint: str | None) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    rows, manifest = {}, []
    for name, controls in conditions.items():
        episodes = []
        for seed in seeds:
            trace = trace_dir / label / name / f"seed-{seed}.jsonl"
            episodes.append(run_m134_episode(policy, seed=seed, condition=name, controls=controls, trace_path=trace, policy_fingerprint=fingerprint))
            manifest.append({"policy": label, "condition": name, "seed": str(seed), "path": str(trace.resolve()), "sha256": _file_sha256(trace)})
        rows[name] = _aggregate(episodes)
    return rows, manifest


def _ceiling_passes(rows: dict[str, dict[str, object]], seeds: tuple[int, ...]) -> bool:
    return all(int(row["survivals"]) == len(seeds) and int(row["completed_maintenance_episodes"]) == len(seeds) and int(row["completed_cycles"]["minimum_feed"]) >= 3 and int(row["completed_cycles"]["minimum_play"]) >= 3 and int(row["completed_cycles"]["minimum_rest"]) >= 2 for row in rows.values())


def _flat(rows: dict[str, dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(bool(episode[field])) for condition in M134_DEVELOPMENT_CONDITIONS for episode in rows[condition]["episodes_detail"]], dtype=np.float64)


def _development_gates(full: dict[str, dict[str, object]], baselines: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    conditions = {name: {"survival_pass": int(row["survivals"]) >= 36, "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 36, "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= .85, "recovery_pass": not bool(M134_DEVELOPMENT_CONDITIONS[name].get("event_relocation_on_first_pickup")) or int(row["forced_recovery_chains"]["completed"]) >= 36, "mask_pass": all(int(episode["interventions"].get("mask_violation", 0)) == 0 and int(episode["interventions"].get("redundant_go", 0)) == 0 for episode in row["episodes_detail"])} for name, row in full.items()}
    full_maintenance = _flat(full, "maintenance_complete")
    deltas = {name: float(np.mean(full_maintenance - _flat(rows, "maintenance_complete"))) for name, rows in baselines.items() if name != "macro_random"}
    random_survival_delta = float(np.mean(_flat(full, "survived") - _flat(baselines["macro_random"], "survived")))
    return {"condition_gates": conditions, "maintenance_deltas": deltas, "random_survival_delta": random_survival_delta, "passes": all(all(item.values()) for item in conditions.values()) and all(value >= .10 for value in deltas.values()) and random_survival_delta >= .20}


def m134_train(*, artifact_dir: str | Path) -> dict[str, object]:
    """Run the entire M13.4 development-only matrix without opening validation."""

    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.4 development artifact directory already exists")
    artifacts.mkdir(parents=True)
    coverage = m10_scan_coverage(M134_DEVELOPMENT_CONDITIONS, seeds=M134_DEVELOPMENT_SEEDS)
    ceiling, manifest = _evaluate(ScriptedM13Oracle(), M134_DEVELOPMENT_CONDITIONS, M134_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label="state_oracle_scripted_ceiling", fingerprint=None)
    replicates: dict[str, object] = {}
    for seed in M134_TRAINING_SEEDS:
        policies = {"complementary_full": CompactM134QPolicy(seed=seed), "interaction_only": CompactM134QPolicy(mask_mode="interaction_only", seed=seed), "mask_null": CompactM134QPolicy(mask_mode="none", seed=seed), "no_drive": CompactM134QPolicy(include_drives=False, seed=seed), "no_memory": CompactM134QPolicy(use_memory=False, seed=seed)}
        training = {name: policy.train() for name, policy in policies.items()}
        serialized = {name: write_m134_policy(artifacts / "policies" / f"seed-{seed}" / f"{name}.json", policy) for name, policy in policies.items()}
        results: dict[str, dict[str, dict[str, object]]] = {}
        for name, policy in policies.items():
            rows, traces = _evaluate(policy, M134_DEVELOPMENT_CONDITIONS, M134_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label=f"seed-{seed}/{name}", fingerprint=m134_policy_fingerprint(policy))
            results[name] = rows; manifest.extend(traces)
            restored = load_m134_policy(serialized[name]["path"])
            for trace in traces:
                replay_m134_trace(trace["path"], restored)
        random = SeededRandomM134Policy(seed=seed)
        random_rows, random_traces = _evaluate(random, M134_DEVELOPMENT_CONDITIONS, M134_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label=f"seed-{seed}/macro_random", fingerprint=f"pcg64-{seed}")
        results["macro_random"] = random_rows; manifest.extend(random_traces)
        for trace in random_traces:
            replay_m134_random_trace(trace["path"], SeededRandomM134Policy(seed=seed))
        gates = _development_gates(results["complementary_full"], {name: rows for name, rows in results.items() if name != "complementary_full"})
        replicates[str(seed)] = {"training": training, "policy_artifacts": serialized, "results": results, "gates": gates}
    passes = all(bool(row["passes"]) for row in coverage.values()) and _ceiling_passes(ceiling, M134_DEVELOPMENT_SEEDS) and all(bool(replicate["gates"]["passes"]) for replicate in replicates.values())
    return {"schema_version": "0.13.4", "protocol_version": M134_PROTOCOL_VERSION, "protocol_fingerprint": m134_protocol_fingerprint(), "split": "development_only", "coverage": coverage, "state_oracle_scripted_ceiling": ceiling, "replicates": replicates, "trace_manifest": manifest, "gate": {"passes": passes}, "limits": ["No validation or audit score is present.", "The mask constrains only public macro preconditions; drive scheduling remains learned."]}


def write_m134_training_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError("M13.4 development report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m134_train(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
