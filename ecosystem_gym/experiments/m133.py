"""M13.3 recovery-safe blocked-action reward baseline (development only)."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..actions import ActionOutcome
from ..drives import Drives
from ..env import EcosystemEnv
from .m13 import (
    M13_EPSILON_DECAY_EPISODES,
    M13_EPSILON_FINAL,
    M13_GAMMA,
    M13_TRAINING_EPISODES,
    M13Macro,
    M13Memory,
    M13Episode,
    RandomM13Policy,
    ScriptedM13Oracle,
    _aggregate,
    _clear_memory,
    _file_sha256,
    _memory_snapshot,
    _options,
    _safe,
    compile_macro,
)
from .m131 import m131_config
from .m132 import (
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_HIDDEN_DIM,
    M132_LEARNING_RATE,
    M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP,
    M132_TARGET_UPDATE_EVERY,
    M132_UPDATE_EVERY,
    _MLP,
    _Replay,
    encode_features,
)
from .m10 import m10_scan_coverage
from ..trajectory import ReplayResult, SCHEMA_VERSION, _json_value, replay_and_validate


M133_PROTOCOL_VERSION = "m13.3-blocked-action-r1"
M133_DEVELOPMENT_SEEDS = tuple(range(2_900, 2_940))
M133_VALIDATION_SEEDS = tuple(range(3_000, 3_020))
M133_AUDIT_SEEDS = tuple(range(3_100, 3_120))
M133_TRAINING_SEED = 20_260_808


def m133_epsilon(episode: int) -> float:
    """The frozen inclusive linear schedule: exactly .05 at episode 9,000."""

    if episode < 0:
        raise ValueError("episode must be non-negative")
    if episode >= M13_EPSILON_DECAY_EPISODES:
        return M13_EPSILON_FINAL
    return max(
        M13_EPSILON_FINAL,
        1.0 - (1.0 - M13_EPSILON_FINAL) * episode / M13_EPSILON_DECAY_EPISODES,
    )


class SeededRandomM133Policy:
    """Predeclared macro-random comparator with replayable PCG64 state."""

    include_drives = True
    use_memory = False

    def __init__(self, seed: int = M133_TRAINING_SEED + 4) -> None:
        self.seed = seed
        self.config = m133_config()
        self.rng = np.random.default_rng(seed)

    def reset(self) -> M13Memory:
        return M13Memory()

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        del observation, memory
        return M13Macro(int(self.rng.integers(len(M13Macro))))

    def observe(self, memory: M13Memory, **_: Any) -> None:
        _clear_memory(memory)


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M133_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m133_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m133_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m133_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m133_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M133_VALIDATION_CONDITIONS = {name: {**controls, "layout_id": f"m133_validation_{controls['layout_id'].removeprefix('m133_dev_')}"} for name, controls in M133_DEVELOPMENT_CONDITIONS.items()}
M133_AUDIT_CONDITIONS = {name: {**controls, "layout_id": f"m133_audit_{controls['layout_id'].removeprefix('m133_dev_')}"} for name, controls in M133_DEVELOPMENT_CONDITIONS.items()}


def m133_config(*, blocked_penalty: float = 0.10):
    """M13.1 rewards plus M13.3's recovery-safe blocked-action penalty."""

    return replace(
        m131_config(),
        blocked_action_penalty=blocked_penalty,
        exempt_forced_relocation_blocked_penalty=blocked_penalty > 0.0,
    )


def _duration(action: dict[str, np.ndarray | int]) -> float:
    return float(np.asarray(action["duration"], dtype=np.float32).item())


def advance_m133_memory(
    memory: M13Memory,
    *,
    observation_before: dict[str, Any],
    macro: M13Macro,
    action: dict[str, np.ndarray | int],
    observation_after: dict[str, Any],
    config: Any,
) -> None:
    """Mirror only public, post-duration eligibility used by the environment."""

    outcome = list(ActionOutcome)[int(observation_after["prior_outcome"])]
    duration = _duration(action)
    drives = np.asarray(observation_before["drives"], dtype=np.float32)
    evolved = Drives(satiety=float(drives[0]), energy=float(drives[1]), boredom=float(drives[2])).evolve(config, duration)
    if macro is M13Macro.CONSUME and outcome is ActionOutcome.SUCCESS:
        memory.feed_count = min(3, memory.feed_count + 1)
        memory.food_cooldown_bucket = 0.0
        memory.pending_recovery = False
    elif float(memory.food_cooldown_bucket) < config.food_respawn_seconds:
        memory.food_cooldown_bucket = min(config.food_respawn_seconds, float(memory.food_cooldown_bucket) + duration)
    if macro is M13Macro.PLAY and outcome is ActionOutcome.SUCCESS and evolved.boredom >= config.play_success_boredom_threshold:
        memory.play_count = min(3, memory.play_count + 1)
    if macro is M13Macro.REST and outcome is ActionOutcome.SUCCESS and evolved.energy <= config.rest_cycle_energy_threshold:
        memory.rest_count = min(2, memory.rest_count + 1)
    if macro is M13Macro.PICK_UP and outcome is ActionOutcome.BLOCKED:
        memory.pending_recovery = True
    memory.last_macro = macro


class CompactM133QPolicy:
    """M13.2 network/feature architecture with corrected M13.3 memory timing."""

    def __init__(self, *, include_drives: bool = True, use_memory: bool = True, blocked_penalty: float = 0.10, seed: int = M133_TRAINING_SEED) -> None:
        self.include_drives, self.use_memory, self.blocked_penalty, self.seed = include_drives, use_memory, blocked_penalty, seed
        self.config = m133_config(blocked_penalty=blocked_penalty)
        self.online = _MLP(np.random.default_rng(seed))
        self.target = self.online.copy()
        self.update_count = 0

    def reset(self) -> M13Memory:
        return M13Memory()

    def features(self, observation: dict[str, Any], memory: M13Memory) -> np.ndarray:
        return encode_features(observation, memory, include_drives=self.include_drives, use_memory=self.use_memory)

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(int(np.argmax(self.online.predict(self.features(observation, memory)))))

    def observe(self, memory: M13Memory, *, observation_before: dict[str, Any], macro: M13Macro, action: dict[str, np.ndarray | int], observation_after: dict[str, Any]) -> None:
        advance_m133_memory(memory, observation_before=observation_before, macro=macro, action=action, observation_after=observation_after, config=self.config)
        if not self.use_memory:
            _clear_memory(memory)

    def _update(self, replay: _Replay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, done = replay.sample(rng)
        selected = np.argmax(self.online.predict(next_features), axis=1)
        targets = rewards + M13_GAMMA * (~done) * self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % M132_TARGET_UPDATE_EVERY == 0:
            self.target = self.online.copy()
        return loss

    def train(self, *, episodes: int = M13_TRAINING_EPISODES) -> dict[str, float]:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13.3 training budget is frozen")
        rng = np.random.default_rng(self.seed)
        self.online, self.target, self.update_count = _MLP(rng), None, 0
        self.target = self.online.copy()
        replay, decisions, losses = _Replay(), 0, []
        env = EcosystemEnv(self.config)
        items = tuple(M133_DEVELOPMENT_CONDITIONS.items())
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                seed = M133_DEVELOPMENT_SEEDS[(episode // len(items)) % len(M133_DEVELOPMENT_SEEDS)]
                observation, _ = env.reset(seed=seed, options=_options(controls))
                memory = self.reset()
                epsilon = m133_epsilon(episode)
                for _ in range(env.config.max_episode_steps):
                    features = self.features(observation, memory)
                    macro = M13Macro(int(rng.integers(len(M13Macro)))) if rng.random() < epsilon else self.choose(observation, memory)
                    action = compile_macro(macro, observation, env.config)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    self.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
                    replay.add(features, int(macro), reward, self.features(next_observation, memory), terminated or truncated)
                    decisions += 1
                    if replay.size >= M132_REPLAY_WARMUP and decisions % M132_UPDATE_EVERY == 0:
                        losses.append(self._update(replay, rng))
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()
        return {"decisions": float(decisions), "updates": float(self.update_count), "mean_huber_loss": float(np.mean(losses)) if losses else 0.0}


def m133_policy_fingerprint(policy: CompactM133QPolicy) -> str:
    digest = hashlib.sha256(M133_PROTOCOL_VERSION.encode())
    digest.update(json.dumps({"include_drives": policy.include_drives, "use_memory": policy.use_memory, "blocked_penalty": policy.blocked_penalty}, sort_keys=True).encode())
    for key in sorted(policy.online.params):
        digest.update(policy.online.params[key].tobytes())
    return digest.hexdigest()


def m133_protocol_fingerprint() -> str:
    source_dir = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in ("m133.py", "m132.py", "m13.py", "env.py", "config.py")}
    payload = {"version": M133_PROTOCOL_VERSION, "config": asdict(m133_config()), "development_seeds": M133_DEVELOPMENT_SEEDS, "validation_seeds": M133_VALIDATION_SEEDS, "audit_seeds": M133_AUDIT_SEEDS, "development_conditions": M133_DEVELOPMENT_CONDITIONS, "validation_conditions": M133_VALIDATION_CONDITIONS, "audit_conditions": M133_AUDIT_CONDITIONS, "features": M132_FEATURE_DIM, "network": [M132_FEATURE_DIM, M132_HIDDEN_DIM, M132_HIDDEN_DIM, len(M13Macro)], "source_hashes": source_hashes, "training": {"episodes": M13_TRAINING_EPISODES, "seed": M133_TRAINING_SEED, "epsilon": {"start": 1.0, "final": M13_EPSILON_FINAL, "inclusive_decay_episodes": M13_EPSILON_DECAY_EPISODES}, "gamma": M13_GAMMA, "replay_capacity": M132_REPLAY_CAPACITY, "warmup": M132_REPLAY_WARMUP, "batch": M132_BATCH_SIZE, "update_every": M132_UPDATE_EVERY, "target_update_every": M132_TARGET_UPDATE_EVERY, "learning_rate": M132_LEARNING_RATE, "adam": [0.9, 0.999, 1e-8], "huber_delta": 1.0}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_m133_policy(path: str | Path, policy: CompactM133QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "m133-policy-v1", "protocol_fingerprint": m133_protocol_fingerprint(), "include_drives": policy.include_drives, "use_memory": policy.use_memory, "blocked_penalty": policy.blocked_penalty, "network": {key: value.tolist() for key, value in policy.online.params.items()}}
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": _file_sha256(output), "policy_fingerprint": m133_policy_fingerprint(policy)}


def load_m133_policy(path: str | Path) -> CompactM133QPolicy:
    """Load only an exact M13.3 artifact produced under this frozen protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m133-policy-v1" or payload.get("protocol_fingerprint") != m133_protocol_fingerprint():
        raise ValueError("M13.3 policy artifact does not match the frozen protocol")
    policy = CompactM133QPolicy(
        include_drives=bool(payload["include_drives"]),
        use_memory=bool(payload["use_memory"]),
        blocked_penalty=float(payload["blocked_penalty"]),
    )
    for key, values in payload["network"].items():
        if key not in policy.online.params or np.asarray(values).shape != policy.online.params[key].shape:
            raise ValueError("M13.3 policy artifact has an invalid parameter")
        policy.online.params[key] = np.asarray(values, dtype=np.float32)
    policy.target = policy.online.copy()
    return policy


def _blocked_classification(info: dict[str, Any]) -> tuple[bool, bool]:
    blocked = info["outcome"] == ActionOutcome.BLOCKED.value
    exempt = blocked and info["disturbance"] == "food_relocated"
    return blocked and not exempt, exempt


def run_m133_episode(
    policy: Any,
    *,
    seed: int,
    condition: str,
    controls: dict[str, Any],
    env: EcosystemEnv | None = None,
    trace_path: str | Path | None = None,
    policy_fingerprint: str | None = None,
) -> M13Episode:
    """Run a frozen M13.3 policy while recording its public inference proof."""

    owns_env = env is None
    if env is None:
        env = EcosystemEnv(policy.config if isinstance(policy, CompactM133QPolicy) else m133_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        records: list[dict[str, object]] = []
        safe_steps = 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        ordinary_blocked_run = 0
        blocked_loop = False
        interventions: dict[str, int] = {}
        for step in range(1, env.config.max_episode_steps + 1):
            memory_before = _memory_snapshot(memory)
            features = policy.features(observation, memory) if isinstance(policy, CompactM133QPolicy) else None
            values = policy.online.predict(features) if isinstance(policy, CompactM133QPolicy) else None
            random_state = _json_value(policy.rng.bit_generator.state) if isinstance(policy, SeededRandomM133Policy) else None
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            next_observation, reward, terminated, truncated, info = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            ordinary_blocked, exempt_blocked = _blocked_classification(info)
            ordinary_blocked_run = ordinary_blocked_run + 1 if ordinary_blocked and records and records[-1]["macro"] == macro.name else int(ordinary_blocked)
            blocked_loop = blocked_loop or ordinary_blocked_run >= 8
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen = relocation_seen or relocation_now
            stale_pickup = stale_pickup or exempt_blocked
            consumed_after_relocation = consumed_after_relocation or (relocation_seen and macro is M13Macro.CONSUME and info["outcome"] == ActionOutcome.SUCCESS.value)
            if relocation_now:
                interventions["food_relocated"] = interventions.get("food_relocated", 0) + 1
            if info["resource_event"] is not None:
                key = str(info["resource_event"])
                interventions[key] = interventions.get(key, 0) + 1
            if info["outcome"] == ActionOutcome.BLOCKED.value:
                interventions["blocked_action"] = interventions.get("blocked_action", 0) + 1
            safe_steps += int(_safe(np.asarray(next_observation["drives"], dtype=np.float32)))
            records.append({
                "step": step,
                "policy_observation": _json_value(observation),
                "features": None if features is None else features.tolist(),
                "q_values": None if values is None else np.asarray(values, dtype=np.float32).tolist(),
                "random_rng_state_before": random_state,
                "memory_before": memory_before,
                "memory_after": _memory_snapshot(memory),
                "macro": macro.name,
                "action": _json_value(action),
                "reward": float(reward),
                "outcome": info["outcome"],
                "ordinary_blocked": ordinary_blocked,
                "exempt_forced_relocation_blocked": exempt_blocked,
                "disturbance": info["disturbance"],
                "post_disturbance_completion": bool(info["post_disturbance_completion"]),
                "resource_event": info["resource_event"],
                "feed_cycles": int(info["feed_cycles"]),
                "play_cycles": int(info["play_cycles"]),
                "rest_cycles": int(info["rest_cycles"]),
                "food_available": bool(info["food_available"]),
                "camera_sector": info["camera_sector"],
                "task_success": bool(info["task_success"]),
                "environment_version": str(info["environment_version"]),
                "observation_after": _json_value(next_observation),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            observation = next_observation
            if terminated or truncated:
                drives = np.asarray(observation["drives"], dtype=np.float32)
                terminal_cause = "survived_horizon" if truncated and bool(info["survived"]) else "energy_depleted" if drives[1] <= 0.0 else "satiety_depleted" if drives[0] <= 0.0 else "terminated"
                episode = M13Episode(
                    seed=seed, condition=condition, survived=bool(info["survived"]), maintenance_complete=bool(info["maintenance_complete"]), terminal_cause=terminal_cause,
                    feed_cycles=int(info["feed_cycles"]), play_cycles=int(info["play_cycles"]), rest_cycles=int(info["rest_cycles"]), safe_drive_fraction=safe_steps / step,
                    forced_recovery={"required": bool(controls.get("event_relocation_on_first_pickup")), "stale_pickup": stale_pickup, "consumed_after_relocation": consumed_after_relocation, "complete": stale_pickup and consumed_after_relocation},
                    interventions={**interventions, "blocked_loop": int(blocked_loop)}, steps=tuple(records),
                )
                if trace_path is not None:
                    _write_m133_trace(trace_path, episode, controls=controls, policy_fingerprint=policy_fingerprint, config=env.config)
                return episode
        raise AssertionError("M13.3 episode did not terminate")
    finally:
        if owns_env:
            env.close()


def _write_m133_trace(path: str | Path, episode: M13Episode, *, controls: dict[str, Any], policy_fingerprint: str | None, config: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "record_type": "episode_metadata", "episode_id": f"m133-{episode.condition}-seed-{episode.seed}", "seed": episode.seed, "reset_options": _json_value(_options(controls)), "observation_mode": "state_oracle", "config": asdict(config), "m133_protocol_fingerprint": m133_protocol_fingerprint(), "m133_policy_fingerprint": policy_fingerprint}
    rows = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for record in episode.steps:
        row = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": header["episode_id"],
            "seed": episode.seed,
            "observation_mode": "state_oracle",
            "observation": record["observation_after"],
            **record,
        }
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def replay_m133_trace(path: str | Path, policy: CompactM133QPolicy) -> ReplayResult:
    """Strictly replay both M13.3 transitions and learned inference."""

    generic = replay_and_validate(path)
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = records[0], records[1:]
    if header.get("m133_protocol_fingerprint") != m133_protocol_fingerprint() or header.get("m133_policy_fingerprint") != m133_policy_fingerprint(policy):
        raise ValueError("M13.3 trace fingerprint mismatch")
    if asdict(policy.config) != header.get("config"):
        raise ValueError("M13.3 trace reward configuration mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            if _json_value(observation) != expected["policy_observation"]:
                raise ValueError(f"M13.3 policy observation mismatch at step {expected['step']}")
            features = policy.features(observation, memory)
            values = policy.online.predict(features)
            if features.tolist() != expected["features"] or np.asarray(values, dtype=np.float32).tolist() != expected["q_values"]:
                raise ValueError(f"M13.3 policy vector mismatch at step {expected['step']}")
            if _memory_snapshot(memory) != expected["memory_before"]:
                raise ValueError(f"M13.3 policy memory mismatch at step {expected['step']}")
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            if macro.name != expected["macro"] or _json_value(action) != expected["action"]:
                raise ValueError(f"M13.3 policy action mismatch at step {expected['step']}")
            next_observation, _, _, _, info = env.step(action)
            ordinary_blocked, exempt_blocked = _blocked_classification(info)
            if ordinary_blocked != expected["ordinary_blocked"] or exempt_blocked != expected["exempt_forced_relocation_blocked"]:
                raise ValueError(f"M13.3 blocked classification mismatch at step {expected['step']}")
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            if _memory_snapshot(memory) != expected["memory_after"]:
                raise ValueError(f"M13.3 policy memory update mismatch at step {expected['step']}")
            observation = next_observation
    finally:
        env.close()
    return generic


def replay_m133_random_trace(path: str | Path, policy: SeededRandomM133Policy) -> ReplayResult:
    """Replay random comparator choices from every recorded pre-choice PCG64 state."""

    generic = replay_and_validate(path)
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    header, steps = records[0], records[1:]
    if header.get("m133_protocol_fingerprint") != m133_protocol_fingerprint():
        raise ValueError("M13.3 random trace protocol fingerprint mismatch")
    env = EcosystemEnv(policy.config)
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        for expected in steps:
            state = expected.get("random_rng_state_before")
            if state is None:
                raise ValueError("M13.3 random trace lacks a pre-choice RNG state")
            policy.rng.bit_generator.state = state
            macro = policy.choose(observation, memory)
            if macro.name != expected["macro"]:
                raise ValueError(f"M13.3 random action mismatch at step {expected['step']}")
            action = compile_macro(macro, observation, env.config)
            next_observation, _, _, _, _ = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            observation = next_observation
    finally:
        env.close()
    return generic


def _evaluate_m133(policy: Any, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], *, trace_dir: Path | None, label: str, fingerprint: str | None) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    results: dict[str, dict[str, object]] = {}
    manifest: list[dict[str, str]] = []
    env = EcosystemEnv(policy.config if hasattr(policy, "config") else m133_config())
    try:
        for name, controls in conditions.items():
            episodes: list[M13Episode] = []
            for seed in seeds:
                trace = None if trace_dir is None else trace_dir / label / name / f"seed-{seed}.jsonl"
                episodes.append(run_m133_episode(policy, seed=seed, condition=name, controls=controls, env=env, trace_path=trace, policy_fingerprint=fingerprint))
                if trace is not None:
                    manifest.append({"policy": label, "condition": name, "seed": str(seed), "path": str(trace.resolve()), "sha256": _file_sha256(trace)})
            results[name] = _aggregate(episodes)
    finally:
        env.close()
    return results, manifest


def _m133_ceiling_passes(rows: dict[str, dict[str, object]], seeds: tuple[int, ...]) -> bool:
    return all(
        int(row["survivals"]) == len(seeds)
        and int(row["completed_maintenance_episodes"]) == len(seeds)
        and int(row["completed_cycles"]["minimum_feed"]) >= 3
        and int(row["completed_cycles"]["minimum_play"]) >= 3
        and int(row["completed_cycles"]["minimum_rest"]) >= 2
        for row in rows.values()
    )


def _m133_development_gates(rows: dict[str, dict[str, object]]) -> dict[str, dict[str, bool]]:
    return {
        name: {
            "survival_pass": int(row["survivals"]) >= 36,
            "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 36,
            "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= 0.85,
            "recovery_pass": not bool(M133_DEVELOPMENT_CONDITIONS[name].get("event_relocation_on_first_pickup")) or int(row["forced_recovery_chains"]["completed"]) >= 36,
            "blocked_loop_pass": all(int(episode["interventions"].get("blocked_loop", 0)) == 0 for episode in row["episodes_detail"]),
        }
        for name, row in rows.items()
    }


def m133_train(*, artifact_dir: str | Path) -> dict[str, object]:
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.3 development artifact directory already exists")
    artifacts.mkdir(parents=True)
    policies = {
        "full_state_oracle_dqn": CompactM133QPolicy(),
        "reward_null_dqn": CompactM133QPolicy(blocked_penalty=0.0, seed=M133_TRAINING_SEED + 1),
        "no_drive_dqn": CompactM133QPolicy(include_drives=False, seed=M133_TRAINING_SEED + 2),
        "no_memory_dqn": CompactM133QPolicy(use_memory=False, seed=M133_TRAINING_SEED + 3),
    }
    training = {name: policy.train() for name, policy in policies.items()}
    serialized = {name: write_m133_policy(artifacts / "policies" / f"{name}.json", policy) for name, policy in policies.items()}
    coverage = m10_scan_coverage(M133_DEVELOPMENT_CONDITIONS, seeds=M133_DEVELOPMENT_SEEDS)
    ceiling, manifest = _evaluate_m133(ScriptedM13Oracle(), M133_DEVELOPMENT_CONDITIONS, M133_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label="state_oracle_scripted_ceiling", fingerprint=None)
    results: dict[str, dict[str, dict[str, object]]] = {}
    for name, policy in policies.items():
        rows, traces = _evaluate_m133(policy, M133_DEVELOPMENT_CONDITIONS, M133_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label=name, fingerprint=m133_policy_fingerprint(policy))
        results[name] = rows
        manifest.extend(traces)
        restored = load_m133_policy(serialized[name]["path"])
        for trace in traces:
            replay_m133_trace(trace["path"], restored)
    random_policy = SeededRandomM133Policy()
    random_rows, random_traces = _evaluate_m133(random_policy, M133_DEVELOPMENT_CONDITIONS, M133_DEVELOPMENT_SEEDS, trace_dir=artifacts / "traces", label="macro_random", fingerprint=f"pcg64-{random_policy.seed}")
    results["macro_random"] = random_rows
    manifest.extend(random_traces)
    for trace in random_traces:
        replay_m133_random_trace(trace["path"], SeededRandomM133Policy())
    for trace in manifest:
        if trace["policy"] == "state_oracle_scripted_ceiling":
            replay_and_validate(trace["path"])
    gates = _m133_development_gates(results["full_state_oracle_dqn"])
    passes = all(bool(row["passes"]) for row in coverage.values()) and _m133_ceiling_passes(ceiling, M133_DEVELOPMENT_SEEDS) and all(all(item.values()) for item in gates.values())
    return {"schema_version": "0.13.3", "protocol_version": M133_PROTOCOL_VERSION, "protocol_fingerprint": m133_protocol_fingerprint(), "split": "development_only", "training": training, "policy_artifacts": serialized, "coverage": coverage, "state_oracle_scripted_ceiling": ceiling, "results": results, "condition_gates": gates, "trace_manifest": manifest, "gate": {"passes": passes}, "limits": ["No validation or audit score is present.", "Reward-null isolates the blocked penalty from M13.3 conformance repairs."]}


def write_m133_training_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError("M13.3 development report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m133_train(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
