"""M13.2 compact-state Double-DQN under the frozen M13.2 protocol."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionOutcome
from .env import EcosystemEnv
from .m13 import (
    M13_EPSILON_DECAY_EPISODES,
    M13_EPSILON_FINAL,
    M13_GAMMA,
    M13_TRAINING_EPISODES,
    M13Macro,
    M13Memory,
    _clear_memory,
    _file_sha256,
    _options,
    advance_memory,
    compile_macro,
)
from .m131 import m131_config


M132_PROTOCOL_VERSION = "m13.2-compact-q-r1"
M132_DEVELOPMENT_SEEDS = tuple(range(2_600, 2_640))
M132_VALIDATION_SEEDS = tuple(range(2_700, 2_720))
M132_AUDIT_SEEDS = tuple(range(2_800, 2_820))
M132_TRAINING_SEED = 20_260_807
M132_FEATURE_DIM = 30
M132_HIDDEN_DIM = 64
M132_REPLAY_CAPACITY = 100_000
M132_REPLAY_WARMUP = 1_000
M132_BATCH_SIZE = 128
M132_UPDATE_EVERY = 256
M132_TARGET_UPDATE_EVERY = 1_000
M132_LEARNING_RATE = 3e-4


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M132_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m132_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m132_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m132_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m132_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M132_VALIDATION_CONDITIONS = {name: {**controls, "layout_id": f"m132_validation_{controls['layout_id'].removeprefix('m132_dev_')}"} for name, controls in M132_DEVELOPMENT_CONDITIONS.items()}
M132_AUDIT_CONDITIONS = {name: {**controls, "layout_id": f"m132_audit_{controls['layout_id'].removeprefix('m132_dev_')}"} for name, controls in M132_DEVELOPMENT_CONDITIONS.items()}


def m132_config():
    """M13.2 deliberately reuses the complete M13.1 reward configuration."""

    return m131_config()


def _outcome_index(observation: dict[str, Any]) -> int:
    return int(observation["prior_outcome"])


def encode_features(observation: dict[str, Any], memory: M13Memory, *, include_drives: bool = True, use_memory: bool = True) -> np.ndarray:
    """The exact M13.2 public 30-feature inference representation."""

    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    targets = np.concatenate([np.asarray(observation[key], dtype=np.float32) - agent for key in ("food_xy", "toy_xy", "rest_xy")])
    relative = np.clip(targets, -1.8, 1.8) / 1.8
    drives = np.asarray(observation["drives"], dtype=np.float32) if include_drives else np.zeros(3, dtype=np.float32)
    outcome = np.zeros(len(ActionOutcome), dtype=np.float32)
    outcome[_outcome_index(observation)] = 1.0
    active = memory if use_memory else M13Memory()
    counts = np.asarray([active.feed_count / 3, active.play_count / 3, active.rest_count / 2, active.food_cooldown_bucket / 5, float(active.pending_recovery)], dtype=np.float32)
    last_macro = np.zeros(len(M13Macro) + 1, dtype=np.float32)
    last_macro[len(M13Macro) if active.last_macro is None else int(active.last_macro)] = 1.0
    features = np.concatenate([relative, drives, np.asarray([observation["holding_food"]], dtype=np.float32), outcome, counts, last_macro]).astype(np.float32)
    if features.shape != (M132_FEATURE_DIM,):
        raise AssertionError(f"expected {M132_FEATURE_DIM} features, got {features.shape}")
    return features


class _MLP:
    """Small fully deterministic NumPy MLP with Adam and Huber regression."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.params = {
            "w1": rng.normal(0.0, np.sqrt(2 / M132_FEATURE_DIM), (M132_FEATURE_DIM, M132_HIDDEN_DIM)).astype(np.float32),
            "b1": np.zeros(M132_HIDDEN_DIM, dtype=np.float32),
            "w2": rng.normal(0.0, np.sqrt(2 / M132_HIDDEN_DIM), (M132_HIDDEN_DIM, M132_HIDDEN_DIM)).astype(np.float32),
            "b2": np.zeros(M132_HIDDEN_DIM, dtype=np.float32),
            "w3": rng.normal(0.0, np.sqrt(2 / M132_HIDDEN_DIM), (M132_HIDDEN_DIM, len(M13Macro))).astype(np.float32),
            "b3": np.zeros(len(M13Macro), dtype=np.float32),
        }
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step = 0

    def copy(self) -> _MLP:
        other = object.__new__(_MLP)
        other.params = {key: value.copy() for key, value in self.params.items()}
        other.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        other.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        other.step = 0
        return other

    def predict(self, features: np.ndarray) -> np.ndarray:
        hidden1 = np.maximum(features @ self.params["w1"] + self.params["b1"], 0.0)
        hidden2 = np.maximum(hidden1 @ self.params["w2"] + self.params["b2"], 0.0)
        return hidden2 @ self.params["w3"] + self.params["b3"]

    def update(self, features: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        pre1 = features @ self.params["w1"] + self.params["b1"]
        hidden1 = np.maximum(pre1, 0.0)
        pre2 = hidden1 @ self.params["w2"] + self.params["b2"]
        hidden2 = np.maximum(pre2, 0.0)
        values = hidden2 @ self.params["w3"] + self.params["b3"]
        row = np.arange(features.shape[0])
        error = values[row, actions] - targets
        gradient = np.where(np.abs(error) <= 1.0, error, np.sign(error)).astype(np.float32) / features.shape[0]
        output_gradient = np.zeros_like(values)
        output_gradient[row, actions] = gradient
        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = hidden2.T @ output_gradient
        gradients["b3"] = output_gradient.sum(axis=0)
        middle = (output_gradient @ self.params["w3"].T) * (pre2 > 0.0)
        gradients["w2"] = hidden1.T @ middle
        gradients["b2"] = middle.sum(axis=0)
        first = (middle @ self.params["w2"].T) * (pre1 > 0.0)
        gradients["w1"] = features.T @ first
        gradients["b1"] = first.sum(axis=0)
        self.step += 1
        for key, grad in gradients.items():
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad * grad
            corrected_m = self.m[key] / (1.0 - 0.9**self.step)
            corrected_v = self.v[key] / (1.0 - 0.999**self.step)
            self.params[key] -= M132_LEARNING_RATE * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        return float(np.mean(np.where(np.abs(error) <= 1.0, 0.5 * error * error, np.abs(error) - 0.5)))


class _Replay:
    def __init__(self) -> None:
        self.features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.actions = np.empty(M132_REPLAY_CAPACITY, dtype=np.int64)
        self.rewards = np.empty(M132_REPLAY_CAPACITY, dtype=np.float32)
        self.next_features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.done = np.empty(M132_REPLAY_CAPACITY, dtype=np.bool_)
        self.size = self.cursor = 0

    def add(self, features: np.ndarray, action: int, reward: float, next_features: np.ndarray, done: bool) -> None:
        index = self.cursor
        self.features[index], self.actions[index], self.rewards[index], self.next_features[index], self.done[index] = features, action, reward, next_features, done
        self.cursor = (index + 1) % M132_REPLAY_CAPACITY
        self.size = min(self.size + 1, M132_REPLAY_CAPACITY)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(self.size, size=M132_BATCH_SIZE, replace=False)
        return self.features[indices], self.actions[indices], self.rewards[indices], self.next_features[indices], self.done[indices]


class CompactM132QPolicy:
    """M13.2 policy: only its continuous Q representation differs from M13.1."""

    def __init__(self, *, include_drives: bool = True, use_memory: bool = True, seed: int = M132_TRAINING_SEED) -> None:
        self.include_drives, self.use_memory, self.seed = include_drives, use_memory, seed
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
        advance_memory(memory, observation_before=observation_before, macro=macro, action=action, observation_after=observation_after)
        if not self.use_memory:
            _clear_memory(memory)

    def _update(self, replay: _Replay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, done = replay.sample(rng)
        selected = np.argmax(self.online.predict(next_features), axis=1)
        bootstrap = self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        targets = rewards + M13_GAMMA * (~done) * bootstrap
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % M132_TARGET_UPDATE_EVERY == 0:
            self.target = self.online.copy()
        return loss

    def train(self, *, episodes: int = M13_TRAINING_EPISODES, seed: int | None = None) -> dict[str, float]:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13.2 training budget is frozen")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        self.online = _MLP(rng)
        self.target = self.online.copy()
        self.update_count = 0
        replay, decisions, losses = _Replay(), 0, []
        env = EcosystemEnv(m132_config())
        items = tuple(M132_DEVELOPMENT_CONDITIONS.items())
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                episode_seed = M132_DEVELOPMENT_SEEDS[(episode // len(items)) % len(M132_DEVELOPMENT_SEEDS)]
                observation, _ = env.reset(seed=episode_seed, options=_options(controls))
                memory = self.reset()
                epsilon = max(M13_EPSILON_FINAL, 1.0 - episode / M13_EPSILON_DECAY_EPISODES)
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


def m132_policy_fingerprint(policy: CompactM132QPolicy) -> str:
    digest = hashlib.sha256(M132_PROTOCOL_VERSION.encode())
    digest.update(json.dumps({"include_drives": policy.include_drives, "use_memory": policy.use_memory}, sort_keys=True).encode())
    for key in sorted(policy.online.params):
        digest.update(policy.online.params[key].tobytes())
    return digest.hexdigest()


def m132_protocol_fingerprint() -> str:
    payload = {"version": M132_PROTOCOL_VERSION, "config": asdict(m132_config()), "development_seeds": M132_DEVELOPMENT_SEEDS, "validation_seeds": M132_VALIDATION_SEEDS, "audit_seeds": M132_AUDIT_SEEDS, "development_conditions": M132_DEVELOPMENT_CONDITIONS, "validation_conditions": M132_VALIDATION_CONDITIONS, "audit_conditions": M132_AUDIT_CONDITIONS, "features": M132_FEATURE_DIM, "network": [M132_FEATURE_DIM, M132_HIDDEN_DIM, M132_HIDDEN_DIM, len(M13Macro)], "training": {"episodes": M13_TRAINING_EPISODES, "seed": M132_TRAINING_SEED, "gamma": M13_GAMMA, "replay_capacity": M132_REPLAY_CAPACITY, "warmup": M132_REPLAY_WARMUP, "batch": M132_BATCH_SIZE, "update_every": M132_UPDATE_EVERY, "target_update_every": M132_TARGET_UPDATE_EVERY, "learning_rate": M132_LEARNING_RATE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_m132_policy(path: str | Path, policy: CompactM132QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "m132-policy-v1", "protocol_fingerprint": m132_protocol_fingerprint(), "include_drives": policy.include_drives, "use_memory": policy.use_memory, "network": {key: value.tolist() for key, value in policy.online.params.items()}}
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": _file_sha256(output), "policy_fingerprint": m132_policy_fingerprint(policy)}


def load_m132_policy(path: str | Path) -> CompactM132QPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m132-policy-v1" or payload.get("protocol_fingerprint") != m132_protocol_fingerprint():
        raise ValueError("M13.2 policy artifact does not match the frozen protocol")
    policy = CompactM132QPolicy(include_drives=bool(payload["include_drives"]), use_memory=bool(payload["use_memory"]))
    for key, values in payload["network"].items():
        if key not in policy.online.params or np.asarray(values).shape != policy.online.params[key].shape:
            raise ValueError("M13.2 policy artifact has an invalid parameter")
        policy.online.params[key] = np.asarray(values, dtype=np.float32)
    policy.target = policy.online.copy()
    return policy


def m132_train(*, artifact_dir: str | Path) -> dict[str, object]:
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError("M13.2 development artifact directory already exists")
    artifacts.mkdir(parents=True)
    policies = {"full_state_oracle_dqn": CompactM132QPolicy(), "no_drive_dqn": CompactM132QPolicy(include_drives=False, seed=M132_TRAINING_SEED + 1), "no_memory_dqn": CompactM132QPolicy(use_memory=False, seed=M132_TRAINING_SEED + 2)}
    training = {name: policy.train() for name, policy in policies.items()}
    serialized = {name: write_m132_policy(artifacts / "policies" / f"{name}.json", policy) for name, policy in policies.items()}
    return {"schema_version": "0.13.2", "protocol_fingerprint": m132_protocol_fingerprint(), "split": "development_only", "training": training, "policy_artifacts": serialized, "limits": ["No validation or audit score is present.", "M13.2 retains M13.1 reward and action semantics; only value representation changed."]}


def write_m132_training_report(path: str | Path) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError("M13.2 development report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = m132_train(artifact_dir=output.parent / f"{output.stem}-artifacts")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
