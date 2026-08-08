"""Corrected deterministic masked semi-Markov NumPy PPO for M13.11-r2.

The frozen :mod:`ppo` implementation is intentionally left untouched.  This
version exposes its objective gradients so the numerical correctness contract
can be tested directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import MaintenanceMacro, MaintenanceMemory, advance_memory, encode_features, eligible_macro_mask

FEATURES, ACTIONS, HIDDEN = 30, len(MaintenanceMacro), 64
GAMMA, LAMBDA, CLIP, VALUE_COEF, ENTROPY_COEF = .99, .95, .20, .50, .01


def _finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(masks, dtype=np.bool_)
    if values.shape != mask.shape or values.shape[-1] != ACTIONS or not np.all(mask.any(axis=-1)):
        raise ValueError("logits and non-empty action masks must have matching final shape")
    masked = np.where(mask, values, -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.where(mask, np.exp(shifted), 0.0)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def semimarkov_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, durations: np.ndarray, done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rewards, values, next_values = (np.asarray(row, dtype=np.float64) for row in (rewards, values, next_values))
    durations, done = np.asarray(durations, dtype=np.float64), np.asarray(done, dtype=np.bool_)
    if not (rewards.shape == values.shape == next_values.shape == durations.shape == done.shape):
        raise ValueError("GAE arrays must share a shape")
    if np.any(durations <= 0) or not np.all(np.isfinite(durations)):
        raise ValueError("GAE durations must be finite and positive")
    advantages = np.zeros_like(rewards)
    carry = 0.0
    for index in range(rewards.size - 1, -1, -1):
        discount = GAMMA ** durations[index]
        continuation = 0.0 if done[index] else 1.0
        delta = rewards[index] + continuation * discount * next_values[index] - values[index]
        carry = delta + continuation * discount * LAMBDA * carry
        advantages[index] = carry
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


@dataclass(slots=True)
class Rollout:
    features: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    durations: np.ndarray
    done: np.ndarray
    values: np.ndarray
    next_values: np.ndarray


class CorrectedMaskedPPOPolicy:
    """30→64→64 masked actor-critic with correct PPO minimization gradient."""

    def __init__(self, *, seed: int, config: Any, protocol_fingerprint: str = "development") -> None:
        self.seed, self.config, self.protocol_fingerprint = int(seed), config, protocol_fingerprint
        rng = np.random.Generator(np.random.PCG64(seed))
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / FEATURES), (FEATURES, HIDDEN)).astype(np.float32), "b1": np.zeros(HIDDEN, np.float32),
            "w2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, HIDDEN)).astype(np.float32), "b2": np.zeros(HIDDEN, np.float32),
            "wa": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, ACTIONS)).astype(np.float32), "ba": np.zeros(ACTIONS, np.float32),
            "wv": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 1)).astype(np.float32), "bv": np.zeros(1, np.float32),
        }
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step = 0

    def reset(self) -> MaintenanceMemory:
        return MaintenanceMemory()

    def features(self, observation: dict[str, Any], memory: MaintenanceMemory) -> np.ndarray:
        return encode_features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return eligible_macro_mask(observation, self.config)

    def observe(self, memory: MaintenanceMemory, *, observation_before: dict[str, Any], macro: MaintenanceMacro, action: dict[str, Any], observation_after: dict[str, Any]) -> None:
        advance_memory(memory, observation_before=observation_before, macro=MaintenanceMacro(int(macro)), action=action, observation_after=observation_after, config=self.config)

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        x = np.asarray(features, dtype=np.float32)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        if x.shape[1:] != (FEATURES,):
            raise ValueError("PPO features must have shape (30,)")
        z1 = x @ self.params["w1"] + self.params["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        h2 = np.maximum(z2, 0)
        logits = h2 @ self.params["wa"] + self.params["ba"]
        values = (h2 @ self.params["wv"] + self.params["bv"]).reshape(-1)
        return (logits[0], values[0], (z1, h1, z2, h2)) if one else (logits, values, (z1, h1, z2, h2))

    def action_distribution(self, features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        logits, value, _ = self.forward(features)
        return masked_probabilities(logits[None, :], np.asarray(mask)[None, :])[0], float(value)

    def choose(self, observation: dict[str, Any], memory: MaintenanceMemory) -> MaintenanceMacro:
        probabilities, _ = self.action_distribution(self.features(observation, memory), self.mask(observation))
        return MaintenanceMacro(int(np.argmax(probabilities)))

    def sample(self, features: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
        probabilities, value = self.action_distribution(features, mask)
        action = int(rng.choice(ACTIONS, p=probabilities))
        return action, float(np.log(probabilities[action])), value

    def value(self, features: np.ndarray) -> float:
        return float(self.forward(features)[1])

    @staticmethod
    def _validate(rollout: Rollout) -> int:
        arrays = (rollout.features, rollout.actions, rollout.masks, rollout.old_log_probs, rollout.rewards, rollout.durations, rollout.done, rollout.values, rollout.next_values)
        size = len(rollout.actions)
        if size == 0 or any(len(row) != size for row in arrays) or rollout.features.shape != (size, FEATURES) or rollout.masks.shape != (size, ACTIONS):
            raise ValueError("PPO rollout is malformed")
        if not np.all(rollout.masks.any(axis=1)):
            raise ValueError("PPO rollout has an empty action mask")
        return size

    def objective_and_gradients(self, rollout: Rollout, *, advantages: np.ndarray, returns: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Return the exact unclipped/entropy/value objective and its gradients.

        ``advantages`` and ``returns`` are explicit to make central finite
        difference tests independent of GAE and advantage normalization.
        """
        size = self._validate(rollout)
        advantages, returns = np.asarray(advantages, np.float32), np.asarray(returns, np.float32)
        if advantages.shape != (size,) or returns.shape != (size,):
            raise ValueError("PPO advantages and returns must match rollout length")
        logits, values, cache = self.forward(rollout.features)
        z1, h1, z2, h2 = cache
        probabilities = masked_probabilities(logits, rollout.masks)
        rows = np.arange(size)
        selected = np.maximum(probabilities[rows, rollout.actions], 1e-12)
        logp = np.log(selected)
        ratio = np.exp(logp - rollout.old_log_probs)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1 - CLIP, 1 + CLIP) * advantages
        actor_loss = -float(np.mean(np.minimum(unclipped, clipped)))
        entropy_rows = -np.sum(np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-12)), 0.0), axis=1)
        critic_loss = float(.5 * np.mean((values - returns) ** 2))
        active = np.where((advantages >= 0) & (ratio > 1 + CLIP), 0.0, np.where((advantages < 0) & (ratio < 1 - CLIP), 0.0, advantages * ratio / size)).astype(np.float32)
        # d(-A * ratio)/d(logits) = +A * ratio * (p-y): this sign is the r2 correction.
        actor_grad = probabilities.copy()
        actor_grad[rows, rollout.actions] -= 1.0
        actor_grad *= active[:, None]
        entropy_grad = -probabilities * (np.log(np.maximum(probabilities, 1e-12)) + entropy_rows[:, None])
        actor_grad -= ENTROPY_COEF * entropy_grad / size
        actor_grad[~rollout.masks] = 0.0
        value_grad = (VALUE_COEF * (values - returns) / size)[:, None]
        gradients = {"wa": h2.T @ actor_grad, "ba": actor_grad.sum(0), "wv": h2.T @ value_grad, "bv": value_grad.sum(0)}
        back = (actor_grad @ self.params["wa"].T + value_grad @ self.params["wv"].T) * (z2 > 0)
        gradients["w2"], gradients["b2"] = h1.T @ back, back.sum(0)
        back = (back @ self.params["w2"].T) * (z1 > 0)
        gradients["w1"], gradients["b1"] = rollout.features.T @ back, back.sum(0)
        metrics = {
            "total_loss": actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * float(entropy_rows.mean()),
            "actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": float(entropy_rows.mean()),
            "approx_kl": float(np.mean(rollout.old_log_probs - logp)),
            "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > CLIP)),
        }
        return metrics, {key: value.astype(np.float32) for key, value in gradients.items()}

    def update(self, rollout: Rollout, *, epochs: int = 4, learning_rate: float = 3e-4, max_grad_norm: float = .5) -> dict[str, Any]:
        self._validate(rollout)
        advantages, returns = semimarkov_gae(rollout.rewards, rollout.values, rollout.next_values, rollout.durations, rollout.done)
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        metrics: dict[str, Any] = {}
        for _ in range(epochs):
            metrics, gradients = self.objective_and_gradients(rollout, advantages=advantages, returns=returns)
            norm = float(np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values())))
            scale = min(1.0, max_grad_norm / max(norm, 1e-12))
            self.step += 1
            for key, gradient in gradients.items():
                gradient = (gradient * scale).astype(np.float32)
                _finite(key, gradient)
                self.m[key] = .9 * self.m[key] + .1 * gradient
                self.v[key] = .999 * self.v[key] + .001 * gradient * gradient
                self.params[key] -= learning_rate * (self.m[key] / (1 - .9 ** self.step)) / (np.sqrt(self.v[key] / (1 - .999 ** self.step)) + 1e-8)
            metrics["gradient_norm"] = norm
            metrics["updates"] = self.step
        metrics["action_histogram"] = {str(index): int(np.count_nonzero(rollout.actions == index)) for index in range(ACTIONS)}
        return metrics

    def fingerprint(self) -> str:
        digest = hashlib.sha256(("m1311-ppo-r2:" + self.protocol_fingerprint).encode())
        for key in sorted(self.params):
            digest.update(self.params[key].tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> dict[str, str]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "m1311-ppo-r2", "protocol_fingerprint": self.protocol_fingerprint, "seed": self.seed, "network": {key: value.tolist() for key, value in self.params.items()}, "policy_fingerprint": self.fingerprint()}
        payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return {"path": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "policy_fingerprint": self.fingerprint()}

    @classmethod
    def load(cls, path: str | Path, *, config: Any, protocol_fingerprint: str) -> "CorrectedMaskedPPOPolicy":
        payload = json.loads(Path(path).read_text())
        expected = dict(payload)
        supplied = expected.pop("content_sha256", None)
        if payload.get("schema_version") != "m1311-ppo-r2" or payload.get("protocol_fingerprint") != protocol_fingerprint or supplied != hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            raise ValueError("invalid r2 PPO artifact schema, protocol, or content hash")
        policy = cls(seed=int(payload["seed"]), config=config, protocol_fingerprint=protocol_fingerprint)
        if set(payload.get("network", {})) != set(policy.params):
            raise ValueError("PPO artifact has invalid parameter names")
        for key, values in payload["network"].items():
            array = np.asarray(values, dtype=np.float32)
            if array.shape != policy.params[key].shape or not np.all(np.isfinite(array)):
                raise ValueError("PPO artifact has invalid parameter")
            policy.params[key] = array
        if payload.get("policy_fingerprint") != policy.fingerprint():
            raise ValueError("PPO artifact fingerprint mismatch")
        return policy
