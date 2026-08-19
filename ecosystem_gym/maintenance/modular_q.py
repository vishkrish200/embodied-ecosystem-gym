"""Shared modular and monolithic Q learners for the unopened M13.14 lane."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .contract import MaintenanceMacro, advance_memory, eligible_macro_mask
from .policy_state import PolicyMemory, public_features
from .reward import duration_discount
from .shielded import ShieldConfig, ShieldDecision, ShieldedLearnedPolicy
from .teacher_data import TeacherDataset, validate_public_observation


ACTION_DIM = len(MaintenanceMacro)
FEATURE_DIM = 30
HEAD_COUNT = 3


class QNetworkKind(StrEnum):
    MODULAR = "modular"
    MONOLITHIC = "monolithic"


class ModularQArm(StrEnum):
    MODULAR_ANCHOR_SHIELD_CANDIDATE = "modular_anchor_shield_candidate"
    MONOLITHIC_ANCHOR_SHIELD_CONTROL = "monolithic_anchor_shield_control"
    MODULAR_NO_ANCHOR_SHIELD_CONTROL = "modular_no_anchor_shield_control"
    MODULAR_ANCHOR_NO_SHIELD_CONTROL = "modular_anchor_no_shield_control"


@dataclass(frozen=True, slots=True)
class ModularQArmSpec:
    network_kind: QNetworkKind
    teacher_margin_weight: float
    shield_enabled: bool
    shared_fit_source: str | None = None


MODULAR_Q_ARM_SPECS = {
    ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE: ModularQArmSpec(
        network_kind=QNetworkKind.MODULAR,
        teacher_margin_weight=1.0,
        shield_enabled=True,
    ),
    ModularQArm.MONOLITHIC_ANCHOR_SHIELD_CONTROL: ModularQArmSpec(
        network_kind=QNetworkKind.MONOLITHIC,
        teacher_margin_weight=1.0,
        shield_enabled=True,
    ),
    ModularQArm.MODULAR_NO_ANCHOR_SHIELD_CONTROL: ModularQArmSpec(
        network_kind=QNetworkKind.MODULAR,
        teacher_margin_weight=0.0,
        shield_enabled=True,
    ),
    ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL: ModularQArmSpec(
        network_kind=QNetworkKind.MODULAR,
        teacher_margin_weight=1.0,
        shield_enabled=False,
        shared_fit_source=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE.value,
    ),
}


@dataclass(frozen=True, slots=True)
class ModularQConfig:
    hidden1: int = 64
    hidden2: int = 64
    learning_rate: float = 1e-3
    teacher_margin: float = 0.8
    target_sync_interval: int = 32
    replay_capacity: int = 4096
    td_batch_size: int = 32
    teacher_batch_size: int = 32
    teacher_warmstart_steps: int = 64
    softmax_temperature: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.teacher_margin,
            self.softmax_temperature,
        )
        if not all(np.isfinite(values)):
            raise ValueError("ModularQConfig scalar values must be finite")
        if self.hidden1 < 1 or self.hidden2 < 1:
            raise ValueError("ModularQConfig hidden sizes must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("ModularQConfig learning_rate must be positive")
        if self.teacher_margin < 0.0:
            raise ValueError("ModularQConfig teacher_margin must be non-negative")
        if self.target_sync_interval < 1:
            raise ValueError("ModularQConfig target_sync_interval must be positive")
        if self.replay_capacity < 1:
            raise ValueError("ModularQConfig replay_capacity must be positive")
        if self.td_batch_size < 1 or self.teacher_batch_size < 1:
            raise ValueError("ModularQConfig batch sizes must be positive")
        if self.teacher_warmstart_steps < 0:
            raise ValueError("ModularQConfig teacher_warmstart_steps must be non-negative")
        if self.softmax_temperature <= 0.0:
            raise ValueError("ModularQConfig softmax_temperature must be positive")


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_features: np.ndarray
    next_masks: np.ndarray
    done: np.ndarray
    durations: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        next_features = np.asarray(self.next_features, dtype=np.float32)
        next_masks = np.asarray(self.next_masks, dtype=np.bool_)
        done = np.asarray(self.done, dtype=np.bool_)
        durations = np.asarray(self.durations, dtype=np.float32)
        rows = int(features.shape[0])
        if features.shape != (rows, FEATURE_DIM):
            raise ValueError("transition features must have shape (rows, 30)")
        if next_features.shape != (rows, FEATURE_DIM):
            raise ValueError("transition next_features must have shape (rows, 30)")
        if actions.shape != (rows,) or rewards.shape != (rows,) or done.shape != (rows,) or durations.shape != (rows,):
            raise ValueError("transition arrays must have matching leading dimensions")
        if next_masks.shape != (rows, ACTION_DIM):
            raise ValueError("transition next_masks must have shape (rows, 8)")
        if rows == 0:
            raise ValueError("transition batches must contain at least one row")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(next_features)) or not np.all(np.isfinite(rewards)):
            raise ValueError("transition numeric arrays must be finite")
        if np.any(actions < 0) or np.any(actions >= ACTION_DIM):
            raise ValueError("transition action is outside the maintenance macro surface")
        if np.any(durations <= 0.0) or not np.all(np.isfinite(durations)):
            raise ValueError("transition durations must be finite and positive")
        if not np.all(np.any(next_masks, axis=1)):
            raise ValueError("transition next_masks must keep at least one legal macro")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "next_features", next_features)
        object.__setattr__(self, "next_masks", next_masks)
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "durations", durations)


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    features: np.ndarray
    masks: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        dataset = TeacherDataset(self.features, self.masks, self.labels)
        object.__setattr__(self, "features", np.asarray(dataset.features, dtype=np.float32))
        object.__setattr__(self, "masks", np.asarray(dataset.masks, dtype=np.bool_))
        object.__setattr__(self, "labels", np.asarray(dataset.labels, dtype=np.int64))


def _masked_argmax(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    q_values = np.asarray(values, dtype=np.float64)
    allowed = np.asarray(masks, dtype=np.bool_)
    if q_values.shape != allowed.shape or q_values.ndim != 2 or q_values.shape[1] != ACTION_DIM:
        raise ValueError("masked argmax expects matching (rows, 8) arrays")
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("every masked argmax row must contain a legal macro")
    return np.argmax(np.where(allowed, q_values, -np.inf), axis=1).astype(np.int64)


def masked_softmax(values: np.ndarray, masks: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    q_values = np.asarray(values, dtype=np.float64)
    allowed = np.asarray(masks, dtype=np.bool_)
    scale = float(temperature)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("softmax temperature must be finite and positive")
    if q_values.shape != allowed.shape or q_values.ndim != 2 or q_values.shape[1] != ACTION_DIM:
        raise ValueError("masked softmax expects matching (rows, 8) arrays")
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("every masked softmax row must contain a legal macro")
    scaled = q_values / scale
    maximum = np.max(np.where(allowed, scaled, -np.inf), axis=1, keepdims=True)
    exp = np.where(allowed, np.exp(scaled - maximum), 0.0)
    normalizer = np.sum(exp, axis=1, keepdims=True)
    return (exp / normalizer).astype(np.float32)


def duration_aware_double_dqn_targets(
    rewards: np.ndarray,
    durations: np.ndarray,
    done: np.ndarray,
    next_online_q: np.ndarray,
    next_target_q: np.ndarray,
    next_masks: np.ndarray,
) -> np.ndarray:
    reward_values = np.asarray(rewards, dtype=np.float32)
    duration_values = np.asarray(durations, dtype=np.float32)
    done_values = np.asarray(done, dtype=np.bool_)
    online_values = np.asarray(next_online_q, dtype=np.float32)
    target_values = np.asarray(next_target_q, dtype=np.float32)
    mask_values = np.asarray(next_masks, dtype=np.bool_)
    rows = int(reward_values.shape[0])
    if reward_values.shape != (rows,) or duration_values.shape != (rows,) or done_values.shape != (rows,):
        raise ValueError("backup inputs must share a one-dimensional row shape")
    if online_values.shape != (rows, ACTION_DIM) or target_values.shape != (rows, ACTION_DIM) or mask_values.shape != (rows, ACTION_DIM):
        raise ValueError("backup Q values and masks must have shape (rows, 8)")
    next_actions = _masked_argmax(online_values, mask_values)
    bootstrap = target_values[np.arange(rows), next_actions]
    discounts = np.asarray([duration_discount(float(value)) for value in duration_values], dtype=np.float32)
    return reward_values + discounts * bootstrap * (~done_values)


def dqfd_teacher_margin_loss(
    q_values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    margin: float,
) -> tuple[float, np.ndarray]:
    values = np.asarray(q_values, dtype=np.float32)
    allowed = np.asarray(masks, dtype=np.bool_)
    target_labels = np.asarray(labels, dtype=np.int64)
    bonus = float(margin)
    rows = int(values.shape[0])
    if values.shape != (rows, ACTION_DIM) or allowed.shape != values.shape or target_labels.shape != (rows,):
        raise ValueError("teacher margin loss expects Q values, masks, and labels with matching shapes")
    if bonus < 0.0 or not np.isfinite(bonus):
        raise ValueError("teacher margin must be finite and non-negative")
    if np.any(target_labels < 0) or np.any(target_labels >= ACTION_DIM):
        raise ValueError("teacher label is outside the maintenance macro surface")
    if not np.all(allowed[np.arange(rows), target_labels]):
        raise ValueError("teacher chose an ineligible maintenance macro")
    margins = np.full((rows, ACTION_DIM), bonus, dtype=np.float32)
    margins[np.arange(rows), target_labels] = 0.0
    augmented = np.where(allowed, values + margins, -np.inf)
    competitors = np.argmax(augmented, axis=1)
    chosen = values[np.arange(rows), target_labels]
    competitor_values = augmented[np.arange(rows), competitors]
    per_row = np.maximum(0.0, competitor_values - chosen)
    gradient = np.zeros_like(values, dtype=np.float32)
    active = per_row > 0.0
    if np.any(active):
        rows_active = np.flatnonzero(active)
        gradient[rows_active, competitors[rows_active]] += 1.0 / rows
        gradient[rows_active, target_labels[rows_active]] -= 1.0 / rows
    return float(np.mean(per_row)), gradient


@dataclass(slots=True)
class _ForwardCache:
    features: np.ndarray
    hidden1_pre: np.ndarray
    hidden1: np.ndarray
    hidden2_pre: np.ndarray
    hidden2: np.ndarray
    components: np.ndarray


@dataclass(slots=True)
class QNetworkState:
    kind: QNetworkKind
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    head_w: np.ndarray
    head_b: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", QNetworkKind(self.kind))
        w1 = np.asarray(self.w1, dtype=np.float32)
        b1 = np.asarray(self.b1, dtype=np.float32)
        w2 = np.asarray(self.w2, dtype=np.float32)
        b2 = np.asarray(self.b2, dtype=np.float32)
        head_w = np.asarray(self.head_w, dtype=np.float32)
        head_b = np.asarray(self.head_b, dtype=np.float32)
        if w1.shape[0] != FEATURE_DIM:
            raise ValueError("Q network first layer must ingest exactly 30 features")
        if b1.shape != (w1.shape[1],):
            raise ValueError("Q network first-layer bias shape is invalid")
        if w2.shape != (w1.shape[1], b2.shape[0]):
            raise ValueError("Q network second layer shape is invalid")
        if self.kind is QNetworkKind.MODULAR:
            if head_w.shape != (HEAD_COUNT, b2.shape[0], ACTION_DIM) or head_b.shape != (HEAD_COUNT, ACTION_DIM):
                raise ValueError("modular Q network head shape is invalid")
        else:
            if head_w.shape != (b2.shape[0], HEAD_COUNT * ACTION_DIM) or head_b.shape != (HEAD_COUNT * ACTION_DIM,):
                raise ValueError("monolithic Q network head shape is invalid")
        for name, value in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("head_w", head_w), ("head_b", head_b)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Q network parameter {name!r} must be finite")
            object.__setattr__(self, name, value.copy())

    @classmethod
    def random(cls, *, rng: np.random.Generator, kind: QNetworkKind, config: ModularQConfig) -> QNetworkState:
        hidden1 = int(config.hidden1)
        hidden2 = int(config.hidden2)
        scale1 = np.sqrt(2.0 / FEATURE_DIM)
        scale2 = np.sqrt(2.0 / hidden1)
        if kind is QNetworkKind.MODULAR:
            head_w = rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(HEAD_COUNT, hidden2, ACTION_DIM)).astype(np.float32)
            head_b = np.zeros((HEAD_COUNT, ACTION_DIM), dtype=np.float32)
        else:
            head_w = rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2, HEAD_COUNT * ACTION_DIM)).astype(np.float32)
            head_b = np.zeros(HEAD_COUNT * ACTION_DIM, dtype=np.float32)
        return cls(
            kind=kind,
            w1=rng.normal(0.0, scale1, size=(FEATURE_DIM, hidden1)).astype(np.float32),
            b1=np.zeros(hidden1, dtype=np.float32),
            w2=rng.normal(0.0, scale2, size=(hidden1, hidden2)).astype(np.float32),
            b2=np.zeros(hidden2, dtype=np.float32),
            head_w=head_w,
            head_b=head_b,
        )

    def copy(self) -> QNetworkState:
        return QNetworkState(
            kind=self.kind,
            w1=self.w1.copy(),
            b1=self.b1.copy(),
            w2=self.w2.copy(),
            b2=self.b2.copy(),
            head_w=self.head_w.copy(),
            head_b=self.head_b.copy(),
        )

    def parameter_count(self) -> int:
        return int(sum(int(np.asarray(value).size) for value in self.parameter_items().values()))

    def parameter_items(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "head_w": self.head_w,
            "head_b": self.head_b,
        }

    def parameter_bytes(self) -> bytes:
        chunks = [f"{self.kind.value}|".encode("utf-8")]
        for name in ("w1", "b1", "w2", "b2", "head_w", "head_b"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            chunks.append(name.encode("utf-8"))
            chunks.append(str(tuple(values.shape)).encode("utf-8"))
            chunks.append(values.tobytes())
        return b"".join(chunks)

    def parameter_fingerprint(self) -> str:
        return hashlib.sha256(b"m1314-q-network-v1" + self.parameter_bytes()).hexdigest()

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, _ForwardCache]:
        feature_values = np.asarray(features, dtype=np.float32)
        if feature_values.ndim != 2 or feature_values.shape[1] != FEATURE_DIM:
            raise ValueError("Q network features must have shape (rows, 30)")
        hidden1_pre = feature_values @ self.w1 + self.b1
        hidden1 = np.maximum(hidden1_pre, 0.0)
        hidden2_pre = hidden1 @ self.w2 + self.b2
        hidden2 = np.maximum(hidden2_pre, 0.0)
        if self.kind is QNetworkKind.MODULAR:
            components = np.einsum("bh,gha->bga", hidden2, self.head_w) + self.head_b[None, :, :]
        else:
            flat = hidden2 @ self.head_w + self.head_b
            components = flat.reshape(feature_values.shape[0], HEAD_COUNT, ACTION_DIM)
        q_values = np.sum(components, axis=1)
        cache = _ForwardCache(
            features=feature_values,
            hidden1_pre=hidden1_pre,
            hidden1=hidden1,
            hidden2_pre=hidden2_pre,
            hidden2=hidden2,
            components=components,
        )
        return components.astype(np.float32), q_values.astype(np.float32), cache

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)[1]

    def component_q_values(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)[0]

    def apply_gradient(self, gradients: dict[str, np.ndarray], *, learning_rate: float) -> None:
        step = float(learning_rate)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        for name, gradient in gradients.items():
            values = np.asarray(getattr(self, name), dtype=np.float32)
            delta = np.asarray(gradient, dtype=np.float32)
            if values.shape != delta.shape or not np.all(np.isfinite(delta)):
                raise ValueError(f"gradient for {name!r} is invalid")
            setattr(self, name, (values - step * delta).astype(np.float32))


def _network_gradients(
    network: QNetworkState,
    cache: _ForwardCache,
    grad_q: np.ndarray,
) -> dict[str, np.ndarray]:
    grad_values = np.asarray(grad_q, dtype=np.float32)
    rows = int(cache.features.shape[0])
    if grad_values.shape != (rows, ACTION_DIM):
        raise ValueError("Q gradient must have shape (rows, 8)")
    component_grad = np.repeat(grad_values[:, None, :], HEAD_COUNT, axis=1)
    gradients: dict[str, np.ndarray]
    if network.kind is QNetworkKind.MODULAR:
        grad_head_w = np.einsum("bh,bga->gha", cache.hidden2, component_grad)
        grad_head_b = np.sum(component_grad, axis=0)
        grad_hidden2 = np.einsum("bga,gha->bh", component_grad, network.head_w)
    else:
        flat = component_grad.reshape(rows, HEAD_COUNT * ACTION_DIM)
        grad_head_w = cache.hidden2.T @ flat
        grad_head_b = np.sum(flat, axis=0)
        grad_hidden2 = flat @ network.head_w.T
    grad_hidden2 *= (cache.hidden2_pre > 0.0).astype(np.float32)
    grad_w2 = cache.hidden1.T @ grad_hidden2
    grad_b2 = np.sum(grad_hidden2, axis=0)
    grad_hidden1 = grad_hidden2 @ network.w2.T
    grad_hidden1 *= (cache.hidden1_pre > 0.0).astype(np.float32)
    grad_w1 = cache.features.T @ grad_hidden1
    grad_b1 = np.sum(grad_hidden1, axis=0)
    gradients = {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
        "head_w": grad_head_w,
        "head_b": grad_head_b,
    }
    return gradients


class DurationReplayBuffer:
    """FIFO replay that keeps macro durations alongside each transition."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.features = np.empty((capacity, FEATURE_DIM), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_features = np.empty((capacity, FEATURE_DIM), dtype=np.float32)
        self.next_masks = np.empty((capacity, ACTION_DIM), dtype=np.bool_)
        self.done = np.empty(capacity, dtype=np.bool_)
        self.durations = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(
        self,
        *,
        features: np.ndarray,
        action: MaintenanceMacro | int,
        reward: float,
        next_features: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        duration: float,
    ) -> None:
        row = TransitionBatch(
            features=np.asarray(features, dtype=np.float32)[None, :],
            actions=np.asarray([int(action)], dtype=np.int64),
            rewards=np.asarray([reward], dtype=np.float32),
            next_features=np.asarray(next_features, dtype=np.float32)[None, :],
            next_masks=np.asarray(next_mask, dtype=np.bool_)[None, :],
            done=np.asarray([done], dtype=np.bool_),
            durations=np.asarray([duration], dtype=np.float32),
        )
        index = self.cursor
        self.features[index] = row.features[0]
        self.actions[index] = row.actions[0]
        self.rewards[index] = row.rewards[0]
        self.next_features[index] = row.next_features[0]
        self.next_masks[index] = row.next_masks[0]
        self.done[index] = row.done[0]
        self.durations[index] = row.durations[0]
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size > 0 and batch_size > 0

    def sample(self, rng: np.random.Generator, *, batch_size: int) -> TransitionBatch:
        if not self.can_sample(batch_size):
            raise ValueError("replay buffer is empty")
        take = int(batch_size)
        replace = self.size < take
        indices = rng.choice(self.size, size=take, replace=replace)
        return TransitionBatch(
            features=self.features[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_features=self.next_features[indices],
            next_masks=self.next_masks[indices],
            done=self.done[indices],
            durations=self.durations[indices],
        )


class TeacherReplayBuffer:
    """Immutable demonstration replay that never mutates the teacher surface."""

    def __init__(self, dataset: TeacherDataset) -> None:
        self.dataset = dataset

    @property
    def size(self) -> int:
        return self.dataset.rows

    def sample(self, rng: np.random.Generator, *, batch_size: int) -> TeacherBatch:
        take = int(batch_size)
        if take < 1:
            raise ValueError("teacher batch size must be positive")
        replace = self.size < take
        indices = rng.choice(self.size, size=take, replace=replace)
        return TeacherBatch(
            features=self.dataset.features[indices],
            masks=self.dataset.masks[indices],
            labels=self.dataset.labels[indices],
        )


class ModularQPolicy:
    """Duration-aware Double-DQN core shared by all M13.14 learner arms."""

    family = "m1314_modular_q"

    def __init__(
        self,
        *,
        arm: ModularQArm | str,
        seed: int,
        config: Any,
        learner: ModularQConfig | None = None,
        protocol_fingerprint: str = "m1314-development",
    ) -> None:
        self.arm = ModularQArm(arm)
        self.arm_spec = MODULAR_Q_ARM_SPECS[self.arm]
        self.seed = int(seed)
        self.config = config
        self.learner = learner or ModularQConfig()
        self.protocol_fingerprint = str(protocol_fingerprint)
        self.rng = np.random.default_rng(self.seed)
        self.online = QNetworkState.random(rng=self.rng, kind=self.arm_spec.network_kind, config=self.learner)
        self.target = self.online.copy()
        self.update_count = 0
        self.replay = DurationReplayBuffer(self.learner.replay_capacity)
        self.teacher_replay: TeacherReplayBuffer | None = None
        self.initial_parameter_fingerprint = self.parameter_fingerprint()
        self.initial_weight_fingerprint = self.weight_fingerprint()

    @property
    def network_kind(self) -> QNetworkKind:
        return self.arm_spec.network_kind

    @property
    def teacher_margin_weight(self) -> float:
        return float(self.arm_spec.teacher_margin_weight)

    def parameter_bytes(self) -> bytes:
        digest = [
            f"{self.family}|{self.arm.value}|{self.protocol_fingerprint}|{self.update_count}|".encode("utf-8"),
            self.online.parameter_bytes(),
            b"|target|",
            self.target.parameter_bytes(),
        ]
        return b"".join(digest)

    def weight_bytes(self) -> bytes:
        return b"".join(
            (
                b"m1314-modular-policy-weights-v1|online|",
                self.online.parameter_bytes(),
                b"|target|",
                self.target.parameter_bytes(),
            )
        )

    def parameter_fingerprint(self) -> str:
        return hashlib.sha256(b"m1314-modular-policy-v1" + self.parameter_bytes()).hexdigest()

    def weight_fingerprint(self) -> str:
        return hashlib.sha256(b"m1314-modular-policy-weights-v1" + self.weight_bytes()).hexdigest()

    def reset(self) -> PolicyMemory:
        return PolicyMemory(food_cooldown_bucket=float(self.config.food_respawn_seconds))

    def features(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        validate_public_observation(observation)
        return public_features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        validate_public_observation(observation)
        return eligible_macro_mask(observation, self.config)

    def _feature_batch(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM:
            raise ValueError("policy features must have shape (30,) or (rows, 30)")
        return rows

    def component_q_values_from_features(self, features: np.ndarray) -> np.ndarray:
        return self.online.component_q_values(self._feature_batch(features))

    def q_values_from_features(self, features: np.ndarray) -> np.ndarray:
        return self.online.q_values(self._feature_batch(features))

    def q_values(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return self.q_values_from_features(self.features(observation, memory))[0]

    def probabilities(
        self,
        observation: dict[str, Any],
        memory: PolicyMemory,
        *,
        mask: np.ndarray | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        allowed = self.mask(observation) if mask is None else np.asarray(mask, dtype=np.bool_)
        values = self.q_values(observation, memory)[None, :]
        probabilities = masked_softmax(values, allowed[None, :], temperature=temperature or self.learner.softmax_temperature)
        return probabilities[0]

    def choose(self, observation: dict[str, Any], memory: PolicyMemory) -> MaintenanceMacro:
        q_values = self.q_values(observation, memory)[None, :]
        mask = self.mask(observation)[None, :]
        return MaintenanceMacro(int(_masked_argmax(q_values, mask)[0]))

    def observe(
        self,
        memory: PolicyMemory,
        *,
        observation_before: dict[str, Any],
        macro: MaintenanceMacro,
        action: dict[str, Any],
        observation_after: dict[str, Any],
    ) -> None:
        advance_memory(
            memory,
            observation_before=observation_before,
            macro=MaintenanceMacro(int(macro)),
            action=action,
            observation_after=observation_after,
            config=self.config,
        )

    def attach_teacher_dataset(self, dataset: TeacherDataset) -> TeacherReplayBuffer:
        self.teacher_replay = TeacherReplayBuffer(dataset)
        return self.teacher_replay

    def td_loss(self, batch: TransitionBatch) -> tuple[float, np.ndarray, _ForwardCache]:
        online_q, target_q = self.online.q_values(batch.next_features), self.target.q_values(batch.next_features)
        targets = duration_aware_double_dqn_targets(
            batch.rewards,
            batch.durations,
            batch.done,
            online_q,
            target_q,
            batch.next_masks,
        )
        _, current_q, cache = self.online.forward(batch.features)
        selected = current_q[np.arange(batch.actions.shape[0]), batch.actions]
        errors = selected - targets
        gradient = np.zeros_like(current_q, dtype=np.float32)
        gradient[np.arange(batch.actions.shape[0]), batch.actions] = errors / batch.actions.shape[0]
        return float(0.5 * np.mean(errors**2)), gradient, cache

    def teacher_margin_loss(self, batch: TeacherBatch) -> tuple[float, np.ndarray, _ForwardCache]:
        _, q_values, cache = self.online.forward(batch.features)
        loss, gradient = dqfd_teacher_margin_loss(
            q_values,
            batch.masks,
            batch.labels,
            margin=self.learner.teacher_margin,
        )
        return loss, gradient, cache

    def _apply_gradients(
        self,
        *,
        td_gradient: np.ndarray,
        td_cache: _ForwardCache,
        teacher_gradient: np.ndarray | None,
        teacher_cache: _ForwardCache | None,
    ) -> None:
        gradients = _network_gradients(self.online, td_cache, td_gradient)
        if self.teacher_margin_weight > 0.0 and teacher_gradient is not None and teacher_cache is not None:
            teacher_gradients = _network_gradients(self.online, teacher_cache, teacher_gradient)
            for name in gradients:
                gradients[name] = gradients[name] + self.teacher_margin_weight * teacher_gradients[name]
        self.online.apply_gradient(gradients, learning_rate=self.learner.learning_rate)
        self.update_count += 1
        if self.update_count % self.learner.target_sync_interval == 0:
            self.target = self.online.copy()

    def update(self, td_batch: TransitionBatch, teacher_batch: TeacherBatch | None = None) -> dict[str, float]:
        td_loss, td_gradient, td_cache = self.td_loss(td_batch)
        teacher_loss = 0.0
        teacher_gradient: np.ndarray | None = None
        teacher_cache: _ForwardCache | None = None
        if teacher_batch is not None:
            teacher_loss, teacher_gradient, teacher_cache = self.teacher_margin_loss(teacher_batch)
        self._apply_gradients(
            td_gradient=td_gradient,
            td_cache=td_cache,
            teacher_gradient=teacher_gradient,
            teacher_cache=teacher_cache,
        )
        total = td_loss + self.teacher_margin_weight * teacher_loss
        return {
            "td_loss": float(td_loss),
            "teacher_margin_loss": float(teacher_loss),
            "teacher_margin_weight": float(self.teacher_margin_weight),
            "total_loss": float(total),
        }

    def warmstart_from_teacher(
        self,
        dataset: TeacherDataset,
        *,
        steps: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        replay = self.attach_teacher_dataset(dataset)
        local_rng = self.rng if rng is None else rng
        iterations = self.learner.teacher_warmstart_steps if steps is None else int(steps)
        if iterations < 0:
            raise ValueError("warmstart steps must be non-negative")
        last_loss = 0.0
        for _ in range(iterations):
            batch = replay.sample(local_rng, batch_size=self.learner.teacher_batch_size)
            current = self.online.q_values(batch.features)
            last_loss, gradient = dqfd_teacher_margin_loss(
                current,
                batch.masks,
                batch.labels,
                margin=self.learner.teacher_margin,
            )
            _, _, cache = self.online.forward(batch.features)
            grads = _network_gradients(self.online, cache, gradient)
            self.online.apply_gradient(grads, learning_rate=self.learner.learning_rate)
        self.target = self.online.copy()
        return {
            "teacher_margin_loss": float(last_loss),
            "steps": float(iterations),
            "policy_fingerprint": self.parameter_fingerprint(),
        }

    def sample_and_update(self, *, rng: np.random.Generator | None = None) -> dict[str, float]:
        if self.teacher_replay is None:
            raise ValueError("teacher replay is not attached")
        local_rng = self.rng if rng is None else rng
        td_batch = self.replay.sample(local_rng, batch_size=self.learner.td_batch_size)
        teacher_batch = self.teacher_replay.sample(local_rng, batch_size=self.learner.teacher_batch_size)
        return self.update(td_batch, teacher_batch)

    def save(self, path: str | Any) -> dict[str, Any]:
        from .modular_q_artifacts import save_modular_q_policy

        return save_modular_q_policy(path, self)

    @classmethod
    def load(
        cls,
        path: str | Any,
        *,
        config: Any,
        expected_protocol_fingerprint: str | None = None,
    ) -> ModularQPolicy:
        from .modular_q_artifacts import load_modular_q_policy

        return load_modular_q_policy(
            path,
            config=config,
            expected_protocol_fingerprint=expected_protocol_fingerprint,
        )


class _QValueActorAdapter:
    """Expose the learner's Q values through the shield's actor interface."""

    def __init__(self, policy: ModularQPolicy) -> None:
        self.policy = policy

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = self.policy.q_values_from_features(np.asarray(features, dtype=np.float32))[0]
        return np.asarray(values, dtype=np.float32)

    def fingerprint(self) -> str:
        return self.policy.parameter_fingerprint()


class ShieldedModularQPolicy:
    """Action-selection wrapper that reuses the published shield semantics."""

    def __init__(
        self,
        *,
        learner: ModularQPolicy,
        shield: ShieldConfig | None = None,
    ) -> None:
        self.learner = learner
        self.shield = shield or ShieldConfig(enabled=self.learner.arm_spec.shield_enabled)
        self.actor = _QValueActorAdapter(learner)
        self._shielded = ShieldedLearnedPolicy(
            actor=self.actor,
            config=self.learner.config,
            shield=self.shield,
            protocol_fingerprint=self.learner.protocol_fingerprint,
        )

    @property
    def family(self) -> str:
        return f"{self.learner.family}_shield"

    @property
    def protocol_fingerprint(self) -> str:
        return self.learner.protocol_fingerprint

    @property
    def config(self) -> Any:
        return self.learner.config

    def reset(self) -> PolicyMemory:
        return self._shielded.reset()

    def features(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return self.learner.features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return self.learner.mask(observation)

    def q_values(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return self.learner.q_values(observation, memory)

    def probabilities(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        decision = self.decide(observation, memory)
        return masked_softmax(
            self.q_values(observation, memory)[None, :],
            decision.allowed_mask[None, :],
            temperature=self.learner.learner.softmax_temperature,
        )[0]

    def decide(self, observation: dict[str, Any], memory: PolicyMemory) -> ShieldDecision:
        return self._shielded.decide(observation, memory)

    def choose(self, observation: dict[str, Any], memory: PolicyMemory) -> MaintenanceMacro:
        return self._shielded.choose(observation, memory)

    def select_action(
        self,
        observation: dict[str, Any],
        memory: PolicyMemory,
        *,
        rng: np.random.Generator,
        epsilon: float = 0.0,
    ) -> MaintenanceMacro:
        chance = float(epsilon)
        if not np.isfinite(chance) or not 0.0 <= chance <= 1.0:
            raise ValueError("epsilon must be finite and in [0, 1]")
        decision = self.decide(observation, memory)
        if chance > 0.0 and rng.random() < chance:
            allowed = np.flatnonzero(decision.allowed_mask)
            return MaintenanceMacro(int(rng.choice(allowed)))
        return decision.macro

    def observe(
        self,
        memory: PolicyMemory,
        *,
        observation_before: dict[str, Any],
        macro: MaintenanceMacro,
        action: dict[str, Any],
        observation_after: dict[str, Any],
    ) -> None:
        self._shielded.observe(
            memory,
            observation_before=observation_before,
            macro=macro,
            action=action,
            observation_after=observation_after,
        )


def modular_q_evaluation_spec(
    learner: ModularQPolicy,
    *,
    shield: ShieldConfig,
) -> dict[str, Any]:
        return {
            "schema_version": "m1314-modular-q-eval-spec-v1",
            "family": learner.family,
            "arm": learner.arm.value,
            "network_kind": learner.network_kind.value,
            "protocol_fingerprint": learner.protocol_fingerprint,
            "policy_fingerprint": learner.parameter_fingerprint(),
            "weight_fingerprint": learner.weight_fingerprint(),
            "parameter_bytes_sha256": hashlib.sha256(learner.parameter_bytes()).hexdigest(),
            "weight_bytes_sha256": hashlib.sha256(learner.weight_bytes()).hexdigest(),
            "shield": asdict(shield),
        }
