"""Learned public actor constrained by a deterministic maintenance supervisor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..actions import ActionOutcome
from .contract import MaintenanceMacro, advance_memory, eligible_macro_mask
from .policy_state import (
    MaintenanceGoal,
    PolicyMemory,
    SafetyLimits,
    goal_margins,
    outcome_from_observation,
    productive_by_completion,
    public_features,
    quota_missing,
    route_macro,
    semantic_macro_mask,
)


_ACTOR_SHAPES = {
    "w1": (30, 64),
    "b1": (64,),
    "w2": (64, 64),
    "b2": (64,),
    "wa": (64, len(MaintenanceMacro)),
    "ba": (len(MaintenanceMacro),),
}


class PublicMLPActor:
    """Inference-only 30->64->64->8 actor compatible with corrected PPO."""

    def __init__(self, parameters: Mapping[str, np.ndarray]) -> None:
        if set(parameters) != set(_ACTOR_SHAPES):
            raise ValueError("public actor parameter names do not match the frozen topology")
        self.parameters: dict[str, np.ndarray] = {}
        for name, shape in _ACTOR_SHAPES.items():
            value = np.asarray(parameters[name], dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"public actor parameter {name!r} is invalid")
            self.parameters[name] = value.copy()

    @classmethod
    def zeros(cls) -> "PublicMLPActor":
        """Deterministic synthetic actor for unit tests, never a fitted policy."""

        return cls({name: np.zeros(shape, dtype=np.float32) for name, shape in _ACTOR_SHAPES.items()})

    @classmethod
    def from_corrected_ppo(cls, policy: Any) -> "PublicMLPActor":
        """Copy only actor weights; critic and optimizer state cannot cross inference."""

        try:
            return cls({name: np.asarray(policy.params[name], dtype=np.float32) for name in _ACTOR_SHAPES})
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("source policy does not expose the corrected public actor topology") from exc

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (30,) or not np.all(np.isfinite(values)):
            raise ValueError("public actor features must be a finite 30-vector")
        h1 = np.maximum(values @ self.parameters["w1"] + self.parameters["b1"], 0.0)
        h2 = np.maximum(h1 @ self.parameters["w2"] + self.parameters["b2"], 0.0)
        logits = h2 @ self.parameters["wa"] + self.parameters["ba"]
        if logits.shape != (len(MaintenanceMacro),) or not np.all(np.isfinite(logits)):
            raise ValueError("public actor emitted invalid logits")
        return logits.astype(np.float32)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"maintenance-public-actor-v1")
        for name in sorted(self.parameters):
            digest.update(name.encode("utf-8"))
            digest.update(self.parameters[name].tobytes())
        return digest.hexdigest()

    def payload(self) -> dict[str, list[Any]]:
        return {name: value.tolist() for name, value in sorted(self.parameters.items())}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PublicMLPActor":
        if not isinstance(payload, dict):
            raise ValueError("public actor payload must be an object")
        return cls({name: np.asarray(value, dtype=np.float32) for name, value in payload.items()})


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    enabled: bool = True
    objective_commitment: bool = True
    feed_activation: float = 0.55
    wait_reserve_seconds: float = 1.0
    preemption_advantage_seconds: float = 1.0
    max_commitment_steps: int = 6

    def __post_init__(self) -> None:
        values = (self.feed_activation, self.wait_reserve_seconds, self.preemption_advantage_seconds)
        if not all(np.isfinite(values)) or not 0.0 < self.feed_activation <= 1.0:
            raise ValueError("shield feed_activation must be finite and in (0, 1]")
        if self.wait_reserve_seconds < 0.0 or self.preemption_advantage_seconds < 0.0:
            raise ValueError("shield reserves must be non-negative")
        if self.max_commitment_steps < 1:
            raise ValueError("shield max_commitment_steps must be positive")


@dataclass(frozen=True, slots=True)
class ShieldDecision:
    macro: MaintenanceMacro
    raw_macro: MaintenanceMacro
    intervened: bool
    reason: str
    allowed_mask: np.ndarray
    margins: dict[MaintenanceGoal, float]


_GOAL_ORDER = {
    MaintenanceGoal.FEED: 0,
    MaintenanceGoal.REST: 1,
    MaintenanceGoal.PLAY: 2,
}


def _masked_argmax(logits: np.ndarray, mask: np.ndarray) -> MaintenanceMacro:
    values = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(mask, dtype=np.bool_)
    if values.shape != (len(MaintenanceMacro),) or allowed.shape != values.shape or not np.any(allowed):
        raise ValueError("learned logits and non-empty macro mask must both have shape (8,)")
    return MaintenanceMacro(int(np.argmax(np.where(allowed, values, -np.inf))))


class ShieldedLearnedPolicy:
    """Hierarchical policy: deterministic obligations, learned safe tie-breaks."""

    family = "shielded_learned"

    def __init__(
        self,
        *,
        actor: PublicMLPActor,
        config: Any,
        shield: ShieldConfig | None = None,
        protocol_fingerprint: str = "maintenance-policy-production-v1",
    ) -> None:
        self.actor = actor
        self.config = config
        self.shield = shield or ShieldConfig()
        self.protocol_fingerprint = str(protocol_fingerprint)

    def reset(self) -> PolicyMemory:
        return PolicyMemory(food_cooldown_bucket=float(self.config.food_respawn_seconds))

    def features(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return public_features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return eligible_macro_mask(observation, self.config)

    def _forced_macro(
        self,
        goal: MaintenanceGoal,
        observation: dict[str, Any],
        memory: PolicyMemory,
    ) -> MaintenanceMacro:
        macro = route_macro(goal, observation, self.config)
        if (
            macro is MaintenanceMacro.PICK_UP
            and float(memory.food_cooldown_bucket) < float(self.config.food_respawn_seconds)
            and not memory.pending_recovery
        ):
            return MaintenanceMacro.WAIT
        return macro

    def _commit(self, memory: PolicyMemory, goal: MaintenanceGoal) -> None:
        if memory.committed_goal is not goal:
            memory.committed_goal = goal
            memory.commitment_steps = 0

    def decide(self, observation: dict[str, Any], memory: PolicyMemory) -> ShieldDecision:
        features = self.features(observation, memory)
        logits = self.actor.logits(features)
        geometry = self.mask(observation)
        raw = _masked_argmax(logits, geometry)
        margins = goal_margins(observation, memory, self.shield.safety, self.config)
        if not self.shield.enabled:
            return ShieldDecision(raw, raw, False, "shield-disabled-control", geometry, margins)

        semantic = semantic_macro_mask(observation, memory, self.config)
        forced: MaintenanceMacro | None = None
        reason = "learned-safe-tiebreak"

        if bool(observation["holding_food"]):
            self._commit(memory, MaintenanceGoal.FEED)
            forced, reason = MaintenanceMacro.CONSUME, "finish-held-food"
        elif memory.pending_recovery:
            self._commit(memory, MaintenanceGoal.FEED)
            forced, reason = self._forced_macro(MaintenanceGoal.FEED, observation, memory), "forced-recovery"
        else:
            if memory.commitment_steps >= self.shield.max_commitment_steps:
                memory.committed_goal = None
                memory.commitment_steps = 0

            committed = memory.committed_goal
            if committed is not None:
                challenger = min(MaintenanceGoal, key=lambda goal: (margins[goal], _GOAL_ORDER[goal]))
                preempt = (
                    challenger is not committed
                    and margins[challenger] <= self.shield.wait_reserve_seconds
                    and margins[challenger] + self.shield.preemption_advantage_seconds < margins[committed]
                )
                if preempt:
                    self._commit(memory, challenger)
                    committed = challenger
                    reason = "safety-preemption"
                else:
                    reason = "continue-supervisor-commitment"
                forced = self._forced_macro(committed, observation, memory)

            if forced is None and self.shield.objective_commitment:
                due_quotas = [
                    goal
                    for goal in MaintenanceGoal
                    if quota_missing(goal, memory, self.config)
                    and productive_by_completion(
                        goal,
                        observation,
                        memory,
                        self.config,
                        feed_activation=self.shield.feed_activation,
                    )
                ]
                if due_quotas:
                    goal = min(due_quotas, key=lambda item: (margins[item], _GOAL_ORDER[item]))
                    self._commit(memory, goal)
                    forced, reason = self._forced_macro(goal, observation, memory), "objective-commitment"

            if forced is None:
                goal = min(MaintenanceGoal, key=lambda item: (margins[item], _GOAL_ORDER[item]))
                if margins[goal] <= self.shield.wait_reserve_seconds:
                    self._commit(memory, goal)
                    forced, reason = self._forced_macro(goal, observation, memory), "unsafe-to-wait"

        if forced is not None:
            allowed = np.zeros(len(MaintenanceMacro), dtype=np.bool_)
            allowed[forced] = True
            chosen = forced
        else:
            allowed = semantic
            chosen = _masked_argmax(logits, allowed)

        intervened = chosen != raw or forced is not None
        if intervened:
            memory.supervisor_interventions += 1
        return ShieldDecision(chosen, raw, intervened, reason, allowed, margins)

    def choose(self, observation: dict[str, Any], memory: PolicyMemory) -> MaintenanceMacro:
        return self.decide(observation, memory).macro

    def observe(
        self,
        memory: PolicyMemory,
        *,
        observation_before: dict[str, Any],
        macro: MaintenanceMacro,
        action: dict[str, Any],
        observation_after: dict[str, Any],
    ) -> None:
        selected = MaintenanceMacro(int(macro))
        advance_memory(
            memory,
            observation_before=observation_before,
            macro=selected,
            action=action,
            observation_after=observation_after,
            config=self.config,
        )
        if memory.committed_goal is not None:
            memory.commitment_steps += 1
        outcome = outcome_from_observation(observation_after)
        completed = (
            (selected is MaintenanceMacro.CONSUME and outcome is ActionOutcome.SUCCESS)
            or (selected is MaintenanceMacro.PLAY and outcome is ActionOutcome.SUCCESS)
            or (selected is MaintenanceMacro.REST and outcome is ActionOutcome.SUCCESS)
        )
        blocked_terminal = selected in {MaintenanceMacro.PLAY, MaintenanceMacro.REST} and outcome is ActionOutcome.BLOCKED
        if completed or blocked_terminal:
            memory.committed_goal = None
            memory.commitment_steps = 0


def shield_config_from_dict(payload: dict[str, Any]) -> ShieldConfig:
    values = dict(payload)
    safety = values.pop("safety", {})
    try:
        return ShieldConfig(safety=SafetyLimits(**safety), **values)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid learned-policy shield configuration") from exc
