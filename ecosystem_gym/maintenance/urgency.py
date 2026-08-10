"""Deterministic urgency, hysteresis, and goal-commitment scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
)


@dataclass(frozen=True, slots=True)
class UrgencySchedulerConfig:
    """Mechanics-derived scheduler parameters, frozen before evaluation."""

    safety: SafetyLimits = field(default_factory=SafetyLimits)
    feed_activation: float = 0.55
    preemption_reserve_seconds: float = 2.0
    preemption_advantage_seconds: float = 1.0
    max_commitment_steps: int = 6
    commitment_enabled: bool = True

    def __post_init__(self) -> None:
        scalars = (
            self.feed_activation,
            self.preemption_reserve_seconds,
            self.preemption_advantage_seconds,
        )
        if not all(np.isfinite(scalars)) or not 0.0 < self.feed_activation <= 1.0:
            raise ValueError("feed_activation must be finite and in (0, 1]")
        if self.preemption_reserve_seconds < 0.0 or self.preemption_advantage_seconds < 0.0:
            raise ValueError("preemption reserves must be non-negative")
        if self.max_commitment_steps < 1:
            raise ValueError("max_commitment_steps must be positive")


@dataclass(frozen=True, slots=True)
class UrgencyDecision:
    macro: MaintenanceMacro
    goal: MaintenanceGoal | None
    reason: str
    margins: dict[MaintenanceGoal, float]


_GOAL_ORDER = {
    MaintenanceGoal.FEED: 0,
    MaintenanceGoal.REST: 1,
    MaintenanceGoal.PLAY: 2,
}


class UrgencySchedulerPolicy:
    """Deadline-based public scheduler with productive interactions and hysteresis."""

    family = "urgency_hysteresis"

    def __init__(
        self,
        *,
        config: Any,
        scheduler: UrgencySchedulerConfig | None = None,
        protocol_fingerprint: str = "maintenance-policy-production-v1",
    ) -> None:
        self.config = config
        self.scheduler = scheduler or UrgencySchedulerConfig()
        self.protocol_fingerprint = str(protocol_fingerprint)

    def reset(self) -> PolicyMemory:
        return PolicyMemory(food_cooldown_bucket=float(self.config.food_respawn_seconds))

    def features(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return public_features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return eligible_macro_mask(observation, self.config)

    def _macro_for_goal(
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

    def _due_goals(
        self,
        observation: dict[str, Any],
        memory: PolicyMemory,
        margins: dict[MaintenanceGoal, float],
    ) -> tuple[MaintenanceGoal, ...]:
        if bool(observation["holding_food"]) or memory.pending_recovery:
            return (MaintenanceGoal.FEED,)
        return tuple(
            goal
            for goal in MaintenanceGoal
            if margins[goal] <= self.scheduler.preemption_reserve_seconds
            or productive_by_completion(
                goal,
                observation,
                memory,
                self.config,
                feed_activation=self.scheduler.feed_activation,
            )
        )

    def decide(self, observation: dict[str, Any], memory: PolicyMemory) -> UrgencyDecision:
        # Encoding is not needed for authored control, but calling it here
        # makes accidental public-contract drift fail at every decision.
        self.features(observation, memory)
        margins = goal_margins(observation, memory, self.scheduler.safety, self.config)

        if bool(observation["holding_food"]):
            if self.scheduler.commitment_enabled:
                memory.committed_goal = MaintenanceGoal.FEED
            return UrgencyDecision(MaintenanceMacro.CONSUME, MaintenanceGoal.FEED, "finish-held-food", margins)

        if memory.commitment_steps >= self.scheduler.max_commitment_steps:
            memory.committed_goal = None
            memory.commitment_steps = 0

        due = self._due_goals(observation, memory, margins)
        committed = memory.committed_goal if self.scheduler.commitment_enabled else None
        if committed is not None:
            alternatives = [goal for goal in due if goal is not committed]
            if alternatives:
                challenger = min(alternatives, key=lambda goal: (margins[goal], _GOAL_ORDER[goal]))
                materially_more_urgent = (
                    margins[challenger] <= self.scheduler.preemption_reserve_seconds
                    and margins[challenger] + self.scheduler.preemption_advantage_seconds < margins[committed]
                )
                if materially_more_urgent:
                    memory.committed_goal = challenger
                    memory.commitment_steps = 0
                    macro = self._macro_for_goal(challenger, observation, memory)
                    return UrgencyDecision(macro, challenger, "safety-preemption", margins)
            macro = self._macro_for_goal(committed, observation, memory)
            return UrgencyDecision(macro, committed, "continue-commitment", margins)

        if not due:
            return UrgencyDecision(MaintenanceMacro.WAIT, None, "all-goals-have-reserve", margins)

        critical = [goal for goal in due if margins[goal] <= self.scheduler.preemption_reserve_seconds]
        if critical:
            selected = min(critical, key=lambda goal: (margins[goal], _GOAL_ORDER[goal]))
            reason = "smallest-safety-margin"
        else:
            selected = min(
                due,
                key=lambda goal: (
                    0 if quota_missing(goal, memory, self.config) else 1,
                    margins[goal],
                    _GOAL_ORDER[goal],
                ),
            )
            reason = "unfinished-quota" if quota_missing(selected, memory, self.config) else "continuous-maintenance"

        if self.scheduler.commitment_enabled:
            memory.committed_goal = selected
            memory.commitment_steps = 0
        return UrgencyDecision(self._macro_for_goal(selected, observation, memory), selected, reason, margins)

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


def urgency_config_from_dict(payload: dict[str, Any]) -> UrgencySchedulerConfig:
    values = dict(payload)
    safety = values.pop("safety", {})
    try:
        return UrgencySchedulerConfig(safety=SafetyLimits(**safety), **values)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid urgency scheduler configuration") from exc
