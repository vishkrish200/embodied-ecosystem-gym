"""Short-horizon receding-horizon scheduler over public maintenance dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..actions import ActionOutcome
from ..drives import Drives
from .contract import MaintenanceMacro, advance_memory, eligible_macro_mask
from .policy_state import (
    PolicyMemory,
    SafetyLimits,
    macro_duration,
    observation_drives,
    project_drives,
    public_features,
    safety_slack,
    semantic_macro_mask,
)


@dataclass(frozen=True, slots=True)
class ModelBasedSchedulerConfig:
    """Bounded planner configuration; no fitted values or trace-derived weights."""

    lookahead_depth: int = 4
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    def __post_init__(self) -> None:
        if not 1 <= self.lookahead_depth <= 5:
            raise ValueError("lookahead_depth must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class PredictedState:
    agent_xy: np.ndarray
    food_xy: np.ndarray
    toy_xy: np.ndarray
    rest_xy: np.ndarray
    drives: Drives
    holding_food: bool
    feed_count: int
    play_count: int
    rest_count: int
    food_cooldown_bucket: float
    pending_recovery: bool

    @classmethod
    def from_public(cls, observation: dict[str, Any], memory: PolicyMemory) -> "PredictedState":
        positions = [np.asarray(observation[key], dtype=np.float64) for key in ("agent_xy", "food_xy", "toy_xy", "rest_xy")]
        if any(value.shape != (2,) for value in positions) or not np.all(np.isfinite(np.concatenate(positions))):
            raise ValueError("public positions must be finite two-vectors")
        return cls(
            agent_xy=positions[0].copy(),
            food_xy=positions[1].copy(),
            toy_xy=positions[2].copy(),
            rest_xy=positions[3].copy(),
            drives=observation_drives(observation),
            holding_food=bool(observation["holding_food"]),
            feed_count=int(memory.feed_count),
            play_count=int(memory.play_count),
            rest_count=int(memory.rest_count),
            food_cooldown_bucket=float(memory.food_cooldown_bucket),
            pending_recovery=bool(memory.pending_recovery),
        )

    def observation(self) -> dict[str, Any]:
        return {
            "agent_xy": self.agent_xy.astype(np.float32),
            "food_xy": self.food_xy.astype(np.float32),
            "toy_xy": self.toy_xy.astype(np.float32),
            "rest_xy": self.rest_xy.astype(np.float32),
            "drives": self.drives.as_array(),
            "holding_food": int(self.holding_food),
            "prior_outcome": list(ActionOutcome).index(ActionOutcome.SUCCESS),
        }

    def memory(self) -> PolicyMemory:
        return PolicyMemory(
            feed_count=self.feed_count,
            play_count=self.play_count,
            rest_count=self.rest_count,
            food_cooldown_bucket=self.food_cooldown_bucket,
            pending_recovery=self.pending_recovery,
        )


@dataclass(frozen=True, slots=True)
class PlanScore:
    violation_cost: float
    minimum_slack: float
    quota_progress: int
    restorative_effect: float
    terminal_slack: float
    elapsed_seconds: float

    def ordering_key(self) -> tuple[float, float, int, float, float, float]:
        return (
            -self.violation_cost,
            self.minimum_slack,
            self.quota_progress,
            self.restorative_effect,
            self.terminal_slack,
            -self.elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class MaintenancePlan:
    macros: tuple[MaintenanceMacro, ...]
    score: PlanScore
    predicted_drives: Drives


@dataclass(frozen=True, slots=True)
class _Branch:
    state: PredictedState
    macros: tuple[MaintenanceMacro, ...]
    slacks: tuple[float, ...]
    elapsed_seconds: float
    restorative_effect: float


class ModelBasedSchedulerPolicy:
    """Enumerate short public macro sequences and execute the best first action."""

    family = "short_horizon_model_based"

    def __init__(
        self,
        *,
        config: Any,
        planner: ModelBasedSchedulerConfig | None = None,
        protocol_fingerprint: str = "maintenance-policy-production-v1",
    ) -> None:
        self.config = config
        self.planner = planner or ModelBasedSchedulerConfig()
        self.protocol_fingerprint = str(protocol_fingerprint)

    def reset(self) -> PolicyMemory:
        return PolicyMemory(food_cooldown_bucket=float(self.config.food_respawn_seconds))

    def features(self, observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
        return public_features(observation, memory)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return eligible_macro_mask(observation, self.config)

    def semantic_actions(self, state: PredictedState) -> tuple[MaintenanceMacro, ...]:
        mask = semantic_macro_mask(state.observation(), state.memory(), self.config)
        return tuple(MaintenanceMacro(index) for index in np.flatnonzero(mask))

    def predict(self, state: PredictedState, macro: MaintenanceMacro) -> tuple[PredictedState, float, float]:
        """Apply the optimistic public model and return state, duration, relief."""

        observation = state.observation()
        if macro not in self.semantic_actions(state):
            raise ValueError(f"macro {macro.name} is not semantically eligible in predicted state")
        duration = macro_duration(macro, observation, self.config)
        evolved = project_drives(state.drives, self.config, duration)
        cooldown = min(
            float(self.config.food_respawn_seconds),
            state.food_cooldown_bucket + duration,
        )
        result = replace(state, drives=evolved, food_cooldown_bucket=cooldown)
        relief = 0.0

        if macro is MaintenanceMacro.GO_FOOD:
            result = replace(result, agent_xy=state.food_xy.copy())
        elif macro is MaintenanceMacro.GO_TOY:
            result = replace(result, agent_xy=state.toy_xy.copy())
        elif macro is MaintenanceMacro.GO_REST:
            result = replace(result, agent_xy=state.rest_xy.copy())
        elif macro is MaintenanceMacro.PICK_UP:
            result = replace(result, holding_food=True)
        elif macro is MaintenanceMacro.CONSUME:
            restored = min(1.0, evolved.satiety + float(self.config.eat_satiety_gain))
            relief = restored - evolved.satiety
            result = replace(
                result,
                drives=Drives(restored, evolved.energy, evolved.boredom),
                holding_food=False,
                feed_count=min(int(self.config.persistent_min_feed_cycles), state.feed_count + 1),
                food_cooldown_bucket=0.0,
                pending_recovery=False,
            )
        elif macro is MaintenanceMacro.PLAY:
            restored = max(0.0, evolved.boredom - float(self.config.play_boredom_reduction))
            relief = evolved.boredom - restored
            productive = evolved.boredom >= float(self.config.play_success_boredom_threshold)
            result = replace(
                result,
                drives=Drives(evolved.satiety, evolved.energy, restored),
                play_count=min(
                    int(self.config.persistent_min_play_cycles),
                    state.play_count + int(productive),
                ),
            )
        elif macro is MaintenanceMacro.REST:
            restored = min(1.0, evolved.energy + float(self.config.rest_energy_gain))
            relief = restored - evolved.energy
            productive = evolved.energy <= float(self.config.rest_cycle_energy_threshold)
            result = replace(
                result,
                drives=Drives(evolved.satiety, restored, evolved.boredom),
                rest_count=min(
                    int(self.config.persistent_min_rest_cycles),
                    state.rest_count + int(productive),
                ),
            )
        return result, duration, relief

    def _score(self, initial: PredictedState, branch: _Branch) -> PlanScore:
        slacks = branch.slacks or (safety_slack(initial.drives, self.planner.safety),)
        violation = float(sum(1.0 + max(0.0, -value) for value in slacks if value <= 0.0))
        quota_progress = (
            branch.state.feed_count - initial.feed_count
            + branch.state.play_count - initial.play_count
            + branch.state.rest_count - initial.rest_count
        )
        return PlanScore(
            violation_cost=violation,
            minimum_slack=float(min(slacks)),
            quota_progress=int(quota_progress),
            restorative_effect=float(branch.restorative_effect),
            terminal_slack=safety_slack(branch.state.drives, self.planner.safety),
            elapsed_seconds=float(branch.elapsed_seconds),
        )

    def plan(self, observation: dict[str, Any], memory: PolicyMemory) -> MaintenancePlan:
        self.features(observation, memory)
        initial = PredictedState.from_public(observation, memory)
        frontier = [_Branch(initial, (), (), 0.0, 0.0)]
        for _ in range(self.planner.lookahead_depth):
            expanded: list[_Branch] = []
            for branch in frontier:
                for macro in self.semantic_actions(branch.state):
                    predicted, duration, relief = self.predict(branch.state, macro)
                    expanded.append(
                        _Branch(
                            state=predicted,
                            macros=branch.macros + (macro,),
                            slacks=branch.slacks + (safety_slack(predicted.drives, self.planner.safety),),
                            elapsed_seconds=branch.elapsed_seconds + duration,
                            restorative_effect=branch.restorative_effect + relief,
                        )
                    )
            frontier = expanded
        if not frontier:
            raise AssertionError("WAIT must keep the public model frontier non-empty")

        def key(branch: _Branch) -> tuple[Any, ...]:
            # Lower enum values break exact numerical ties deterministically.
            tie = tuple(-int(macro) for macro in branch.macros)
            return (*self._score(initial, branch).ordering_key(), tie)

        best = max(frontier, key=key)
        return MaintenancePlan(best.macros, self._score(initial, best), best.state.drives)

    def choose(self, observation: dict[str, Any], memory: PolicyMemory) -> MaintenanceMacro:
        return self.plan(observation, memory).macros[0]

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


def model_config_from_dict(payload: dict[str, Any]) -> ModelBasedSchedulerConfig:
    values = dict(payload)
    safety = values.pop("safety", {})
    try:
        return ModelBasedSchedulerConfig(safety=SafetyLimits(**safety), **values)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid model-based scheduler configuration") from exc
