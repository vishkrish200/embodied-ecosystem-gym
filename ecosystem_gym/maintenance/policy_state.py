"""Shared public-state mechanics for persistent-maintenance policy families.

The helpers in this module deliberately accept only the public observation,
the frozen maintenance memory, and the public environment configuration.  The
extra commitment fields in :class:`PolicyMemory` are policy-owned state; they
are not appended to the frozen 30-feature representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import inf
from typing import Any

import numpy as np

from ..actions import ActionOutcome
from ..drives import Drives
from .contract import (
    MaintenanceMacro,
    MaintenanceMemory,
    compile_macro,
    eligible_macro_mask,
    encode_features,
)


class MaintenanceGoal(StrEnum):
    FEED = "feed"
    PLAY = "play"
    REST = "rest"


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """The already-published M13 safe-drive boundary."""

    satiety_min: float = 0.15
    energy_min: float = 0.15
    boredom_max: float = 0.90

    def __post_init__(self) -> None:
        values = (self.satiety_min, self.energy_min, self.boredom_max)
        if not all(np.isfinite(values)) or not (0.0 <= self.satiety_min < 1.0):
            raise ValueError("satiety_min must be finite and in [0, 1)")
        if not 0.0 <= self.energy_min < 1.0:
            raise ValueError("energy_min must be in [0, 1)")
        if not 0.0 < self.boredom_max <= 1.0:
            raise ValueError("boredom_max must be in (0, 1]")


@dataclass(slots=True)
class PolicyMemory(MaintenanceMemory):
    """Public contract memory plus unencoded policy-owned control state."""

    committed_goal: MaintenanceGoal | None = None
    commitment_steps: int = 0
    supervisor_interventions: int = 0


def memory_snapshot(memory: PolicyMemory) -> dict[str, Any]:
    """Return a JSON-safe checkpoint of all public and policy-owned memory."""

    return {
        "schema_version": "maintenance-policy-memory-v1",
        "feed_count": int(memory.feed_count),
        "play_count": int(memory.play_count),
        "rest_count": int(memory.rest_count),
        "food_cooldown_bucket": float(memory.food_cooldown_bucket),
        "last_macro": None if memory.last_macro is None else memory.last_macro.name,
        "pending_recovery": bool(memory.pending_recovery),
        "committed_goal": None if memory.committed_goal is None else memory.committed_goal.value,
        "commitment_steps": int(memory.commitment_steps),
        "supervisor_interventions": int(memory.supervisor_interventions),
    }


def restore_memory(payload: dict[str, Any]) -> PolicyMemory:
    if payload.get("schema_version") != "maintenance-policy-memory-v1":
        raise ValueError("unsupported maintenance policy memory schema")
    try:
        last = payload["last_macro"]
        goal = payload["committed_goal"]
        memory = PolicyMemory(
            feed_count=int(payload["feed_count"]),
            play_count=int(payload["play_count"]),
            rest_count=int(payload["rest_count"]),
            food_cooldown_bucket=float(payload["food_cooldown_bucket"]),
            last_macro=None if last is None else MaintenanceMacro[str(last)],
            pending_recovery=bool(payload["pending_recovery"]),
            committed_goal=None if goal is None else MaintenanceGoal(str(goal)),
            commitment_steps=int(payload["commitment_steps"]),
            supervisor_interventions=int(payload["supervisor_interventions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed maintenance policy memory") from exc
    if (
        not 0 <= memory.feed_count <= 3
        or not 0 <= memory.play_count <= 3
        or not 0 <= memory.rest_count <= 2
        or not np.isfinite(memory.food_cooldown_bucket)
        or memory.food_cooldown_bucket < 0.0
        or memory.commitment_steps < 0
        or memory.supervisor_interventions < 0
    ):
        raise ValueError("maintenance policy memory is outside its public bounds")
    return memory


def public_dynamics_signature(config: Any) -> dict[str, float]:
    """Capture every public scalar used by the three policy implementations."""

    names = (
        "walk_speed_per_second",
        "pickup_radius",
        "toy_interaction_radius",
        "rest_interaction_radius",
        "satiety_decay_per_second",
        "energy_decay_per_second",
        "boredom_gain_per_second",
        "eat_satiety_gain",
        "play_success_boredom_threshold",
        "play_boredom_reduction",
        "rest_cycle_energy_threshold",
        "rest_energy_gain",
        "food_respawn_seconds",
        "persistent_min_feed_cycles",
        "persistent_min_play_cycles",
        "persistent_min_rest_cycles",
    )
    result = {name: float(getattr(config, name)) for name in names}
    if not all(np.isfinite(tuple(result.values()))):
        raise ValueError("public dynamics configuration must be finite")
    if result["walk_speed_per_second"] <= 0.0 or any(
        result[name] < 0.0
        for name in ("satiety_decay_per_second", "energy_decay_per_second", "boredom_gain_per_second")
    ):
        raise ValueError("public movement and drive rates are invalid")
    return result


def public_features(observation: dict[str, Any], memory: PolicyMemory) -> np.ndarray:
    """Expose the frozen encoder while making the 30-feature invariant explicit."""

    features = encode_features(observation, memory)
    if features.shape != (30,):
        raise AssertionError("maintenance policy crossed the frozen 30-feature boundary")
    return features


def observation_drives(observation: dict[str, Any]) -> Drives:
    values = np.asarray(observation["drives"], dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("public drives must contain three finite values in [0, 1]")
    return Drives(satiety=float(values[0]), energy=float(values[1]), boredom=float(values[2]))


def project_drives(drives: Drives, config: Any, duration: float) -> Drives:
    elapsed = float(duration)
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("projected duration must be finite and non-negative")
    return drives.evolve(config, elapsed)


def macro_duration(macro: MaintenanceMacro, observation: dict[str, Any], config: Any) -> float:
    action = compile_macro(macro, observation, config)
    duration = float(np.asarray(action["duration"], dtype=np.float64).item())
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("compiled macro duration must be finite and positive")
    return duration


def _distance(observation: dict[str, Any], key: str) -> float:
    agent = np.asarray(observation["agent_xy"], dtype=np.float64)
    target = np.asarray(observation[key], dtype=np.float64)
    if agent.shape != (2,) or target.shape != (2,) or not np.all(np.isfinite(np.concatenate((agent, target)))):
        raise ValueError("public positions must be finite two-vectors")
    return float(np.linalg.norm(target - agent))


def route_macro(goal: MaintenanceGoal, observation: dict[str, Any], config: Any) -> MaintenanceMacro:
    """Return the next geometrically eligible macro on a goal's public route."""

    mask = eligible_macro_mask(observation, config)
    if goal is MaintenanceGoal.FEED:
        if bool(observation["holding_food"]):
            macro = MaintenanceMacro.CONSUME
        else:
            macro = MaintenanceMacro.PICK_UP if bool(mask[MaintenanceMacro.PICK_UP]) else MaintenanceMacro.GO_FOOD
    elif goal is MaintenanceGoal.PLAY:
        macro = MaintenanceMacro.PLAY if bool(mask[MaintenanceMacro.PLAY]) else MaintenanceMacro.GO_TOY
    else:
        macro = MaintenanceMacro.REST if bool(mask[MaintenanceMacro.REST]) else MaintenanceMacro.GO_REST
    if not bool(mask[macro]):
        raise AssertionError(f"public route selected ineligible macro {macro.name}")
    return macro


def goal_service_duration(
    goal: MaintenanceGoal,
    observation: dict[str, Any],
    memory: PolicyMemory,
    config: Any,
) -> float:
    """Optimistic time until the goal's restorative effect can complete."""

    if goal is MaintenanceGoal.FEED:
        if bool(observation["holding_food"]):
            return 0.1
        distance = _distance(observation, "food_xy")
        travel = 0.0 if distance <= float(config.pickup_radius) else distance / float(config.walk_speed_per_second)
        cooldown = max(0.0, float(config.food_respawn_seconds) - float(memory.food_cooldown_bucket))
        return max(travel, cooldown) + 0.2
    target, radius = (
        ("toy_xy", float(config.toy_interaction_radius))
        if goal is MaintenanceGoal.PLAY
        else ("rest_xy", float(config.rest_interaction_radius))
    )
    distance = _distance(observation, target)
    travel = 0.0 if distance <= radius else distance / float(config.walk_speed_per_second)
    return travel + 1.0


def safety_deadlines(observation: dict[str, Any], limits: SafetyLimits, config: Any) -> dict[MaintenanceGoal, float]:
    drives = observation_drives(observation)

    def falling(value: float, boundary: float, rate: float) -> float:
        if value <= boundary:
            return 0.0
        return inf if rate <= 0.0 else (value - boundary) / rate

    def rising(value: float, boundary: float, rate: float) -> float:
        if value >= boundary:
            return 0.0
        return inf if rate <= 0.0 else (boundary - value) / rate

    return {
        MaintenanceGoal.FEED: falling(drives.satiety, limits.satiety_min, float(config.satiety_decay_per_second)),
        MaintenanceGoal.REST: falling(drives.energy, limits.energy_min, float(config.energy_decay_per_second)),
        MaintenanceGoal.PLAY: rising(drives.boredom, limits.boredom_max, float(config.boredom_gain_per_second)),
    }


def goal_margins(
    observation: dict[str, Any],
    memory: PolicyMemory,
    limits: SafetyLimits,
    config: Any,
) -> dict[MaintenanceGoal, float]:
    deadlines = safety_deadlines(observation, limits, config)
    return {
        goal: deadlines[goal] - goal_service_duration(goal, observation, memory, config)
        for goal in MaintenanceGoal
    }


def quota_missing(goal: MaintenanceGoal, memory: PolicyMemory, config: Any) -> bool:
    if goal is MaintenanceGoal.FEED:
        return memory.feed_count < int(config.persistent_min_feed_cycles)
    if goal is MaintenanceGoal.PLAY:
        return memory.play_count < int(config.persistent_min_play_cycles)
    return memory.rest_count < int(config.persistent_min_rest_cycles)


def productive_by_completion(
    goal: MaintenanceGoal,
    observation: dict[str, Any],
    memory: PolicyMemory,
    config: Any,
    *,
    feed_activation: float,
) -> bool:
    """Whether starting the public route now can end in a useful interaction."""

    duration = goal_service_duration(goal, observation, memory, config)
    drives = project_drives(observation_drives(observation), config, duration)
    if goal is MaintenanceGoal.FEED:
        cooldown_ready = float(memory.food_cooldown_bucket) + duration >= float(config.food_respawn_seconds)
        return bool(observation["holding_food"]) or (cooldown_ready and drives.satiety <= feed_activation)
    if goal is MaintenanceGoal.PLAY:
        return drives.boredom >= float(config.play_success_boredom_threshold)
    return drives.energy <= float(config.rest_cycle_energy_threshold)


def semantic_macro_mask(
    observation: dict[str, Any],
    memory: PolicyMemory,
    config: Any,
) -> np.ndarray:
    """Remove legal-but-useless interactions without changing geometry masking."""

    mask = eligible_macro_mask(observation, config).copy()
    drives = observation_drives(observation)
    after_interaction = project_drives(drives, config, 1.0)
    if float(memory.food_cooldown_bucket) < float(config.food_respawn_seconds) and not memory.pending_recovery:
        mask[MaintenanceMacro.PICK_UP] = False
    if after_interaction.boredom < float(config.play_success_boredom_threshold):
        mask[MaintenanceMacro.PLAY] = False
    if after_interaction.energy > float(config.rest_cycle_energy_threshold):
        mask[MaintenanceMacro.REST] = False
    if not np.any(mask):
        mask[MaintenanceMacro.WAIT] = True
    return mask


def outcome_from_observation(observation: dict[str, Any]) -> ActionOutcome:
    try:
        return list(ActionOutcome)[int(observation["prior_outcome"])]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("public prior_outcome is invalid") from exc


def safety_slack(drives: Drives, limits: SafetyLimits) -> float:
    """Normalized signed distance to the nearest published safety boundary."""

    satiety = (drives.satiety - limits.satiety_min) / max(1e-12, 1.0 - limits.satiety_min)
    energy = (drives.energy - limits.energy_min) / max(1e-12, 1.0 - limits.energy_min)
    boredom = (limits.boredom_max - drives.boredom) / max(1e-12, limits.boredom_max)
    return float(min(satiety, energy, boredom))


def policy_parameter_dict(value: Any) -> dict[str, Any]:
    """Dataclass-to-dict helper kept here to centralize manifest encoding."""

    return asdict(value)
