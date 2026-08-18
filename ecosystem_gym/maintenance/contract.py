"""Public, deterministic maintenance-policy contract.

No function in this module accepts environment ``info``, reset options, or a
task identifier.  Those values remain evaluation/reward-sidecar only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..drives import Drives
from ..policies import skill_action


class MaintenanceMacro(IntEnum):
    GO_FOOD = 0
    PICK_UP = 1
    CONSUME = 2
    GO_TOY = 3
    PLAY = 4
    GO_REST = 5
    REST = 6
    WAIT = 7


@dataclass(slots=True)
class MaintenanceMemory:
    feed_count: int = 0
    play_count: int = 0
    rest_count: int = 0
    food_cooldown_bucket: float = 5.0
    last_macro: MaintenanceMacro | None = None
    pending_recovery: bool = False


def _public_observation(observation: dict[str, Any]) -> None:
    required = {"agent_xy", "food_xy", "toy_xy", "rest_xy", "drives", "holding_food", "prior_outcome"}
    missing = required.difference(observation)
    if missing:
        raise ValueError(f"public maintenance observation is missing {sorted(missing)!r}")


def encode_features(observation: dict[str, Any], memory: MaintenanceMemory) -> np.ndarray:
    """The frozen 30-feature public representation, recreated independently."""

    _public_observation(observation)
    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    relative = np.concatenate([
        np.asarray(observation[key], dtype=np.float32) - agent for key in ("food_xy", "toy_xy", "rest_xy")
    ])
    relative = np.clip(relative, -1.8, 1.8) / 1.8
    drives = np.asarray(observation["drives"], dtype=np.float32)
    if drives.shape != (3,):
        raise ValueError("public drives must have shape (3,)")
    outcome = np.zeros(len(ActionOutcome), dtype=np.float32)
    outcome[int(observation["prior_outcome"])] = 1.0
    counts = np.asarray([
        memory.feed_count / 3, memory.play_count / 3, memory.rest_count / 2,
        memory.food_cooldown_bucket / 5, float(memory.pending_recovery),
    ], dtype=np.float32)
    last = np.zeros(len(MaintenanceMacro) + 1, dtype=np.float32)
    last[len(MaintenanceMacro) if memory.last_macro is None else int(memory.last_macro)] = 1.0
    result = np.concatenate([relative, drives, np.asarray([observation["holding_food"]], dtype=np.float32), outcome, counts, last]).astype(np.float32)
    if result.shape != (30,):
        raise AssertionError(f"expected 30 public features, got {result.shape}")
    return result


def eligible_macro_mask(observation: dict[str, Any], config: Any) -> np.ndarray:
    """Exact complementary M13.4 mask, derived solely from public geometry."""

    _public_observation(observation)
    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    near = {
        "food": bool(np.linalg.norm(np.asarray(observation["food_xy"], dtype=np.float32) - agent) <= config.pickup_radius),
        "toy": bool(np.linalg.norm(np.asarray(observation["toy_xy"], dtype=np.float32) - agent) <= config.toy_interaction_radius),
        "rest": bool(np.linalg.norm(np.asarray(observation["rest_xy"], dtype=np.float32) - agent) <= config.rest_interaction_radius),
    }
    mask = np.zeros(len(MaintenanceMacro), dtype=np.bool_)
    mask[MaintenanceMacro.PICK_UP] = near["food"] and not bool(observation["holding_food"])
    mask[MaintenanceMacro.CONSUME] = bool(observation["holding_food"])
    mask[MaintenanceMacro.PLAY], mask[MaintenanceMacro.REST], mask[MaintenanceMacro.WAIT] = near["toy"], near["rest"], True
    mask[MaintenanceMacro.GO_FOOD], mask[MaintenanceMacro.GO_TOY], mask[MaintenanceMacro.GO_REST] = not near["food"], not near["toy"], not near["rest"]
    return mask


def compile_macro(macro: MaintenanceMacro, observation: dict[str, Any], config: Any) -> dict[str, np.ndarray | int]:
    _public_observation(observation)
    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    target = {
        MaintenanceMacro.GO_FOOD: "food_xy", MaintenanceMacro.GO_TOY: "toy_xy", MaintenanceMacro.GO_REST: "rest_xy",
    }.get(macro)
    if target is not None:
        point = np.asarray(observation[target], dtype=np.float32)
        return skill_action(ActionKind.WALK_TO, point, max(0.1, float(np.linalg.norm(point - agent)) / config.walk_speed_per_second))
    kinds = {MaintenanceMacro.PICK_UP: ActionKind.PICK_UP, MaintenanceMacro.CONSUME: ActionKind.CONSUME,
             MaintenanceMacro.PLAY: ActionKind.RUN_AROUND, MaintenanceMacro.REST: ActionKind.REST,
             MaintenanceMacro.WAIT: ActionKind.IDLE}
    return skill_action(kinds[macro], np.zeros(2, dtype=np.float32), 0.1 if macro in {MaintenanceMacro.PICK_UP, MaintenanceMacro.CONSUME} else 1.0)


def advance_memory(memory: MaintenanceMemory, *, observation_before: dict[str, Any], macro: MaintenanceMacro,
                   action: dict[str, np.ndarray | int], observation_after: dict[str, Any], config: Any) -> None:
    """M13.3's public post-duration memory timing."""

    _public_observation(observation_before); _public_observation(observation_after)
    outcome = list(ActionOutcome)[int(observation_after["prior_outcome"])]
    duration = float(np.asarray(action["duration"], dtype=np.float32).item())
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("macro duration must be finite and positive")
    drives = np.asarray(observation_before["drives"], dtype=np.float32)
    evolved = Drives(satiety=float(drives[0]), energy=float(drives[1]), boredom=float(drives[2])).evolve(config, duration)
    if macro is MaintenanceMacro.CONSUME and outcome is ActionOutcome.SUCCESS:
        memory.feed_count, memory.food_cooldown_bucket, memory.pending_recovery = min(3, memory.feed_count + 1), 0.0, False
    elif memory.food_cooldown_bucket < config.food_respawn_seconds:
        memory.food_cooldown_bucket = min(config.food_respawn_seconds, memory.food_cooldown_bucket + duration)
    if macro is MaintenanceMacro.PLAY and outcome is ActionOutcome.SUCCESS and evolved.boredom >= config.play_success_boredom_threshold:
        memory.play_count = min(3, memory.play_count + 1)
    if macro is MaintenanceMacro.REST and outcome is ActionOutcome.SUCCESS and evolved.energy <= config.rest_cycle_energy_threshold:
        memory.rest_count = min(2, memory.rest_count + 1)
    if macro is MaintenanceMacro.PICK_UP and outcome is ActionOutcome.BLOCKED:
        memory.pending_recovery = True
    memory.last_macro = macro
