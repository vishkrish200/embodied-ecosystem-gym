from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np


class ActionKind(IntEnum):
    WALK_TO = 0
    PICK_UP = 1
    CONSUME = 2
    PLACE = 3
    RUN_AROUND = 4
    IDLE = 5
    WALK_RELATIVE = 6
    SCAN = 7
    PICK_UP_RELATIVE = 8


class ActionOutcome(StrEnum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    TARGET_NOT_VISIBLE = "target_not_visible"
    NOT_HOLDING_OBJECT = "not_holding_object"
    NOT_EDIBLE = "not_edible"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class SkillAction:
    kind: ActionKind
    target_xy: np.ndarray
    duration_seconds: float

    @classmethod
    def from_gym(cls, action: dict[str, np.ndarray | int]) -> "SkillAction":
        target = np.asarray(action["target"], dtype=np.float32)
        if target.shape != (2,):
            raise ValueError("action.target must have shape (2,)")
        duration = float(np.asarray(action["duration"], dtype=np.float32).item())
        if duration <= 0:
            raise ValueError("action.duration must be positive")
        return cls(kind=ActionKind(int(action["kind"])), target_xy=target, duration_seconds=duration)
