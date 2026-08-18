"""Exact M13.9 public-drive-potential reward recomposition for successors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..actions import ActionKind, ActionOutcome

GAMMA = 0.99
HORIZON = 200


def duration_discount(duration: float) -> float:
    value = float(duration)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("macro duration must be finite and positive")
    return float(GAMMA ** value)


def safe_error(drives: np.ndarray) -> float:
    values = np.asarray(drives, dtype=np.float64)
    if values.shape != (3,): raise ValueError("public drive vector must have shape (3,)")
    return max(float(np.clip((.30-values[0])/.15, 0, 1))**2, float(np.clip((.30-values[1])/.15, 0, 1))**2, float(np.clip((values[2]-.70)/.20, 0, 1))**2)


def _safe(drives: np.ndarray) -> bool:
    values = np.asarray(drives, dtype=np.float32)
    return bool(values[0] > .15 and values[1] > .15 and values[2] < .90)


@dataclass(slots=True)
class MaintenanceRewardState:
    required_recovery: bool
    feed_cycles: int = 0; play_cycles: int = 0; rest_cycles: int = 0
    decision_steps: int = 0; decision_safe_steps: int = 0
    total_duration: float = 0.; duration_safe_seconds: float = 0.
    relocation_seen: bool = False; stale_pickup: bool = False
    recovery_complete: bool = False; recovery_rewarded: bool = False
    @property
    def decision_safe_fraction(self) -> float: return self.decision_safe_steps / self.decision_steps if self.decision_steps else 0.
    @property
    def duration_safe_fraction(self) -> float: return self.duration_safe_seconds / self.total_duration if self.total_duration else 0.
    def snapshot(self) -> dict[str, Any]: return asdict(self) | {"decision_safe_fraction": self.decision_safe_fraction, "duration_safe_fraction": self.duration_safe_fraction}


def learning_reward(*, environment_reward: float, state: MaintenanceRewardState, observation_before: dict[str, Any], observation_after: dict[str, Any], action: dict[str, Any], info: dict[str, Any], terminated: bool, truncated: bool) -> tuple[float, dict[str, Any]]:
    duration = float(np.asarray(action["duration"], dtype=np.float32).item())
    discount = duration_discount(duration); before, after = np.asarray(observation_before["drives"]), np.asarray(observation_after["drives"])
    error_before, error_after = safe_error(before), safe_error(after); potential = discount * -error_after + error_before
    state.decision_steps += 1; state.decision_safe_steps += int(_safe(after)); state.total_duration += duration; state.duration_safe_seconds += duration * int(_safe(after))
    outcome = str(info["outcome"]); kind = ActionKind(int(np.asarray(action["kind"]).item()))
    forced = kind is ActionKind.PICK_UP and outcome == ActionOutcome.BLOCKED.value and info.get("disturbance") == "food_relocated"; ordinary = outcome == ActionOutcome.BLOCKED.value and not forced; invalid = outcome not in {ActionOutcome.SUCCESS.value, ActionOutcome.BLOCKED.value}
    if forced: state.relocation_seen = state.stale_pickup = True
    old = (state.feed_cycles, state.play_cycles, state.rest_cycles); new = tuple(int(info.get(k, old[i])) for i, k in enumerate(("feed_cycles", "play_cycles", "rest_cycles")))
    deltas = tuple(max(0, min(limit, new[i])-min(limit, old[i])) for i, limit in enumerate((3,3,2))); state.feed_cycles, state.play_cycles, state.rest_cycles = new
    if state.required_recovery and state.stale_pickup and kind is ActionKind.CONSUME and outcome == ActionOutcome.SUCCESS.value: state.recovery_complete = True
    recovery = state.required_recovery and state.recovery_complete and not state.recovery_rewarded
    if recovery: state.recovery_rewarded = True
    end = bool(terminated or truncated); full = bool(end and truncated and not terminated and state.decision_steps == HORIZON and info.get("survived", False) and state.feed_cycles >= 3 and state.play_cycles >= 3 and state.rest_cycles >= 2 and state.decision_safe_fraction >= .85 and state.duration_safe_fraction >= .85 and (not state.required_recovery or state.recovery_complete)); failure = end and not full
    components: dict[str, Any] = {"duration":duration, "safe_error_before":error_before, "safe_error_after":error_after, "phi_before":-error_before, "phi_after":-error_after, "duration_discount":discount, "potential_reward":potential, "safety_integral":duration*(error_before+error_after)/2, "duration_cost":-.01*duration, "invalid":invalid, "invalid_cost":-.1*int(invalid), "ordinary_blocked":ordinary, "blocked_cost":-.1*int(ordinary), "forced_stale_exempt":forced, "feed_milestone_delta":deltas[0], "play_milestone_delta":deltas[1], "rest_milestone_delta":deltas[2], "feed_milestone_reward":.25*deltas[0], "play_milestone_reward":.25*deltas[1], "rest_milestone_reward":.375*deltas[2], "recovery_milestone":recovery, "recovery_reward":.25*int(recovery), "post_safe":_safe(after), "full_terminal_success":full, "terminal_success_reward":5.*int(full), "terminal_failure":failure, "terminal_failure_cost":-2.*int(failure), "environment_reward":float(environment_reward)}
    components["safety_cost"] = -.02*float(components["safety_integral"])
    static = sum(float(components[k]) for k in ("duration_cost","safety_cost","invalid_cost","blocked_cost","feed_milestone_reward","play_milestone_reward","rest_milestone_reward","recovery_reward","terminal_success_reward","terminal_failure_cost")); components["static_cost_recomposed"] = static; components["candidate_recomposed"] = static - float(components["safety_cost"]) + potential; components["learning_reward"] = components["candidate_recomposed"]
    return float(components["learning_reward"]), components
