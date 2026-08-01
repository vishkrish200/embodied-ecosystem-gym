from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EcosystemConfig:
    """Versioned environment knobs; task rewards belong here, not in policies."""

    observation_mode: Literal["state_oracle"] = "state_oracle"
    max_episode_steps: int = 200
    world_radius: float = 1.0
    walk_speed_per_second: float = 0.30
    satiety_decay_per_second: float = 0.002
    energy_decay_per_second: float = 0.001
    boredom_gain_per_second: float = 0.003
    eat_satiety_gain: float = 0.45
    task_success_reward: float = 1.0
    step_penalty_per_second: float = 0.01
    invalid_action_penalty: float = 0.1
