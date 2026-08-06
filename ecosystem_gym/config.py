from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EcosystemConfig:
    """Versioned environment knobs; task rewards belong here, not in policies."""

    observation_mode: Literal["state_oracle", "hybrid", "rgb"] = "state_oracle"
    max_episode_steps: int = 200
    world_radius: float = 1.0
    body_clearance: float = 0.10
    walk_speed_per_second: float = 0.30
    pickup_radius: float = 0.12
    satiety_decay_per_second: float = 0.002
    energy_decay_per_second: float = 0.001
    boredom_gain_per_second: float = 0.003
    toy_interaction_radius: float = 0.22
    play_boredom_reduction: float = 0.60
    play_success_boredom_threshold: float = 0.35
    eat_satiety_gain: float = 0.45
    food_respawn_seconds: float = 5.0
    rest_interaction_radius: float = 0.22
    rest_cycle_energy_threshold: float = 0.45
    rest_energy_gain: float = 0.65
    persistent_min_feed_cycles: int = 3
    persistent_min_play_cycles: int = 3
    persistent_min_rest_cycles: int = 2
    task_success_reward: float = 1.0
    step_penalty_per_second: float = 0.01
    invalid_action_penalty: float = 0.1
    disturbance_recovery_reward: float = 0.05
    play_success_reward: float = 1.0
    rgb_width: int = 64
    rgb_height: int = 64
