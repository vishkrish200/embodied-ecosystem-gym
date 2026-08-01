from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import ActionKind, ActionOutcome, SkillAction
from .config import EcosystemConfig
from .drives import Drives


@dataclass(slots=True)
class WorldState:
    agent_xy: np.ndarray
    food_xy: np.ndarray
    holding_food: bool
    drives: Drives
    step_count: int


class EcosystemEnv(gym.Env[dict[str, Any], dict[str, np.ndarray | int]]):
    """A deterministic, non-physics M0 reference environment.

    M1 replaces the transition implementation with MuJoCo while preserving this
    observation/action contract and the explicit action outcomes.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EcosystemConfig | None = None) -> None:
        super().__init__()
        self.config = config or EcosystemConfig()
        if self.config.observation_mode != "state_oracle":
            raise NotImplementedError("M0 implements state_oracle only; hybrid and rgb arrive in M3")
        radius = self.config.world_radius
        self.action_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(ActionKind)),
                "target": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "duration": spaces.Box(low=np.asarray(0.1, dtype=np.float32), high=np.asarray(5.0, dtype=np.float32), shape=(), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "agent_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "food_xy": spaces.Box(low=-radius, high=radius, shape=(2,), dtype=np.float32),
                "drives": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "holding_food": spaces.Discrete(2),
                "prior_outcome": spaces.Discrete(len(ActionOutcome)),
            }
        )
        self._state: WorldState | None = None
        self._prior_outcome = ActionOutcome.SUCCESS

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        radius = self.config.world_radius * 0.8
        self._state = WorldState(
            agent_xy=np.zeros(2, dtype=np.float32),
            food_xy=self.np_random.uniform(-radius, radius, size=2).astype(np.float32),
            holding_food=False,
            drives=Drives(satiety=0.45),
            step_count=0,
        )
        self._prior_outcome = ActionOutcome.SUCCESS
        return self._observation(), {"environment_version": "0.0.1", "seed": seed}

    def step(self, action: dict[str, np.ndarray | int]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        skill = SkillAction.from_gym(action)
        state.drives = state.drives.evolve(self.config, skill.duration_seconds)
        outcome, task_success = self._execute(state, skill)
        state.step_count += 1
        self._prior_outcome = outcome
        terminated = task_success or state.drives.satiety <= 0.0 or state.drives.energy <= 0.0
        truncated = state.step_count >= self.config.max_episode_steps and not terminated
        reward = -self.config.step_penalty_per_second * skill.duration_seconds
        if outcome not in {ActionOutcome.SUCCESS, ActionOutcome.BLOCKED}:
            reward -= self.config.invalid_action_penalty
        if task_success:
            reward += self.config.task_success_reward
        if state.drives.satiety <= 0.0 or state.drives.energy <= 0.0:
            reward -= self.config.task_success_reward
        return self._observation(), float(reward), terminated, truncated, self._info(outcome, task_success)

    def _execute(self, state: WorldState, action: SkillAction) -> tuple[ActionOutcome, bool]:
        if action.kind is ActionKind.WALK_TO:
            delta = np.clip(action.target_xy, -self.config.world_radius, self.config.world_radius) - state.agent_xy
            distance = float(np.linalg.norm(delta))
            if distance == 0:
                return ActionOutcome.SUCCESS, False
            step_distance = min(distance, self.config.walk_speed_per_second * action.duration_seconds)
            state.agent_xy = (state.agent_xy + delta / distance * step_distance).astype(np.float32)
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.PICK_UP:
            if float(np.linalg.norm(state.agent_xy - state.food_xy)) > 0.12:
                return ActionOutcome.BLOCKED, False
            state.holding_food = True
            return ActionOutcome.SUCCESS, False
        if action.kind is ActionKind.CONSUME:
            if not state.holding_food:
                return ActionOutcome.NOT_HOLDING_OBJECT, False
            state.holding_food = False
            state.drives = Drives(
                satiety=min(1.0, state.drives.satiety + self.config.eat_satiety_gain),
                energy=state.drives.energy,
                boredom=state.drives.boredom,
            )
            return ActionOutcome.SUCCESS, True
        if action.kind is ActionKind.PLACE and not state.holding_food:
            return ActionOutcome.NOT_HOLDING_OBJECT, False
        if action.kind is ActionKind.PLACE:
            state.food_xy = np.clip(action.target_xy, -self.config.world_radius, self.config.world_radius).astype(np.float32)
            state.holding_food = False
            return ActionOutcome.SUCCESS, False
        return ActionOutcome.SUCCESS, False

    def _observation(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "agent_xy": state.agent_xy.copy(),
            "food_xy": state.food_xy.copy(),
            "drives": state.drives.as_array(),
            "holding_food": int(state.holding_food),
            "prior_outcome": list(ActionOutcome).index(self._prior_outcome),
        }

    def _info(self, outcome: ActionOutcome, task_success: bool) -> dict[str, Any]:
        state = self._require_state()
        return {
            "outcome": outcome.value,
            "task_success": task_success,
            "step_count": state.step_count,
            "environment_version": "0.0.1",
        }

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise gym.error.ResetNeeded("Call reset() before requesting observations or stepping the environment")
        return self._state
