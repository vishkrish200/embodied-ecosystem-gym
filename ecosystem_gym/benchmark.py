"""M2 benchmark runner: registered task splits, metrics, and state-oracle Q learning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol

import numpy as np

from .actions import ActionKind
from .config import EcosystemConfig
from .env import EcosystemEnv
from .policies import FIXED_EVALUATION_SEEDS, skill_action
from .tasks import TaskSpec, get_task


class MacroAction(IntEnum):
    WALK_TO_FOOD = 0
    PICK_UP = 1
    CONSUME = 2
    IDLE = 3


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    seed: int
    layout_id: str
    task_success: bool
    total_reward: float
    steps: int
    terminal_reason: str


class Policy(Protocol):
    def act(self, observation: dict[str, np.ndarray | int], env: EcosystemEnv) -> dict[str, np.ndarray | int]: ...


def _oracle_state(observation: dict[str, np.ndarray | int], pickup_radius: float) -> tuple[bool, bool]:
    distance = float(np.linalg.norm(np.asarray(observation["food_xy"]) - np.asarray(observation["agent_xy"])))
    return bool(observation["holding_food"]), distance <= pickup_radius


def _macro_to_skill(
    action: MacroAction, observation: dict[str, np.ndarray | int], env: EcosystemEnv
) -> dict[str, np.ndarray | int]:
    if action is MacroAction.WALK_TO_FOOD:
        distance = float(np.linalg.norm(np.asarray(observation["food_xy"]) - np.asarray(observation["agent_xy"])))
        duration = max(0.1, distance / env.config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_TO, np.asarray(observation["food_xy"]), duration)
    if action is MacroAction.PICK_UP:
        return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
    if action is MacroAction.CONSUME:
        return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
    return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)


class RawRandomPolicy:
    """Samples the public continuous skill space without privileged target selection."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, np.ndarray | int], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        del observation
        radius = env.config.world_radius
        return skill_action(
            ActionKind(int(self.rng.integers(len(ActionKind)))),
            self.rng.uniform(-radius, radius, size=2).astype(np.float32),
            0.1,
        )


class TabularOracleQPolicy:
    """A small learned state-oracle policy over the project’s bounded skill API."""

    def __init__(self) -> None:
        self.q_values: dict[tuple[bool, bool], np.ndarray] = {}

    def _q(self, state: tuple[bool, bool]) -> np.ndarray:
        return self.q_values.setdefault(state, np.zeros(len(MacroAction), dtype=np.float64))

    def act(self, observation: dict[str, np.ndarray | int], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        state = _oracle_state(observation, env.config.pickup_radius)
        action = MacroAction(int(np.argmax(self._q(state))))
        return _macro_to_skill(action, observation, env)

    def train(
        self,
        task: TaskSpec,
        *,
        episodes: int = 400,
        seed: int = 20260803,
        alpha: float = 0.25,
        discount: float = 0.95,
    ) -> None:
        rng = np.random.default_rng(seed)
        for episode in range(episodes):
            layout_id = task.train_layout_ids[episode % len(task.train_layout_ids)]
            env = EcosystemEnv()
            observation, _ = env.reset(seed=seed + episode, options={"layout_id": layout_id})
            epsilon = max(0.02, 1.0 - episode / (episodes * 0.8))
            try:
                for _ in range(env.config.max_episode_steps):
                    state = _oracle_state(observation, env.config.pickup_radius)
                    if rng.random() < epsilon:
                        macro_action = MacroAction(int(rng.integers(len(MacroAction))))
                    else:
                        macro_action = MacroAction(int(np.argmax(self._q(state))))
                    next_observation, reward, terminated, truncated, _ = env.step(
                        _macro_to_skill(macro_action, observation, env)
                    )
                    next_state = _oracle_state(next_observation, env.config.pickup_radius)
                    target = reward if terminated or truncated else reward + discount * float(np.max(self._q(next_state)))
                    self._q(state)[int(macro_action)] += alpha * (target - self._q(state)[int(macro_action)])
                    observation = next_observation
                    if terminated or truncated:
                        break
            finally:
                env.close()


def run_episode(policy: Policy, *, seed: int, layout_id: str) -> EpisodeMetrics:
    env = EcosystemEnv()
    observation, _ = env.reset(seed=seed, options={"layout_id": layout_id})
    total_reward = 0.0
    try:
        for step in range(1, env.config.max_episode_steps + 1):
            observation, reward, terminated, truncated, info = env.step(policy.act(observation, env))
            total_reward += reward
            if terminated or truncated:
                if info["task_success"]:
                    reason = "task_success"
                elif observation["drives"][0] <= 0.0:
                    reason = "satiety_depleted"
                elif observation["drives"][1] <= 0.0:
                    reason = "energy_depleted"
                else:
                    reason = "time_limit"
                return EpisodeMetrics(seed, layout_id, bool(info["task_success"]), total_reward, step, reason)
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _aggregate(episodes: list[EpisodeMetrics]) -> dict[str, float | int | dict[str, int]]:
    reasons: dict[str, int] = {}
    for episode in episodes:
        reasons[episode.terminal_reason] = reasons.get(episode.terminal_reason, 0) + 1
    return {
        "episodes": len(episodes),
        "successes": sum(episode.task_success for episode in episodes),
        "success_rate": float(np.mean([episode.task_success for episode in episodes])),
        "mean_total_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([episode.steps for episode in episodes])),
        "terminal_reasons": reasons,
    }


def _evaluate_policy(policy: Policy, layouts: tuple[str, ...]) -> dict[str, float | int | dict[str, int]]:
    episodes = [
        run_episode(policy, seed=seed, layout_id=layout_id)
        for layout_id in layouts
        for seed in FIXED_EVALUATION_SEEDS
    ]
    return _aggregate(episodes)


def benchmark(task_name: str = "find_and_eat", *, training_episodes: int = 400) -> dict[str, object]:
    """Run comparable raw-random and learned-oracle evaluations over fixed splits."""

    task = get_task(task_name)
    learned = TabularOracleQPolicy()
    learned.train(task, episodes=training_episodes)
    random_train = _evaluate_policy(RawRandomPolicy(seed=101), task.train_layout_ids)
    random_heldout = _evaluate_policy(RawRandomPolicy(seed=202), task.heldout_layout_ids)
    learned_train = _evaluate_policy(learned, task.train_layout_ids)
    learned_heldout = _evaluate_policy(learned, task.heldout_layout_ids)
    random_rate = float(random_heldout["success_rate"])
    learned_rate = float(learned_heldout["success_rate"])
    return {
        "schema_version": "0.2",
        "environment_version": "0.2.0",
        "task": asdict(task),
        "reward_config": asdict(EcosystemConfig()),
        "evaluation_seeds": list(FIXED_EVALUATION_SEEDS),
        "training": {"algorithm": "tabular_q_learning", "episodes": training_episodes, "seed": 20260803},
        "baselines": {
            "raw_random": {"train": random_train, "heldout": random_heldout},
            "state_oracle_q_learning": {"train": learned_train, "heldout": learned_heldout},
        },
        "comparison": {
            "heldout_success_rate_delta": learned_rate - random_rate,
            "learned_beats_random": learned_rate > random_rate,
        },
        "limits": [
            "Layouts are held-out creature and food spawn distributions in the same M1 room geometry.",
            "This is a state-oracle skill-selection baseline, not an RGB or contact-control result.",
        ],
    }


def write_benchmark_report(path: str | Path, **kwargs: object) -> dict[str, object]:
    report = benchmark(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
