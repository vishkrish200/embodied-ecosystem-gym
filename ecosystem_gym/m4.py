"""M4 drive-task evaluation over object-appearance and dynamics splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionKind
from .env import EcosystemEnv
from .policies import skill_action
from .tasks import get_task


M4_EVALUATION_SEEDS = tuple(range(10))
TRAIN_CONDITIONS = (
    {"food_variant": "red", "toy_variant": "ball", "dynamics_variant": "nominal"},
    {"food_variant": "orange", "toy_variant": "cube", "dynamics_variant": "grippy"},
)
HELDOUT_CONDITIONS = (
    {"food_variant": "blue", "toy_variant": "ball", "dynamics_variant": "nominal"},
    {"food_variant": "red", "toy_variant": "capsule", "dynamics_variant": "nominal"},
    {"food_variant": "red", "toy_variant": "ball", "dynamics_variant": "slippery"},
    {"food_variant": "purple", "toy_variant": "capsule", "dynamics_variant": "slippery"},
)


@dataclass(frozen=True, slots=True)
class M4Episode:
    task_id: str
    layout_id: str
    seed: int
    condition: dict[str, str]
    task_success: bool
    survived: bool
    total_reward: float
    steps: int
    boundary_contacts: int


class DriveAwareOraclePolicy:
    """Small inspectable controller that prioritizes food under competing drives."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def act(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        drives = np.asarray(observation["drives"], dtype=np.float32)
        prioritize_food = self.task_id == "competing_drives" and drives[0] <= drives[2]
        if self.task_id == "play_when_bored" or (self.task_id == "competing_drives" and not prioritize_food):
            target = np.asarray(observation["toy_xy"], dtype=np.float32)
            if float(np.linalg.norm(target - observation["agent_xy"])) > env.config.toy_interaction_radius:
                return self._walk(target, observation, env)
            return skill_action(ActionKind.RUN_AROUND, target, 0.5)
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        target = np.asarray(observation["food_xy"], dtype=np.float32)
        if float(np.linalg.norm(target - observation["agent_xy"])) <= env.config.pickup_radius:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        return self._walk(target, observation, env)

    @staticmethod
    def _walk(target: np.ndarray, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        distance = float(np.linalg.norm(target - observation["agent_xy"]))
        # This controller does not receive the reset condition or private
        # dynamics scale, so held-out movement changes remain a real test.
        duration = max(0.1, distance / env.config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_TO, target, duration)


def _run_episode(task_id: str, layout_id: str, seed: int, condition: dict[str, str]) -> M4Episode:
    env = EcosystemEnv()
    try:
        observation, _ = env.reset(seed=seed, options={"task_id": task_id, "layout_id": layout_id, **condition})
        policy = DriveAwareOraclePolicy(task_id)
        total_reward = 0.0
        for step in range(1, env.config.max_episode_steps + 1):
            observation, reward, terminated, truncated, info = env.step(policy.act(observation, env))
            total_reward += reward
            if terminated or truncated:
                return M4Episode(
                    task_id, layout_id, seed, condition, bool(info["task_success"]), bool(info["survived"]),
                    total_reward, step, int(info["boundary_contacts"]),
                )
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _aggregate(episodes: list[M4Episode]) -> dict[str, float | int]:
    return {
        "episodes": len(episodes),
        "successes": sum(episode.task_success for episode in episodes),
        "task_success_rate": float(np.mean([episode.task_success for episode in episodes])),
        "survival_rate": float(np.mean([episode.survived for episode in episodes])),
        "mean_total_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "boundary_contacts": sum(episode.boundary_contacts for episode in episodes),
        "mean_boundary_contacts": float(np.mean([episode.boundary_contacts for episode in episodes])),
    }


def _condition_name(condition: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in condition.items())


def _evaluate(
    task_id: str, layouts: tuple[str, ...], conditions: tuple[dict[str, str], ...], seeds: tuple[int, ...]
) -> dict[str, object]:
    all_episodes = [
        _run_episode(task_id, layout_id, seed, condition)
        for layout_id in layouts
        for condition in conditions
        for seed in seeds
    ]
    result: dict[str, object] = _aggregate(all_episodes)
    result["by_condition"] = {
        _condition_name(condition): _aggregate([episode for episode in all_episodes if episode.condition == condition])
        for condition in conditions
    }
    return result


def drive_benchmark(*, seeds: tuple[int, ...] = M4_EVALUATION_SEEDS) -> dict[str, object]:
    """Report drive-task success, survival, reward, boundary contact, and split robustness."""
    results: dict[str, dict[str, dict[str, object]]] = {}
    for task_id in ("play_when_bored", "competing_drives"):
        task = get_task(task_id)
        results[task_id] = {
            "train": _evaluate(task_id, task.train_layout_ids, TRAIN_CONDITIONS, seeds),
            "heldout": _evaluate(task_id, task.heldout_layout_ids, HELDOUT_CONDITIONS, seeds),
        }
    train_rates = [float(result["train"]["task_success_rate"]) for result in results.values()]
    heldout_rates = [float(result["heldout"]["task_success_rate"]) for result in results.values()]
    heldout_condition_rates = [
        float(condition_result["task_success_rate"])
        for result in results.values()
        for condition_result in result["heldout"]["by_condition"].values()  # type: ignore[index,union-attr]
    ]
    return {
        "schema_version": "0.4",
        "environment_version": "0.4.0",
        "evaluation_seeds": list(seeds),
        "tasks": {task_id: asdict(get_task(task_id)) for task_id in results},
        "conditions": {"train": list(TRAIN_CONDITIONS), "heldout": list(HELDOUT_CONDITIONS)},
        "results": results,
        "robustness": {
            "mean_train_task_success_rate": float(np.mean(train_rates)),
            "mean_heldout_task_success_rate": float(np.mean(heldout_rates)),
            "heldout_success_rate_delta": float(np.mean(heldout_rates) - np.mean(train_rates)),
            "all_heldout_conditions_successful": all(rate == 1.0 for rate in heldout_condition_rates),
        },
        "limits": [
            "Dynamics variants change both floor friction metadata and the bounded skill's movement scale; this is not raw torque control.",
            "Food colors and toy shapes are visual variants; the shared toy play transition is shape-invariant.",
            "Held-out conditions include one-factor food, toy, and dynamics changes plus a compositional shift.",
        ],
    }


def write_drive_report(path: str | Path, **kwargs: object) -> dict[str, object]:
    report = drive_benchmark(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
