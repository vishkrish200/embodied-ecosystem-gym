"""Inspectable state-oracle baselines for the M1 Find and eat task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import ActionKind
from .env import EcosystemEnv
from .trajectory import TrajectoryWriter


FIXED_EVALUATION_SEEDS = tuple(range(20))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    steps: int
    task_success: bool
    total_reward: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    seeds: tuple[int, ...]
    successes: int
    mean_reward: float

    @property
    def success_rate(self) -> float:
        return self.successes / len(self.seeds)


def skill_action(kind: ActionKind, target: np.ndarray, duration: float) -> dict[str, np.ndarray | int]:
    return {
        "kind": int(kind),
        "target": np.asarray(target, dtype=np.float32),
        "duration": np.asarray(duration, dtype=np.float32),
    }


def scripted_find_and_eat(
    seed: int,
    *,
    trajectory_path: str | Path | None = None,
) -> EpisodeResult:
    """Use the permitted state-oracle food position and guarded skill API only."""

    env = EcosystemEnv()
    observation, _ = env.reset(seed=seed)
    distance = float(np.linalg.norm(observation["food_xy"] - observation["agent_xy"]))
    walk_duration = max(0.1, distance / env.config.walk_speed_per_second)
    plan = (
        skill_action(ActionKind.WALK_TO, observation["food_xy"], walk_duration),
        skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1),
        skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1),
    )
    total_reward = 0.0
    terminated = truncated = False
    writer = (
        TrajectoryWriter(trajectory_path, episode_id=f"find-eat_seed-{seed:04d}_run-01", seed=seed)
        if trajectory_path is not None
        else None
    )
    try:
        for step, action in enumerate(plan, start=1):
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if writer is not None:
                writer.record(
                    step=step,
                    action=action,
                    observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            if terminated or truncated:
                break
        return EpisodeResult(
            seed=seed,
            steps=step,
            task_success=bool(info["task_success"]),
            total_reward=total_reward,
        )
    finally:
        if writer is not None:
            writer.close()
        env.close()


def evaluate_scripted_policy(
    seeds: tuple[int, ...] = FIXED_EVALUATION_SEEDS,
    *,
    trajectory_dir: str | Path | None = None,
) -> EvaluationResult:
    """Run the fixed in-distribution suite, optionally keeping every replay trace."""

    directory = Path(trajectory_dir) if trajectory_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    episodes = [
        scripted_find_and_eat(
            seed,
            trajectory_path=directory / f"find-eat_seed-{seed:04d}.jsonl" if directory is not None else None,
        )
        for seed in seeds
    ]
    return EvaluationResult(
        seeds=seeds,
        successes=sum(episode.task_success for episode in episodes),
        mean_reward=float(np.mean([episode.total_reward for episode in episodes])),
    )
