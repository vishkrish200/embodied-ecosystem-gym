"""M3 perception, disturbance-recovery, and behavior-cloning benchmark tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from .actions import ActionKind
from .benchmark import TabularOracleQPolicy
from .config import EcosystemConfig
from .env import EcosystemEnv
from .perception import detection_to_relative_xy, food_detection_from_rgb
from .policies import skill_action
from .tasks import get_task


M3_EVALUATION_SEEDS = tuple(range(10))
CAMERA_VARIANTS = ("center", "left", "right")
ObservationMode = Literal["state_oracle", "hybrid", "rgb"]


@dataclass(frozen=True, slots=True)
class PerceptionEpisode:
    seed: int
    layout_id: str
    camera_variant: str
    task_success: bool
    total_reward: float
    steps: int
    disturbance_events: int
    recovery_actions: int
    failure_reasons: tuple[str, ...]


class M3Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]: ...


class OraclePolicy:
    def __init__(self, policy: TabularOracleQPolicy) -> None:
        self.policy = policy

    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        return self.policy.act(observation, env)


class VisualServoPolicy:
    """A transparent visual controller calibrated only to the RGB camera footprint."""

    def __init__(self, mode: Literal["hybrid", "rgb"]) -> None:
        self.mode = mode
        self._estimated_xy = np.zeros(2, dtype=np.float32)
        self._awaiting_pickup = False

    def reset(self) -> None:
        # The M3 task registry fixes the creature's initial spawn at the room origin.
        self._estimated_xy = np.zeros(2, dtype=np.float32)
        self._awaiting_pickup = False

    def act(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        if self._awaiting_pickup:
            self._awaiting_pickup = False
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        detection = (
            np.asarray(observation["food_detection"], dtype=np.float32)
            if self.mode == "hybrid"
            else food_detection_from_rgb(np.asarray(observation["rgb"], dtype=np.uint8))
        )
        if detection[0] <= 0.0:
            # An inspectable failure mode: without a detection, return to the known room origin.
            target = np.zeros(2, dtype=np.float32)
            duration = max(0.1, float(np.linalg.norm(target - self._estimated_xy)) / env.config.walk_speed_per_second)
            self._estimated_xy = target
            return skill_action(ActionKind.WALK_TO, target, duration)
        relative_xy = detection_to_relative_xy(detection)
        if float(np.linalg.norm(relative_xy)) <= env.config.pickup_radius:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        target = np.clip(self._estimated_xy + relative_xy, -0.9, 0.9).astype(np.float32)
        duration = max(0.1, float(np.linalg.norm(target - self._estimated_xy)) / env.config.walk_speed_per_second)
        self._estimated_xy = target
        self._awaiting_pickup = True
        return skill_action(ActionKind.WALK_TO, target, duration)


def _run_episode(
    policy: M3Policy,
    env: EcosystemEnv,
    *,
    seed: int,
    layout_id: str,
    camera_variant: str,
    disturbed: bool,
) -> PerceptionEpisode:
    options: dict[str, Any] = {"layout_id": layout_id, "camera_variant": camera_variant}
    if disturbed:
        options["disturbance_step"] = 1
    observation, _ = env.reset(seed=seed, options=options)
    policy.reset()
    total_reward = 0.0
    disturbances = 0
    recoveries = 0
    failures: list[str] = []
    for step in range(1, env.config.max_episode_steps + 1):
        observation, reward, terminated, truncated, info = env.step(policy.act(observation, env))
        total_reward += reward
        disturbances += int(info["disturbance"] is not None)
        recoveries += int(info["recovery_action"])
        if info["failure_reason"] is not None:
            failures.append(str(info["failure_reason"]))
        if terminated or truncated:
            return PerceptionEpisode(
                seed,
                layout_id,
                camera_variant,
                bool(info["task_success"]),
                total_reward,
                step,
                disturbances,
                recoveries,
                tuple(failures),
            )
    raise AssertionError("environment did not terminate at its declared step limit")


def _aggregate(episodes: list[PerceptionEpisode]) -> dict[str, object]:
    failures: dict[str, int] = {}
    for episode in episodes:
        for reason in episode.failure_reasons:
            failures[reason] = failures.get(reason, 0) + 1
    disturbed_episodes = [episode for episode in episodes if episode.disturbance_events]
    return {
        "episodes": len(episodes),
        "successes": sum(episode.task_success for episode in episodes),
        "success_rate": float(np.mean([episode.task_success for episode in episodes])),
        "mean_total_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([episode.steps for episode in episodes])),
        "disturbance_events": sum(episode.disturbance_events for episode in episodes),
        "recovery_actions": sum(episode.recovery_actions for episode in episodes),
        "disturbed_success_rate": (
            float(np.mean([episode.task_success for episode in disturbed_episodes])) if disturbed_episodes else None
        ),
        "failure_reasons": failures,
    }


def _evaluate(policy: M3Policy, mode: ObservationMode, layout_ids: tuple[str, ...], *, disturbed: bool) -> dict[str, object]:
    env = EcosystemEnv(EcosystemConfig(observation_mode=mode))
    try:
        episodes = [
            _run_episode(
                policy,
                env,
                seed=seed,
                layout_id=layout_id,
                camera_variant=camera_variant,
                disturbed=disturbed,
            )
            for layout_id in layout_ids
            for camera_variant in CAMERA_VARIANTS
            for seed in M3_EVALUATION_SEEDS
        ]
        return _aggregate(episodes)
    finally:
        env.close()


def perception_benchmark(*, training_episodes: int = 400) -> dict[str, object]:
    task = get_task("find_and_eat_perception")
    oracle_q = TabularOracleQPolicy()
    oracle_q.train(task, episodes=training_episodes)
    baselines: tuple[tuple[str, ObservationMode, M3Policy], ...] = (
        ("state_oracle_q_learning", "state_oracle", OraclePolicy(oracle_q)),
        ("hybrid_visual_servo", "hybrid", VisualServoPolicy("hybrid")),
        ("rgb_visual_servo", "rgb", VisualServoPolicy("rgb")),
    )
    results: dict[str, object] = {}
    for name, mode, policy in baselines:
        results[name] = {
            "observation_mode": mode,
            "train": {
                "clean": _evaluate(policy, mode, task.train_layout_ids, disturbed=False),
                "disturbed": _evaluate(policy, mode, task.train_layout_ids, disturbed=True),
            },
            "heldout": {
                "clean": _evaluate(policy, mode, task.heldout_layout_ids, disturbed=False),
                "disturbed": _evaluate(policy, mode, task.heldout_layout_ids, disturbed=True),
            },
        }
    return {
        "schema_version": "0.3",
        "environment_version": "0.3.0",
        "task": asdict(task),
        "evaluation_seeds": list(M3_EVALUATION_SEEDS),
        "camera_variants": list(CAMERA_VARIANTS),
        "training": {"state_oracle_algorithm": "tabular_q_learning", "episodes": training_episodes},
        "results": results,
        "limits": [
            "The RGB and hybrid baselines are calibrated visual-servo controllers, not learned vision models.",
            "Hybrid food detections are red-pixel segmentations computed from RGB, not simulator coordinates.",
            "The M3 split varies spawn layouts and camera offsets in the same room geometry.",
        ],
    }


def write_perception_report(path: str | Path, **kwargs: object) -> dict[str, object]:
    report = perception_benchmark(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def collect_rgb_behavior_cloning_data(
    path: str | Path,
    *,
    seeds: tuple[int, ...] = M3_EVALUATION_SEEDS,
) -> dict[str, object]:
    """Collect RGB frames and privileged teacher labels, including recovery examples."""

    frames: list[np.ndarray] = []
    kinds: list[int] = []
    targets: list[np.ndarray] = []
    durations: list[float] = []
    episode_seeds: list[int] = []
    recovered: list[bool] = []
    relative_targets: list[np.ndarray] = []
    holding_states: list[int] = []
    food_variants: list[str] = []
    task = get_task("find_and_eat_perception")
    for food_variant in ("red", "orange"):
        for layout_id in task.train_layout_ids:
            for camera_variant in CAMERA_VARIANTS:
                for seed in seeds:
                    env = EcosystemEnv(EcosystemConfig(observation_mode="rgb"))
                    observation, _ = env.reset(
                        seed=seed,
                        options={
                            "layout_id": layout_id,
                            "camera_variant": camera_variant,
                            "food_variant": food_variant,
                            "disturbance_step": 1 if seed % 2 else None,
                        },
                    )
                    try:
                        for _ in range(8):
                            state = env._require_state()  # Privileged teacher state; never fed to the RGB policy.
                            if state.holding_food:
                                action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
                            elif env._distance_to_food() <= env.config.pickup_radius:
                                action = skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
                            else:
                                target = env._food_xy().copy()
                                duration = max(0.1, float(np.linalg.norm(target - env._agent_xy())) / env.config.walk_speed_per_second)
                                action = skill_action(ActionKind.WALK_TO, target, duration)
                            frames.append(np.asarray(observation["rgb"], dtype=np.uint8))
                            kinds.append(int(action["kind"]))
                            targets.append(np.asarray(action["target"], dtype=np.float32))
                            relative_targets.append(np.asarray(action["target"], dtype=np.float32) - env._agent_xy())
                            durations.append(float(action["duration"]))
                            episode_seeds.append(seed)
                            holding_states.append(int(observation["holding_food"]))
                            food_variants.append(food_variant)
                            observation, _, terminated, truncated, info = env.step(action)
                            recovered.append(bool(info["recovery_action"]))
                            if terminated or truncated:
                                break
                    finally:
                        env.close()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rgb=np.stack(frames),
        action_kind=np.asarray(kinds, dtype=np.int8),
        action_target=np.stack(targets),
        action_target_relative=np.stack(relative_targets),
        action_duration=np.asarray(durations, dtype=np.float32),
        episode_seed=np.asarray(episode_seeds, dtype=np.int32),
        recovery_label=np.asarray(recovered, dtype=np.bool_),
        holding_food=np.asarray(holding_states, dtype=np.int8),
        food_variant=np.asarray(food_variants),
    )
    return {
        "path": str(output),
        "samples": len(frames),
        "image_shape": list(frames[0].shape),
        "recovery_samples": int(np.sum(recovered)),
        "observation_mode": "rgb",
        "teacher": "state_oracle_scripted",
        "train_food_variants": ["red", "orange"],
    }
