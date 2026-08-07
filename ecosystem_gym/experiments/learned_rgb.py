"""Deterministic NumPy behavior cloning from RGB to local skill targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..actions import ActionKind
from ..config import EcosystemConfig
from ..env import EcosystemEnv
from .m3 import M3_EVALUATION_SEEDS, collect_rgb_behavior_cloning_data
from ..policies import skill_action
from ..tasks import get_task


def _features(rgb: np.ndarray, holding_food: np.ndarray | float) -> np.ndarray:
    """Extract a color-agnostic candidate location from policy-visible RGB.

    This is deliberately not the M3 red-pixel detector: it retains every
    sufficiently saturated object except the fixed cyan agent and yellow toy,
    then the fitted regressor learns the camera-pixel-to-local-target mapping
    from demonstrations.  It therefore keeps blue and purple food in the
    held-out evaluation path without consuming oracle coordinates.
    """

    image = np.asarray(rgb, dtype=np.float32)
    images = image[None, ...] if image.ndim == 3 else image
    red, green, blue = (images[..., channel] for channel in range(3))
    saturation = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    yellow_toy = (red > 140.0) & (green > 100.0) & (blue < 120.0)
    cyan_agent = (blue > 180.0) & (green > 120.0) & (red < 140.0)
    candidate = (saturation > 45.0) & ~yellow_toy & ~cyan_agent
    rows, columns = np.indices(candidate.shape[1:])
    count = candidate.sum(axis=(1, 2))
    denominator = np.maximum(count, 1)
    x = (candidate * columns).sum(axis=(1, 2)) / denominator
    y = (candidate * rows).sum(axis=(1, 2)) / denominator
    width = candidate.shape[2] - 1
    height = candidate.shape[1] - 1
    held = np.asarray(holding_food, dtype=np.float32).reshape(-1)
    return np.column_stack(
        [x / width * 2.0 - 1.0, y / height * 2.0 - 1.0, count > 0, held, np.ones(len(images))]
    ).astype(np.float32)


class LearnedRgbBCPolicy:
    """A learned local-target regressor with the same odometry adapter as M3."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights
        self.estimated_xy = np.zeros(2, dtype=np.float32)
        self.awaiting_pickup = False

    @classmethod
    def fit(cls, dataset_path: str | Path, *, ridge: float = 1e-3) -> "LearnedRgbBCPolicy":
        data = np.load(dataset_path)
        walk = data["action_kind"] == int(ActionKind.WALK_TO)
        features = _features(data["rgb"][walk], data["holding_food"][walk])
        targets = data["action_target_relative"][walk].astype(np.float32)
        weights = np.linalg.solve(features.T @ features + ridge * np.eye(features.shape[1]), features.T @ targets)
        return cls(weights.astype(np.float32))

    def reset(self) -> None:
        self.estimated_xy[:] = 0.0
        self.awaiting_pickup = False

    def act(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        if self.awaiting_pickup:
            self.awaiting_pickup = False
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        relative = (_features(np.asarray(observation["rgb"]), float(observation["holding_food"])) @ self.weights)[0]
        target = np.clip(self.estimated_xy + relative, -0.9, 0.9).astype(np.float32)
        duration = max(0.1, float(np.linalg.norm(target - self.estimated_xy)) / env.config.walk_speed_per_second)
        self.estimated_xy = target
        self.awaiting_pickup = True
        return skill_action(ActionKind.WALK_TO, target, duration)


def _evaluate(policy: LearnedRgbBCPolicy, appearance_ids: tuple[str, ...]) -> dict[str, object]:
    task = get_task("find_and_eat_perception")
    successes = 0
    total = 0
    for appearance in appearance_ids:
        for layout_id in task.heldout_layout_ids:
            for seed in M3_EVALUATION_SEEDS:
                env = EcosystemEnv(EcosystemConfig(observation_mode="rgb"))
                observation, _ = env.reset(seed=seed, options={"layout_id": layout_id, "food_variant": appearance})
                policy.reset()
                succeeded = False
                try:
                    for _ in range(8):
                        observation, _, terminated, truncated, info = env.step(policy.act(observation, env))
                        if terminated or truncated:
                            succeeded = bool(info["task_success"])
                            break
                finally:
                    env.close()
                successes += int(succeeded)
                total += 1
    return {"episodes": total, "successes": successes, "success_rate": successes / total}


def _evaluate_random(appearance_ids: tuple[str, ...]) -> dict[str, object]:
    """Measure a fixed-seed RGB-agnostic skill sampler as the named gate baseline."""

    task = get_task("find_and_eat_perception")
    successes = 0
    total = 0
    for appearance_index, appearance in enumerate(appearance_ids):
        for layout_index, layout_id in enumerate(task.heldout_layout_ids):
            for seed in M3_EVALUATION_SEEDS:
                rng = np.random.default_rng(10_000 * appearance_index + 1_000 * layout_index + seed)
                env = EcosystemEnv(EcosystemConfig(observation_mode="rgb"))
                env.reset(seed=seed, options={"layout_id": layout_id, "food_variant": appearance})
                succeeded = False
                try:
                    for _ in range(8):
                        action = skill_action(
                            ActionKind(int(rng.integers(len(ActionKind)))),
                            rng.uniform(-0.9, 0.9, size=2).astype(np.float32),
                            float(rng.uniform(0.1, 2.0)),
                        )
                        _, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            succeeded = bool(info["task_success"])
                            break
                finally:
                    env.close()
                successes += int(succeeded)
                total += 1
    return {"episodes": total, "successes": successes, "success_rate": successes / total}


def learned_rgb_gate(output: str | Path, *, dataset_path: str | Path | None = None) -> dict[str, object]:
    """Fit RGB BC and report fixed train versus held-out appearance performance."""

    dataset = Path(dataset_path) if dataset_path is not None else Path("artifacts/datasets/m35-rgb-bc.npz")
    if not dataset.exists():
        collect_rgb_behavior_cloning_data(dataset)
    policy = LearnedRgbBCPolicy.fit(dataset)
    report = {
        "schema_version": "0.35",
        "policy": "numpy_ridge_rgb_behavior_cloning",
        "feature_extractor": "fixed RGB candidate geometry plus learned local-target regression",
        "dataset": str(dataset),
        "train_appearance": _evaluate(policy, ("red", "orange")),
        "heldout_appearance": _evaluate(policy, ("blue", "purple")),
        "heldout_random_rgb": _evaluate_random(("blue", "purple")),
        "acceptance": {
            "heldout_success_rate_minimum": 0.50,
            "named_baseline": "measured_random_rgb",
        },
    }
    report["heldout_beats_random"] = (
        report["heldout_appearance"]["success_rate"] > report["heldout_random_rgb"]["success_rate"]
    )
    report["passed"] = bool(
        report["heldout_beats_random"]
        and report["heldout_appearance"]["success_rate"] >= report["acceptance"]["heldout_success_rate_minimum"]
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
