from __future__ import annotations

import numpy as np

from ecosystem_gym import EcosystemEnv
from ecosystem_gym.actions import ActionKind
from ecosystem_gym.experiments.m4 import drive_benchmark, write_drive_report
from ecosystem_gym.policies import skill_action


def _walk(env: EcosystemEnv, observation: dict[str, object], target: np.ndarray) -> dict[str, np.ndarray | int]:
    distance = float(np.linalg.norm(target - np.asarray(observation["agent_xy"])))
    return skill_action(
        ActionKind.WALK_TO,
        target,
        max(0.1, distance / (env.config.walk_speed_per_second * env._movement_speed_scale)),
    )


def test_play_when_bored_requires_toy_proximity_and_relieves_boredom() -> None:
    env = EcosystemEnv()
    observation, reset_info = env.reset(
        seed=7,
        options={"task_id": "play_when_bored", "toy_variant": "cube", "dynamics_variant": "slippery"},
    )
    assert reset_info["environment_version"] == "0.4.0"
    before = float(observation["drives"][2])
    _, _, terminated, _, info = env.step(skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 0.5))
    assert not terminated
    assert info["outcome"] == "blocked"
    observation, _, _, _, _ = env.step(_walk(env, observation, np.asarray(observation["toy_xy"])))
    observation, _, terminated, _, info = env.step(
        skill_action(ActionKind.RUN_AROUND, np.asarray(observation["toy_xy"]), 0.5)
    )
    assert terminated and info["task_success"]
    assert float(observation["drives"][2]) < before
    env.close()


def test_competing_drives_and_heldout_appearance_variants_are_reported() -> None:
    env = EcosystemEnv()
    observation, reset_info = env.reset(
        seed=4,
        options={"task": "competing_drives", "food_variant": "purple", "toy_variant": "capsule"},
    )
    assert reset_info["food_variant"] == "purple"
    assert reset_info["toy_variant"] == "capsule"
    assert np.isclose(float(observation["drives"][0]), 0.2)
    env.close()

    report = drive_benchmark(seeds=(0,))
    assert report["results"]["play_when_bored"]["heldout"]["survival_rate"] == 1.0
    assert report["robustness"]["all_heldout_conditions_successful"]
    assert len(report["results"]["competing_drives"]["heldout"]["by_condition"]) == 4


def test_competing_drives_rejects_play_before_food() -> None:
    env = EcosystemEnv()
    observation, _ = env.reset(seed=7, options={"task_id": "competing_drives"})
    observation, _, _, _, _ = env.step(_walk(env, observation, np.asarray(observation["toy_xy"])))
    observation, _, _, _, _ = env.step(skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 0.5))
    observation, _, _, _, _ = env.step(_walk(env, observation, np.asarray(observation["food_xy"])))
    observation, _, _, _, _ = env.step(skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1))
    _, _, terminated, _, info = env.step(skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1))
    assert not terminated
    assert not info["task_success"]
    env.close()


def test_drive_report_is_serializable(tmp_path) -> None:
    path = tmp_path / "m4-report.json"
    report = write_drive_report(path, seeds=(0,))
    assert path.is_file()
    assert report["results"]["competing_drives"]["train"]["boundary_contacts"] >= 0
