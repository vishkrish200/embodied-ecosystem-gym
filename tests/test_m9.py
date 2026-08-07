from __future__ import annotations

import numpy as np

from ecosystem_gym.actions import ActionKind
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m9 import (
    M9_AUDIT_CONDITIONS,
    M9_AUDIT_SEEDS,
    M9_TRAIN_CONDITIONS,
    M9_TRAIN_SEEDS,
    M9_VALIDATION_CONDITIONS,
    M9_VALIDATION_SEEDS,
    m9_scan_coverage,
    m9_validation,
)
from ecosystem_gym.policies import skill_action


def test_maintain_needs_requires_both_food_and_play() -> None:
    env = EcosystemEnv(EcosystemConfig(observation_mode="state_oracle", max_episode_steps=20))
    try:
        observation, _ = env.reset(seed=1_400, options={"task_id": "maintain_needs", "layout_id": "m9_validation_northeast"})
        for action in (
            skill_action(ActionKind.WALK_TO, observation["food_xy"], 5.0),
            skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1),
            skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1),
        ):
            observation, _, terminated, _, info = env.step(action)
        assert not terminated
        assert not info["task_success"]
        observation, _, _, _, _ = env.step(skill_action(ActionKind.WALK_TO, observation["toy_xy"], 5.0))
        _, _, terminated, _, info = env.step(skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 1.0))
        assert terminated
        assert info["task_success"]
    finally:
        env.close()


def test_m9_splits_are_disjoint_and_coverage_is_publicly_valid() -> None:
    assert set(M9_TRAIN_SEEDS).isdisjoint(M9_VALIDATION_SEEDS)
    assert set(M9_TRAIN_SEEDS).isdisjoint(M9_AUDIT_SEEDS)
    assert set(M9_VALIDATION_SEEDS).isdisjoint(M9_AUDIT_SEEDS)
    train_layouts = {row["layout_id"] for row in M9_TRAIN_CONDITIONS.values()}
    validation_layouts = {row["layout_id"] for row in M9_VALIDATION_CONDITIONS.values()}
    audit_layouts = {row["layout_id"] for row in M9_AUDIT_CONDITIONS.values()}
    assert train_layouts.isdisjoint(validation_layouts | audit_layouts)
    assert validation_layouts.isdisjoint(audit_layouts)
    coverage = m9_scan_coverage(M9_VALIDATION_CONDITIONS, seeds=M9_VALIDATION_SEEDS[:2])
    assert all(row["passes"] for row in coverage.values())


def test_m9_validation_is_frozen_and_reports_its_failed_gate() -> None:
    try:
        m9_validation(seeds=(1_401,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("M9 validation must reject custom seeds")
