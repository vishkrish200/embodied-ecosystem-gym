from __future__ import annotations

import json

import numpy as np

from ecosystem_gym.actions import ActionKind
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m84 import (
    M84_TRAIN_CONDITIONS,
    M84_TRAIN_SEEDS,
    M84_VALIDATION_CONDITIONS,
    M84_VALIDATION_SEEDS,
    _options,
    m84_config,
    m84_scan_coverage,
    write_m84_report,
)
from ecosystem_gym.policies import skill_action


def test_scan_v2_changes_public_rgb_without_exposing_camera_metadata() -> None:
    env = EcosystemEnv(m84_config())
    try:
        observation, info = env.reset(seed=500, options=_options(M84_VALIDATION_CONDITIONS["northwest_orange_east"]))
        scanned, _, _, _, scanned_info = env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
        assert info["environment_version"] == "0.6.0"
        assert info["camera_sector"] == "east"
        assert scanned_info["camera_sector"] == "south"
        assert not np.array_equal(observation["rgb"], scanned["rgb"])
        assert set(observation) == {"rgb", "drives", "holding_food", "prior_outcome"}
    finally:
        env.close()


def test_m84_coverage_and_frozen_validation_gate(tmp_path) -> None:
    train = m84_scan_coverage(M84_TRAIN_CONDITIONS, seeds=M84_TRAIN_SEEDS)
    validation = m84_scan_coverage(M84_VALIDATION_CONDITIONS, seeds=M84_VALIDATION_SEEDS)
    assert all(result["passes"] for result in train.values())
    assert all(result["passes"] for result in validation.values())

    report = write_m84_report(tmp_path / "m84.json")
    assert json.loads((tmp_path / "m84.json").read_text(encoding="utf-8")) == report
    assert report["m82_training_episodes"] == 0
    assert report["validation_gate"]["episodes"] == 80
    assert report["validation_gate"]["passes"]
    assert all(rate >= 0.75 for rate in report["validation_gate"]["condition_success_rates"].values())
    for result in report["results"].values():
        assert result["state_oracle_ceiling"]["successes"] == len(M84_VALIDATION_SEEDS)
    blocked = report["grounding_metrics"]["northwest_purple_blocked"]
    assert blocked["matched_target_components"] + blocked["false_negative_components"] > 0
