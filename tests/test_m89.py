from __future__ import annotations

import json

import numpy as np
import pytest

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m84 import m84_config
from ecosystem_gym.experiments.m87 import M87_TRAIN_CONDITIONS, M87_VALIDATION_CONDITIONS
from ecosystem_gym.experiments.m88 import M88_CONDITIONS
from ecosystem_gym.experiments.m89 import (
    M87_FROZEN_POLICY_FINGERPRINT,
    M87_FROZEN_PROTOCOL_FINGERPRINT,
    M89_CONDITIONS,
    M89_FROZEN_PROTOCOL_FINGERPRINT,
    M89_TEST_SEEDS,
    m89_benchmark,
    m89_protocol_fingerprint,
    run_m89_viewer_demo,
    write_m89_report,
)


def test_m89_visual_variants_change_rgb_without_expanding_public_observation() -> None:
    env = EcosystemEnv(m84_config())
    try:
        sphere, _ = env.reset(seed=12, options={"task_id": "find_and_eat", "camera_control": "scan_v2"})
        shifted, _ = env.reset(
            seed=12,
            options={
                "task_id": "find_and_eat",
                "camera_control": "scan_v2",
                "agent_shape_variant": "capsule",
                "food_shape_variant": "box",
            },
        )
    finally:
        env.close()
    assert set(shifted) == {"rgb", "drives", "holding_food", "prior_outcome"}
    assert not np.array_equal(sphere["rgb"], shifted["rgb"])


def test_m89_is_disjoint_coverage_gated_and_freezes_m87(tmp_path) -> None:
    m87_layouts = {controls["layout_id"] for controls in M87_TRAIN_CONDITIONS.values()} | {
        controls["layout_id"] for controls in M87_VALIDATION_CONDITIONS.values()
    }
    assert {controls["layout_id"] for controls in M89_CONDITIONS.values()}.isdisjoint(m87_layouts)
    assert {controls["layout_id"] for controls in M89_CONDITIONS.values()}.isdisjoint(
        {controls["layout_id"] for controls in M88_CONDITIONS.values()}
    )
    assert M89_TEST_SEEDS == tuple(range(1_200, 1_220))
    assert m89_protocol_fingerprint() == M89_FROZEN_PROTOCOL_FINGERPRINT

    report = write_m89_report(tmp_path / "m89.json")
    assert json.loads((tmp_path / "m89.json").read_text(encoding="utf-8")) == report
    assert all(result["passes"] for result in report["scan_coverage"].values())
    assert report["m87_freeze"]["m89_training_episodes"] == 0
    assert report["m87_freeze"]["observed_protocol_fingerprint"] == M87_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m87_freeze"]["observed_policy_fingerprint"] == M87_FROZEN_POLICY_FINGERPRINT
    assert report["external_validity_gate"]["oracle_successes"] == 80
    assert report["external_validity_gate"]["successes"] == 75
    assert report["external_validity_gate"]["passes"]


def test_m89_rejects_custom_seeds_and_replays_its_predeclared_demo(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        m89_benchmark(test_seeds=(1_200,))

    trace = tmp_path / "m89.jsonl"
    result = run_m89_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert result.steps == len(records) - 1
    assert any(record["disturbance"] == "food_relocated" for record in records[1:])
