from __future__ import annotations

import json

import pytest

from ecosystem_gym.experiments.m84 import M84_TRAIN_CONDITIONS, M84_VALIDATION_CONDITIONS
from ecosystem_gym.experiments.m85 import (
    M84_FROZEN_POLICY_FINGERPRINT,
    M84_FROZEN_PROTOCOL_FINGERPRINT,
    M85_CONDITIONS,
    M85_FROZEN_PROTOCOL_FINGERPRINT,
    M85_TEST_SEEDS,
    m85_benchmark,
    m85_protocol_fingerprint,
    run_m85_viewer_demo,
    write_m85_report,
)


def test_m85_is_disjoint_coverage_gated_and_freezes_m84(tmp_path) -> None:
    development_layouts = {controls["layout_id"] for controls in M84_TRAIN_CONDITIONS.values()} | {
        controls["layout_id"] for controls in M84_VALIDATION_CONDITIONS.values()
    }
    assert {controls["layout_id"] for controls in M85_CONDITIONS.values()}.isdisjoint(development_layouts)
    assert M85_TEST_SEEDS == tuple(range(600, 620))
    assert m85_protocol_fingerprint() == M85_FROZEN_PROTOCOL_FINGERPRINT

    report = write_m85_report(tmp_path / "m85.json")
    assert json.loads((tmp_path / "m85.json").read_text(encoding="utf-8")) == report
    assert all(result["passes"] for result in report["scan_coverage"].values())
    assert report["m84_freeze"]["m85_training_episodes"] == 0
    assert report["m84_freeze"]["observed_protocol_fingerprint"] == M84_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m84_freeze"]["observed_policy_fingerprint"] == M84_FROZEN_POLICY_FINGERPRINT
    assert report["external_validity_gate"]["oracle_successes"] == 80
    assert report["external_validity_gate"]["passes"] is False
    assert report["external_validity_gate"]["condition_success_rates"]["northeast_blue_grippy"] == 0.15


def test_m85_rejects_custom_seeds_and_replays_its_predeclared_demo(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        m85_benchmark(test_seeds=(600,))

    trace = tmp_path / "m85.jsonl"
    result = run_m85_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert result.steps == len(records) - 1
    assert any(record["disturbance"] == "food_relocated" for record in records[1:])
