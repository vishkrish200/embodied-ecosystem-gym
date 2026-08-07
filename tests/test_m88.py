from __future__ import annotations

import json

import pytest

from ecosystem_gym.experiments.m87 import M87_TRAIN_CONDITIONS, M87_VALIDATION_CONDITIONS
from ecosystem_gym.experiments.m88 import (
    M87_FROZEN_POLICY_FINGERPRINT,
    M87_FROZEN_PROTOCOL_FINGERPRINT,
    M88_CONDITIONS,
    M88_FROZEN_PROTOCOL_FINGERPRINT,
    M88_TEST_SEEDS,
    m88_benchmark,
    m88_protocol_fingerprint,
    run_m88_viewer_demo,
    write_m88_report,
)


def test_m88_is_disjoint_coverage_gated_and_freezes_m87(tmp_path) -> None:
    development_layouts = {controls["layout_id"] for controls in M87_TRAIN_CONDITIONS.values()} | {
        controls["layout_id"] for controls in M87_VALIDATION_CONDITIONS.values()
    }
    assert {controls["layout_id"] for controls in M88_CONDITIONS.values()}.isdisjoint(development_layouts)
    assert M88_TEST_SEEDS == tuple(range(1_100, 1_120))
    assert m88_protocol_fingerprint() == M88_FROZEN_PROTOCOL_FINGERPRINT

    report = write_m88_report(tmp_path / "m88.json")
    assert json.loads((tmp_path / "m88.json").read_text(encoding="utf-8")) == report
    assert all(result["passes"] for result in report["scan_coverage"].values())
    assert report["m87_freeze"]["m88_training_episodes"] == 0
    assert report["m87_freeze"]["observed_protocol_fingerprint"] == M87_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m87_freeze"]["observed_policy_fingerprint"] == M87_FROZEN_POLICY_FINGERPRINT
    assert report["external_validity_gate"]["oracle_successes"] == 80
    assert report["external_validity_gate"]["passes"]
    assert report["external_validity_gate"]["successes"] == 79


def test_m88_rejects_custom_seeds_and_replays_its_predeclared_demo(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        m88_benchmark(test_seeds=(1_100,))

    trace = tmp_path / "m88.jsonl"
    result = run_m88_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert result.steps == len(records) - 1
    assert any(record["disturbance"] == "food_relocated" for record in records[1:])
