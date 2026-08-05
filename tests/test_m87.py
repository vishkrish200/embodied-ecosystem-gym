from __future__ import annotations

import json

from ecosystem_gym.m85 import M85_TEST_SEEDS
from ecosystem_gym.m87 import (
    M87_TRAIN_CONDITIONS,
    M87_TRAIN_SEEDS,
    M87_VALIDATION_CONDITIONS,
    M87_VALIDATION_SEEDS,
    m87_validation,
    run_m87_viewer_demo,
    write_m87_report,
)


def test_m87_uses_disjoint_blue_hard_negatives_and_passes_its_frozen_gate(tmp_path) -> None:
    report = write_m87_report(tmp_path / "m87.json")
    assert json.loads((tmp_path / "m87.json").read_text(encoding="utf-8")) == report
    assert not (set(M87_TRAIN_SEEDS) & set(M85_TEST_SEEDS))
    assert not (set(M87_VALIDATION_SEEDS) & set(M85_TEST_SEEDS))
    assert {controls["layout_id"] for controls in M87_VALIDATION_CONDITIONS.values()}.isdisjoint(
        {"m85_northeast", "m85_southwest", "m85_northwest", "m85_southeast"}
    )
    assert "blue_hard_northeast" in M87_TRAIN_CONDITIONS
    assert all(result["passes"] for group in report["scan_coverage"].values() for result in group.values())
    assert report["m85_training_episodes"] == 0
    assert report["validation_gate"]["passes"]
    assert report["validation_gate"]["condition_success_rates"]["northwest_blue_nominal"] == 0.8
    assert report["grounding_metrics"]["northwest_blue_nominal"]["visible_recall"] > 0.85


def test_m87_rejects_custom_validation_seeds_and_replays(tmp_path) -> None:
    try:
        m87_validation(seeds=(1_000,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("custom M8.7 validation seeds must be rejected")

    trace = tmp_path / "m87.jsonl"
    result = run_m87_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert result.steps == len(records) - 1
