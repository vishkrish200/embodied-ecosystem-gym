from __future__ import annotations

import json

import pytest

from ecosystem_gym.m85 import M85_TEST_SEEDS
from ecosystem_gym.m86 import (
    M86_DIAGNOSTIC_SEEDS,
    m86_diagnostics,
    run_m86_viewer_demo,
    write_m86_report,
)


def test_m86_is_disjoint_factorial_diagnosis_of_the_frozen_grounder(tmp_path) -> None:
    report = write_m86_report(tmp_path / "m86.json")
    assert json.loads((tmp_path / "m86.json").read_text(encoding="utf-8")) == report
    assert tuple(report["diagnostic_seeds"]) == M86_DIAGNOSTIC_SEEDS
    assert not (set(M86_DIAGNOSTIC_SEEDS) & set(M85_TEST_SEEDS))
    assert report["m84_freeze"]["m86_training_episodes"] == 0
    assert report["m85_exclusion"]["reused_m85_seeds"] == 0
    assert report["m85_exclusion"]["reused_m85_layouts"] == 0
    assert all(result["passes"] for result in report["scan_coverage"].values())
    assert report["results"]["new_northeast_red_nominal"]["frozen_m84_rgb_heatmap"]["successes"] == 20
    assert report["results"]["new_northeast_blue_nominal"]["frozen_m84_rgb_heatmap"]["successes"] == 10
    assert report["grounding_metrics"]["new_northeast_blue_nominal"]["visible_recall"] < 0.5
    assert report["grounding_metrics"]["new_northeast_red_nominal"]["visible_recall"] > 0.9


def test_m86_rejects_custom_seeds_and_replays_a_diagnostic_trace(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        m86_diagnostics(seeds=(700,))

    trace = tmp_path / "m86.jsonl"
    result = run_m86_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.steps == len(records) - 1
    assert records[0]["episode_id"] == "m86-blue-grippy-diagnosis"
