from __future__ import annotations

import inspect
import json

from ecosystem_gym.m81 import (
    M81_CONDITIONS,
    M81_DIAGNOSTIC_CONDITIONS,
    M81_PAIRED_WIN_MARGIN,
    M81_TEST_SEEDS,
    M81_TRAIN_SEEDS,
    FeedForwardRgbBCPolicy,
    RecurrentRgbBCPolicy,
    m81_benchmark,
    write_m81_report,
)


def test_learned_policy_act_boundary_has_no_privileged_arguments() -> None:
    assert tuple(inspect.signature(FeedForwardRgbBCPolicy.act).parameters) == ("self", "rgb")
    assert tuple(inspect.signature(RecurrentRgbBCPolicy.act).parameters) == ("self", "rgb", "memory")
    forbidden = {"env", "info", "task", "reset", "options"}
    assert forbidden.isdisjoint(inspect.signature(FeedForwardRgbBCPolicy.act).parameters)
    assert forbidden.isdisjoint(inspect.signature(RecurrentRgbBCPolicy.act).parameters)


def test_m81_protocol_has_frozen_disjoint_seeds_conditions_and_diagnostic() -> None:
    assert M81_TEST_SEEDS == tuple(range(20))
    assert set(M81_TRAIN_SEEDS).isdisjoint(M81_TEST_SEEDS)
    assert tuple(M81_CONDITIONS) == ("reference", "occlusion", "blocked_distractor", "relocation")
    assert M81_DIAGNOSTIC_CONDITIONS["landmark_geometry"]["geometry_variant"] == "m81_landmark"


def test_m81_report_is_deterministic_paired_and_honest(tmp_path) -> None:
    first = write_m81_report(tmp_path / "first.json")
    second = write_m81_report(tmp_path / "second.json")
    assert first == second
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == first
    assert tuple(first["test_seeds"]) == M81_TEST_SEEDS
    assert set(first["results"]) == set(M81_CONDITIONS)
    assert set(first["diagnostic_results"]) == set(M81_DIAGNOSTIC_CONDITIONS)
    verdict = first["recurrence_verdict"]
    assert verdict["paired_episodes"] == len(M81_CONDITIONS) * len(M81_TEST_SEEDS)
    assert verdict["predeclared_net_advantage_margin"] == M81_PAIRED_WIN_MARGIN
    assert not verdict["diagnostics_included"]
    assert verdict["recurrence_win"] == (verdict["paired_net_advantage"] >= M81_PAIRED_WIN_MARGIN)
    for condition in first["results"].values():
        assert set(condition) == {
            "m8_fixed_rgb_baseline",
            "feed_forward_rgb_bc",
            "recurrent_rgb_bc",
            "state_oracle_ceiling",
        }
        for metrics in condition.values():
            low, high = metrics["task_success_wilson_95"]
            assert metrics["episodes"] == 20
            assert 0.0 <= low <= high <= 1.0


def test_m81_rejects_non_protocol_test_seeds() -> None:
    try:
        m81_benchmark(test_seeds=(0,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("custom M8.1 test seeds must be rejected")
