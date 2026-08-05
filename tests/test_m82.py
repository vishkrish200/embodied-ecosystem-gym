from __future__ import annotations

import json

import pytest

import ecosystem_gym.m81 as m81
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m8 import _agent_image_centre
from ecosystem_gym.m81 import M81_TEST_SEEDS, M81_TRAIN_SEEDS
from ecosystem_gym.m82 import (
    M81_FROZEN_POLICY_FINGERPRINT,
    M81_FROZEN_PROTOCOL_FINGERPRINT,
    M82_CONDITIONS,
    M82_PAIRED_WIN_MARGIN,
    M82_TEST_SEEDS,
    m82_benchmark,
    run_m82_viewer_demo,
    write_m82_report,
)


def test_m82_protocol_is_sealed_and_uses_new_external_conditions() -> None:
    assert M82_TEST_SEEDS == tuple(range(200, 220))
    assert not (set(M82_TEST_SEEDS) & (set(M81_TRAIN_SEEDS) | set(M81_TEST_SEEDS)))
    assert tuple(M82_CONDITIONS) == (
        "northwest_blue_west_sector",
        "southeast_purple_south_sector",
        "northwest_blue_relocation",
        "southeast_purple_blocked_landmark",
    )
    assert {controls["layout_id"] for controls in M82_CONDITIONS.values()} == {"m82_northwest", "m82_southeast"}
    assert {controls["food_variant"] for controls in M82_CONDITIONS.values()} == {"blue", "purple"}
    assert M82_CONDITIONS["southeast_purple_blocked_landmark"]["geometry_variant"] == "m82_landmark"


def test_m82_report_locks_weights_and_reports_the_stop_rule(tmp_path) -> None:
    report = write_m82_report(tmp_path / "m82.json")
    assert json.loads((tmp_path / "m82.json").read_text(encoding="utf-8")) == report
    assert report["m81_freeze"]["m82_training_episodes"] == 0
    assert report["m81_freeze"]["frozen_protocol_fingerprint"] == M81_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m81_freeze"]["observed_protocol_fingerprint"] == M81_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m81_freeze"]["frozen_policy_fingerprint"] == M81_FROZEN_POLICY_FINGERPRINT
    assert report["m81_freeze"]["observed_policy_fingerprint"] == M81_FROZEN_POLICY_FINGERPRINT
    assert set(report["results"]) == set(M82_CONDITIONS)
    verdict = report["transfer_verdict"]
    assert verdict["paired_episodes"] == len(M82_CONDITIONS) * len(M82_TEST_SEEDS)
    assert verdict["predeclared_net_advantage_margin"] == M82_PAIRED_WIN_MARGIN
    low, high = verdict["paired_net_advantage_bootstrap_95"]
    assert -1.0 <= low <= high <= 1.0
    assert verdict["transfer_win"] == (
        verdict["paired_net_advantage"] >= M82_PAIRED_WIN_MARGIN and low > 0.0
    )
    for condition in report["results"].values():
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


def test_m82_rejects_non_protocol_test_seeds() -> None:
    try:
        m82_benchmark(test_seeds=(200,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("custom M8.2 test seeds must be rejected")


def test_m82_rejects_an_edited_m81_test_protocol(monkeypatch) -> None:
    monkeypatch.setattr(m81, "M81_TEST_SEEDS", (0, 1))
    with pytest.raises(AssertionError, match="protocol fingerprint drifted"):
        m82_benchmark()


def test_m82_lighting_and_landmark_leave_the_agent_visible_and_replay(tmp_path) -> None:
    env = EcosystemEnv(EcosystemConfig(observation_mode="rgb"))
    try:
        observation, info = env.reset(seed=200, options={"task_id": "find_and_eat_perception", **M82_CONDITIONS["northwest_blue_west_sector"]})
        assert info["lighting_variant"] == "dim"
        assert _agent_image_centre(observation["rgb"]) is not None
        observation, _ = env.reset(seed=200, options={"task_id": "find_and_eat_perception", **M82_CONDITIONS["southeast_purple_blocked_landmark"]})
        assert _agent_image_centre(observation["rgb"]) is not None
    finally:
        env.close()

    trace = tmp_path / "m82.jsonl"
    result = run_m82_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.steps == len(records) - 1
    assert any(record["disturbance"] == "food_relocated" for record in records[1:])
