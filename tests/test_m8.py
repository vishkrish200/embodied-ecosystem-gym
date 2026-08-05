from __future__ import annotations

import json

import numpy as np

from ecosystem_gym.actions import ActionKind
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m8 import (
    M8_CONDITIONS,
    M8_EVALUATION_SEEDS,
    FixedRgbScanRecoveryPolicy,
    _run_rgb_episode,
    m8_benchmark,
    m8_config,
    run_m8_viewer_demo,
    wilson_interval,
)
from ecosystem_gym.policies import skill_action


def test_scan_changes_only_the_public_rgb_view_and_replays_through_m8_demo(tmp_path) -> None:
    env = EcosystemEnv(EcosystemConfig(observation_mode="rgb"))
    try:
        observation, info = env.reset(seed=7, options={"camera_control": "scan", "initial_scan_sector": "north"})
        scanned, _, _, _, scanned_info = env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
        assert info["camera_sector"] == "north"
        assert scanned_info["camera_sector"] == "east"
        assert not np.array_equal(observation["rgb"], scanned["rgb"])
        assert set(observation) == {"rgb", "drives", "holding_food", "prior_outcome"}
    finally:
        env.close()

    trace = tmp_path / "m8.jsonl"
    result = run_m8_viewer_demo(trace)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    steps = records[1:]
    assert result.task_success
    assert any(record["action"]["kind"] == int(ActionKind.SCAN) for record in steps)
    assert any(record["disturbance"] == "food_relocated" for record in steps)
    assert steps[-1]["post_disturbance_completion"]
    assert all("camera_sector" in record for record in steps)


def test_rgb_policy_is_pure_observation_and_memory_interface() -> None:
    env = EcosystemEnv(m8_config())
    try:
        observation, _ = env.reset(seed=7, options={"camera_control": "scan", "initial_scan_sector": "north"})
        with_metadata = {**observation, "task_id": "secret", "layout_id": "secret", "seed": 999, "info": object()}
        policy = FixedRgbScanRecoveryPolicy()
        first = policy.act(observation, policy.reset())
        second = policy.act(with_metadata, policy.reset())
        assert first["kind"] == second["kind"]
        assert np.array_equal(first["target"], second["target"])
        assert first["duration"] == second["duration"]
    finally:
        env.close()


def test_m8_protocol_is_frozen_complete_and_reports_uncertainty_without_a_success_gate() -> None:
    report = m8_benchmark()
    assert tuple(report["evaluation_seeds"]) == M8_EVALUATION_SEEDS
    assert tuple(report["conditions"]) == tuple(M8_CONDITIONS)
    assert "gate" not in report
    assert report["policy_input"] == ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"]
    for condition, result in report["results"].items():
        assert set(result) == {"fixed_rgb_scan_recovery_baseline", "state_oracle_ceiling"}
        for metrics in result.values():
            assert metrics["episodes"] == 20
            assert len(metrics["task_success_wilson_95"]) == 2
            assert 0.0 <= metrics["task_success_wilson_95"][0] <= metrics["task_success_wilson_95"][1] <= 1.0
    assert report["results"]["occlusion"]["fixed_rgb_scan_recovery_baseline"]["scan_actions"] > 0
    assert report["results"]["blocked_distractor"]["fixed_rgb_scan_recovery_baseline"]["blocked_outcomes"] > 0
    assert report["results"]["relocation"]["fixed_rgb_scan_recovery_baseline"]["post_disturbance_completions"] > 0


def test_wilson_interval_has_a_non_degenerate_small_sample_bound() -> None:
    assert wilson_interval(0, 20)[1] > 0.0
    assert wilson_interval(20, 20)[0] < 1.0


def test_blocked_distractor_episode_reobserves_and_completes() -> None:
    episode = _run_rgb_episode(FixedRgbScanRecoveryPolicy(), seed=7, condition="blocked_distractor")
    assert episode.blocked_outcomes > 0
    assert episode.task_success
