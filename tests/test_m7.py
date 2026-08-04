from __future__ import annotations

import json

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m7 import (
    TabularRgbDriveBCPolicy,
    _gate_passed,
    _rgb_config,
    m7_benchmark,
    observable_rgb_drive_state,
    rgb_arbitration_probe,
    run_m7_viewer_demo,
)


def test_rgb_selector_state_ignores_task_and_reset_metadata() -> None:
    env = EcosystemEnv(_rgb_config())
    try:
        observation, _ = env.reset(seed=7, options={"task_id": "competing_drives", "layout_id": "m4_train_competing"})
        state = observable_rgb_drive_state(
            observation,
            env.config,
            played_toy=False,
            pending_food_pickup=False,
            pending_toy_play=False,
        )
        with_metadata = {
            **observation,
            "task_id": "competing_drives",
            "dynamics_variant": "slippery",
            "secret": object(),
        }
        assert observable_rgb_drive_state(
            with_metadata,
            env.config,
            played_toy=False,
            pending_food_pickup=False,
            pending_toy_play=False,
        ) == state
    finally:
        env.close()


def test_learned_rgb_policy_passes_same_frame_drive_counterfactual() -> None:
    policy = TabularRgbDriveBCPolicy()
    policy.train(episodes=80)
    probe = rgb_arbitration_probe(policy)
    assert probe["passed"]
    assert probe["food_first_action"] == "PURSUE_FOOD"
    assert probe["play_first_action"] == "PURSUE_PLAY"


def test_m7_report_passes_visual_heldout_gate_and_reports_spatial_diagnostic() -> None:
    report = m7_benchmark(training_episodes=80, seeds=(0, 1))
    assert report["gate"]["passed"]
    for task_id in ("play_when_bored", "competing_drives"):
        result = report["results"][task_id]["learned_rgb_drive_bc"]
        assert len(result["visual_heldout"]["by_condition"]) == 4
        assert "spatial_heldout_diagnostic" in result
    assert report["results"]["competing_drives"]["learned_rgb_drive_bc"]["visual_heldout"]["play_before_food_episodes"] == 0


def test_m7_gate_rejects_higher_return_when_macro_random_matches_success() -> None:
    assert not _gate_passed(
        learned_rates=[1.0, 1.0],
        learned_survival_rates=[1.0, 1.0],
        condition_rates=[1.0, 1.0],
        no_drive_rates=[0.7, 1.0],
        random_rates=[1.0, 1.0],
        learned_returns=[4.0, 4.0],
        random_returns=[2.0, 2.0],
        arbitration_passed=True,
        competing_play_before_food_episodes=0,
    )


def test_m7_demo_records_an_rgb_viewer_trace_that_replays(tmp_path) -> None:
    trace = tmp_path / "m7-viewer.jsonl"
    result = run_m7_viewer_demo(trace, training_episodes=80)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert records[0]["record_type"] == "episode_metadata"
    assert records[0]["observation_mode"] == "rgb"
