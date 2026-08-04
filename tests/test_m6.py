from __future__ import annotations

import json

import numpy as np

from ecosystem_gym.actions import ActionOutcome
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.m6 import (
    DriveMacroAction,
    TabularDriveQPolicy,
    _gate_passed,
    arbitration_probe,
    m6_benchmark,
    observable_drive_state,
    run_m6_viewer_demo,
)


def _observation(*, satiety: float, boredom: float) -> dict[str, object]:
    return {
        "agent_xy": np.zeros(2, dtype=np.float32),
        "food_xy": np.asarray((0.5, 0.0), dtype=np.float32),
        "toy_xy": np.asarray((-0.5, 0.0), dtype=np.float32),
        "drives": np.asarray((satiety, 1.0, boredom), dtype=np.float32),
        "holding_food": 0,
        "prior_outcome": list(ActionOutcome).index(ActionOutcome.SUCCESS),
    }


def test_observable_drive_state_ignores_task_and_reset_metadata() -> None:
    observation = _observation(satiety=0.2, boredom=0.85)
    state = observable_drive_state(observation, EcosystemConfig(), played_toy=False)
    with_metadata = {**observation, "task_id": "competing_drives", "dynamics_variant": "slippery", "secret": object()}
    assert observable_drive_state(with_metadata, EcosystemConfig(), played_toy=False) == state


def test_learned_policy_passes_the_counterfactual_food_vs_play_probe() -> None:
    policy = TabularDriveQPolicy()
    policy.train(episodes=400)
    probe = arbitration_probe(policy)
    assert probe["passed"]
    assert probe["food_first_action"] == DriveMacroAction.PURSUE_FOOD.name
    assert probe["play_first_action"] == DriveMacroAction.PURSUE_PLAY.name


def test_m6_report_uses_the_shared_heldout_matrix_and_passes_its_gate() -> None:
    report = m6_benchmark(training_episodes=400, seeds=(0, 1))
    assert report["gate"]["passed"]
    for task in ("play_when_bored", "competing_drives"):
        heldout = report["results"][task]["learned_state_oracle_q"]["heldout"]
        assert len(heldout["by_condition"]) == 4
        assert heldout["task_success_rate"] == 1.0
    assert report["results"]["competing_drives"]["learned_state_oracle_q"]["heldout"]["play_before_food_episodes"] == 0


def test_m6_gate_rejects_higher_return_when_macro_random_matches_success() -> None:
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


def test_m6_demo_records_a_viewer_trace_that_replays(tmp_path) -> None:
    trace = tmp_path / "m6-viewer.jsonl"
    result = run_m6_viewer_demo(trace, training_episodes=400)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert result.task_success
    assert records[0]["record_type"] == "episode_metadata"
    assert records[0]["reset_options"]["task_id"] == "competing_drives"
