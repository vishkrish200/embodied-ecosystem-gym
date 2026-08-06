from __future__ import annotations

import numpy as np

from ecosystem_gym.actions import ActionKind
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m10 import (
    M10_AUDIT_CONDITIONS,
    M10_AUDIT_SEEDS,
    M10_TRAIN_CONDITIONS,
    M10_TRAIN_SEEDS,
    M10_VALIDATION_CONDITIONS,
    M10_VALIDATION_SEEDS,
    _options,
    _run_oracle,
    m10_config,
    m10_scan_coverage,
    m10_validation,
    run_m10_viewer_demo,
)
from ecosystem_gym.policies import skill_action


def test_m10_splits_and_layouts_are_frozen_and_disjoint() -> None:
    assert set(M10_TRAIN_SEEDS).isdisjoint(M10_VALIDATION_SEEDS)
    assert set(M10_TRAIN_SEEDS).isdisjoint(M10_AUDIT_SEEDS)
    assert set(M10_VALIDATION_SEEDS).isdisjoint(M10_AUDIT_SEEDS)
    train = {row["layout_id"] for row in M10_TRAIN_CONDITIONS.values()}
    validation = {row["layout_id"] for row in M10_VALIDATION_CONDITIONS.values()}
    audit = {row["layout_id"] for row in M10_AUDIT_CONDITIONS.values()}
    assert train.isdisjoint(validation | audit)
    assert validation.isdisjoint(audit)


def test_persistent_food_replenishes_without_terminating() -> None:
    env = EcosystemEnv(m10_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=1_800, options=_options(M10_VALIDATION_CONDITIONS["persistent_reference"]))
        env._set_agent_xy(np.asarray(observation["food_xy"], dtype=np.float32))
        observation, _, terminated, _, info = env.step(skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1))
        assert info["outcome"] == "success"
        observation, _, terminated, _, info = env.step(skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1))
        assert not terminated
        assert info["feed_cycles"] == 1
        assert not info["food_available"]
        _, _, _, _, info = env.step(skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), env.config.food_respawn_seconds))
        assert info["resource_event"] == "food_replenished"
        assert info["food_available"]
    finally:
        env.close()


def test_event_relocation_forces_the_stale_pickup() -> None:
    controls = M10_VALIDATION_CONDITIONS["event_relocation"]
    env = EcosystemEnv(m10_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=1_800, options=_options(controls))
        env._set_agent_xy(np.asarray(observation["food_xy"], dtype=np.float32))
        _, _, terminated, _, info = env.step(skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1))
        assert not terminated
        assert info["disturbance"] == "food_relocated"
        assert info["outcome"] == "blocked"
    finally:
        env.close()


def test_m10_oracle_completes_one_persistent_horizon() -> None:
    episode = _run_oracle(
        seed=1_800,
        condition="event_relocation",
        controls=M10_VALIDATION_CONDITIONS["event_relocation"],
    )
    config = m10_config()
    assert episode.success
    assert episode.feed_cycles >= config.persistent_min_feed_cycles
    assert episode.play_cycles >= config.persistent_min_play_cycles
    assert episode.rest_cycles >= config.persistent_min_rest_cycles
    assert episode.relocation_failures == 1
    assert episode.recovered_relocations == 1


def test_m10_public_coverage_and_replay_demo(tmp_path) -> None:
    coverage = m10_scan_coverage(
        {"event_relocation": M10_VALIDATION_CONDITIONS["event_relocation"]},
        seeds=(1_800,),
    )
    assert coverage["event_relocation"]["passes"]
    replay = run_m10_viewer_demo(tmp_path / "m10.jsonl", seed=1_807)
    assert replay.task_success


def test_m10_validation_rejects_custom_seeds() -> None:
    try:
        m10_validation(seeds=(1_800,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("M10 validation must reject custom seeds")
