from __future__ import annotations

import numpy as np
import pytest

from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.maintenance import (
    MODULAR_Q_ARTIFACT_SCHEMA_M1314R2,
    TEACHER_DATASET_SCHEMA_M1314R2,
    MaintenanceMacro,
    ModularQArm,
    ModularQArmM1314R2,
    ModularQPolicy,
    ModularQPolicyM1314R2,
    TeacherDataset,
    TeacherDatasetM1314R2,
    TeacherMinibatchM1314R2,
    TeacherReplayBufferM1314R2,
    load_modular_q_policy_m1314r2,
    load_teacher_dataset_m1314r2,
    modular_q_artifact_payload_m1314r2,
    save_modular_q_policy,
    save_modular_q_policy_m1314r2,
    save_teacher_dataset,
    save_teacher_dataset_m1314r2,
)


def _config() -> EcosystemConfig:
    return EcosystemConfig(
        observation_mode="state_oracle",
        max_episode_steps=200,
        satiety_decay_per_second=0.018,
        energy_decay_per_second=0.012,
        boredom_gain_per_second=0.020,
        play_success_boredom_threshold=0.60,
        play_boredom_reduction=0.55,
        eat_satiety_gain=0.55,
        food_respawn_seconds=5.0,
        rest_cycle_energy_threshold=0.45,
        rest_energy_gain=0.65,
    )


def _full_teacher_dataset(rows_per_macro: int = 4) -> TeacherDatasetM1314R2:
    labels = np.repeat(np.arange(len(MaintenanceMacro), dtype=np.int64), rows_per_macro)
    features = np.zeros((len(labels), 30), dtype=np.float32)
    features[np.arange(len(labels)), np.arange(len(labels)) % 30] = 1.0
    masks = np.ones((len(labels), len(MaintenanceMacro)), dtype=np.bool_)
    return TeacherDatasetM1314R2(features=features, masks=masks, labels=labels)


def _partial_teacher_rows(rows: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = [
        MaintenanceMacro.GO_FOOD,
        MaintenanceMacro.PICK_UP,
        MaintenanceMacro.GO_TOY,
        MaintenanceMacro.PLAY,
        MaintenanceMacro.REST,
        MaintenanceMacro.WAIT,
    ]
    labels = np.asarray([int(allowed[index % len(allowed)]) for index in range(rows)], dtype=np.int64)
    features = np.zeros((rows, 30), dtype=np.float32)
    features[np.arange(rows), (np.arange(rows) * 3) % 30] = 1.0
    masks = np.ones((rows, len(MaintenanceMacro)), dtype=np.bool_)
    return features, masks, labels


def _fill_replay(policy: ModularQPolicyM1314R2, rows: int = 32) -> None:
    features = np.zeros((rows, 30), dtype=np.float32)
    next_features = np.zeros((rows, 30), dtype=np.float32)
    next_mask = np.ones(len(MaintenanceMacro), dtype=np.bool_)
    for index in range(rows):
        policy.replay.add(
            features=features[index],
            action=int(index % len(MaintenanceMacro)),
            reward=1.0,
            next_features=next_features[index],
            next_mask=next_mask,
            done=False,
            duration=1.0,
        )


def test_teacher_minibatch_allows_partial_macro_coverage_but_dataset_requires_all_macros() -> None:
    features, masks, labels = _partial_teacher_rows()
    minibatch = TeacherMinibatchM1314R2(features=features, masks=masks, labels=labels)
    assert minibatch.rows == 32
    assert set(minibatch.labels.tolist()) == {
        int(MaintenanceMacro.GO_FOOD),
        int(MaintenanceMacro.PICK_UP),
        int(MaintenanceMacro.GO_TOY),
        int(MaintenanceMacro.PLAY),
        int(MaintenanceMacro.REST),
        int(MaintenanceMacro.WAIT),
    }
    with pytest.raises(ValueError, match=r"missing .*CONSUME.*GO_REST"):
        TeacherDatasetM1314R2(features=features, masks=masks, labels=labels)


def test_teacher_replay_returns_minibatches_and_policy_paths_accept_them() -> None:
    dataset = _full_teacher_dataset()
    replay = TeacherReplayBufferM1314R2(dataset)
    sampled = replay.sample(np.random.default_rng(0), batch_size=32)
    assert isinstance(sampled, TeacherMinibatchM1314R2)

    partial = TeacherMinibatchM1314R2(*_partial_teacher_rows())
    policy = ModularQPolicyM1314R2(
        arm=ModularQArmM1314R2.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=7,
        config=_config(),
        protocol_fingerprint="r2-test",
    )
    _fill_replay(policy)
    metrics = policy.update(policy.replay.sample(np.random.default_rng(1), batch_size=32), partial)
    assert metrics["teacher_margin_loss"] >= 0.0

    warmstart = policy.warmstart_from_teacher(dataset, steps=1, rng=np.random.default_rng(2))
    assert warmstart["steps"] == pytest.approx(1.0)

    policy.attach_teacher_dataset(dataset)
    sampled_metrics = policy.sample_and_update(rng=np.random.default_rng(3))
    assert sampled_metrics["total_loss"] >= sampled_metrics["td_loss"]


def test_m1314r2_teacher_datasets_are_schema_isolated_from_m1314(tmp_path) -> None:
    dataset = _full_teacher_dataset()
    artifact = save_teacher_dataset_m1314r2(tmp_path / "teacher_r2.npz", dataset)
    assert artifact["schema_version"] == TEACHER_DATASET_SCHEMA_M1314R2
    restored = load_teacher_dataset_m1314r2(artifact["path"])
    assert restored.fingerprint() == dataset.fingerprint()

    legacy_dataset = TeacherDataset(
        features=dataset.features.copy(),
        masks=dataset.masks.copy(),
        labels=dataset.labels.copy(),
    )
    legacy = save_teacher_dataset(tmp_path / "teacher_legacy.npz", legacy_dataset)
    with pytest.raises(ValueError, match="invalid field set"):
        load_teacher_dataset_m1314r2(legacy["path"])
    with pytest.raises(ValueError, match="legacy M13.14/M13.13"):
        save_teacher_dataset_m1314r2(tmp_path / "m1314" / "teacher.npz", dataset)


def test_m1314r2_artifacts_use_new_schema_and_reject_legacy_m1314_artifacts(tmp_path) -> None:
    config = _config()
    policy = ModularQPolicyM1314R2(
        arm=ModularQArmM1314R2.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=19,
        config=config,
        protocol_fingerprint="r2-artifact-test",
    )
    artifact = save_modular_q_policy_m1314r2(tmp_path / "policy_r2.json", policy)
    payload = modular_q_artifact_payload_m1314r2(policy)
    assert payload["schema_version"] == MODULAR_Q_ARTIFACT_SCHEMA_M1314R2
    assert "ecosystem_gym/maintenance/modular_q_m1314r2.py" in payload["source_hashes"]
    assert "ecosystem_gym/maintenance/teacher_data_m1314r2.py" in payload["source_hashes"]

    restored = load_modular_q_policy_m1314r2(
        artifact["path"],
        config=config,
        expected_protocol_fingerprint="r2-artifact-test",
        require_current_source_hashes=True,
    )
    assert restored.parameter_fingerprint() == policy.parameter_fingerprint()

    legacy_policy = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=19,
        config=config,
        protocol_fingerprint="legacy-artifact-test",
    )
    legacy = save_modular_q_policy(tmp_path / "policy_legacy.json", legacy_policy)
    with pytest.raises(ValueError, match="unsupported M13.14-r2 modular-Q artifact schema"):
        load_modular_q_policy_m1314r2(legacy["path"], config=config)
    with pytest.raises(ValueError, match="legacy M13.14/M13.13"):
        save_modular_q_policy_m1314r2(tmp_path / "m1314" / "policy.json", policy)
