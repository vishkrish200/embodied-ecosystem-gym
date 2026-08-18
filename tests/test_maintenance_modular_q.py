from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecosystem_gym.actions import ActionOutcome
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.maintenance import (
    DurationReplayBuffer,
    MaintenanceMacro,
    ModularQArm,
    ModularQPolicy,
    ShieldConfig,
    ShieldedModularQPolicy,
    TeacherDataset,
    duration_aware_double_dqn_targets,
    load_modular_q_policy,
    load_teacher_dataset,
    modular_q_artifact_payload,
    modular_q_evaluation_spec,
    modular_q_policy_fingerprint,
    modular_q_weight_fingerprint,
    save_modular_q_policy,
    save_teacher_dataset,
    teacher_dataset_fingerprint,
    teacher_example,
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


def _observation(
    *,
    drives: tuple[float, float, float] = (0.45, 0.65, 0.65),
    agent: tuple[float, float] = (0.0, 0.0),
    food: tuple[float, float] = (0.6, 0.0),
    toy: tuple[float, float] = (0.0, 0.6),
    rest: tuple[float, float] = (-0.6, 0.0),
    holding_food: bool = False,
    outcome: ActionOutcome = ActionOutcome.SUCCESS,
) -> dict[str, object]:
    return {
        "agent_xy": np.asarray(agent, dtype=np.float32),
        "food_xy": np.asarray(food, dtype=np.float32),
        "toy_xy": np.asarray(toy, dtype=np.float32),
        "rest_xy": np.asarray(rest, dtype=np.float32),
        "drives": np.asarray(drives, dtype=np.float32),
        "holding_food": int(holding_food),
        "prior_outcome": list(ActionOutcome).index(outcome),
    }


def _teacher_dataset() -> TeacherDataset:
    labels = np.arange(len(MaintenanceMacro), dtype=np.int64)
    features = np.zeros((len(labels), 30), dtype=np.float32)
    features[np.arange(len(labels)), np.arange(len(labels))] = 1.0
    masks = np.ones((len(labels), len(MaintenanceMacro)), dtype=np.bool_)
    return TeacherDataset(features=features, masks=masks, labels=labels)


def _zero_network(policy: ModularQPolicy) -> None:
    for name in policy.online.parameter_items():
        getattr(policy.online, name).fill(0.0)
        getattr(policy.target, name).fill(0.0)


def _td_batch(rows: int = 8):
    features = np.zeros((rows, 30), dtype=np.float32)
    next_features = np.zeros((rows, 30), dtype=np.float32)
    actions = np.arange(rows, dtype=np.int64) % len(MaintenanceMacro)
    rewards = np.full(rows, 1.0, dtype=np.float32)
    next_masks = np.ones((rows, len(MaintenanceMacro)), dtype=np.bool_)
    done = np.zeros(rows, dtype=np.bool_)
    durations = np.ones(rows, dtype=np.float32)
    return features, actions, rewards, next_features, next_masks, done, durations


def test_modular_and_monolithic_q_surfaces_match_public_contract_and_budget() -> None:
    config = _config()
    observation = _observation()
    modular = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=7,
        config=config,
        protocol_fingerprint="test",
    )
    monolithic = ModularQPolicy(
        arm=ModularQArm.MONOLITHIC_ANCHOR_SHIELD_CONTROL,
        seed=7,
        config=config,
        protocol_fingerprint="test",
    )
    memory = modular.reset()
    features = modular.features(observation, memory)
    assert features.shape == (30,)
    component_values = modular.component_q_values_from_features(features)[0]
    q_values = modular.q_values(observation, memory)
    assert component_values.shape == (3, 8)
    assert q_values.shape == (8,)
    assert np.allclose(np.sum(component_values, axis=0), q_values)
    assert modular.mask(observation).shape == monolithic.mask(observation).shape == (8,)
    assert modular.online.parameter_count() == monolithic.online.parameter_count()

    private = dict(observation)
    private["task_id"] = "forbidden"
    with pytest.raises(ValueError, match="private keys"):
        modular.features(private, memory)
    with pytest.raises(ValueError, match="private keys"):
        teacher_example(private, memory, label=MaintenanceMacro.WAIT, config=config)


def test_duration_aware_targets_and_anchor_losses_remain_separate() -> None:
    rewards = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
    durations = np.asarray((0.1, 1.0, 2.0), dtype=np.float32)
    done = np.asarray((False, False, True))
    next_online = np.zeros((3, 8), dtype=np.float32)
    next_target = np.full((3, 8), 2.0, dtype=np.float32)
    next_masks = np.ones((3, 8), dtype=np.bool_)
    targets = duration_aware_double_dqn_targets(rewards, durations, done, next_online, next_target, next_masks)
    assert targets[0] == pytest.approx(1.0 + 0.99**0.1 * 2.0)
    assert targets[1] == pytest.approx(1.0 + 0.99 * 2.0)
    assert targets[2] == pytest.approx(1.0)

    config = _config()
    dataset = _teacher_dataset()
    candidate = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=11,
        config=config,
        protocol_fingerprint="loss-test",
    )
    no_anchor = ModularQPolicy(
        arm=ModularQArm.MODULAR_NO_ANCHOR_SHIELD_CONTROL,
        seed=11,
        config=config,
        protocol_fingerprint="loss-test",
    )
    for policy in (candidate, no_anchor):
        _zero_network(policy)
        policy.target.head_b.fill(1.0)
        policy.attach_teacher_dataset(dataset)
        features, actions, rewards, next_features, next_masks, done, durations = _td_batch()
        for index in range(features.shape[0]):
            policy.replay.add(
                features=features[index],
                action=int(actions[index]),
                reward=float(rewards[index]),
                next_features=next_features[index],
                next_mask=next_masks[index],
                done=bool(done[index]),
                duration=float(durations[index]),
            )

    candidate_metrics = candidate.sample_and_update()
    no_anchor_metrics = no_anchor.sample_and_update()
    assert candidate_metrics["td_loss"] > 0.0
    assert candidate_metrics["teacher_margin_loss"] > 0.0
    assert candidate_metrics["total_loss"] > candidate_metrics["td_loss"]
    assert no_anchor_metrics["teacher_margin_loss"] > 0.0
    assert no_anchor_metrics["total_loss"] == pytest.approx(no_anchor_metrics["td_loss"])


def test_teacher_dataset_round_trips_without_mutation_and_stays_immutable(tmp_path: Path) -> None:
    dataset = _teacher_dataset()
    path = tmp_path / "teacher.npz"
    metadata = save_teacher_dataset(path, dataset)
    restored = load_teacher_dataset(path)
    assert metadata["fingerprint"] == dataset.fingerprint() == restored.fingerprint()
    assert metadata["public_fields_only"] is True
    assert restored.class_counts == {macro.name: 1 for macro in MaintenanceMacro}
    assert not restored.features.flags.writeable
    assert not restored.masks.flags.writeable
    assert not restored.labels.flags.writeable
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"features", "masks", "labels"}

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        features=dataset.features[:-1],
        masks=dataset.masks[:-1],
        labels=dataset.labels[:-1],
    )
    with pytest.raises(ValueError, match="cover every macro"):
        load_teacher_dataset(invalid)


def test_modular_q_artifact_round_trip_binds_fingerprint_and_rejects_m1313_paths(tmp_path: Path) -> None:
    config = _config()
    policy = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=19,
        config=config,
        protocol_fingerprint="artifact-test",
    )
    artifact = save_modular_q_policy(tmp_path / "policy.json", policy)
    restored = load_modular_q_policy(
        artifact["path"],
        config=config,
        expected_protocol_fingerprint="artifact-test",
    )
    assert artifact["policy_fingerprint"] == modular_q_policy_fingerprint(policy) == modular_q_policy_fingerprint(restored)
    assert artifact["weight_fingerprint"] == modular_q_weight_fingerprint(policy) == modular_q_weight_fingerprint(restored)
    assert artifact["parameter_bytes_sha256"] == hashlib.sha256(policy.parameter_bytes()).hexdigest()
    assert artifact["weight_bytes_sha256"] == hashlib.sha256(policy.weight_bytes()).hexdigest()
    payload = modular_q_artifact_payload(restored)
    assert payload["parameter_bytes_sha256"] == artifact["parameter_bytes_sha256"]
    assert payload["weight_bytes_sha256"] == artifact["weight_bytes_sha256"]

    tampered = tmp_path / "tampered.json"
    body = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    body["update_count"] = 999
    tampered.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_modular_q_policy(tampered, config=config)
    with pytest.raises(ValueError, match="protocol"):
        load_modular_q_policy(artifact["path"], config=config, expected_protocol_fingerprint="wrong")
    with pytest.raises(ValueError, match="M13.14 artifact paths must reject frozen M13.13 paths"):
        save_modular_q_policy(tmp_path / "m1313" / "policy.json", policy)


def test_shielded_wrapper_reuses_supervisor_semantics_and_specs_keep_identical_q_bytes() -> None:
    config = _config()
    learner = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=23,
        config=config,
        protocol_fingerprint="shield-test",
    )
    _zero_network(learner)
    learner.online.head_b[:, MaintenanceMacro.WAIT].fill(100.0)
    shielded = ShieldedModularQPolicy(learner=learner)
    unshielded = ShieldedModularQPolicy(learner=learner, shield=ShieldConfig(enabled=False))
    observation = _observation(drives=(0.20, 0.80, 0.10))
    assert unshielded.choose(observation, unshielded.reset()) is MaintenanceMacro.WAIT
    decision = shielded.decide(observation, shielded.reset())
    assert decision.macro is MaintenanceMacro.GO_FOOD
    assert decision.intervened is True
    assert decision.reason == "objective-commitment"

    enabled_spec = modular_q_evaluation_spec(learner, shield=ShieldConfig(enabled=True))
    disabled_spec = modular_q_evaluation_spec(learner, shield=ShieldConfig(enabled=False))
    assert enabled_spec["parameter_bytes_sha256"] == disabled_spec["parameter_bytes_sha256"]
    assert enabled_spec["policy_fingerprint"] == disabled_spec["policy_fingerprint"]
    common_enabled = dict(enabled_spec)
    common_disabled = dict(disabled_spec)
    common_enabled.pop("shield")
    common_disabled.pop("shield")
    assert common_enabled == common_disabled
    assert enabled_spec["shield"]["enabled"] is True
    assert disabled_spec["shield"]["enabled"] is False


def test_no_shield_control_can_reuse_candidate_q_weights_while_arm_identity_differs() -> None:
    config = _config()
    candidate = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=31,
        config=config,
        protocol_fingerprint="reuse-test",
    )
    no_shield = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_NO_SHIELD_CONTROL,
        seed=31,
        config=config,
        protocol_fingerprint="reuse-test",
    )
    no_shield.online = candidate.online.copy()
    no_shield.target = candidate.target.copy()
    no_shield.update_count = candidate.update_count
    no_shield.initial_parameter_fingerprint = candidate.initial_parameter_fingerprint
    no_shield.initial_weight_fingerprint = candidate.initial_weight_fingerprint

    assert candidate.arm is not no_shield.arm
    assert modular_q_policy_fingerprint(candidate) != modular_q_policy_fingerprint(no_shield)
    assert modular_q_weight_fingerprint(candidate) == modular_q_weight_fingerprint(no_shield)
    assert hashlib.sha256(candidate.weight_bytes()).hexdigest() == hashlib.sha256(no_shield.weight_bytes()).hexdigest()

    candidate_spec = modular_q_evaluation_spec(candidate, shield=ShieldConfig(enabled=True))
    no_shield_spec = modular_q_evaluation_spec(no_shield, shield=ShieldConfig(enabled=False))
    assert candidate_spec["arm"] != no_shield_spec["arm"]
    assert candidate_spec["policy_fingerprint"] != no_shield_spec["policy_fingerprint"]
    assert candidate_spec["weight_fingerprint"] == no_shield_spec["weight_fingerprint"]
    assert candidate_spec["weight_bytes_sha256"] == no_shield_spec["weight_bytes_sha256"]
    assert candidate_spec["parameter_bytes_sha256"] != no_shield_spec["parameter_bytes_sha256"]


def test_teacher_replay_remains_immutable_after_sampling_and_update() -> None:
    config = _config()
    dataset = _teacher_dataset()
    learner = ModularQPolicy(
        arm=ModularQArm.MODULAR_ANCHOR_SHIELD_CANDIDATE,
        seed=29,
        config=config,
        protocol_fingerprint="immutability-test",
    )
    _zero_network(learner)
    learner.target.head_b.fill(1.0)
    learner.attach_teacher_dataset(dataset)
    before_features = dataset.features.copy()
    before_masks = dataset.masks.copy()
    before_labels = dataset.labels.copy()
    before_fingerprint = teacher_dataset_fingerprint(dataset.features, dataset.masks, dataset.labels)

    replay = DurationReplayBuffer(4)
    features, actions, rewards, next_features, next_masks, done, durations = _td_batch(rows=4)
    for index in range(4):
        replay.add(
            features=features[index],
            action=int(actions[index]),
            reward=float(rewards[index]),
            next_features=next_features[index],
            next_mask=next_masks[index],
            done=bool(done[index]),
            duration=float(durations[index]),
        )
    learner.replay = replay
    learner.sample_and_update()

    assert np.array_equal(before_features, dataset.features)
    assert np.array_equal(before_masks, dataset.masks)
    assert np.array_equal(before_labels, dataset.labels)
    assert teacher_dataset_fingerprint(dataset.features, dataset.masks, dataset.labels) == before_fingerprint
