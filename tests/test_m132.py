import numpy as np

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m13 import M13Memory
from ecosystem_gym.experiments.m132 import (
    M132_AUDIT_SEEDS,
    M132_DEVELOPMENT_CONDITIONS,
    M132_DEVELOPMENT_SEEDS,
    M132_FEATURE_DIM,
    M132_VALIDATION_SEEDS,
    CompactM132QPolicy,
    encode_features,
    load_m132_policy,
    m132_config,
    m132_protocol_fingerprint,
    write_m132_policy,
)


def _observation():
    env = EcosystemEnv(m132_config())
    try:
        observation, _ = env.reset(seed=M132_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M132_DEVELOPMENT_CONDITIONS["persistent_reference"]})
        return observation
    finally:
        env.close()


def test_m132_feature_boundary_shape_and_ablations() -> None:
    observation = _observation()
    memory = M13Memory(feed_count=2, play_count=1, rest_count=1, food_cooldown_bucket=3, pending_recovery=True)
    full = encode_features(observation, memory)
    no_drive = encode_features(observation, memory, include_drives=False)
    no_memory = encode_features(observation, memory, use_memory=False)
    assert full.shape == (M132_FEATURE_DIM,)
    assert full.dtype == np.float32
    assert np.array_equal(no_drive[:6], full[:6])
    assert np.array_equal(no_drive[6:9], np.zeros(3, dtype=np.float32))
    assert not np.array_equal(no_memory[16:], full[16:])


def test_m132_network_is_deterministic_and_fresh_splits_are_disjoint() -> None:
    observation = _observation()
    first, second = CompactM132QPolicy(), CompactM132QPolicy()
    assert np.array_equal(first.online.predict(first.features(observation, first.reset())), second.online.predict(second.features(observation, second.reset())))
    assert set(M132_DEVELOPMENT_SEEDS).isdisjoint(M132_VALIDATION_SEEDS)
    assert set(M132_DEVELOPMENT_SEEDS).isdisjoint(M132_AUDIT_SEEDS)
    assert m132_protocol_fingerprint()


def test_m132_policy_artifact_preserves_inference(tmp_path) -> None:
    observation = _observation()
    policy = CompactM132QPolicy()
    artifact = write_m132_policy(tmp_path / "policy.json", policy)
    restored = load_m132_policy(artifact["path"])
    assert np.array_equal(policy.online.predict(policy.features(observation, policy.reset())), restored.online.predict(restored.features(observation, restored.reset())))
