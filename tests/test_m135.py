import numpy as np

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m13 import M13Macro
from ecosystem_gym.m135 import (
    M135_AUDIT_SEEDS,
    M135_CANDIDATE_UPDATE_EVERY,
    M135_DEVELOPMENT_CONDITIONS,
    M135_DEVELOPMENT_SEEDS,
    M135_NULL_UPDATE_EVERY,
    M135_VALIDATION_SEEDS,
    CompactM135QPolicy,
    SeededRandomM135Policy,
    load_m135_policy,
    m135_policy_fingerprint,
    m135_protocol_fingerprint,
    m135_update_every,
    replay_m135_random_trace,
    replay_m135_trace,
    run_m135_episode,
    write_m135_policy,
)
from ecosystem_gym.m133 import m133_config


def test_m135_update_cadence_is_the_only_policy_hyperparameter_change() -> None:
    candidate, control = CompactM135QPolicy(arm="update_ratio", seed=20_260_812), CompactM135QPolicy(arm="cadence_null", seed=20_260_812)
    assert candidate.update_every == M135_CANDIDATE_UPDATE_EVERY == 4
    assert control.update_every == M135_NULL_UPDATE_EVERY == 256
    assert candidate.mask_mode == control.mask_mode == "complementary"
    assert candidate.include_drives and candidate.use_memory
    assert candidate.config == control.config
    assert m135_update_every("update_ratio") == 4
    assert m135_update_every("cadence_null") == 256


def test_m135_artifact_and_learned_random_replay(tmp_path) -> None:
    policy = CompactM135QPolicy(arm="update_ratio", seed=20_260_812)
    artifact = write_m135_policy(tmp_path / "policy.json", policy)
    restored = load_m135_policy(artifact["path"])
    assert m135_policy_fingerprint(policy) == m135_policy_fingerprint(restored)
    trace = tmp_path / "learned.jsonl"
    run_m135_episode(policy, seed=M135_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M135_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=trace, policy_fingerprint=m135_policy_fingerprint(policy))
    assert replay_m135_trace(trace, restored).steps > 0
    random = SeededRandomM135Policy(20_260_812)
    random_trace = tmp_path / "random.jsonl"
    run_m135_episode(random, seed=M135_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M135_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=random_trace, policy_fingerprint="pcg64")
    assert replay_m135_random_trace(random_trace, SeededRandomM135Policy(20_260_812)).steps > 0


def test_m135_fresh_splits_and_public_mask() -> None:
    assert set(M135_DEVELOPMENT_SEEDS).isdisjoint(M135_VALIDATION_SEEDS)
    assert set(M135_DEVELOPMENT_SEEDS).isdisjoint(M135_AUDIT_SEEDS)
    assert set(M135_VALIDATION_SEEDS).isdisjoint(M135_AUDIT_SEEDS)
    env = EcosystemEnv(m133_config())
    try:
        observation, _ = env.reset(seed=M135_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M135_DEVELOPMENT_CONDITIONS["persistent_reference"]})
    finally:
        env.close()
    policy = CompactM135QPolicy(arm="update_ratio", seed=20_260_812)
    mask = policy.mask(observation)
    assert mask.dtype == np.bool_ and mask[M13Macro.WAIT]
    assert m135_protocol_fingerprint()
