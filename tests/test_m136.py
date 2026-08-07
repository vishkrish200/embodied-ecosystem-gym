import numpy as np

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m13 import M13Macro
from ecosystem_gym.experiments.m136 import (
    M136_AUDIT_SEEDS,
    M136_CANDIDATE_UPDATE_EVERY,
    M136_DEVELOPMENT_CONDITIONS,
    M136_DEVELOPMENT_SEEDS,
    M136_NULL_UPDATE_EVERY,
    M136_TRAINING_SEEDS,
    M136_VALIDATION_SEEDS,
    CompactM136QPolicy,
    SeededRandomM136Policy,
    load_m136_policy,
    m136_policy_fingerprint,
    m136_protocol_fingerprint,
    replay_m136_random_trace,
    replay_m136_trace,
    run_m136_episode,
    write_m136_policy,
)
from ecosystem_gym.experiments.m133 import m133_config


def test_m136_freezes_fresh_splits_and_eight_paired_seeds() -> None:
    assert len(M136_TRAINING_SEEDS) == 8
    assert set(M136_DEVELOPMENT_SEEDS).isdisjoint(M136_VALIDATION_SEEDS)
    assert set(M136_DEVELOPMENT_SEEDS).isdisjoint(M136_AUDIT_SEEDS)
    assert set(M136_VALIDATION_SEEDS).isdisjoint(M136_AUDIT_SEEDS)
    candidate = CompactM136QPolicy(arm="update_ratio", seed=M136_TRAINING_SEEDS[0])
    null = CompactM136QPolicy(arm="cadence_null", seed=M136_TRAINING_SEEDS[0])
    assert candidate.update_every == M136_CANDIDATE_UPDATE_EVERY == 4
    assert null.update_every == M136_NULL_UPDATE_EVERY == 256
    assert candidate.config == null.config and candidate.mask_mode == null.mask_mode == "complementary"
    assert m136_protocol_fingerprint()


def test_m136_artifact_and_trace_replay(tmp_path) -> None:
    policy = CompactM136QPolicy(arm="update_ratio", seed=M136_TRAINING_SEEDS[0])
    artifact = write_m136_policy(tmp_path / "policy.json", policy)
    restored = load_m136_policy(artifact["path"])
    assert m136_policy_fingerprint(policy) == m136_policy_fingerprint(restored)
    trace = tmp_path / "learned.jsonl"
    run_m136_episode(policy, seed=M136_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M136_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=trace, policy_fingerprint=m136_policy_fingerprint(policy))
    assert replay_m136_trace(trace, restored).steps > 0
    random_trace = tmp_path / "random.jsonl"
    run_m136_episode(SeededRandomM136Policy(M136_TRAINING_SEEDS[0]), seed=M136_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M136_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=random_trace, policy_fingerprint="pcg64")
    assert replay_m136_random_trace(random_trace, SeededRandomM136Policy(M136_TRAINING_SEEDS[0])).steps > 0
    env = EcosystemEnv(m133_config())
    try:
        observation, _ = env.reset(seed=M136_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M136_DEVELOPMENT_CONDITIONS["persistent_reference"]})
    finally:
        env.close()
    assert CompactM136QPolicy(arm="update_ratio", seed=M136_TRAINING_SEEDS[0]).mask(observation).dtype == np.bool_
    assert CompactM136QPolicy(arm="update_ratio", seed=M136_TRAINING_SEEDS[0]).mask(observation)[M13Macro.WAIT]
