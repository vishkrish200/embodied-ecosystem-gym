import numpy as np

from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m13 import M13Macro, compile_macro
from ecosystem_gym.m134 import (
    M134_AUDIT_SEEDS,
    M134_DEVELOPMENT_CONDITIONS,
    M134_DEVELOPMENT_SEEDS,
    M134_VALIDATION_SEEDS,
    CompactM134QPolicy,
    SeededRandomM134Policy,
    load_m134_policy,
    m134_mask,
    m134_policy_fingerprint,
    m134_protocol_fingerprint,
    replay_m134_random_trace,
    replay_m134_trace,
    run_m134_episode,
    write_m134_policy,
)
from ecosystem_gym.m133 import m133_config


def _observation(condition: str = "persistent_reference"):
    env = EcosystemEnv(m133_config())
    try:
        return env.reset(seed=M134_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M134_DEVELOPMENT_CONDITIONS[condition]})[0], env.config
    finally:
        env.close()


def test_m134_complementary_mask_is_public_and_removes_arrival_navigation() -> None:
    observation, config = _observation()
    mask = m134_mask(observation, config)
    assert mask.dtype == np.bool_
    assert mask[M13Macro.WAIT]
    # Move public targets to the agent: interactions are enabled, GO macros are not.
    at_targets = {**observation, "food_xy": observation["agent_xy"].copy(), "toy_xy": observation["agent_xy"].copy(), "rest_xy": observation["agent_xy"].copy()}
    at_mask = m134_mask(at_targets, config)
    assert not at_mask[M13Macro.GO_FOOD] and at_mask[M13Macro.PICK_UP]
    assert not at_mask[M13Macro.GO_TOY] and at_mask[M13Macro.PLAY]
    assert not at_mask[M13Macro.GO_REST] and at_mask[M13Macro.REST]
    # A public drive change cannot alter the macro-precondition mask.
    assert np.array_equal(at_mask, m134_mask({**at_targets, "drives": np.asarray([0.0, 1.0, 1.0], dtype=np.float32)}, config))


def test_m134_forced_stale_pickup_remains_eligible() -> None:
    env = EcosystemEnv(m133_config())
    try:
        observation, _ = env.reset(seed=M134_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M134_DEVELOPMENT_CONDITIONS["event_relocation"]})
        action = compile_macro(M13Macro.GO_FOOD, observation, env.config)
        observation, _, _, _, _ = env.step(action)
        assert m134_mask(observation, env.config)[M13Macro.PICK_UP]
        action = compile_macro(M13Macro.PICK_UP, observation, env.config)
        observation, _, _, _, info = env.step(action)
        assert info["disturbance"] == "food_relocated"
        assert m134_mask(observation, env.config)[M13Macro.GO_FOOD]
    finally:
        env.close()


def test_m134_artifact_and_strict_replay(tmp_path) -> None:
    policy = CompactM134QPolicy()
    artifact = write_m134_policy(tmp_path / "policy.json", policy)
    restored = load_m134_policy(artifact["path"])
    assert m134_policy_fingerprint(policy) == m134_policy_fingerprint(restored)
    trace = tmp_path / "trace.jsonl"
    run_m134_episode(policy, seed=M134_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M134_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=trace, policy_fingerprint=m134_policy_fingerprint(policy))
    assert replay_m134_trace(trace, restored).steps > 0
    random_trace = tmp_path / "random.jsonl"
    random = SeededRandomM134Policy()
    run_m134_episode(random, seed=M134_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M134_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=random_trace, policy_fingerprint="pcg64")
    assert replay_m134_random_trace(random_trace, SeededRandomM134Policy()).steps > 0


def test_m134_masks_and_splits_are_frozen() -> None:
    assert set(M134_DEVELOPMENT_SEEDS).isdisjoint(M134_VALIDATION_SEEDS)
    assert set(M134_DEVELOPMENT_SEEDS).isdisjoint(M134_AUDIT_SEEDS)
    assert set(M134_VALIDATION_SEEDS).isdisjoint(M134_AUDIT_SEEDS)
    assert np.array_equal(m134_mask(_observation()[0], _observation()[1], "none"), np.ones(len(M13Macro), dtype=np.bool_))
    assert SeededRandomM134Policy().mask_mode == "complementary"
    assert m134_protocol_fingerprint()
