from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m131 import (
    M131_AUDIT_SEEDS,
    M131_DEVELOPMENT_CONDITIONS,
    M131_DEVELOPMENT_SEEDS,
    M131_VALIDATION_SEEDS,
    TabularM131QPolicy,
    _run,
    m131_config,
    m131_policy_fingerprint,
    m131_protocol_fingerprint,
    replay_m131_trace,
)
from ecosystem_gym.m13 import ScriptedM13Oracle, m13_config, run_m13_episode


def test_m131_reward_revision_is_cycle_only_and_legacy_defaults_are_zero() -> None:
    assert EcosystemConfig().persistent_feed_cycle_reward == 0.0
    assert EcosystemConfig().persistent_play_cycle_reward == 0.0
    assert EcosystemConfig().persistent_rest_cycle_reward == 0.0
    config = m131_config()
    assert (config.persistent_feed_cycle_reward, config.persistent_play_cycle_reward, config.persistent_rest_cycle_reward) == (0.25, 0.25, 0.25)


def test_m131_has_disjoint_fresh_splits_and_replayable_policy_trace(tmp_path) -> None:
    assert set(M131_DEVELOPMENT_SEEDS).isdisjoint(M131_VALIDATION_SEEDS)
    assert set(M131_DEVELOPMENT_SEEDS).isdisjoint(M131_AUDIT_SEEDS)
    policy = TabularM131QPolicy()
    trace = tmp_path / "m131.jsonl"
    _run(policy, seed=M131_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M131_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=trace, policy_fingerprint=m131_policy_fingerprint(policy))
    assert m131_protocol_fingerprint()
    assert replay_m131_trace(trace, policy).steps == 160


def test_m131_environment_keeps_public_state_contract() -> None:
    env = EcosystemEnv(m131_config())
    try:
        observation, _ = env.reset(seed=M131_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M131_DEVELOPMENT_CONDITIONS["persistent_reference"]})
        assert env.observation_space.contains(observation)
        assert "rest_xy" in observation
    finally:
        env.close()


def test_m131_pays_only_for_authoritative_cycle_counter_increments() -> None:
    controls = M131_DEVELOPMENT_CONDITIONS["persistent_reference"]
    baseline_env, revised_env = EcosystemEnv(m13_config()), EcosystemEnv(m131_config())
    try:
        baseline = run_m13_episode(ScriptedM13Oracle(), seed=M131_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=controls, env=baseline_env)
        revised = run_m13_episode(ScriptedM13Oracle(), seed=M131_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=controls, env=revised_env)
    finally:
        baseline_env.close()
        revised_env.close()
    assert len(baseline.steps) == len(revised.steps)
    before = (0, 0, 0)
    for old, new in zip(baseline.steps, revised.steps, strict=True):
        after = (new["feed_cycles"], new["play_cycles"], new["rest_cycles"])
        increments = sum(int(now > was) for was, now in zip(before, after, strict=True))
        assert new["reward"] - old["reward"] == 0.25 * increments
        before = after
