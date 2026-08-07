import numpy as np

from ecosystem_gym.actions import ActionOutcome
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.m13 import M13Macro, M13Memory, compile_macro
from ecosystem_gym.m131 import m131_config
from ecosystem_gym.m133 import (
    M133_AUDIT_SEEDS,
    M133_DEVELOPMENT_CONDITIONS,
    M133_DEVELOPMENT_SEEDS,
    M133_VALIDATION_SEEDS,
    CompactM133QPolicy,
    SeededRandomM133Policy,
    advance_m133_memory,
    load_m133_policy,
    m133_config,
    m133_epsilon,
    m133_policy_fingerprint,
    m133_protocol_fingerprint,
    replay_m133_random_trace,
    replay_m133_trace,
    run_m133_episode,
    write_m133_policy,
)


def _reset(env: EcosystemEnv):
    return env.reset(
        seed=M133_DEVELOPMENT_SEEDS[0],
        options={"task_id": "persistent_maintenance", **M133_DEVELOPMENT_CONDITIONS["event_relocation"]},
    )[0]


def test_m133_reward_is_narrow_and_forced_relocation_is_exempt() -> None:
    assert EcosystemConfig().blocked_action_penalty == 0.0
    assert EcosystemConfig().exempt_forced_relocation_blocked_penalty is False
    baseline, revised = EcosystemEnv(m131_config()), EcosystemEnv(m133_config())
    try:
        old, new = _reset(baseline), _reset(revised)
        for macro in (M13Macro.GO_FOOD, M13Macro.PICK_UP):
            old_action, new_action = compile_macro(macro, old, baseline.config), compile_macro(macro, new, revised.config)
            old, old_reward, _, _, old_info = baseline.step(old_action)
            new, new_reward, _, _, new_info = revised.step(new_action)
        assert old_info["disturbance"] == new_info["disturbance"] == "food_relocated"
        assert old_info["outcome"] == new_info["outcome"] == ActionOutcome.BLOCKED.value
        assert new_reward == old_reward
        old_action, new_action = compile_macro(M13Macro.PICK_UP, old, baseline.config), compile_macro(M13Macro.PICK_UP, new, revised.config)
        _, old_reward, _, _, old_info = baseline.step(old_action)
        _, new_reward, _, _, new_info = revised.step(new_action)
        assert old_info["disturbance"] is new_info["disturbance"] is None
        assert old_info["outcome"] == new_info["outcome"] == ActionOutcome.BLOCKED.value
        assert np.isclose(new_reward - old_reward, -0.10)
    finally:
        baseline.close()
        revised.close()


def test_m133_memory_uses_exact_duration_and_epsilon_schedule() -> None:
    env = EcosystemEnv(m133_config())
    try:
        observation, _ = env.reset(seed=M133_DEVELOPMENT_SEEDS[0], options={"task_id": "persistent_maintenance", **M133_DEVELOPMENT_CONDITIONS["persistent_reference"]})
    finally:
        env.close()
    memory = M13Memory(food_cooldown_bucket=0)
    after = {**observation, "prior_outcome": list(ActionOutcome).index(ActionOutcome.SUCCESS)}
    advance_m133_memory(memory, observation_before=observation, macro=M13Macro.WAIT, action={"duration": np.asarray(0.1, dtype=np.float32)}, observation_after=after, config=m133_config())
    assert np.isclose(memory.food_cooldown_bucket, 0.1)
    assert m133_epsilon(0) == 1.0
    assert m133_epsilon(9_000) == 0.05
    assert m133_epsilon(8_999) > 0.05


def test_m133_artifacts_and_strict_policy_and_random_replay(tmp_path) -> None:
    policy = CompactM133QPolicy()
    artifact = write_m133_policy(tmp_path / "policy.json", policy)
    restored = load_m133_policy(artifact["path"])
    assert m133_policy_fingerprint(policy) == m133_policy_fingerprint(restored)
    trace = tmp_path / "learned.jsonl"
    run_m133_episode(policy, seed=M133_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M133_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=trace, policy_fingerprint=m133_policy_fingerprint(policy))
    assert replay_m133_trace(trace, restored).steps > 0
    random_trace = tmp_path / "random.jsonl"
    random = SeededRandomM133Policy()
    run_m133_episode(random, seed=M133_DEVELOPMENT_SEEDS[0], condition="persistent_reference", controls=M133_DEVELOPMENT_CONDITIONS["persistent_reference"], trace_path=random_trace, policy_fingerprint="pcg64")
    assert replay_m133_random_trace(random_trace, SeededRandomM133Policy()).steps > 0


def test_m133_splits_are_fresh_and_fingerprinted() -> None:
    assert set(M133_DEVELOPMENT_SEEDS).isdisjoint(M133_VALIDATION_SEEDS)
    assert set(M133_DEVELOPMENT_SEEDS).isdisjoint(M133_AUDIT_SEEDS)
    assert set(M133_VALIDATION_SEEDS).isdisjoint(M133_AUDIT_SEEDS)
    assert m133_protocol_fingerprint()
