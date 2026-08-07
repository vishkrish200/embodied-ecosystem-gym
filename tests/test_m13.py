import numpy as np
import pytest

from ecosystem_gym.actions import ActionKind, ActionOutcome
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments.m13 import (
    M13_AUDIT_SEEDS,
    M13_DEVELOPMENT_CONDITIONS,
    M13_DEVELOPMENT_SEEDS,
    M13_VALIDATION_CONDITIONS,
    M13_VALIDATION_SEEDS,
    M13Macro,
    M13Memory,
    ScriptedM13Oracle,
    TabularM13QPolicy,
    advance_memory,
    compile_macro,
    encode_state,
    load_m13_policy,
    m13_audit,
    m13_config,
    m13_policy_fingerprint,
    replay_m13_trace,
    run_m13_episode,
    write_m13_policy,
)
from ecosystem_gym.policies import skill_action


def _reset():
    env = EcosystemEnv(m13_config())
    observation, _ = env.reset(
        seed=M13_DEVELOPMENT_SEEDS[0],
        options={"task_id": "persistent_maintenance", **M13_DEVELOPMENT_CONDITIONS["persistent_reference"]},
    )
    return env, observation


def test_m13_state_oracle_adds_public_rest_target_and_fresh_splits() -> None:
    env, observation = _reset()
    try:
        assert "rest_xy" in observation
        assert env.observation_space.contains(observation)
    finally:
        env.close()
    assert set(M13_DEVELOPMENT_SEEDS).isdisjoint(M13_VALIDATION_SEEDS)
    assert set(M13_DEVELOPMENT_SEEDS).isdisjoint(M13_AUDIT_SEEDS)
    assert set(M13_VALIDATION_SEEDS).isdisjoint(M13_AUDIT_SEEDS)
    assert {row["layout_id"] for row in M13_DEVELOPMENT_CONDITIONS.values()}.isdisjoint(
        {row["layout_id"] for row in M13_VALIDATION_CONDITIONS.values()}
    )


def test_m13_encoder_and_macro_compiler_use_only_public_observation() -> None:
    env, observation = _reset()
    try:
        state = encode_state(observation, M13Memory())
        assert state
        action = compile_macro(M13Macro.GO_REST, observation, env.config)
        assert ActionKind(int(action["kind"])) is ActionKind.WALK_TO
        assert np.array_equal(np.asarray(action["target"]), observation["rest_xy"])
    finally:
        env.close()


def test_m13_memory_updates_from_own_action_and_prior_outcome_only() -> None:
    env, observation = _reset()
    try:
        memory = M13Memory()
        action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        after = {**observation, "prior_outcome": list(ActionOutcome).index(ActionOutcome.SUCCESS)}
        advance_memory(memory, observation_before=observation, macro=M13Macro.CONSUME, action=action, observation_after=after)
        assert memory.feed_count == 1
        assert memory.food_cooldown_bucket == 0
    finally:
        env.close()


def test_m13_public_state_scripted_ceiling_completes_one_fresh_development_episode() -> None:
    result = run_m13_episode(
        ScriptedM13Oracle(),
        seed=M13_DEVELOPMENT_SEEDS[0],
        condition="persistent_reference",
        controls=M13_DEVELOPMENT_CONDITIONS["persistent_reference"],
    )
    assert result.survived
    assert result.maintenance_complete
    assert result.feed_cycles >= 3 and result.play_cycles >= 3 and result.rest_cycles >= 2


def test_m13_policy_and_trace_replay_preserve_inference(tmp_path) -> None:
    policy = TabularM13QPolicy()
    artifact = write_m13_policy(tmp_path / "policy.json", policy)
    restored = load_m13_policy(artifact["path"])
    trace = tmp_path / "episode.jsonl"
    run_m13_episode(
        restored,
        seed=M13_DEVELOPMENT_SEEDS[0],
        condition="persistent_reference",
        controls=M13_DEVELOPMENT_CONDITIONS["persistent_reference"],
        trace_path=trace,
        policy_fingerprint=m13_policy_fingerprint(restored),
    )
    replay = replay_m13_trace(trace, restored)
    assert replay.steps == 160


def test_m13_audit_refuses_a_non_passing_validation_report(tmp_path) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text('{"gate":{"passes":false}}\n', encoding="utf-8")
    audit_dir = tmp_path / "audit-artifacts"
    with pytest.raises(ValueError, match="validation-passing"):
        m13_audit(validation_report_path=validation, artifact_dir=audit_dir)
    assert not audit_dir.exists()
