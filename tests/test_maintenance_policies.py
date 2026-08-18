from __future__ import annotations

import json

import numpy as np
import pytest

from ecosystem_gym.actions import ActionOutcome
from ecosystem_gym.config import EcosystemConfig
from ecosystem_gym.maintenance import (
    MaintenanceGoal,
    MaintenanceMacro,
    ModelBasedSchedulerConfig,
    ModelBasedSchedulerPolicy,
    PolicyMemory,
    PublicMLPActor,
    ShieldConfig,
    ShieldedLearnedPolicy,
    UrgencySchedulerConfig,
    UrgencySchedulerPolicy,
    compile_macro,
    encode_features,
    load_policy,
    memory_snapshot,
    policy_fingerprint,
    restore_memory,
    save_policy,
)
from ecosystem_gym.maintenance.model_based import PredictedState


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


def test_private_commitment_memory_does_not_create_feature_31_and_round_trips() -> None:
    observation = _observation()
    plain = PolicyMemory(feed_count=1, play_count=2, rest_count=1, food_cooldown_bucket=3.5)
    private = PolicyMemory(
        feed_count=1,
        play_count=2,
        rest_count=1,
        food_cooldown_bucket=3.5,
        committed_goal=MaintenanceGoal.REST,
        commitment_steps=4,
        supervisor_interventions=9,
    )
    assert encode_features(observation, plain).shape == (30,)
    assert np.array_equal(encode_features(observation, plain), encode_features(observation, private))
    assert memory_snapshot(restore_memory(memory_snapshot(private))) == memory_snapshot(private)


def test_urgency_scheduler_commits_recovers_and_maintains_after_quotas() -> None:
    policy = UrgencySchedulerPolicy(config=_config(), protocol_fingerprint="test")
    memory = policy.reset()
    low_satiety = _observation(drives=(0.20, 0.80, 0.10))
    first = policy.decide(low_satiety, memory)
    assert first.macro is MaintenanceMacro.GO_FOOD
    assert first.goal is MaintenanceGoal.FEED
    assert memory.committed_goal is MaintenanceGoal.FEED

    near_food = _observation(drives=(0.18, 0.78, 0.20), agent=(0.6, 0.0))
    continued = policy.decide(near_food, memory)
    assert continued.macro is MaintenanceMacro.PICK_UP
    assert continued.reason == "continue-commitment"

    memory.pending_recovery = True
    relocated = _observation(drives=(0.18, 0.78, 0.20), agent=(0.6, 0.0), food=(-0.6, 0.0))
    assert policy.choose(relocated, memory) is MaintenanceMacro.GO_FOOD

    memory = policy.reset()
    memory.feed_count, memory.play_count, memory.rest_count = 3, 3, 2
    bored_at_toy = _observation(drives=(0.80, 0.80, 0.80), agent=(0.0, 0.6))
    decision = policy.decide(bored_at_toy, memory)
    assert decision.macro is MaintenanceMacro.PLAY
    assert decision.reason == "continuous-maintenance"


def test_urgency_hysteresis_is_the_only_candidate_control_difference() -> None:
    candidate = UrgencySchedulerConfig()
    control = UrgencySchedulerConfig(commitment_enabled=False)
    assert candidate.commitment_enabled is True and control.commitment_enabled is False
    candidate_values = candidate.__dict__ if hasattr(candidate, "__dict__") else {
        name: getattr(candidate, name) for name in candidate.__dataclass_fields__
    }
    control_values = control.__dict__ if hasattr(control, "__dict__") else {
        name: getattr(control, name) for name in control.__dataclass_fields__
    }
    assert {key for key in candidate_values if candidate_values[key] != control_values[key]} == {"commitment_enabled"}


def test_urgency_scheduler_never_selects_unproductive_play_or_rest() -> None:
    config = _config()
    policy = UrgencySchedulerPolicy(config=config)
    observation = _observation(
        drives=(0.80, 0.80, 0.10),
        agent=(0.0, 0.0),
        toy=(0.0, 0.0),
        rest=(0.0, 0.0),
    )
    assert policy.choose(observation, policy.reset()) is MaintenanceMacro.WAIT


def test_depth_four_plans_feed_chain_that_myopic_control_cannot_value() -> None:
    observation = _observation(drives=(0.20, 0.80, 0.10))
    config = _config()
    myopic = ModelBasedSchedulerPolicy(
        config=config,
        planner=ModelBasedSchedulerConfig(lookahead_depth=1),
    )
    horizon = ModelBasedSchedulerPolicy(
        config=config,
        planner=ModelBasedSchedulerConfig(lookahead_depth=4),
    )
    assert myopic.choose(observation, myopic.reset()) is MaintenanceMacro.WAIT
    plan = horizon.plan(observation, horizon.reset())
    assert plan.macros[:3] == (
        MaintenanceMacro.GO_FOOD,
        MaintenanceMacro.PICK_UP,
        MaintenanceMacro.CONSUME,
    )
    assert plan.score.quota_progress == 1
    assert plan.score.restorative_effect > 0.5


def test_public_model_numerics_and_semantic_interaction_filter() -> None:
    config = _config()
    policy = ModelBasedSchedulerPolicy(config=config)
    state = PredictedState.from_public(_observation(drives=(0.20, 0.80, 0.10)), policy.reset())
    assert MaintenanceMacro.PLAY not in policy.semantic_actions(state)
    assert MaintenanceMacro.REST not in policy.semantic_actions(state)
    arrived, duration, relief = policy.predict(state, MaintenanceMacro.GO_FOOD)
    assert duration == pytest.approx(2.0)
    assert relief == 0.0
    assert arrived.drives.satiety == pytest.approx(0.20 - 2.0 * 0.018, abs=1e-7)
    picked, _, _ = policy.predict(arrived, MaintenanceMacro.PICK_UP)
    consumed, _, relief = policy.predict(picked, MaintenanceMacro.CONSUME)
    assert consumed.feed_count == 1 and not consumed.holding_food
    assert relief == pytest.approx(0.55)


def test_shield_overrides_wait_and_useless_interactions_but_control_does_not() -> None:
    config = _config()
    actor = PublicMLPActor.zeros()
    actor.parameters["ba"][MaintenanceMacro.WAIT] = 100.0
    candidate = ShieldedLearnedPolicy(actor=actor, config=config)
    control = ShieldedLearnedPolicy(actor=actor, config=config, shield=ShieldConfig(enabled=False))
    low_satiety = _observation(drives=(0.20, 0.80, 0.10))
    assert control.decide(low_satiety, control.reset()).macro is MaintenanceMacro.WAIT
    decision = candidate.decide(low_satiety, candidate.reset())
    assert decision.macro is MaintenanceMacro.GO_FOOD
    assert decision.intervened and decision.reason == "objective-commitment"

    actor = PublicMLPActor.zeros()
    actor.parameters["ba"][MaintenanceMacro.REST] = 100.0
    actor.parameters["ba"][MaintenanceMacro.WAIT] = 50.0
    at_rest = _observation(drives=(0.80, 0.80, 0.10), rest=(0.0, 0.0))
    raw = ShieldedLearnedPolicy(actor=actor, config=config, shield=ShieldConfig(enabled=False))
    shielded = ShieldedLearnedPolicy(actor=actor, config=config)
    assert raw.choose(at_rest, raw.reset()) is MaintenanceMacro.REST
    semantic = shielded.decide(at_rest, shielded.reset())
    assert semantic.raw_macro is MaintenanceMacro.REST
    assert semantic.macro is MaintenanceMacro.WAIT
    assert not semantic.allowed_mask[MaintenanceMacro.REST]


def test_every_policy_family_serializes_reloads_and_rejects_tampering(tmp_path) -> None:
    config = _config()
    actor = PublicMLPActor.zeros()
    actor.parameters["ba"][MaintenanceMacro.WAIT] = 2.0
    policies = (
        UrgencySchedulerPolicy(config=config, protocol_fingerprint="artifact-test"),
        ModelBasedSchedulerPolicy(config=config, protocol_fingerprint="artifact-test"),
        ShieldedLearnedPolicy(actor=actor, config=config, protocol_fingerprint="artifact-test"),
    )
    observation = _observation(drives=(0.20, 0.80, 0.10))
    for index, policy in enumerate(policies):
        artifact = save_policy(tmp_path / f"policy-{index}.json", policy)
        restored = load_policy(
            artifact["path"],
            config=config,
            expected_protocol_fingerprint="artifact-test",
        )
        assert policy_fingerprint(restored) == policy_fingerprint(policy) == artifact["policy_fingerprint"]
        assert restored.choose(observation, restored.reset()) == policy.choose(observation, policy.reset())

    payload = json.loads((tmp_path / "policy-0.json").read_text())
    payload["parameters"]["feed_activation"] = 0.99
    (tmp_path / "tampered.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash"):
        load_policy(tmp_path / "tampered.json", config=config)
    with pytest.raises(ValueError, match="protocol"):
        load_policy(tmp_path / "policy-0.json", config=config, expected_protocol_fingerprint="wrong")
