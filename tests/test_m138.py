from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import inspect
import json

import numpy as np
import pytest

from ecosystem_gym.actions import ActionKind, ActionOutcome
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments import m138
from ecosystem_gym.experiments.m13 import M13_GAMMA, M13Macro, compile_macro
from ecosystem_gym.experiments.m132 import (
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_HIDDEN_DIM,
    M132_LEARNING_RATE,
    M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP,
)
from ecosystem_gym.experiments.m133 import m133_config, m133_epsilon
from ecosystem_gym.experiments.m138 import (
    M138_AUDIT_CONDITIONS,
    M138_AUDIT_SEEDS,
    M138_CONFIRMATION_EPISODES,
    M138_CONFIRMATION_EVALUATION_CONDITIONS,
    M138_CONFIRMATION_EVALUATION_SEEDS,
    M138_CONFIRMATION_FIT_CONDITIONS,
    M138_CONFIRMATION_FIT_SEEDS,
    M138_CONFIRMATION_TRAINING_SEEDS,
    M138_SCREEN_EPISODES,
    M138_SCREEN_FIT_CONDITIONS,
    M138_SCREEN_FIT_SEEDS,
    M138_SCREEN_PROBE_CONDITIONS,
    M138_SCREEN_PROBE_SEEDS,
    M138_SCREEN_TRAINING_SEEDS,
    CompactM138QPolicy,
    M138RewardState,
    m138_config,
    m138_learning_reward,
    m138_safe,
    m138_safe_error,
)
from ecosystem_gym.experiments import m138_support
from ecosystem_gym.experiments.m138_support import (
    CompactEpisodeSummary,
    M138ProtocolError,
    SplitAlreadyOpenedError,
    SplitDependencyError,
    SplitLedgerError,
    SplitOpenLedger,
    ScreenReportVerificationError,
    assert_compact_report,
    compact_episode_summary,
    m138_confirmation_passes,
    m138_content_hash,
    m138_screen_promotes,
    seal_screen_report,
    verify_screen_report,
)
from ecosystem_gym.tasks import LAYOUTS
from ecosystem_gym.trajectory import replay_and_validate


def _observation(drives: tuple[float, float, float]) -> dict[str, np.ndarray]:
    return {"drives": np.asarray(drives, dtype=np.float32)}


def _action(duration: float, kind: ActionKind = ActionKind.IDLE) -> dict[str, np.ndarray | int]:
    return {
        "kind": int(kind),
        "target": np.zeros(2, dtype=np.float32),
        "duration": np.asarray(duration, dtype=np.float32),
    }


def _info(
    *,
    outcome: ActionOutcome = ActionOutcome.SUCCESS,
    disturbance: str | None = None,
    recovery_action: bool = False,
    post_disturbance_completion: bool = False,
    feed_cycles: int = 0,
    play_cycles: int = 0,
    rest_cycles: int = 0,
    survived: bool = True,
    task_success: bool = False,
    maintenance_complete: bool = False,
) -> dict[str, object]:
    return {
        "outcome": outcome.value,
        "disturbance": disturbance,
        "recovery_action": recovery_action,
        "post_disturbance_completion": post_disturbance_completion,
        "resource_event": None,
        "feed_cycles": feed_cycles,
        "play_cycles": play_cycles,
        "rest_cycles": rest_cycles,
        "survived": survived,
        "task_success": task_success,
        "maintenance_complete": maintenance_complete,
    }


def _candidate_step(
    state: M138RewardState,
    *,
    before: tuple[float, float, float] = (0.50, 0.50, 0.50),
    after: tuple[float, float, float] = (0.50, 0.50, 0.50),
    duration: float = 1.0,
    kind: ActionKind = ActionKind.IDLE,
    macro: M13Macro | None = None,
    info: dict[str, object] | None = None,
    terminated: bool = False,
    truncated: bool = False,
    environment_reward: float = 123.0,
) -> tuple[float, dict[str, float | bool | int]]:
    if macro is None:
        macro = {
            ActionKind.PICK_UP: M13Macro.PICK_UP,
            ActionKind.CONSUME: M13Macro.CONSUME,
            ActionKind.WALK_TO: M13Macro.GO_FOOD,
            ActionKind.RUN_AROUND: M13Macro.PLAY,
            ActionKind.REST: M13Macro.REST,
            ActionKind.IDLE: M13Macro.WAIT,
        }.get(kind)
    return m138_learning_reward(
        arm="candidate",
        environment_reward=environment_reward,
        state=state,
        observation_before=_observation(before),
        observation_after=_observation(after),
        action=_action(duration, kind),
        info=_info() if info is None else info,
        terminated=terminated,
        truncated=truncated,
        macro=macro,
    )


@pytest.mark.parametrize(
    ("drives", "expected"),
    [
        ((0.30, 0.30, 0.70), 0.0),
        ((0.225, 0.30, 0.70), 0.25),
        ((0.30, 0.225, 0.70), 0.25),
        ((0.30, 0.30, 0.80), 0.25),
        ((0.225, 0.225, 0.80), 0.25),  # maximum, not a sum or mean
        ((0.15, 0.30, 0.70), 1.0),
        ((0.30, 0.15, 0.70), 1.0),
        ((0.30, 0.30, 0.90), 1.0),
        ((-1.0, 2.0, -1.0), 1.0),
    ],
)
def test_m138_safe_error_is_the_exact_buffered_worst_drive_error(
    drives: tuple[float, float, float], expected: float
) -> None:
    assert m138_safe_error(np.asarray(drives, dtype=np.float32)) == pytest.approx(expected)

    rng = np.random.default_rng(20_260_901)
    errors = [m138_safe_error(row) for row in rng.uniform(-1.0, 2.0, size=(256, 3))]
    assert min(errors) >= 0.0
    assert max(errors) <= 1.0


def test_m138_hard_safe_band_is_strict_at_the_declared_limits() -> None:
    assert m138_safe(np.asarray((0.15001, 0.15001, 0.89999), dtype=np.float32))
    assert not m138_safe(np.asarray((0.15, 0.50, 0.50), dtype=np.float32))
    assert not m138_safe(np.asarray((0.50, 0.15, 0.50), dtype=np.float32))
    assert not m138_safe(np.asarray((0.50, 0.50, 0.90), dtype=np.float32))


def test_m138_safety_cost_and_episode_safe_metrics_use_their_distinct_duration_rules() -> None:
    # E(before)=0 and E(after)=1, hence C=d/2.  Duration and safety costs
    # together are -.02*d for this transition.
    state_two = M138RewardState(required_recovery=False)
    reward_two, _ = _candidate_step(
        state_two,
        before=(0.30, 0.30, 0.70),
        after=(0.15, 0.30, 0.70),
        duration=2.0,
    )
    state_four = M138RewardState(required_recovery=False)
    reward_four, _ = _candidate_step(
        state_four,
        before=(0.30, 0.30, 0.70),
        after=(0.15, 0.30, 0.70),
        duration=4.0,
    )
    assert reward_two == pytest.approx(-0.04)
    assert reward_four == pytest.approx(-0.08)
    assert reward_four == pytest.approx(2.0 * reward_two)

    # Episode occupancy samples only the post-action hard predicate.  It does
    # not reuse the trapezoidal error integral above.
    assert state_two.decision_steps == 1
    assert state_two.decision_safe_steps == 0
    assert state_two.total_duration == pytest.approx(2.0)
    assert state_two.duration_safe_seconds == pytest.approx(0.0)
    safe_state = M138RewardState(required_recovery=False)
    _candidate_step(safe_state, before=(0.15, 0.30, 0.70), after=(0.50, 0.50, 0.50), duration=3.0)
    assert safe_state.decision_safe_steps == 1
    assert safe_state.duration_safe_seconds == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("field", "before_count", "quota_count", "farmed_count", "award", "duration"),
    [
        ("feed_cycles", 2, 3, 30, 0.25, 0.1),
        ("play_cycles", 2, 3, 30, 0.25, 1.0),
        ("rest_cycles", 1, 2, 30, 0.375, 1.0),
    ],
)
def test_m138_cycle_rewards_are_bounded_332_milestones(
    field: str,
    before_count: int,
    quota_count: int,
    farmed_count: int,
    award: float,
    duration: float,
) -> None:
    state = M138RewardState(required_recovery=False, **{field: before_count})
    counts = {"feed_cycles": 0, "play_cycles": 0, "rest_cycles": 0}
    counts[field] = quota_count
    quota_reward, _ = _candidate_step(state, duration=duration, info=_info(**counts))
    assert quota_reward == pytest.approx(award - 0.01 * duration)

    counts[field] = farmed_count
    farmed_reward, _ = _candidate_step(state, duration=duration, info=_info(**counts))
    assert farmed_reward == pytest.approx(-0.01 * duration)


def test_m138_forced_stale_block_is_exempt_only_on_relocation_and_recovery_pays_once() -> None:
    state = M138RewardState(required_recovery=True)
    stale_reward, _ = _candidate_step(
        state,
        duration=0.1,
        kind=ActionKind.PICK_UP,
        info=_info(outcome=ActionOutcome.BLOCKED, disturbance="food_relocated"),
    )
    assert stale_reward == pytest.approx(-0.001)
    assert state.relocation_seen and state.stale_pickup

    next_block_reward, _ = _candidate_step(
        state,
        duration=0.1,
        kind=ActionKind.PICK_UP,
        info=_info(outcome=ActionOutcome.BLOCKED),
    )
    assert next_block_reward == pytest.approx(-0.101)

    recovery_walk_reward, _ = _candidate_step(
        state,
        duration=1.0,
        kind=ActionKind.WALK_TO,
        info=_info(recovery_action=True),
    )
    assert recovery_walk_reward == pytest.approx(-0.01)

    completed_reward, _ = _candidate_step(
        state,
        duration=0.1,
        kind=ActionKind.CONSUME,
        info=_info(post_disturbance_completion=True),
    )
    assert completed_reward == pytest.approx(0.249)
    assert state.recovery_complete and state.recovery_rewarded

    repeated_reward, _ = _candidate_step(
        state,
        duration=0.1,
        kind=ActionKind.CONSUME,
        info=_info(post_disturbance_completion=True),
    )
    assert repeated_reward == pytest.approx(-0.001)

    not_required = M138RewardState(required_recovery=False, relocation_seen=True, stale_pickup=True)
    nonrequired_reward, _ = _candidate_step(
        not_required,
        duration=0.1,
        kind=ActionKind.CONSUME,
        info=_info(post_disturbance_completion=True),
    )
    assert nonrequired_reward == pytest.approx(-0.001)
    assert not not_required.recovery_rewarded

    no_stale = M138RewardState(required_recovery=True, relocation_seen=True)
    no_stale_reward, _ = _candidate_step(
        no_stale,
        duration=0.1,
        kind=ActionKind.CONSUME,
        info=_info(post_disturbance_completion=True),
    )
    assert no_stale_reward == pytest.approx(-0.001)
    assert not no_stale.recovery_complete and not no_stale.recovery_rewarded

    stale_but_not_consume = M138RewardState(
        required_recovery=True,
        relocation_seen=True,
        stale_pickup=True,
    )
    fabricated_reward, _ = _candidate_step(
        stale_but_not_consume,
        duration=1.0,
        kind=ActionKind.IDLE,
        info=_info(post_disturbance_completion=True),
    )
    assert fabricated_reward == pytest.approx(-0.01)
    assert not stale_but_not_consume.recovery_complete


def test_m138_invalid_outcome_keeps_ordinary_invalid_penalty() -> None:
    reward, _ = _candidate_step(
        M138RewardState(required_recovery=False),
        duration=1.0,
        info=_info(outcome=ActionOutcome.NOT_HOLDING_OBJECT),
    )
    assert reward == pytest.approx(-0.11)


def test_m138_terminal_gate_includes_final_safety_sample_and_ignores_legacy_task_success() -> None:
    # The final safe decision/duration moves both exact ratios from 169/199 to
    # 170/200 == .85.  False legacy task flags cannot suppress the +5 gate.
    success_state = M138RewardState(
        required_recovery=False,
        feed_cycles=3,
        play_cycles=3,
        rest_cycles=2,
        decision_steps=199,
        decision_safe_steps=169,
        total_duration=199.0,
        duration_safe_seconds=169.0,
    )
    success_reward, success_components = _candidate_step(
        success_state,
        duration=1.0,
        info=_info(
            feed_cycles=3,
            play_cycles=3,
            rest_cycles=2,
            survived=True,
            task_success=False,
            maintenance_complete=False,
        ),
        truncated=True,
    )
    assert success_state.decision_safe_steps / success_state.decision_steps == pytest.approx(0.85)
    assert success_state.duration_safe_seconds / success_state.total_duration == pytest.approx(0.85)
    assert success_reward == pytest.approx(4.99)
    assert success_components["full_terminal_success"] is True
    assert success_components["terminal_failure"] is False

    # Quotas and even misleading legacy success flags cannot turn early drive
    # depletion into candidate success.
    death_state = M138RewardState(
        required_recovery=False,
        feed_cycles=3,
        play_cycles=3,
        rest_cycles=2,
        decision_steps=9,
        decision_safe_steps=9,
        total_duration=9.0,
        duration_safe_seconds=9.0,
    )
    death_reward, death_components = _candidate_step(
        death_state,
        before=(0.50, 0.50, 0.50),
        after=(0.0, 0.50, 0.50),
        duration=1.0,
        info=_info(
            feed_cycles=3,
            play_cycles=3,
            rest_cycles=2,
            survived=False,
            task_success=True,
            maintenance_complete=True,
        ),
        terminated=True,
    )
    assert death_reward == pytest.approx(-2.02)
    assert death_components["full_terminal_success"] is False
    assert death_components["terminal_failure"] is True

    # At the horizon, missing either safety or required recovery selects -2,
    # never +5, even when the legacy environment reports cycle maintenance.
    incomplete_state = M138RewardState(
        required_recovery=True,
        feed_cycles=3,
        play_cycles=3,
        rest_cycles=2,
        decision_steps=199,
        decision_safe_steps=168,
        total_duration=199.0,
        duration_safe_seconds=168.0,
        relocation_seen=True,
        stale_pickup=True,
        recovery_complete=False,
    )
    incomplete_reward, incomplete_components = _candidate_step(
        incomplete_state,
        duration=1.0,
        info=_info(
            feed_cycles=3,
            play_cycles=3,
            rest_cycles=2,
            survived=True,
            task_success=True,
            maintenance_complete=True,
        ),
        truncated=True,
    )
    assert incomplete_reward == pytest.approx(-2.01)
    assert incomplete_components["full_terminal_success"] is False
    assert incomplete_components["terminal_failure"] is True

    short_horizon_state = M138RewardState(
        required_recovery=False,
        feed_cycles=3,
        play_cycles=3,
        rest_cycles=2,
        decision_steps=8,
        decision_safe_steps=8,
        total_duration=8.0,
        duration_safe_seconds=8.0,
    )
    short_reward, short_components = _candidate_step(
        short_horizon_state,
        duration=1.0,
        info=_info(
            feed_cycles=3,
            play_cycles=3,
            rest_cycles=2,
            survived=True,
            task_success=True,
            maintenance_complete=True,
        ),
        truncated=True,
    )
    assert short_horizon_state.decision_steps == 9
    assert short_reward == pytest.approx(-2.01)
    assert short_components["full_terminal_success"] is False
    assert short_components["terminal_failure"] is True

    recovered_state = M138RewardState(
        required_recovery=True,
        feed_cycles=3,
        play_cycles=3,
        rest_cycles=2,
        decision_steps=199,
        decision_safe_steps=169,
        total_duration=199.0,
        duration_safe_seconds=169.0,
        relocation_seen=True,
        stale_pickup=True,
        recovery_complete=True,
        recovery_rewarded=True,
    )
    recovered_reward, recovered_components = _candidate_step(
        recovered_state,
        duration=1.0,
        info=_info(feed_cycles=3, play_cycles=3, rest_cycles=2, survived=True),
        truncated=True,
    )
    assert recovered_reward == pytest.approx(4.99)
    assert recovered_components["full_terminal_success"] is True
    assert recovered_components["terminal_failure"] is False


def test_m138_candidate_is_recomposed_from_zero_and_legacy_control_is_raw_environment_reward() -> None:
    candidate_a, components = _candidate_step(
        M138RewardState(required_recovery=False), environment_reward=123.0
    )
    candidate_b, _ = _candidate_step(M138RewardState(required_recovery=False), environment_reward=-999.0)
    assert candidate_a == candidate_b == pytest.approx(-0.01)
    component_sum = sum(
        float(components[key])
        for key in (
            "duration_cost",
            "safety_cost",
            "invalid_cost",
            "blocked_cost",
            "feed_milestone_reward",
            "play_milestone_reward",
            "rest_milestone_reward",
            "recovery_reward",
            "terminal_success_reward",
            "terminal_failure_cost",
        )
    )
    assert component_sum == pytest.approx(candidate_a)
    assert components["candidate_recomposed"] == pytest.approx(candidate_a)
    assert components["learning_reward"] == pytest.approx(candidate_a)
    assert components["environment_reward"] == 123.0

    legacy_state = M138RewardState(required_recovery=False)
    legacy, _ = m138_learning_reward(
        arm="legacy_control",
        environment_reward=7.25,
        state=legacy_state,
        observation_before=_observation((0.50, 0.50, 0.50)),
        observation_after=_observation((0.50, 0.50, 0.50)),
        action=_action(1.0),
        info=_info(),
        terminated=False,
        truncated=False,
    )
    assert legacy == 7.25
    assert legacy_state.decision_steps == 1
    assert legacy_state.decision_safe_steps == 1
    assert legacy_state.total_duration == pytest.approx(1.0)
    assert legacy_state.duration_safe_seconds == pytest.approx(1.0)


def test_m138_candidate_and_control_share_only_the_declared_200_step_config_change() -> None:
    historical, current = asdict(m133_config()), asdict(m138_config())
    changed = {key for key in historical if historical[key] != current[key]}
    assert changed == {"max_episode_steps"}
    assert historical["max_episode_steps"] == 160
    assert current["max_episode_steps"] == 200

    seed = M138_SCREEN_TRAINING_SEEDS[0]
    candidate = CompactM138QPolicy(arm="candidate", seed=seed, stage="screen")
    control = CompactM138QPolicy(arm="legacy_control", seed=seed, stage="screen")
    assert candidate.config == control.config == m138_config()
    assert candidate.config.max_episode_steps == control.config.max_episode_steps == 200
    assert candidate.update_every == control.update_every == 4
    assert candidate.target_update_every == control.target_update_every == 1_000
    assert candidate.mask_mode == control.mask_mode == "complementary"
    assert candidate.online.params["w1"].shape == (30, M132_HIDDEN_DIM) == (30, 64)
    assert candidate.online.params["w2"].shape == (64, 64)
    assert candidate.online.params["w3"].shape == (64, len(M13Macro)) == (64, 8)
    assert M132_REPLAY_CAPACITY == 100_000
    assert M132_REPLAY_WARMUP == 1_000
    assert M132_BATCH_SIZE == 128
    assert M132_LEARNING_RATE == pytest.approx(3e-4)
    assert M13_GAMMA == pytest.approx(0.99)
    assert m133_epsilon(0) == 1.0
    assert m133_epsilon(M138_SCREEN_EPISODES) > 0.05  # no compressed screen schedule
    assert m133_epsilon(9_000) == 0.05
    for key in candidate.online.params:
        assert np.array_equal(candidate.online.params[key], control.online.params[key])
    with pytest.raises(ValueError, match="frozen|budget"):
        candidate.train(episodes=1)


def _layout_geometry(name: str) -> tuple[float, ...]:
    layout = LAYOUTS[name]
    return tuple(
        float(value)
        for point in (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)
        for value in point
    )


def _layout_points(name: str) -> set[tuple[float, float]]:
    layout = LAYOUTS[name]
    return {
        tuple(map(float, point))
        for point in (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)
    }


def test_m138_splits_rng_seeds_and_layout_coordinates_are_fresh_and_disjoint() -> None:
    assert M138_SCREEN_FIT_SEEDS == tuple(range(4_800, 4_820))
    assert M138_SCREEN_PROBE_SEEDS == tuple(range(4_820, 4_828))
    assert M138_CONFIRMATION_FIT_SEEDS == tuple(range(4_900, 4_940))
    assert M138_CONFIRMATION_EVALUATION_SEEDS == tuple(range(5_000, 5_020))
    assert M138_AUDIT_SEEDS == tuple(range(5_100, 5_120))
    assert M138_SCREEN_TRAINING_SEEDS == (20_260_901, 20_260_902)
    assert M138_CONFIRMATION_TRAINING_SEEDS == tuple(range(20_260_903, 20_260_911))
    assert M138_SCREEN_EPISODES == 4_000
    assert M138_CONFIRMATION_EPISODES == 12_000
    assert M138_SCREEN_EPISODES == 4 * len(M138_SCREEN_FIT_SEEDS) * 50
    assert M138_CONFIRMATION_EPISODES == 4 * len(M138_CONFIRMATION_FIT_SEEDS) * 75

    environment_groups = (
        M138_SCREEN_FIT_SEEDS,
        M138_SCREEN_PROBE_SEEDS,
        M138_CONFIRMATION_FIT_SEEDS,
        M138_CONFIRMATION_EVALUATION_SEEDS,
        M138_AUDIT_SEEDS,
    )
    for index, left in enumerate(environment_groups):
        for right in environment_groups[index + 1 :]:
            assert set(left).isdisjoint(right)
    assert set(M138_SCREEN_TRAINING_SEEDS).isdisjoint(M138_CONFIRMATION_TRAINING_SEEDS)

    condition_groups = (
        (M138_SCREEN_FIT_CONDITIONS, "m138_screen_"),
        (M138_SCREEN_PROBE_CONDITIONS, "m138_probe_"),
        (M138_CONFIRMATION_FIT_CONDITIONS, "m138_confirm_fit_"),
        (M138_CONFIRMATION_EVALUATION_CONDITIONS, "m138_confirm_eval_"),
        (M138_AUDIT_CONDITIONS, "m138_audit_"),
    )
    new_layouts: list[str] = []
    for conditions, prefix in condition_groups:
        assert set(conditions) == {
            "persistent_reference",
            "renewal_and_morphology",
            "event_relocation",
            "compound",
        }
        names = [str(row["layout_id"]) for row in conditions.values()]
        assert all(name.startswith(prefix) for name in names)
        new_layouts.extend(names)
    assert len(new_layouts) == len(set(new_layouts)) == 20
    new_geometries = {_layout_geometry(name) for name in new_layouts}
    old_geometries = {_layout_geometry(name) for name in LAYOUTS if name not in new_layouts}
    assert len(new_geometries) == 20
    assert new_geometries.isdisjoint(old_geometries)
    new_points = [point for name in new_layouts for point in _layout_points(name)]
    prior_m13_points = {
        point
        for name in LAYOUTS
        if name.startswith("m13") and not name.startswith("m138")
        for point in _layout_points(name)
    }
    assert len(new_points) == len(set(new_points)) == 100
    assert set(new_points).isdisjoint(prior_m13_points)


def test_m138_protocol_fingerprint_is_stable_and_covers_the_frozen_budget(monkeypatch) -> None:
    fingerprint = m138.m138_protocol_fingerprint()
    assert fingerprint == m138.m138_protocol_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)

    monkeypatch.setattr(m138, "M138_SCREEN_EPISODES", M138_SCREEN_EPISODES + 1)
    assert m138.m138_protocol_fingerprint() != fingerprint


def test_m138_policy_artifact_roundtrip_is_exact_and_refuses_overwrite_or_tamper(tmp_path) -> None:
    policy = CompactM138QPolicy(
        arm="candidate",
        seed=M138_SCREEN_TRAINING_SEEDS[0],
        stage="screen",
    )
    path = tmp_path / "candidate.json"
    artifact = m138.write_m138_policy(path, policy)
    restored = m138.load_m138_policy(artifact["path"])
    assert m138.m138_policy_fingerprint(restored) == m138.m138_policy_fingerprint(policy)
    assert restored.arm == policy.arm
    assert restored.seed == policy.seed
    assert restored.stage == policy.stage
    for key in policy.online.params:
        assert np.array_equal(restored.online.params[key], policy.online.params[key])
    with pytest.raises(FileExistsError):
        m138.write_m138_policy(path, policy)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol_fingerprint"] = "0" * 64
    protocol_tamper = tmp_path / "protocol-tamper.json"
    protocol_tamper.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        m138.load_m138_policy(protocol_tamper)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["network"]["w1"] = [[0.0]]
    parameter_tamper = tmp_path / "parameter-tamper.json"
    parameter_tamper.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter"):
        m138.load_m138_policy(parameter_tamper)


def _trace_rows(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_m138_candidate_control_and_random_traces_strictly_replay_raw_and_learning_rewards(tmp_path) -> None:
    training_seed = M138_SCREEN_TRAINING_SEEDS[0]
    condition = "persistent_reference"
    controls = M138_SCREEN_PROBE_CONDITIONS[condition]
    env_seed = M138_SCREEN_PROBE_SEEDS[0]

    candidate = CompactM138QPolicy(
        arm="candidate", seed=training_seed, stage="screen"
    )
    candidate_artifact = m138.write_m138_policy(tmp_path / "candidate.json", candidate)
    candidate_reloaded = m138.load_m138_policy(candidate_artifact["path"])
    candidate_trace = tmp_path / "candidate.jsonl"
    m138.run_m138_episode(
        candidate_reloaded,
        arm="candidate",
        training_seed=training_seed,
        seed=env_seed,
        condition=condition,
        controls=controls,
        trace_path=candidate_trace,
        policy_label="candidate-reloaded",
        policy_fingerprint=m138.m138_policy_fingerprint(candidate_reloaded),
    )
    assert m138.replay_m138_trace(candidate_trace, candidate_reloaded).steps > 0
    candidate_rows = _trace_rows(candidate_trace)
    assert candidate_rows[0]["config"]["max_episode_steps"] == 200
    steps = candidate_rows[1:]
    assert steps
    assert all("reward" in row and "learning_reward" in row for row in steps)
    assert all("reward_components" in row for row in steps)
    assert any(
        not np.isclose(float(row["reward"]), float(row["learning_reward"]))
        for row in steps
    )

    # Generic environment replay deliberately ignores the shaped sidecar.
    # Mutating a component leaves generic replay valid but strict M13.8 replay
    # must reject it.
    component_tamper = deepcopy(candidate_rows)
    component_tamper[1]["reward_components"]["duration_cost"] += 0.123
    component_path = tmp_path / "candidate-component-tamper.jsonl"
    component_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in component_tamper)
        + "\n",
        encoding="utf-8",
    )
    assert replay_and_validate(component_path).steps > 0
    with pytest.raises(ValueError, match="reward_components"):
        m138.replay_m138_trace(component_path, candidate_reloaded)

    learning_tamper = deepcopy(candidate_rows)
    learning_tamper[1]["learning_reward"] += 0.5
    learning_path = tmp_path / "candidate-learning-tamper.jsonl"
    learning_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in learning_tamper)
        + "\n",
        encoding="utf-8",
    )
    assert replay_and_validate(learning_path).steps > 0
    with pytest.raises(ValueError, match="learning reward"):
        m138.replay_m138_trace(learning_path, candidate_reloaded)

    control = CompactM138QPolicy(
        arm="legacy_control", seed=training_seed, stage="screen"
    )
    control_artifact = m138.write_m138_policy(tmp_path / "control.json", control)
    control_reloaded = m138.load_m138_policy(control_artifact["path"])
    control_trace = tmp_path / "control.jsonl"
    m138.run_m138_episode(
        control_reloaded,
        arm="legacy_control",
        training_seed=training_seed,
        seed=env_seed,
        condition=condition,
        controls=controls,
        trace_path=control_trace,
        policy_label="control-reloaded",
        policy_fingerprint=m138.m138_policy_fingerprint(control_reloaded),
    )
    assert m138.replay_m138_trace(control_trace, control_reloaded).steps > 0
    for row in _trace_rows(control_trace)[1:]:
        assert float(row["learning_reward"]) == pytest.approx(float(row["reward"]))

    random = m138.SeededRandomM138Policy(training_seed)
    random_trace = tmp_path / "random.jsonl"
    m138.run_m138_episode(
        random,
        arm="candidate",
        training_seed=training_seed,
        seed=env_seed,
        condition=condition,
        controls=controls,
        trace_path=random_trace,
        policy_label="matched-random",
        policy_fingerprint=f"pcg64-{training_seed}",
    )
    fresh_random = m138.SeededRandomM138Policy(training_seed)
    assert m138.replay_m138_random_trace(random_trace, fresh_random).steps > 0


def test_m138_policy_boundary_is_exactly_public_30_features_and_ignores_private_injections() -> None:
    policy = CompactM138QPolicy(
        arm="candidate",
        seed=M138_SCREEN_TRAINING_SEEDS[0],
        stage="screen",
    )
    env = EcosystemEnv(m138_config())
    try:
        observation, _ = env.reset(
            seed=M138_SCREEN_FIT_SEEDS[0],
            options={
                "task_id": "persistent_maintenance",
                **M138_SCREEN_FIT_CONDITIONS["persistent_reference"],
            },
        )
        assert set(observation) == {
            "agent_xy",
            "food_xy",
            "toy_xy",
            "rest_xy",
            "drives",
            "holding_food",
            "prior_outcome",
        }
        memory = policy.reset()
        features = policy.features(observation, memory)
        mask = policy.mask(observation)
        macro = policy.choose(observation, memory)
        assert features.shape == (M132_FEATURE_DIM,) == (30,)

        injected = {
            **observation,
            "task_id": "private-task",
            "reset_options": {"private": True},
            "info": {"feed_cycles": 999},
            "timer": 999_999,
            "food_available": False,
            "resource_flags": [1, 2, 3],
            "reward_state": M138RewardState(required_recovery=True, feed_cycles=999),
        }
        assert np.array_equal(features, policy.features(injected, memory))
        assert np.array_equal(mask, policy.mask(injected))
        assert macro is policy.choose(injected, memory)

        action = compile_macro(M13Macro.WAIT, observation, env.config)
        next_observation, _, _, _, _ = env.step(action)
        next_injected = {**next_observation, **{key: injected[key] for key in injected if key not in observation}}
        plain_memory, injected_memory = policy.reset(), policy.reset()
        plain_memory.food_cooldown_bucket = 0
        injected_memory.food_cooldown_bucket = 0
        policy.observe(
            plain_memory,
            observation_before=observation,
            macro=M13Macro.WAIT,
            action=action,
            observation_after=next_observation,
        )
        policy.observe(
            injected_memory,
            observation_before=injected,
            macro=M13Macro.WAIT,
            action=action,
            observation_after=next_injected,
        )
        assert asdict(plain_memory) == asdict(injected_memory)
        selected_duration = float(np.asarray(action["duration"], dtype=np.float32).item())
        assert plain_memory.food_cooldown_bucket == pytest.approx(selected_duration)
    finally:
        env.close()

    assert "info" not in inspect.signature(policy.features).parameters
    assert "info" not in inspect.signature(policy.mask).parameters
    assert "info" not in inspect.signature(policy.choose).parameters
    assert "info" not in inspect.signature(policy.observe).parameters


def _compact_summary(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "training_seed": M138_SCREEN_TRAINING_SEEDS[0],
        "condition": "persistent_reference",
        "env_seed": M138_SCREEN_PROBE_SEEDS[0],
        "survived": True,
        "maintenance_complete": True,
        "decision_safe_fraction": 0.90,
        "duration_safe_fraction": 0.90,
        "recovery_required": False,
        "recovery_complete": False,
        "conformance_violations": 0,
        "replay_violations": 0,
    }
    row.update(updates)
    return row


def test_m138_compact_evidence_excludes_transition_payloads_and_rejects_false_maintenance() -> None:
    oversized = {
        **_compact_summary(),
        "steps": [{"observation": [1, 2, 3]}],
        "episodes_detail": [{"private": True}],
    }
    compact = compact_episode_summary(oversized)
    assert set(compact) == {
        "training_seed",
        "condition",
        "env_seed",
        "survived",
        "maintenance_complete",
        "decision_safe_fraction",
        "duration_safe_fraction",
        "recovery_required",
        "recovery_complete",
        "conformance_violations",
        "replay_violations",
    }
    assert "steps" not in compact and "episodes_detail" not in compact
    assert_compact_report({"episode_summaries": [compact]})
    with pytest.raises(M138ProtocolError, match="steps|episodes_detail"):
        assert_compact_report({"episode_summaries": [oversized]})
    with pytest.raises(M138ProtocolError, match="maintenance_complete"):
        CompactEpisodeSummary(**_compact_summary(survived=False))
    with pytest.raises(M138ProtocolError, match="recovery_complete"):
        CompactEpisodeSummary(**_compact_summary(recovery_complete=True))


def test_m138_split_ledger_is_canonical_one_way_and_output_independent(tmp_path) -> None:
    ledger_path = tmp_path / "protocol-state" / "m138-ledger.json"
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint="frozen-fingerprint")
    assert ledger.snapshot()["opened_partitions"] == []
    with pytest.raises(SplitDependencyError):
        ledger.open_partition("confirmation_fit")

    opened = ledger.open_partition("screen_fit")
    assert opened["opened_partitions"] == ["screen_fit"]
    assert ledger_path.exists()
    # Report/artifact output locations are irrelevant: a fresh ledger object
    # at the canonical injected path sees and rejects the prior opening.
    (tmp_path / "output-a").mkdir()
    (tmp_path / "output-b").mkdir()
    reopened = SplitOpenLedger(ledger_path, protocol_fingerprint="frozen-fingerprint")
    assert reopened.snapshot()["opened_partitions"] == ["screen_fit"]
    with pytest.raises(SplitAlreadyOpenedError):
        reopened.open_partition("screen_fit")
    assert reopened.open_partition("screen_probe")["opened_partitions"] == [
        "screen_fit",
        "screen_probe",
    ]
    with pytest.raises(SplitLedgerError, match="fingerprint"):
        SplitOpenLedger(ledger_path, protocol_fingerprint="forged").snapshot()


def test_m138_split_ledger_atomic_write_failure_preserves_previous_marker(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "m138-ledger.json"
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint="frozen-fingerprint")
    ledger.open_partition("screen_fit")
    before = ledger_path.read_bytes()

    def fail_replace(source, destination) -> None:
        del source, destination
        raise OSError("injected replace failure")

    real_replace = m138_support.os.replace
    monkeypatch.setattr(m138_support.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        ledger.open_partition("screen_probe")
    monkeypatch.setattr(m138_support.os, "replace", real_replace)

    assert ledger_path.read_bytes() == before
    assert ledger.snapshot()["opened_partitions"] == ["screen_fit"]
    assert not list(ledger_path.parent.glob(f".{ledger_path.name}.*.tmp"))


def _summary_matrix(
    *,
    training_seeds: tuple[int, ...],
    conditions: tuple[str, ...],
    env_seeds: tuple[int, ...],
    survival_count: int,
    maintenance_count: int,
    decision_safe: float,
    duration_safe: float,
    recovery_count: int,
) -> list[CompactEpisodeSummary]:
    rows: list[CompactEpisodeSummary] = []
    for training_seed in training_seeds:
        for condition in conditions:
            required = condition in {"event_relocation", "compound"}
            for index, env_seed in enumerate(env_seeds):
                survived = index < survival_count
                rows.append(
                    CompactEpisodeSummary(
                        training_seed=training_seed,
                        condition=condition,
                        env_seed=env_seed,
                        survived=survived,
                        maintenance_complete=survived and index < maintenance_count,
                        decision_safe_fraction=decision_safe,
                        duration_safe_fraction=duration_safe,
                        recovery_required=required,
                        recovery_complete=required and index < recovery_count,
                    )
                )
    return rows


def _screen_evidence() -> tuple[
    list[CompactEpisodeSummary],
    list[CompactEpisodeSummary],
    list[CompactEpisodeSummary],
]:
    conditions = tuple(M138_SCREEN_PROBE_CONDITIONS)
    candidate = _summary_matrix(
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=conditions,
        env_seeds=M138_SCREEN_PROBE_SEEDS,
        survival_count=8,
        maintenance_count=7,
        decision_safe=0.86,
        duration_safe=0.86,
        recovery_count=8,
    )
    control = _summary_matrix(
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=conditions,
        env_seeds=M138_SCREEN_PROBE_SEEDS,
        survival_count=8,
        maintenance_count=6,
        decision_safe=0.86,
        duration_safe=0.86,
        recovery_count=8,
    )
    random = _summary_matrix(
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=conditions,
        env_seeds=M138_SCREEN_PROBE_SEEDS,
        survival_count=6,
        maintenance_count=0,
        decision_safe=0.50,
        duration_safe=0.50,
        recovery_count=0,
    )
    return candidate, control, random


def _screen_gate(
    candidate: list[CompactEpisodeSummary],
    control: list[CompactEpisodeSummary],
    random: list[CompactEpisodeSummary],
) -> dict[str, object]:
    return m138_screen_promotes(
        candidate,
        control,
        random,
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
    )


def test_m138_screen_promotion_requires_both_replicas_all_rows_random_delta_and_positive_effect() -> None:
    candidate, control, random = _screen_evidence()
    passing = _screen_gate(candidate, list(reversed(control)), list(reversed(random)))
    assert passing["passes"] is True
    assert len(passing["per_replica"]) == 2
    assert len(passing["positive_effect_pairs"]) == 2
    for replica in passing["per_replica"].values():
        assert replica["random_survival_pass"] is True
        assert replica["control_noninferiority_pass"] is True
        assert all(row["passes"] for row in replica["conditions"].values())

    one_replica = [
        row for row in candidate if row.training_seed == M138_SCREEN_TRAINING_SEEDS[0]
    ]
    with pytest.raises(M138ProtocolError, match="missing|mismatched"):
        _screen_gate(one_replica, control, random)

    no_effect = _screen_gate(candidate, deepcopy(candidate), random)
    assert no_effect["positive_effect_pass"] is False
    assert no_effect["passes"] is False

    # Starting from 24/32 random survivals, flip two paired rows to reach
    # 26/32; candidate-random is then .1875 and must fail the +.20 gate.
    high_random = list(random)
    flipped = 0
    for index, row in enumerate(high_random):
        if (
            row.training_seed == M138_SCREEN_TRAINING_SEEDS[0]
            and not row.survived
            and flipped < 2
        ):
            high_random[index] = replace(row, survived=True)
            flipped += 1
    random_failure = _screen_gate(candidate, control, high_random)
    first = random_failure["per_replica"][str(M138_SCREEN_TRAINING_SEEDS[0])]
    assert first["deltas"]["random_survival"] == pytest.approx(0.1875)
    assert first["random_survival_pass"] is False
    assert random_failure["passes"] is False


def test_m138_screen_rejects_any_condition_recovery_integrity_or_control_noninferiority_failure() -> None:
    candidate, control, random = _screen_evidence()
    target_seed = M138_SCREEN_TRAINING_SEEDS[0]

    row_failure = list(candidate)
    index = next(
        index
        for index, row in enumerate(row_failure)
        if row.training_seed == target_seed
        and row.condition == "persistent_reference"
        and row.env_seed == M138_SCREEN_PROBE_SEEDS[0]
    )
    row_failure[index] = replace(
        row_failure[index], survived=False, maintenance_complete=False
    )
    assert _screen_gate(row_failure, control, random)["passes"] is False

    recovery_failure = list(candidate)
    index = next(
        index
        for index, row in enumerate(recovery_failure)
        if row.training_seed == target_seed
        and row.condition == "event_relocation"
        and row.env_seed == M138_SCREEN_PROBE_SEEDS[0]
    )
    recovery_failure[index] = replace(recovery_failure[index], recovery_complete=False)
    assert _screen_gate(recovery_failure, control, random)["passes"] is False

    integrity_failure = list(candidate)
    integrity_failure[0] = replace(integrity_failure[0], conformance_violations=1)
    assert _screen_gate(integrity_failure, control, random)["passes"] is False

    forged_recovery_requirement = list(candidate)
    forged_recovery_requirement[0] = replace(
        forged_recovery_requirement[0], recovery_required=True
    )
    with pytest.raises(M138ProtocolError, match="recovery requirement"):
        _screen_gate(forged_recovery_requirement, control, random)

    superior_control = _summary_matrix(
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
        survival_count=8,
        maintenance_count=8,
        decision_safe=0.88,
        duration_safe=0.88,
        recovery_count=8,
    )
    assert _screen_gate(candidate, superior_control, random)["passes"] is False


def _confirmation_evidence() -> tuple[
    list[CompactEpisodeSummary],
    list[CompactEpisodeSummary],
    list[CompactEpisodeSummary],
]:
    conditions = tuple(M138_CONFIRMATION_EVALUATION_CONDITIONS)
    candidate = _summary_matrix(
        training_seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        conditions=conditions,
        env_seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
        survival_count=18,
        maintenance_count=18,
        decision_safe=0.86,
        duration_safe=0.86,
        recovery_count=18,
    )
    # Seven controls have a lower worst margin.  The eighth exactly ties.
    control = [
        replace(
            row,
            decision_safe_fraction=(
                0.84
                if row.training_seed in M138_CONFIRMATION_TRAINING_SEEDS[:7]
                else 0.86
            ),
        )
        for row in candidate
    ]
    random = _summary_matrix(
        training_seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        conditions=conditions,
        env_seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
        survival_count=13,
        maintenance_count=0,
        decision_safe=0.50,
        duration_safe=0.50,
        recovery_count=0,
    )
    return candidate, control, random


def _confirmation_gate(
    candidate: list[CompactEpisodeSummary],
    control: list[CompactEpisodeSummary],
    random: list[CompactEpisodeSummary],
) -> dict[str, object]:
    return m138_confirmation_passes(
        candidate,
        control,
        random,
        training_seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M138_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
    )


def test_m138_confirmation_requires_all_eight_hard_gates_and_seven_paired_margin_wins() -> None:
    candidate, control, random = _confirmation_evidence()
    passing = _confirmation_gate(candidate, list(reversed(control)), list(reversed(random)))
    assert passing["hard_gate_pass"] is True
    assert passing["paired_margin_wins"] == 7
    assert passing["passes"] is True

    only_six_better = [
        replace(
            row,
            decision_safe_fraction=(
                0.84
                if row.training_seed in M138_CONFIRMATION_TRAINING_SEEDS[:6]
                else 0.86
            ),
        )
        for row in candidate
    ]
    causal_failure = _confirmation_gate(candidate, only_six_better, random)
    assert causal_failure["hard_gate_pass"] is True
    assert causal_failure["paired_margin_wins"] == 6
    assert causal_failure["passes"] is False

    hard_failure = list(candidate)
    failing_seed = M138_CONFIRMATION_TRAINING_SEEDS[-1]
    failing_condition = "compound"
    changed = 0
    for index, row in enumerate(hard_failure):
        if (
            row.training_seed == failing_seed
            and row.condition == failing_condition
            and row.survived
            and changed < 2
        ):
            hard_failure[index] = replace(
                row, survived=False, maintenance_complete=False, recovery_complete=False
            )
            changed += 1
    result = _confirmation_gate(hard_failure, control, random)
    assert result["hard_gate_pass"] is False
    assert result["passes"] is False

    high_random = _summary_matrix(
        training_seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M138_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
        survival_count=15,
        maintenance_count=0,
        decision_safe=0.50,
        duration_safe=0.50,
        recovery_count=0,
    )
    random_failure = _confirmation_gate(candidate, control, high_random)
    assert random_failure["hard_gate_pass"] is False


def test_m138_screen_report_hash_and_gate_are_recomputed_from_compact_evidence() -> None:
    candidate, control, random = _screen_evidence()
    axes = {
        "expected_protocol_fingerprint": "frozen-fingerprint",
        "training_seeds": M138_SCREEN_TRAINING_SEEDS,
        "conditions": tuple(M138_SCREEN_PROBE_CONDITIONS),
        "env_seeds": M138_SCREEN_PROBE_SEEDS,
    }
    sealed = seal_screen_report(
        {
            "stage": "screen",
            "protocol_fingerprint": "frozen-fingerprint",
            "episode_summaries": {
                "candidate": [row.compact_dict() for row in candidate],
                "legacy_control": [row.compact_dict() for row in control],
                "random": [row.compact_dict() for row in random],
            },
        },
        **axes,
    )
    assert verify_screen_report(sealed, **axes)["passes"] is True
    assert_compact_report(sealed)

    tampered = deepcopy(sealed)
    tampered["episode_summaries"]["candidate"][0]["decision_safe_fraction"] = 0.0
    with pytest.raises(ScreenReportVerificationError, match="hash"):
        verify_screen_report(tampered, **axes)

    wrong_protocol = deepcopy(sealed)
    wrong_protocol["protocol_fingerprint"] = "forged"
    wrong_protocol["content_hash"] = m138_content_hash(
        {key: value for key, value in wrong_protocol.items() if key != "content_hash"}
    )
    with pytest.raises(ScreenReportVerificationError, match="protocol"):
        verify_screen_report(wrong_protocol, **axes)

    no_effect_report = seal_screen_report(
        {
            "stage": "screen",
            "protocol_fingerprint": "frozen-fingerprint",
            "passes": True,
            "episode_summaries": {
                "candidate": [row.compact_dict() for row in candidate],
                "legacy_control": [row.compact_dict() for row in candidate],
                "random": [row.compact_dict() for row in random],
            },
        },
        **axes,
    )
    assert no_effect_report["promotion"]["passes"] is False
    with pytest.raises(ScreenReportVerificationError, match="did not promote"):
        verify_screen_report(no_effect_report, **axes)

    forged_gate = deepcopy(no_effect_report)
    forged_gate["promotion"]["passes"] = True
    forged_gate["content_hash"] = m138_content_hash(
        {key: value for key, value in forged_gate.items() if key != "content_hash"}
    )
    with pytest.raises(ScreenReportVerificationError, match="disagrees"):
        verify_screen_report(forged_gate, **axes)


def _sealed_screen_report(
    *,
    protocol_fingerprint: str,
    promotes: bool,
    policy_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    candidate, control, random = _screen_evidence()
    if not promotes:
        control = deepcopy(candidate)
    unsealed: dict[str, object] = {
        "stage": "screen",
        "protocol_fingerprint": protocol_fingerprint,
        "episode_summaries": {
            "candidate": [row.compact_dict() for row in candidate],
            "legacy_control": [row.compact_dict() for row in control],
            "random": [row.compact_dict() for row in random],
        },
    }
    if policy_artifacts is not None:
        unsealed["policy_artifacts"] = policy_artifacts
    return seal_screen_report(
        unsealed,
        expected_protocol_fingerprint=protocol_fingerprint,
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
    )


def test_m138_confirmation_rejects_unpromoted_or_tampered_screen_before_any_confirmation_access(
    tmp_path, monkeypatch
) -> None:
    protocol_fingerprint = m138.m138_protocol_fingerprint()
    ledger_path = tmp_path / "canonical-ledger.json"
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.open_partition("screen_fit")
    ledger.open_partition("screen_probe")
    before = ledger.snapshot()

    def forbidden_reset(*args, **kwargs):
        del args, kwargs
        raise AssertionError("confirmation reset happened before screen verification")

    def forbidden_train(*args, **kwargs):
        del args, kwargs
        raise AssertionError("confirmation fitting happened before screen verification")

    monkeypatch.setattr(m138.EcosystemEnv, "reset", forbidden_reset)
    monkeypatch.setattr(m138, "_train_paired", forbidden_train)

    unpromoted = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=False,
    )
    with pytest.raises(ScreenReportVerificationError, match="did not promote"):
        m138.m138_confirm(
            screen_report=unpromoted,
            artifact_dir=tmp_path / "unpromoted-artifacts",
            ledger_path=ledger_path,
        )

    tampered = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=True,
    )
    tampered["episode_summaries"]["candidate"][0]["decision_safe_fraction"] = 0.0
    with pytest.raises(ScreenReportVerificationError, match="hash"):
        m138.m138_confirm(
            screen_report=tampered,
            artifact_dir=tmp_path / "tampered-artifacts",
            ledger_path=ledger_path,
        )

    wrong_protocol = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=True,
    )
    wrong_protocol["protocol_fingerprint"] = "forged"
    wrong_protocol["content_hash"] = m138_content_hash(
        {key: value for key, value in wrong_protocol.items() if key != "content_hash"}
    )
    with pytest.raises(ScreenReportVerificationError, match="protocol"):
        m138.m138_confirm(
            screen_report=wrong_protocol,
            artifact_dir=tmp_path / "wrong-protocol-artifacts",
            ledger_path=ledger_path,
        )

    assert ledger.snapshot() == before
    assert not (tmp_path / "unpromoted-artifacts").exists()
    assert not (tmp_path / "tampered-artifacts").exists()
    assert not (tmp_path / "wrong-protocol-artifacts").exists()


def test_m138_confirmation_rejects_missing_or_tampered_screen_policy_files_before_split_access(
    tmp_path, monkeypatch
) -> None:
    protocol_fingerprint = m138.m138_protocol_fingerprint()

    def forbidden_access(*args, **kwargs):
        del args, kwargs
        raise AssertionError("confirmation accessed a split before artifact verification")

    monkeypatch.setattr(m138, "SplitOpenLedger", forbidden_access)
    monkeypatch.setattr(m138.EcosystemEnv, "reset", forbidden_access)
    monkeypatch.setattr(m138, "_train_paired", forbidden_access)

    missing = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=True,
    )
    with pytest.raises(ValueError, match="incomplete policy artifact"):
        m138.m138_confirm(
            screen_report=missing,
            artifact_dir=tmp_path / "missing-artifacts",
            ledger_path=tmp_path / "ledger.json",
        )

    policies, _, _ = _fake_training_result(
        stage="screen", seeds=M138_SCREEN_TRAINING_SEEDS
    )
    policy_artifacts, _ = m138._serialize_all_policies(
        policies,
        seeds=M138_SCREEN_TRAINING_SEEDS,
        policy_dir=tmp_path / "screen-policies",
    )
    tampered = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=True,
        policy_artifacts=policy_artifacts,
    )
    first_path = policy_artifacts[str(M138_SCREEN_TRAINING_SEEDS[0])]["candidate"]["path"]
    with open(first_path, "a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        m138.m138_confirm(
            screen_report=tampered,
            artifact_dir=tmp_path / "tampered-policy-artifacts",
            ledger_path=tmp_path / "ledger.json",
        )


def _fake_training_result(
    *, stage: str, seeds: tuple[int, ...]
) -> tuple[dict[tuple[str, int], CompactM138QPolicy], dict[tuple[str, int], dict[str, float]], dict[str, object]]:
    policies: dict[tuple[str, int], CompactM138QPolicy] = {}
    diagnostics: dict[tuple[str, int], dict[str, float]] = {}
    for seed in seeds:
        for arm in ("candidate", "legacy_control"):
            policies[(arm, seed)] = CompactM138QPolicy(arm=arm, seed=seed, stage=stage)
            diagnostics[(arm, seed)] = {"episodes": 0.0}
    return policies, diagnostics, {"workers": 8, "submission_order": [], "completion_order": []}


def test_m138_screen_serializes_all_four_then_opens_probe_and_scores_reloaded_objects(
    tmp_path, monkeypatch
) -> None:
    candidate, control, random = _screen_evidence()
    evidence = {
        "candidate": candidate,
        "legacy_control": control,
        "random": random,
    }
    artifacts = tmp_path / "screen-artifacts"
    phase: dict[str, object] = {"serialized": False, "restored": set(), "learned_evaluations": 0}
    events: list[str] = []

    class PhaseLedger:
        def __init__(self, path, *, protocol_fingerprint) -> None:
            del path
            self.protocol_fingerprint = protocol_fingerprint

        def open_partition(self, name: str) -> dict[str, object]:
            if name == "screen_probe":
                assert phase["serialized"] is True
                assert len(list((artifacts / "policies").glob("seed-*/*.json"))) == 4
                assert len(phase["restored"]) == 4
            events.append(name)
            return self.snapshot()

        def snapshot(self) -> dict[str, object]:
            return {
                "protocol_fingerprint": self.protocol_fingerprint,
                "opened_partitions": list(events),
            }

    original_serialize = m138._serialize_all_policies

    def serialize_spy(policies, *, seeds, policy_dir):
        result = original_serialize(policies, seeds=seeds, policy_dir=policy_dir)
        phase["serialized"] = True
        phase["restored"] = {id(policy) for policy in result[1].values()}
        return result

    def evaluate_spy(
        policy,
        *,
        arm,
        training_seed,
        conditions,
        seeds,
        trace_dir,
        label,
        policy_fingerprint,
        replay_mode,
    ):
        del seeds, trace_dir, label, policy_fingerprint, replay_mode
        key = "random" if isinstance(policy, m138.SeededRandomM138Policy) else arm
        if key != "random":
            assert id(policy) in phase["restored"]
            phase["learned_evaluations"] = int(phase["learned_evaluations"]) + 1
        rows = [
            row.compact_dict()
            for row in evidence[key]
            if row.training_seed == training_seed
        ]
        return rows, {condition: {"episodes": 8} for condition in conditions}, []

    monkeypatch.setattr(m138, "SplitOpenLedger", PhaseLedger)
    monkeypatch.setattr(m138, "_preflight", lambda **kwargs: ({"passes": True}, []))
    monkeypatch.setattr(
        m138,
        "_train_paired",
        lambda **kwargs: _fake_training_result(
            stage="screen", seeds=M138_SCREEN_TRAINING_SEEDS
        ),
    )
    monkeypatch.setattr(m138, "_serialize_all_policies", serialize_spy)
    monkeypatch.setattr(m138, "_evaluate_policy", evaluate_spy)

    report = m138.m138_screen(
        artifact_dir=artifacts,
        ledger_path=tmp_path / "ignored-ledger.json",
    )
    assert events == ["screen_fit", "screen_probe"]
    assert phase["learned_evaluations"] == 4
    assert report["screen"]["status"] == "promote"
    assert verify_screen_report(
        report,
        expected_protocol_fingerprint=report["protocol_fingerprint"],
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
    )["passes"] is True
    assert_compact_report(report)


def test_m138_confirmation_serializes_all_sixteen_before_evaluation_and_uses_reloads(
    tmp_path, monkeypatch
) -> None:
    protocol_fingerprint = m138.m138_protocol_fingerprint()
    screen_report = _sealed_screen_report(
        protocol_fingerprint=protocol_fingerprint,
        promotes=True,
    )
    candidate, control, random = _confirmation_evidence()
    evidence = {
        "candidate": candidate,
        "legacy_control": control,
        "random": random,
    }
    artifacts = tmp_path / "confirmation-artifacts"
    phase: dict[str, object] = {"serialized": False, "restored": set(), "learned_evaluations": 0}
    screen_artifacts_verified = {"value": False}
    events: list[str] = []

    class PhaseLedger:
        def __init__(self, path, *, protocol_fingerprint) -> None:
            del path
            assert screen_artifacts_verified["value"] is True
            self.protocol_fingerprint = protocol_fingerprint

        def open_partition(self, name: str) -> dict[str, object]:
            if name == "confirmation_evaluation":
                assert phase["serialized"] is True
                assert len(list((artifacts / "policies").glob("seed-*/*.json"))) == 16
                assert len(phase["restored"]) == 16
            events.append(name)
            return self.snapshot()

        def snapshot(self) -> dict[str, object]:
            return {
                "protocol_fingerprint": self.protocol_fingerprint,
                "opened_partitions": list(events),
            }

    original_serialize = m138._serialize_all_policies

    def serialize_spy(policies, *, seeds, policy_dir):
        result = original_serialize(policies, seeds=seeds, policy_dir=policy_dir)
        phase["serialized"] = True
        phase["restored"] = {id(policy) for policy in result[1].values()}
        return result

    def evaluate_spy(
        policy,
        *,
        arm,
        training_seed,
        conditions,
        seeds,
        trace_dir,
        label,
        policy_fingerprint,
        replay_mode,
    ):
        del seeds, trace_dir, label, policy_fingerprint, replay_mode
        key = "random" if isinstance(policy, m138.SeededRandomM138Policy) else arm
        if key != "random":
            assert id(policy) in phase["restored"]
            phase["learned_evaluations"] = int(phase["learned_evaluations"]) + 1
        rows = [
            row.compact_dict()
            for row in evidence[key]
            if row.training_seed == training_seed
        ]
        return rows, {condition: {"episodes": 20} for condition in conditions}, []

    monkeypatch.setattr(m138, "SplitOpenLedger", PhaseLedger)
    monkeypatch.setattr(
        m138,
        "_verify_screen_policy_artifacts",
        lambda report: screen_artifacts_verified.__setitem__("value", True),
    )
    monkeypatch.setattr(
        m138,
        "_train_paired",
        lambda **kwargs: _fake_training_result(
            stage="confirmation", seeds=M138_CONFIRMATION_TRAINING_SEEDS
        ),
    )
    monkeypatch.setattr(m138, "_serialize_all_policies", serialize_spy)
    monkeypatch.setattr(m138, "_evaluate_policy", evaluate_spy)

    report = m138.m138_confirm(
        screen_report=screen_report,
        artifact_dir=artifacts,
        ledger_path=tmp_path / "ignored-ledger.json",
    )
    assert events == ["confirmation_fit", "confirmation_evaluation"]
    assert phase["learned_evaluations"] == 16
    assert report["confirmation"]["status"] == "pass"
    assert report["content_hash"] == m138_content_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    assert_compact_report(report)


def test_m138_report_writers_refuse_existing_output_or_artifact_targets_before_running(
    tmp_path, monkeypatch
) -> None:
    def must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("writer entered experiment after detecting an existing target")

    monkeypatch.setattr(m138, "m138_screen", must_not_run)
    screen_output = tmp_path / "screen.json"
    screen_output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        m138.write_m138_screen_report(screen_output, ledger_path=tmp_path / "ledger.json")

    screen_output.unlink()
    (tmp_path / "screen-artifacts").mkdir()
    with pytest.raises(FileExistsError):
        m138.write_m138_screen_report(screen_output, ledger_path=tmp_path / "ledger.json")

    monkeypatch.setattr(m138, "m138_confirm", must_not_run)
    confirmation_output = tmp_path / "confirmation.json"
    confirmation_output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        m138.write_m138_confirmation_report(
            confirmation_output,
            screen_report=tmp_path / "screen-source.json",
            ledger_path=tmp_path / "ledger.json",
        )

    confirmation_output.unlink()
    (tmp_path / "confirmation-artifacts").mkdir()
    with pytest.raises(FileExistsError):
        m138.write_m138_confirmation_report(
            confirmation_output,
            screen_report=tmp_path / "screen-source.json",
            ledger_path=tmp_path / "ledger.json",
        )


def test_m138_wrong_worker_count_rejects_before_fingerprint_or_split_access(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("protocol or split touched before worker-count validation")

    monkeypatch.setattr(m138, "m138_protocol_fingerprint", forbidden)
    monkeypatch.setattr(m138, "SplitOpenLedger", forbidden)
    with pytest.raises(ValueError, match="worker count"):
        m138.m138_screen(
            artifact_dir=tmp_path / "screen-artifacts",
            ledger_path=tmp_path / "ledger.json",
            workers=7,
        )
    with pytest.raises(ValueError, match="worker count"):
        m138.m138_confirm(
            screen_report={},
            artifact_dir=tmp_path / "confirmation-artifacts",
            ledger_path=tmp_path / "ledger.json",
            workers=7,
        )

def test_m138_cli_routes_screen_and_confirmation_with_required_paths(tmp_path, monkeypatch, capsys) -> None:
    from ecosystem_gym.experiments import cli

    calls: list[tuple[str, object, object | None]] = []

    def fake_screen(path):
        calls.append(("screen", path, None))
        return {"screen": {"status": "reject"}}

    def fake_confirmation(path, *, screen_report):
        calls.append(("confirmation", path, screen_report))
        return {"confirmation": {"status": "fail"}}

    monkeypatch.setattr(cli, "write_m138_screen_report", fake_screen)
    monkeypatch.setattr(cli, "write_m138_confirmation_report", fake_confirmation)
    screen_output = tmp_path / "screen.json"
    confirmation_output = tmp_path / "confirmation.json"
    cli.main(["m138-screen", "--output", str(screen_output)])
    cli.main(
        [
            "m138-confirm",
            "--screen-report",
            str(screen_output),
            "--output",
            str(confirmation_output),
        ]
    )
    assert calls == [
        ("screen", screen_output, None),
        ("confirmation", confirmation_output, screen_output),
    ]
    assert '"status": "reject"' in capsys.readouterr().out
