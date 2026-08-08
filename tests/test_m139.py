from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ecosystem_gym.actions import ActionKind, ActionOutcome
from ecosystem_gym.env import EcosystemEnv
from ecosystem_gym.experiments import cli, m139
from ecosystem_gym.experiments.m13 import M13Macro
from ecosystem_gym.experiments.m132 import M132_FEATURE_DIM
from ecosystem_gym.experiments.m138 import M138RewardState, m138_learning_reward
from ecosystem_gym.experiments.m139 import (
    M139_AUDIT_CONDITIONS,
    M139_AUDIT_SEEDS,
    M139_CONFIRMATION_ARMS,
    M139_CONFIRMATION_EVALUATION_CONDITIONS,
    M139_CONFIRMATION_EVALUATION_SEEDS,
    M139_CONFIRMATION_FIT_CONDITIONS,
    M139_CONFIRMATION_FIT_SEEDS,
    M139_CONFIRMATION_TRAINING_SEEDS,
    M139_SCREEN_ARMS,
    M139_SCREEN_EPISODES,
    M139_SCREEN_FIT_CONDITIONS,
    M139_SCREEN_FIT_SEEDS,
    M139_SCREEN_PROBE_CONDITIONS,
    M139_SCREEN_PROBE_SEEDS,
    M139_SCREEN_TRAINING_SEEDS,
    M139_WORKERS,
    CompactM139QPolicy,
    M139RewardState,
    m139_backup_targets,
    m139_duration_discount,
    m139_learning_reward,
    m139_potential,
    m139_protocol_fingerprint,
    m139_safe_error,
)
from ecosystem_gym.experiments.m139_support import (
    SplitAlreadyOpenedError,
    SplitDependencyError,
    SplitLedgerError,
    SplitOpenLedger,
    compact_episode_summary,
    m139_confirmation_passes,
    m139_screen_promotes,
)
from ecosystem_gym.tasks import LAYOUTS


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
    feed_cycles: int = 0,
    play_cycles: int = 0,
    rest_cycles: int = 0,
    survived: bool = True,
) -> dict[str, object]:
    return {
        "outcome": outcome.value,
        "disturbance": disturbance,
        "resource_event": None,
        "feed_cycles": feed_cycles,
        "play_cycles": play_cycles,
        "rest_cycles": rest_cycles,
        "survived": survived,
        "task_success": False,
        "maintenance_complete": False,
    }


def _reward_step(
    arm: m139.Arm,
    state: M139RewardState,
    *,
    before: tuple[float, float, float] = (0.50, 0.50, 0.50),
    after: tuple[float, float, float] = (0.50, 0.50, 0.50),
    duration: float = 1.0,
    kind: ActionKind = ActionKind.IDLE,
    info: dict[str, object] | None = None,
    terminated: bool = False,
    truncated: bool = False,
    environment_reward: float = 123.0,
):
    macro = {
        ActionKind.PICK_UP: M13Macro.PICK_UP,
        ActionKind.CONSUME: M13Macro.CONSUME,
        ActionKind.RUN_AROUND: M13Macro.PLAY,
        ActionKind.REST: M13Macro.REST,
        ActionKind.IDLE: M13Macro.WAIT,
    }.get(kind, M13Macro.GO_FOOD)
    return m139_learning_reward(
        arm=arm,
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
    ("drives", "error"),
    [
        ((0.30, 0.30, 0.70), 0.0),
        ((0.225, 0.30, 0.70), 0.25),
        ((0.30, 0.15, 0.70), 1.0),
        ((0.30, 0.30, 0.90), 1.0),
        ((-10.0, 10.0, -10.0), 1.0),
    ],
)
def test_m139_potential_boundaries_and_clipping(drives: tuple[float, float, float], error: float) -> None:
    values = np.asarray(drives, dtype=np.float32)
    assert m139_safe_error(values) == pytest.approx(error)
    assert m139_potential(values) == pytest.approx(-error)

    rng = np.random.default_rng(139)
    errors = [m139_safe_error(row) for row in rng.uniform(-2.0, 3.0, size=(256, 3))]
    assert min(errors) >= 0.0
    assert max(errors) <= 1.0


def test_m139_wait_and_restoration_have_declared_signs() -> None:
    comfort = (0.60, 0.60, 0.30)
    unsafe = (0.15, 0.60, 0.30)
    reward, components = _reward_step(
        "public_potential_candidate", M139RewardState(False), before=comfort, after=comfort
    )
    assert components["potential_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(-0.01)
    _, worsening = _reward_step(
        "public_potential_candidate", M139RewardState(False), before=comfort, after=unsafe
    )
    _, restoration = _reward_step(
        "public_potential_candidate", M139RewardState(False), before=unsafe, after=comfort
    )
    assert float(worsening["potential_reward"]) < 0.0
    assert float(restoration["potential_reward"]) > 0.0


def test_m139_discounted_potential_telescopes_over_variable_durations() -> None:
    rng = np.random.default_rng(20260909)
    for _ in range(100):
        count = int(rng.integers(2, 12))
        drives = rng.uniform(0.0, 1.0, size=(count + 1, 3))
        durations = rng.uniform(0.1, 2.0, size=count)
        elapsed = total = 0.0
        for index, duration in enumerate(durations):
            term = m139_duration_discount(duration) * m139_potential(drives[index + 1]) - m139_potential(
                drives[index]
            )
            total += 0.99**elapsed * term
            elapsed += duration
        endpoint = 0.99**elapsed * m139_potential(drives[-1]) - m139_potential(drives[0])
        assert total == pytest.approx(endpoint, abs=1e-12)

    # Unsafe -> comfort -> unsafe has a positive bounded endpoint term under
    # discounting, but no residual beyond that exact telescoping endpoint.
    unsafe, comfort = np.asarray((0.15, 0.6, 0.3)), np.asarray((0.6, 0.6, 0.3))
    terms = [
        0.99 * m139_potential(comfort) - m139_potential(unsafe),
        0.99 * m139_potential(unsafe) - m139_potential(comfort),
    ]
    assert terms[0] + 0.99 * terms[1] == pytest.approx(0.99**2 * -1.0 - -1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_m139_duration_discount_rejects_nonpositive_or_nonfinite(bad: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        m139_duration_discount(bad)


def test_m139_all_reward_arms_recompose_and_static_matches_m138() -> None:
    kwargs = {
        "environment_reward": 7.25,
        "observation_before": _observation((0.20, 0.50, 0.50)),
        "observation_after": _observation((0.50, 0.50, 0.50)),
        "action": _action(1.25, ActionKind.CONSUME),
        "info": _info(feed_cycles=1),
        "terminated": False,
        "truncated": False,
        "macro": M13Macro.CONSUME,
    }
    candidate, candidate_components = m139_learning_reward(
        arm="public_potential_candidate", state=M139RewardState(False), **kwargs
    )
    static, static_components = m139_learning_reward(
        arm="static_cost_control", state=M139RewardState(False), **kwargs
    )
    legacy, legacy_components = m139_learning_reward(
        arm="legacy_guardrail", state=M139RewardState(False), **kwargs
    )
    m138_static, _ = m138_learning_reward(arm="candidate", state=M138RewardState(False), **kwargs)
    assert candidate == pytest.approx(candidate_components["candidate_recomposed"])
    assert static == pytest.approx(static_components["static_cost_recomposed"])
    assert static == pytest.approx(m138_static)
    assert legacy == pytest.approx(7.25) == pytest.approx(legacy_components["environment_reward"])


def test_m139_terminal_potential_uses_post_drives_without_bootstrap() -> None:
    state = M139RewardState(False, feed_cycles=3, play_cycles=3, rest_cycles=2)
    state.decision_steps = state.decision_safe_steps = 199
    state.total_duration = state.duration_safe_seconds = 199.0
    reward, components = _reward_step(
        "public_potential_candidate",
        state,
        before=(0.15, 0.5, 0.5),
        after=(0.6, 0.6, 0.3),
        truncated=True,
        info=_info(feed_cycles=3, play_cycles=3, rest_cycles=2, survived=True),
    )
    assert components["phi_after"] == pytest.approx(0.0)
    assert components["potential_reward"] == pytest.approx(1.0)
    assert components["full_terminal_success"] is True
    assert reward == pytest.approx(components["candidate_recomposed"])


def test_m139_duration_aware_backup_is_shared_and_terminal_zeroes_only_bootstrap() -> None:
    rewards = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
    durations = np.asarray((0.1, 1.0, 2.0), dtype=np.float32)
    done = np.asarray((False, False, True))
    bootstrap = np.asarray((10.0, 10.0, 10.0), dtype=np.float32)
    targets = m139_backup_targets(rewards, durations, done, bootstrap)
    assert targets[0] == pytest.approx(1.0 + 0.99**0.1 * 10.0)
    assert targets[1] == pytest.approx(1.0 + 0.99 * 10.0)
    assert targets[2] == pytest.approx(1.0)
    for arm in M139_SCREEN_ARMS:
        assert (
            CompactM139QPolicy(arm=arm, seed=1, stage="screen")._update.__func__ is CompactM139QPolicy._update
        )


def test_m139_policy_boundary_is_exactly_public_30_features() -> None:
    policy = CompactM139QPolicy(
        arm="public_potential_candidate", seed=M139_SCREEN_TRAINING_SEEDS[0], stage="screen"
    )
    env = EcosystemEnv(m139.m139_config())
    try:
        observation, _ = env.reset(
            seed=M139_SCREEN_FIT_SEEDS[0],
            options={
                "task_id": "persistent_maintenance",
                **M139_SCREEN_FIT_CONDITIONS["persistent_reference"],
            },
        )
        memory = policy.reset()
        features = policy.features(observation, memory)
        mask = policy.mask(observation)
        macro = policy.choose(observation, memory)
        injected = {
            **observation,
            "task_id": "private",
            "info": {"feed_cycles": 999},
            "timer": 999,
            "food_available": False,
            "reward_state": M139RewardState(True, feed_cycles=999),
        }
        assert features.shape == (M132_FEATURE_DIM,) == (30,)
        assert np.array_equal(features, policy.features(injected, memory))
        assert np.array_equal(mask, policy.mask(injected))
        assert macro is policy.choose(injected, memory)
    finally:
        env.close()
    for method in (policy.features, policy.mask, policy.choose, policy.observe):
        assert "info" not in inspect.signature(method).parameters


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


def test_m139_splits_budgets_arms_workers_and_layouts_are_fresh() -> None:
    assert M139_SCREEN_FIT_SEEDS == tuple(range(5200, 5220))
    assert M139_SCREEN_PROBE_SEEDS == tuple(range(5220, 5228))
    assert M139_CONFIRMATION_FIT_SEEDS == tuple(range(5300, 5340))
    assert M139_CONFIRMATION_EVALUATION_SEEDS == tuple(range(5400, 5420))
    assert M139_AUDIT_SEEDS == tuple(range(5500, 5520))
    assert M139_SCREEN_EPISODES == 4 * 20 * 25 == 2000
    assert M139_WORKERS == 6
    assert M139_SCREEN_ARMS == ("public_potential_candidate", "static_cost_control", "legacy_guardrail")
    assert M139_CONFIRMATION_ARMS == ("public_potential_candidate", "static_cost_control")
    assert len(m139.m139_screen_jobs()) == len(set(m139.m139_screen_jobs())) == 6

    groups = (
        (M139_SCREEN_FIT_CONDITIONS, "m139_screen_"),
        (M139_SCREEN_PROBE_CONDITIONS, "m139_probe_"),
        (M139_CONFIRMATION_FIT_CONDITIONS, "m139_confirm_fit_"),
        (M139_CONFIRMATION_EVALUATION_CONDITIONS, "m139_confirm_eval_"),
        (M139_AUDIT_CONDITIONS, "m139_audit_"),
    )
    names = [
        str(row["layout_id"])
        for conditions, prefix in groups
        for row in conditions.values()
        if str(row["layout_id"]).startswith(prefix)
    ]
    assert len(names) == len(set(names)) == 20
    geometries = {_layout_geometry(name) for name in names}
    old = {
        _layout_geometry(name) for name in LAYOUTS if name not in names and name.startswith(("m10", "m13"))
    }
    assert len(geometries) == 20 and geometries.isdisjoint(old)
    points = [point for name in names for point in _layout_points(name)]
    old_points = {
        point
        for name in LAYOUTS
        if name not in names and name.startswith(("m10", "m13"))
        for point in _layout_points(name)
    }
    assert len(points) == len(set(points)) == 100
    assert set(points).isdisjoint(old_points)


def test_m139_fingerprint_covers_budget_and_layout(monkeypatch) -> None:
    fingerprint = m139_protocol_fingerprint()
    assert len(fingerprint) == 64
    monkeypatch.setattr(m139, "M139_SCREEN_EPISODES", M139_SCREEN_EPISODES + 1)
    assert m139_protocol_fingerprint() != fingerprint


def test_m139_policy_artifact_roundtrip_refuses_overwrite_and_tamper(tmp_path: Path) -> None:
    policy = CompactM139QPolicy(arm="static_cost_control", seed=20260911, stage="screen")
    path = tmp_path / "policy.json"
    artifact = m139.write_m139_policy(path, policy)
    restored = m139.load_m139_policy(path)
    assert m139.m139_policy_fingerprint(restored) == artifact["policy_fingerprint"]
    with pytest.raises(FileExistsError):
        m139.write_m139_policy(path, policy)
    payload = json.loads(path.read_text())
    payload["arm"] = "legacy_guardrail"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initial|invalid|fingerprint"):
        m139.load_m139_policy(tampered)


def test_m139_strict_replay_rejects_potential_or_duration_tamper(tmp_path: Path) -> None:
    policy = CompactM139QPolicy(arm="public_potential_candidate", seed=20260911, stage="screen")
    trace = tmp_path / "episode.jsonl"
    episode = m139.run_m139_episode(
        policy,
        arm="public_potential_candidate",
        training_seed=20260911,
        seed=5200,
        condition="persistent_reference",
        controls=M139_SCREEN_FIT_CONDITIONS["persistent_reference"],
        trace_path=trace,
        policy_label="test",
        policy_fingerprint=m139.m139_policy_fingerprint(policy),
    )
    assert m139.replay_m139_trace(trace, policy).steps > 0
    assert m139._compact_summary_from_trace(trace, policy) == compact_episode_summary(episode.gate_dict())
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[1]["reward_components"]["duration_discount"] += 0.01
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    with pytest.raises(ValueError, match="reward_components|differs"):
        m139.replay_m139_trace(tampered, policy)


def test_m139_split_ledger_hashes_markers_and_only_fit_is_idempotent(tmp_path: Path) -> None:
    ledger = SplitOpenLedger(tmp_path / "ledger.json", protocol_fingerprint="frozen")
    with pytest.raises(SplitDependencyError):
        ledger.open_partition("confirmation_fit")
    opened = ledger.ensure_screen_fit_open()
    assert opened["opened_partitions"] == ["screen_fit"]
    assert len(opened["marker_hashes"]["screen_fit"]) == 64
    assert ledger.ensure_screen_fit_open()["opened_partitions"] == ["screen_fit"]
    with pytest.raises(SplitAlreadyOpenedError):
        ledger.open_partition("screen_fit")
    ledger.open_partition("screen_probe")
    with pytest.raises(SplitLedgerError, match="unavailable"):
        ledger.ensure_screen_fit_open()
    with pytest.raises(SplitLedgerError, match="fingerprint"):
        SplitOpenLedger(tmp_path / "ledger.json", protocol_fingerprint="forged").snapshot()


def _summary(
    *,
    training_seed: int,
    condition: str,
    env_seed: int,
    survived: bool = True,
    maintenance: bool = True,
    full: bool = True,
    safe: float = 0.90,
    waits: int = 10,
    unsafe_waits: int = 0,
    integrity: int = 0,
) -> dict[str, object]:
    recovery_required = condition in {"event_relocation", "compound"}
    return {
        "training_seed": training_seed,
        "condition": condition,
        "env_seed": env_seed,
        "survived": survived,
        "maintenance_complete": maintenance,
        "full_gate_success": full,
        "decision_safe_fraction": safe,
        "duration_safe_fraction": safe,
        "recovery_required": recovery_required,
        "recovery_complete": recovery_required and survived,
        "decision_steps": 200,
        "wait_decisions": waits,
        "unsafe_wait_decisions": unsafe_waits,
        "unsafe_wait_fraction": unsafe_waits / waits if waits else 0.0,
        "conformance_violations": integrity,
        "replay_violations": 0,
    }


def _matrix(seeds, env_seeds, **overrides):
    return [
        _summary(training_seed=training_seed, condition=condition, env_seed=env_seed, **overrides)
        for training_seed in seeds
        for condition in M139_SCREEN_PROBE_CONDITIONS
        for env_seed in env_seeds
    ]


def test_m139_screen_gate_enforces_static_legacy_random_and_unsafe_wait_rules() -> None:
    candidate = _matrix(M139_SCREEN_TRAINING_SEEDS, M139_SCREEN_PROBE_SEEDS)
    static = _matrix(
        M139_SCREEN_TRAINING_SEEDS, M139_SCREEN_PROBE_SEEDS, full=False, safe=0.75, unsafe_waits=4
    )
    legacy = _matrix(M139_SCREEN_TRAINING_SEEDS, M139_SCREEN_PROBE_SEEDS)
    random = _matrix(
        M139_SCREEN_TRAINING_SEEDS,
        M139_SCREEN_PROBE_SEEDS,
        survived=False,
        maintenance=False,
        full=False,
        safe=0.20,
    )
    gate = m139_screen_promotes(
        candidate,
        static,
        legacy,
        random,
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
        env_seeds=M139_SCREEN_PROBE_SEEDS,
    )
    assert gate["passes"] is True
    broken = deepcopy(candidate)
    for row in broken[: len(M139_SCREEN_PROBE_SEEDS)]:
        row["unsafe_wait_decisions"] = 2
        row["unsafe_wait_fraction"] = 0.2
    assert (
        m139_screen_promotes(
            broken,
            static,
            legacy,
            random,
            training_seeds=M139_SCREEN_TRAINING_SEEDS,
            conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
            env_seeds=M139_SCREEN_PROBE_SEEDS,
        )["passes"]
        is False
    )


def test_m139_confirmation_gate_requires_all_eight_and_seven_margin_wins() -> None:
    candidate = _matrix(M139_CONFIRMATION_TRAINING_SEEDS, M139_CONFIRMATION_EVALUATION_SEEDS, safe=0.90)
    static = _matrix(
        M139_CONFIRMATION_TRAINING_SEEDS, M139_CONFIRMATION_EVALUATION_SEEDS, safe=0.86, unsafe_waits=1
    )
    random = _matrix(
        M139_CONFIRMATION_TRAINING_SEEDS,
        M139_CONFIRMATION_EVALUATION_SEEDS,
        survived=False,
        maintenance=False,
        full=False,
        safe=0.20,
    )
    gate = m139_confirmation_passes(
        candidate,
        static,
        random,
        training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M139_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
    )
    assert gate["passes"] is True and gate["paired_margin_wins"] == 8
    candidate[0]["conformance_violations"] = 1
    assert (
        m139_confirmation_passes(
            candidate,
            static,
            random,
            training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
            conditions=tuple(M139_CONFIRMATION_EVALUATION_CONDITIONS),
            env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
        )["passes"]
        is False
    )


def test_m139_confirmation_rejects_cross_arm_recovery_mismatch() -> None:
    candidate = _matrix(
        M139_CONFIRMATION_TRAINING_SEEDS,
        M139_CONFIRMATION_EVALUATION_SEEDS,
    )
    static = deepcopy(candidate)
    random = deepcopy(candidate)
    relocation = next(row for row in static if row["condition"] == "event_relocation")
    relocation["recovery_required"] = False
    relocation["recovery_complete"] = False
    with pytest.raises(ValueError, match="paired arms disagree"):
        m139_confirmation_passes(
            candidate,
            static,
            random,
            training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
            conditions=tuple(M139_CONFIRMATION_EVALUATION_CONDITIONS),
            env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
        )


def test_m139_trace_evidence_rejects_forged_rehashed_summary(tmp_path: Path) -> None:
    seed = M139_SCREEN_TRAINING_SEEDS[0]
    arm = "public_potential_candidate"
    condition = "persistent_reference"
    env_seed = M139_SCREEN_FIT_SEEDS[0]
    policy = CompactM139QPolicy(arm=arm, seed=seed, stage="screen")
    trace = tmp_path / "trace.jsonl"
    episode = m139.run_m139_episode(
        policy,
        arm=arm,
        training_seed=seed,
        seed=env_seed,
        condition=condition,
        controls=M139_SCREEN_FIT_CONDITIONS[condition],
        trace_path=trace,
        policy_label=f"seed-{seed}/{arm}",
        policy_fingerprint=m139.m139_policy_fingerprint(policy),
    )
    forged = episode.gate_dict()
    forged["decision_safe_fraction"] = 0.0 if float(forged["decision_safe_fraction"]) > 0.5 else 1.0
    report = {
        "episode_summaries": {arm: [forged]},
        "trace_manifest": [
            {
                "policy": f"seed-{seed}/{arm}",
                "condition": condition,
                "env_seed": env_seed,
                "path": str(trace),
                "sha256": m139._file_sha256(trace),
            }
        ],
    }
    report["content_hash"] = m139._report_hash(report)
    with pytest.raises(ValueError, match="not derived from trace"):
        m139._verify_report_trace_evidence(
            report,
            learned_policies={(arm, seed): policy},
            arms=(arm,),
            training_seeds=(seed,),
            conditions={condition: M139_SCREEN_FIT_CONDITIONS[condition]},
            env_seeds=(env_seed,),
        )


def test_m139_confirmation_audit_rejects_bogus_screen_authorization() -> None:
    payload = {
        "schema_version": "m13.9-confirmation-report-r1",
        "stage": "confirmation",
        "protocol_fingerprint": m139_protocol_fingerprint(),
        "screen_authorization": {
            "path": "/bogus/screen.json",
            "file_sha256": "0" * 64,
            "content_hash": "1" * 64,
            "trusted_promotion": {"passes": False},
        },
    }
    payload["content_hash"] = m139._report_hash(payload)
    with pytest.raises(ValueError, match="originating screen"):
        m139._verify_confirmation_report(payload)


def test_m139_compact_summary_rejects_inconsistent_unsafe_wait_fraction() -> None:
    value = _summary(training_seed=1, condition="persistent_reference", env_seed=1)
    assert compact_episode_summary(value)["unsafe_wait_fraction"] == 0.0
    value["unsafe_wait_fraction"] = 0.5
    with pytest.raises(ValueError, match="unsafe_wait_fraction"):
        compact_episode_summary(value)


def test_m139_fit_smoke_is_structurally_fit_only(tmp_path: Path, monkeypatch) -> None:
    policies = {
        (arm, seed): CompactM139QPolicy(arm=arm, seed=seed, stage="fit_smoke")
        for seed in M139_SCREEN_TRAINING_SEEDS
        for arm in M139_SCREEN_ARMS
    }
    training = {(arm, seed): {"episodes": 16.0, "decisions": 1.0, "updates": 0.0} for arm, seed in policies}
    monkeypatch.setattr(
        m139,
        "_train_smoke_wave",
        lambda: (
            policies,
            training,
            {
                "workers": 6,
                "start_method": "spawn",
                "worker_threads": 1,
                "submission_order": [],
                "completion_order": [],
            },
        ),
    )
    monkeypatch.setattr(
        m139,
        "_serialize_all_policies",
        lambda *_args, **_kwargs: (
            {
                str(seed): {arm: {"path": "non-protocol"} for arm in M139_SCREEN_ARMS}
                for seed in M139_SCREEN_TRAINING_SEEDS
            },
            policies,
        ),
    )
    monkeypatch.setattr(
        m139,
        "_evaluate_policy",
        lambda *_args, **_kwargs: ([{"replay_violations": 0}], {"ok": True}, [{"replay_pass": True}]),
    )
    report = m139.m139_fit_smoke(artifact_dir=tmp_path / "smoke")
    assert report["promotional"] is False
    assert report["eligible_policy_artifacts"] is False
    assert report["opened_partitions"] == ["screen_fit"]
    assert report["forbidden_partitions_opened"] is False
    assert report["ledger_scope"] == "smoke-only-non-protocol"
    assert report["canonical_protocol_ledger_accessed"] is False
    assert (tmp_path / "smoke" / "split-open-ledger.non-protocol.json").is_file()


def test_m139_cli_routes_all_protocol_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "write_m139_fit_smoke_report",
        lambda path: {"label": "smoke", "elapsed_seconds": 1.0, "replay_pass": True},
    )
    monkeypatch.setattr(cli, "write_m139_screen_report", lambda path: {"screen": {"passes": False}})
    monkeypatch.setattr(
        cli, "write_m139_confirmation_report", lambda path, screen_report: {"confirmation": {"passes": False}}
    )
    monkeypatch.setattr(
        cli, "write_m139_audit_report", lambda path, confirmation_report: {"audit_result": {"passes": False}}
    )
    cli.main(["m139-fit-smoke", "--output", str(tmp_path / "smoke.json")])
    cli.main(["m139-screen", "--output", str(tmp_path / "screen.json")])
    cli.main(
        [
            "m139-confirm",
            "--screen-report",
            str(tmp_path / "screen.json"),
            "--output",
            str(tmp_path / "confirm.json"),
        ]
    )
    cli.main(
        [
            "m139-audit",
            "--confirmation-report",
            str(tmp_path / "confirm.json"),
            "--output",
            str(tmp_path / "audit.json"),
        ]
    )
    assert len(capsys.readouterr().out.splitlines()) == 4
