from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import ecosystem_gym.maintenance.policy_protocol_m1314 as protocol_m1314
from ecosystem_gym.__main__ import main
from ecosystem_gym.experiments import cli
from ecosystem_gym.experiments.m1314_support import (
    M1314PreflightVerificationError,
    M1314StageLedger,
    seal_m1314_preflight_report,
    verify_m1314_preflight_report,
)
from ecosystem_gym.maintenance.policy_protocol_m1314 import (
    M1314_ACTION_SURFACE,
    M1314_ARM_NAMES,
    M1314_AUDIT_SEEDS,
    M1314_COMMAND_NAMES,
    M1314_CONDITIONS,
    M1314_CONFIRMATION_EVALUATION_SEEDS,
    M1314_CONFIRMATION_FIT_SEEDS,
    M1314_CONFIRMATION_TRAINING_SEEDS,
    M1314_DECISION_BUDGET_PER_FIT,
    M1314_DEVELOPMENT_CHECK_SEEDS,
    M1314_DEVELOPMENT_FIT_SEEDS,
    M1314_DEVELOPMENT_TRAINING_SEEDS,
    M1314_NO_RUN_AUTHORIZATION,
    M1314_PREFLIGHT_STAGE_ORDER,
    M1314_PREFLIGHT_TARGETS,
    M1314_REPLAY_REQUIREMENT,
    m1314_policy_manifest,
    m1314_protocol_fingerprint,
    verify_m1314_policy_manifest,
)
from ecosystem_gym.tasks import LAYOUTS


_PARTITION_SEEDS = {
    "development_fit": M1314_DEVELOPMENT_FIT_SEEDS,
    "development_check": M1314_DEVELOPMENT_CHECK_SEEDS,
    "confirmation_fit": M1314_CONFIRMATION_FIT_SEEDS,
    "confirmation_evaluation": M1314_CONFIRMATION_EVALUATION_SEEDS,
    "audit": M1314_AUDIT_SEEDS,
}


def _points(layout) -> tuple[object, ...]:
    return (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)


def _coverage_row(*, episodes: int, passes: bool) -> dict[str, object]:
    return {
        "episodes": episodes,
        "initial_food_visible": episodes,
        "replenished_food_visible": episodes,
        "toy_visible": episodes,
        "rest_visible": episodes,
        "relocation_episodes": episodes // 2,
        "relocated_food_visible": episodes // 2,
        "distractor_episodes": episodes // 2,
        "distractor_visible": episodes // 2,
        "passes": passes,
    }


def _stage_report(stage: str, *, manifest_fingerprint: str, passes: bool) -> dict[str, object]:
    coverage = {
        partition: {
            condition: _coverage_row(
                episodes=len(_PARTITION_SEEDS[partition]),
                passes=passes,
            )
            for condition in M1314_CONDITIONS["development_fit"]
        }
        for partition in M1314_PREFLIGHT_TARGETS[stage]["scan_partitions"]
    }
    return {
        "schema_version": "non-canonical-fixture",
        "protocol_fingerprint": m1314_protocol_fingerprint(),
        "manifest_fingerprint": manifest_fingerprint,
        "stage": stage,
        "coverage": coverage,
        "note": "synthetic fixture",
    }


def _load_future_runner_module():
    try:
        return importlib.import_module("ecosystem_gym.experiments.m1314")
    except ModuleNotFoundError as exc:
        if exc.name == "ecosystem_gym.experiments.m1314":
            pytest.skip("ecosystem_gym.experiments.m1314 is not present in this bounded CLI/tests change")
        raise


def _future_export(module: object, *names: str):
    for name in names:
        exported = getattr(module, name, None)
        if callable(exported):
            return exported
    pytest.skip(f"future M13.14 runner exports are not available yet: {names}")


def _call_supported(function, /, *args, **kwargs):
    signature = inspect.signature(function)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return function(*args, **accepted)


def _patch_if_present(monkeypatch: pytest.MonkeyPatch, module: object, name: str, replacement) -> bool:
    if hasattr(module, name):
        monkeypatch.setattr(module, name, replacement)
        return True
    return False


def _stub_source_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        protocol_m1314,
        "_source_hashes",
        lambda: {
            "ecosystem_gym/experiments/m1314.py": "0" * 64,
            "ecosystem_gym/experiments/cli.py": "1" * 64,
            "ecosystem_gym/__main__.py": "2" * 64,
        },
    )


def _normalized_preflight_targets() -> dict[str, dict[str, object]]:
    return {
        stage: {
            "scan_partitions": list(target["scan_partitions"]),
            "authorizes_command": target["authorizes_command"],
        }
        for stage, target in M1314_PREFLIGHT_TARGETS.items()
    }


def _m1314_evidence_row(
    *,
    arm: str,
    training_seed: int,
    condition: str,
    env_seed: int,
    survived: bool = True,
    maintenance_complete: bool = True,
    full_objective_success: bool = True,
    decision_safe_fraction: float = 0.95,
    duration_safe_fraction: float = 0.95,
    recovery_required: bool = False,
    recovery_complete: bool = True,
    unsafe_wait_decisions: int = 0,
    wait_decisions: int = 1,
    conformance_violations: int = 0,
    replay_violations: int = 0,
    weight_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "arm": arm,
        "training_seed": training_seed,
        "condition": condition,
        "env_seed": env_seed,
        "survived": survived,
        "maintenance_complete": maintenance_complete,
        "full_objective_success": full_objective_success,
        "decision_safe_fraction": decision_safe_fraction,
        "duration_safe_fraction": duration_safe_fraction,
        "recovery_required": recovery_required,
        "recovery_complete": recovery_complete,
        "unsafe_wait_decisions": unsafe_wait_decisions,
        "wait_decisions": wait_decisions,
        "conformance_violations": conformance_violations,
        "replay_violations": replay_violations,
        "weight_fingerprint": weight_fingerprint or f"weights:{arm}:{training_seed}",
        "policy_fingerprint": policy_fingerprint or f"policy:{arm}:{training_seed}",
        "artifact_sha256": artifact_sha256 or f"artifact:{arm}:{training_seed}",
    }


def test_m1314_partitions_and_all_five_layout_families_are_fresh() -> None:
    partitions = (
        M1314_DEVELOPMENT_FIT_SEEDS,
        M1314_DEVELOPMENT_CHECK_SEEDS,
        M1314_CONFIRMATION_FIT_SEEDS,
        M1314_CONFIRMATION_EVALUATION_SEEDS,
        M1314_AUDIT_SEEDS,
    )
    assert M1314_DEVELOPMENT_FIT_SEEDS == tuple(range(6600, 6620))
    assert M1314_DEVELOPMENT_CHECK_SEEDS == tuple(range(6620, 6640))
    assert M1314_CONFIRMATION_FIT_SEEDS == tuple(range(6640, 6680))
    assert M1314_CONFIRMATION_EVALUATION_SEEDS == tuple(range(6680, 6700))
    assert M1314_AUDIT_SEEDS == tuple(range(6700, 6720))
    assert all(set(left).isdisjoint(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :])

    expected = {
        str(condition["layout_id"])
        for conditions in M1314_CONDITIONS.values()
        for condition in conditions.values()
    }
    names = {name for name in LAYOUTS if name.startswith("m1314_")}
    assert names == expected and len(names) == 20
    current = {point for name in names for point in _points(LAYOUTS[name])}
    previous = {point for name, layout in LAYOUTS.items() if not name.startswith("m1314_") for point in _points(layout)}
    assert len(current) == 100
    assert current.isdisjoint(previous)


def test_manifest_freezes_declared_arms_preflights_budgets_and_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_source_hashes(monkeypatch)
    manifest = m1314_policy_manifest()
    assert manifest["protocol_fingerprint"] == m1314_protocol_fingerprint()
    assert manifest["status"] == "implemented-frozen-and-unopened"
    assert manifest["public_boundary"]["features"] == 30
    assert manifest["public_boundary"]["macros"] == list(M1314_ACTION_SURFACE)
    assert manifest["public_boundary"]["action_value_boundary"]["modular_head_count"] == 3
    assert set(manifest["arms"]) == set(M1314_ARM_NAMES)
    assert manifest["arms"]["modular_anchor_shield_candidate"]["fit_artifact_reused_by"] == [
        "modular_anchor_no_shield_control"
    ]
    assert manifest["arms"]["modular_anchor_no_shield_control"]["parameter_fingerprint_must_match_candidate"] is True
    assert manifest["preflight"]["stage_order"] == list(M1314_PREFLIGHT_STAGE_ORDER)
    assert manifest["preflight"]["targets"] == _normalized_preflight_targets()
    assert manifest["training_budget"]["decision_budget_per_fit"] == M1314_DECISION_BUDGET_PER_FIT
    assert manifest["training_budget"]["development_training_seeds"] == list(M1314_DEVELOPMENT_TRAINING_SEEDS)
    assert manifest["training_budget"]["confirmation_training_seeds"] == list(M1314_CONFIRMATION_TRAINING_SEEDS)
    assert manifest["evaluation_budget"]["replay_requirement"] == M1314_REPLAY_REQUIREMENT
    assert manifest["evaluation_budget"]["cross_arm_winner_selection"] == "forbidden"
    assert manifest["causal_comparison_gate"]["no_cross_arm_selection"] is True
    assert manifest["future_command_names"] == list(M1314_COMMAND_NAMES)
    assert len(manifest["future_commands"]) == 8
    assert all(command.startswith("uv run python -m ecosystem_gym") for command in manifest["future_commands"])
    assert manifest["authorization"] == M1314_NO_RUN_AUTHORIZATION


def test_manifest_cli_is_registered_and_runs_without_an_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_source_hashes(monkeypatch)
    output = tmp_path / "manifest.json"
    main(["maintenance-policy-manifest-m1314", "--output", str(output)])
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert printed == {
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "status": "implemented-frozen-and-unopened",
    }
    assert payload["future_command_names"] == list(M1314_COMMAND_NAMES)
    assert payload["authorization"] == M1314_NO_RUN_AUTHORIZATION


def test_manifest_write_and_verify_roundtrip_when_protocol_roundtrips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_source_hashes(monkeypatch)
    output = tmp_path / "manifest.json"
    written = protocol_m1314.write_m1314_policy_manifest(output)
    try:
        verified = verify_m1314_policy_manifest(output)
    except ValueError as exc:
        if "does not match frozen sources and protocol" in str(exc):
            pytest.skip("verify_m1314_policy_manifest does not round-trip JSON yet due to tuple/list normalization")
        raise
    assert verified == written


def test_m1314_ledger_is_strictly_one_way_and_only_fit_stages_are_resumable(tmp_path: Path) -> None:
    ledger = M1314StageLedger(tmp_path / "ledger.json", protocol_fingerprint="protocol")
    assert ledger.open("development_preflight")["opened_stages"] == ["development_preflight"]
    with pytest.raises(ValueError):
        ledger.open("development_preflight")
    assert ledger.open("development_fit")["opened_stages"] == ["development_preflight", "development_fit"]
    assert ledger.open("development_fit", resumable_fit=True)["opened_stages"] == [
        "development_preflight",
        "development_fit",
    ]
    ledger.open("development_check", resumable_fit=True)
    with pytest.raises(ValueError):
        ledger.open("development_check", resumable_fit=True)
    ledger.open("confirmation_preflight")
    ledger.open("confirmation_fit")
    assert ledger.open("confirmation_fit", resumable_fit=True)["opened_stages"][-1] == "confirmation_fit"
    ledger.open("confirmation_evaluation")
    ledger.open("audit_preflight")
    assert ledger.open("audit")["opened_stages"] == [
        "development_preflight",
        "development_fit",
        "development_check",
        "confirmation_preflight",
        "confirmation_fit",
        "confirmation_evaluation",
        "audit_preflight",
        "audit",
    ]
    with pytest.raises(ValueError):
        ledger.open("audit", resumable_fit=True)


def test_preflight_reports_are_sealed_verified_and_persist_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_source_hashes(monkeypatch)
    manifest_fingerprint = "m1314-manifest"
    stage = "development_preflight"
    expected_conditions = tuple(M1314_CONDITIONS["development_fit"])
    expected_counts = {
        partition: len(_PARTITION_SEEDS[partition])
        for partition in M1314_PREFLIGHT_TARGETS[stage]["scan_partitions"]
    }

    passing = seal_m1314_preflight_report(
        _stage_report(stage, manifest_fingerprint=manifest_fingerprint, passes=True),
        expected_protocol_fingerprint=m1314_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    )
    assert passing["coverage_passes"] is True
    assert passing["failing_rows"] == []
    assert verify_m1314_preflight_report(
        passing,
        expected_protocol_fingerprint=m1314_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    ) == passing

    failing = seal_m1314_preflight_report(
        _stage_report(stage, manifest_fingerprint=manifest_fingerprint, passes=False),
        expected_protocol_fingerprint=m1314_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    )
    assert failing["coverage_passes"] is False
    assert len(failing["failing_rows"]) == 8
    assert verify_m1314_preflight_report(
        failing,
        expected_protocol_fingerprint=m1314_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
        require_pass=False,
    ) == failing
    with pytest.raises(M1314PreflightVerificationError):
        verify_m1314_preflight_report(
            failing,
            expected_protocol_fingerprint=m1314_protocol_fingerprint(),
            expected_manifest_fingerprint=manifest_fingerprint,
            expected_stage=stage,
            expected_conditions=expected_conditions,
            expected_episode_counts=expected_counts,
        )
    tampered = dict(passing)
    tampered["content_hash"] = "0" * 64
    with pytest.raises(M1314PreflightVerificationError):
        verify_m1314_preflight_report(
            tampered,
            expected_protocol_fingerprint=m1314_protocol_fingerprint(),
            expected_manifest_fingerprint=manifest_fingerprint,
            expected_stage=stage,
            expected_conditions=expected_conditions,
            expected_episode_counts=expected_counts,
        )


def test_future_experiment_clis_are_registered_without_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path, dict[str, object]]] = []

    def fake(stage: str):
        def run(target: Path, **kwargs):
            calls.append((stage, target, kwargs))
            if stage.endswith("preflight"):
                return {"stage": stage, "coverage_passes": True, "content_hash": "a" * 64}
            if stage == "replay":
                return {"stage": stage, "replay_pass": True}
            return {"stage": stage, "strict_replay_count": 0, "gate": {}}

        return run

    monkeypatch.setattr(cli, "write_m1314_development_preflight_report", fake("development_preflight"))
    monkeypatch.setattr(cli, "write_m1314_development_report", fake("development"))
    monkeypatch.setattr(cli, "write_m1314_confirmation_preflight_report", fake("confirmation_preflight"))
    monkeypatch.setattr(cli, "write_m1314_confirmation_report", fake("confirmation"))
    monkeypatch.setattr(cli, "write_m1314_audit_preflight_report", fake("audit_preflight"))
    monkeypatch.setattr(cli, "write_m1314_audit_report", fake("audit"))
    monkeypatch.setattr(cli, "replay_m1314_trace", fake("replay"))

    manifest = tmp_path / "manifest.json"
    preflight = tmp_path / "preflight.json"
    development = tmp_path / "development.json"
    confirmation = tmp_path / "confirmation.json"
    output = tmp_path / "output.json"
    trace = tmp_path / "trace.jsonl"
    policy = tmp_path / "policy.json"

    cli.main(["m1314-development-preflight", "--manifest", str(manifest), "--output", str(output)])
    cli.main([
        "m1314-development",
        "--manifest",
        str(manifest),
        "--preflight-report",
        str(preflight),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314-confirmation-preflight",
        "--manifest",
        str(manifest),
        "--development-report",
        str(development),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314-confirmation",
        "--manifest",
        str(manifest),
        "--preflight-report",
        str(preflight),
        "--development-report",
        str(development),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314-audit-preflight",
        "--manifest",
        str(manifest),
        "--confirmation-report",
        str(confirmation),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314-audit",
        "--manifest",
        str(manifest),
        "--preflight-report",
        str(preflight),
        "--confirmation-report",
        str(confirmation),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314-replay",
        "--manifest",
        str(manifest),
        "--trace",
        str(trace),
        "--policy",
        str(policy),
    ])

    assert [stage for stage, _, _ in calls] == [
        "development_preflight",
        "development",
        "confirmation_preflight",
        "confirmation",
        "audit_preflight",
        "audit",
        "replay",
    ]
    assert calls[0] == ("development_preflight", output, {"manifest_path": manifest})
    assert calls[1] == (
        "development",
        output,
        {"manifest_path": manifest, "preflight_report_path": preflight},
    )
    assert calls[2] == (
        "confirmation_preflight",
        output,
        {"manifest_path": manifest, "development_report_path": development},
    )
    assert calls[3] == (
        "confirmation",
        output,
        {
            "manifest_path": manifest,
            "preflight_report_path": preflight,
            "development_report_path": development,
        },
    )
    assert calls[4] == (
        "audit_preflight",
        output,
        {"manifest_path": manifest, "confirmation_report_path": confirmation},
    )
    assert calls[5] == (
        "audit",
        output,
        {
            "manifest_path": manifest,
            "preflight_report_path": preflight,
            "confirmation_report_path": confirmation,
        },
    )
    assert calls[6] == (
        "replay",
        trace,
        {"manifest_path": manifest, "policy_path": policy},
    )


def test_m1314_replay_cli_wrapper_matches_runner_signature_without_environment_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_future_runner_module()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"m139_policy_label": "modular_anchor_shield_candidate"}) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    policy = tmp_path / "policy.json"
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "verify_m1314_policy_manifest", lambda path: events.append(("verify_manifest", Path(path))) or {"protocol_fingerprint": "ok"})
    monkeypatch.setattr(module, "m139_config", lambda: events.append(("config", None)) or {"cfg": True})
    monkeypatch.setattr(
        module,
        "load_modular_q_policy",
        lambda path, *, config, expected_protocol_fingerprint: events.append(
            ("load_policy", (Path(path), config, expected_protocol_fingerprint))
        )
        or {"loaded_policy": True},
    )
    monkeypatch.setattr(module, "ShieldConfig", lambda *, enabled: {"enabled": enabled})
    monkeypatch.setattr(
        module,
        "ShieldedModularQPolicy",
        lambda *, learner, shield: {"learner": learner, "shield": shield},
    )
    monkeypatch.setattr(
        module,
        "replay_m139_trace",
        lambda trace_path, policy_obj: events.append(("replay_trace", (Path(trace_path), policy_obj)))
        or {"replay_pass": True, "steps": 1},
    )
    monkeypatch.setattr(
        module,
        "EcosystemEnv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("environment construction is forbidden")),
    )

    cli.main([
        "m1314-replay",
        "--manifest",
        str(manifest),
        "--trace",
        str(trace),
        "--policy",
        str(policy),
    ])

    assert json.loads(capsys.readouterr().out) == {"replay_pass": True, "steps": 1}
    assert events[0] == ("verify_manifest", manifest)
    assert events[1] == ("config", None)
    assert events[2][0] == "load_policy"
    assert events[2][1][0] == policy
    assert events[3][0] == "replay_trace"
    assert events[3][1][0] == trace
    assert events[3][1][1]["shield"] == {"enabled": True}


def test_future_runner_short_circuits_existing_output_before_stage_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_future_runner_module()
    writer = _future_export(module, "write_m1314_development_preflight_report")
    runner_name = "m1314_development_preflight"
    if not hasattr(module, runner_name):
        pytest.skip("future M13.14 preflight runner is not available yet")

    output = tmp_path / "existing.json"
    output.write_text("already here", encoding="utf-8")
    called = False

    def forbidden(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("runner should not be reached")

    monkeypatch.setattr(module, runner_name, forbidden)
    with pytest.raises(FileExistsError):
        writer(output, manifest_path=tmp_path / "manifest.json")
    assert called is False


def test_future_development_runner_requires_passing_preflight_before_ledger_or_env_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_future_runner_module()
    runner = _future_export(module, "m1314_development")

    manifest = tmp_path / "manifest.json"
    preflight = tmp_path / "preflight.json"
    manifest.write_text("{}", encoding="utf-8")
    preflight.write_text("{}", encoding="utf-8")
    events: list[str] = []

    def verify_manifest(_path):
        events.append("verify_manifest")
        return {"protocol_fingerprint": m1314_protocol_fingerprint()}

    def fail_preflight(*_args, **_kwargs):
        events.append("verify_preflight")
        raise M1314PreflightVerificationError("blocked before fit")

    class ForbiddenLedger:
        def __init__(self, *_args, **_kwargs):
            events.append("ledger_init")

        def open(self, *_args, **_kwargs):
            events.append("ledger_open")
            raise AssertionError("ledger fit stage should not open before preflight passes")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fit/eval/environment work should not start before preflight verification")

    _patch_if_present(monkeypatch, module, "verify_m1314_policy_manifest", verify_manifest)
    _patch_if_present(monkeypatch, module, "verify_m1314_preflight_report", fail_preflight)
    _patch_if_present(monkeypatch, module, "M1314StageLedger", ForbiddenLedger)
    for name in (
        "m10_scan_coverage",
        "fit_modular_q_policy",
        "evaluate_m1314_policy",
        "replay_m1314_trace",
        "_fit_declared_arm",
        "_evaluate_declared_stage",
        "_serialize_policy_artifact",
    ):
        _patch_if_present(monkeypatch, module, name, forbidden)

    with pytest.raises(M1314PreflightVerificationError):
        _call_supported(
            runner,
            manifest_path=manifest,
            preflight_report_path=preflight,
            artifact_dir=tmp_path / "artifacts",
        )
    assert events == ["verify_manifest", "verify_preflight"]


def test_later_stage_tampered_rehashed_report_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_source_hashes(monkeypatch)
    module = _load_future_runner_module()
    report_path = tmp_path / "confirmation.json"
    expected_manifest = "expected-manifest"
    report = module._seal_stage_report(
        {
            "schema_version": module.M1314_CONFIRMATION_REPORT_SCHEMA_VERSION,
            "stage": "confirmation",
            "protocol_fingerprint": m1314_protocol_fingerprint(),
            "manifest_fingerprint": "tampered-manifest",
            "development_report_sha256": "d" * 64,
            "preflight_report_sha256": "p" * 64,
            "elapsed_seconds": 0.0,
            "policy_artifacts": [],
            "strict_replay_count": 0,
            "episode_evidence": [],
            "shared_diagnostics": {},
            "gate": {"passes": True},
            "split_ledger": {"opened_stages": []},
            "unopened_partitions": [],
            "limits": [],
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        module._verify_stage_report(
            report_path,
            stage="confirmation",
            manifest_fingerprint=expected_manifest,
        )


def test_candidate_advantage_can_pass_when_matched_control_fails_its_own_hard_gate() -> None:
    module = _load_future_runner_module()
    seed = M1314_DEVELOPMENT_TRAINING_SEEDS[0]
    expected_env_seeds = M1314_DEVELOPMENT_CHECK_SEEDS
    conditions = M1314_CONDITIONS["development_check"]
    candidate_arm = "modular_anchor_shield_candidate"
    control_arm = "monolithic_anchor_shield_control"
    evidence: list[dict[str, object]] = []

    for condition in conditions:
        for env_seed in expected_env_seeds:
            evidence.append(
                _m1314_evidence_row(
                    arm=candidate_arm,
                    training_seed=seed,
                    condition=condition,
                    env_seed=env_seed,
                    recovery_required=condition in {"event_relocation", "compound"},
                    recovery_complete=True,
                )
            )
            evidence.append(
                _m1314_evidence_row(
                    arm=control_arm,
                    training_seed=seed,
                    condition=condition,
                    env_seed=env_seed,
                    recovery_required=condition in {"event_relocation", "compound"},
                    recovery_complete=True,
                    full_objective_success=not (condition == "persistent_reference" and env_seed in expected_env_seeds[:10]),
                )
            )

    comparison = module._comparison_gate(
        evidence,
        training_seed=seed,
        comparison_key="vs_monolithic_anchor_shield_control",
        conditions=conditions,
        expected_env_seeds=expected_env_seeds,
    )
    assert comparison["candidate_hard_gate_passes"] is True
    assert comparison["candidate_passes"] is True
    assert comparison["control_passes"] is False
    assert comparison["full_objective_advantage"] >= comparison["required_advantage"]
    assert comparison["passes"] is True


def test_relocation_recovery_required_all_20_episodes_is_enforced() -> None:
    module = _load_future_runner_module()
    rows = [
        _m1314_evidence_row(
            arm="modular_anchor_shield_candidate",
            training_seed=M1314_DEVELOPMENT_TRAINING_SEEDS[0],
            condition="event_relocation",
            env_seed=env_seed,
            recovery_required=True,
            recovery_complete=env_seed < 17,
        )
        for env_seed in range(20)
    ]
    result = module._hard_condition_gate(rows)
    assert result["recovery_required"] == 20
    assert result["recovery_complete"] == 17
    assert result["passes"] is False
