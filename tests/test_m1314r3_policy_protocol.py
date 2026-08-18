from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ecosystem_gym.__main__ import main
from ecosystem_gym.experiments import cli
from ecosystem_gym.tasks import LAYOUTS


def _load_protocol_module():
    try:
        return importlib.import_module("ecosystem_gym.maintenance.policy_protocol_m1314r3")
    except ModuleNotFoundError as exc:
        if exc.name == "ecosystem_gym.maintenance.policy_protocol_m1314r3":
            pytest.skip("ecosystem_gym.maintenance.policy_protocol_m1314r3 is not present in this bounded CLI/tests change")
        raise


def _load_support_module():
    try:
        return importlib.import_module("ecosystem_gym.experiments.m1314r3_support")
    except ModuleNotFoundError as exc:
        if exc.name == "ecosystem_gym.experiments.m1314r3_support":
            pytest.skip("ecosystem_gym.experiments.m1314r3_support is not present in this bounded CLI/tests change")
        raise


def _load_future_runner_module():
    try:
        return importlib.import_module("ecosystem_gym.experiments.m1314r3")
    except ModuleNotFoundError as exc:
        if exc.name in {
            "ecosystem_gym.experiments.m1314r3",
            "ecosystem_gym.experiments.m1314r3_support",
        }:
            pytest.skip(f"{exc.name} is not present in this bounded CLI/tests change")
        raise


def _future_export(module: object, *names: str):
    for name in names:
        exported = getattr(module, name, None)
        if callable(exported):
            return exported
    pytest.skip(f"future M13.14-r3 runner exports are not available yet: {names}")


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


def _partition_seeds(protocol) -> dict[str, tuple[int, ...]]:
    return {
        "development_fit": protocol.M1314R3_DEVELOPMENT_FIT_SEEDS,
        "development_check": protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS,
        "confirmation_fit": protocol.M1314R3_CONFIRMATION_FIT_SEEDS,
        "confirmation_evaluation": protocol.M1314R3_CONFIRMATION_EVALUATION_SEEDS,
        "audit": protocol.M1314R3_AUDIT_SEEDS,
    }


def _stage_report(protocol, stage: str, *, manifest_fingerprint: str, passes: bool) -> dict[str, object]:
    coverage = {
        partition: {
            condition: _coverage_row(
                episodes=len(_partition_seeds(protocol)[partition]),
                passes=passes,
            )
            for condition in protocol.M1314R3_CONDITIONS["development_fit"]
        }
        for partition in protocol.M1314R3_PREFLIGHT_TARGETS[stage]["scan_partitions"]
    }
    return {
        "schema_version": "non-canonical-fixture",
        "protocol_fingerprint": protocol.m1314r3_protocol_fingerprint(),
        "manifest_fingerprint": manifest_fingerprint,
        "stage": stage,
        "coverage": coverage,
        "note": "synthetic fixture",
    }


def _normalized_preflight_targets(protocol) -> dict[str, dict[str, object]]:
    return {
        stage: {
            "scan_partitions": list(target["scan_partitions"]),
            "authorizes_command": target["authorizes_command"],
        }
        for stage, target in protocol.M1314R3_PREFLIGHT_TARGETS.items()
    }


def _stub_source_hashes(monkeypatch: pytest.MonkeyPatch, protocol) -> None:
    monkeypatch.setattr(
        protocol,
        "_source_hashes",
        lambda: {
            "ecosystem_gym/experiments/cli.py": "0" * 64,
            "ecosystem_gym/__main__.py": "1" * 64,
            "ecosystem_gym/maintenance/policy_protocol_m1314r3.py": "2" * 64,
            "ecosystem_gym/experiments/m1314r3.py": "3" * 64,
        },
    )


def _m1314r3_evidence_row(
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
    teacher_dataset_path: str | None = None,
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
        "teacher_dataset_path": teacher_dataset_path or f"artifacts/teachers/{arm}-{training_seed}.npz",
        "weight_fingerprint": weight_fingerprint or f"weights:{arm}:{training_seed}",
        "policy_fingerprint": policy_fingerprint or f"policy:{arm}:{training_seed}",
        "artifact_sha256": artifact_sha256 or f"artifact:{arm}:{training_seed}",
    }


def test_manifest_cli_is_registered_and_runs_without_an_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "manifest.json"
    payload = {
        "protocol_fingerprint": "r3-fingerprint",
        "status": "implemented-frozen-and-unopened",
        "future_command_names": [
            "maintenance-policy-manifest-m1314r3",
            "m1314r3-development-preflight",
            "m1314r3-development",
            "m1314r3-confirmation-preflight",
            "m1314r3-confirmation",
            "m1314r3-audit-preflight",
            "m1314r3-audit",
            "m1314r3-replay",
        ],
    }
    module = types.ModuleType("ecosystem_gym.maintenance.policy_protocol_m1314r3")

    def write_manifest(path: Path) -> dict[str, object]:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    module.write_m1314r3_policy_manifest = write_manifest
    monkeypatch.setitem(sys.modules, "ecosystem_gym.maintenance.policy_protocol_m1314r3", module)

    main(["maintenance-policy-manifest-m1314r3", "--output", str(output)])
    assert json.loads(capsys.readouterr().out) == {
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "status": payload["status"],
    }
    assert json.loads(output.read_text(encoding="utf-8"))["future_command_names"] == payload["future_command_names"]


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

    monkeypatch.setattr(cli, "write_m1314r3_development_preflight_report", fake("development_preflight"))
    monkeypatch.setattr(cli, "write_m1314r3_development_report", fake("development"))
    monkeypatch.setattr(cli, "write_m1314r3_confirmation_preflight_report", fake("confirmation_preflight"))
    monkeypatch.setattr(cli, "write_m1314r3_confirmation_report", fake("confirmation"))
    monkeypatch.setattr(cli, "write_m1314r3_audit_preflight_report", fake("audit_preflight"))
    monkeypatch.setattr(cli, "write_m1314r3_audit_report", fake("audit"))
    monkeypatch.setattr(cli, "replay_m1314r3_trace", fake("replay"))

    manifest = tmp_path / "manifest.json"
    preflight = tmp_path / "preflight.json"
    development = tmp_path / "development.json"
    confirmation = tmp_path / "confirmation.json"
    output = tmp_path / "output.json"
    trace = tmp_path / "trace.jsonl"
    policy = tmp_path / "policy.json"

    cli.main(["m1314r3-development-preflight", "--manifest", str(manifest), "--output", str(output)])
    cli.main([
        "m1314r3-development",
        "--manifest",
        str(manifest),
        "--preflight-report",
        str(preflight),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314r3-confirmation-preflight",
        "--manifest",
        str(manifest),
        "--development-report",
        str(development),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314r3-confirmation",
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
        "m1314r3-audit-preflight",
        "--manifest",
        str(manifest),
        "--confirmation-report",
        str(confirmation),
        "--output",
        str(output),
    ])
    cli.main([
        "m1314r3-audit",
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
        "m1314r3-replay",
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


def test_m1314r3_partitions_and_all_five_layout_families_are_fresh() -> None:
    protocol = _load_protocol_module()
    partitions = (
        protocol.M1314R3_DEVELOPMENT_FIT_SEEDS,
        protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS,
        protocol.M1314R3_CONFIRMATION_FIT_SEEDS,
        protocol.M1314R3_CONFIRMATION_EVALUATION_SEEDS,
        protocol.M1314R3_AUDIT_SEEDS,
    )
    assert protocol.M1314R3_DEVELOPMENT_FIT_SEEDS == tuple(range(6840, 6860))
    assert protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS == tuple(range(6860, 6880))
    assert protocol.M1314R3_CONFIRMATION_FIT_SEEDS == tuple(range(6880, 6920))
    assert protocol.M1314R3_CONFIRMATION_EVALUATION_SEEDS == tuple(range(6920, 6940))
    assert protocol.M1314R3_AUDIT_SEEDS == tuple(range(6940, 6960))
    assert all(set(left).isdisjoint(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :])

    expected = {
        str(condition["layout_id"])
        for conditions in protocol.M1314R3_CONDITIONS.values()
        for condition in conditions.values()
    }
    names = {name for name in LAYOUTS if name.startswith("m1314r3_")}
    if not names:
        pytest.skip("M13.14-r3 layouts are not present on this checkout")
    assert names == expected and len(names) == 20
    current = {point for name in names for point in _points(LAYOUTS[name])}
    previous = {point for name, layout in LAYOUTS.items() if not name.startswith("m1314r3_") for point in _points(layout)}
    assert len(current) == 100
    assert current.isdisjoint(previous)


def test_manifest_freezes_declared_arms_preflights_budgets_and_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _load_protocol_module()
    _stub_source_hashes(monkeypatch, protocol)
    manifest = protocol.m1314r3_policy_manifest()
    assert manifest["protocol_fingerprint"] == protocol.m1314r3_protocol_fingerprint()
    assert manifest["status"] == "implemented-frozen-and-unopened"
    assert manifest["public_boundary"]["features"] == 30
    assert manifest["public_boundary"]["macros"] == list(protocol.M1314R3_ACTION_SURFACE)
    assert set(manifest["arms"]) == set(protocol.M1314R3_ARM_NAMES)
    assert manifest["teacher_dataset"]["legacy_teacher_reuse_forbidden"] is True
    assert manifest["serialization_and_reuse"]["legacy_teacher_or_artifact_reuse_forbidden"] is True
    assert any(
        key.endswith("paths_rejected") and value is True
        for key, value in manifest["serialization_and_reuse"].items()
    )
    assert manifest["preflight"]["stage_order"] == list(protocol.M1314R3_PREFLIGHT_STAGE_ORDER)
    assert manifest["preflight"]["targets"] == _normalized_preflight_targets(protocol)
    assert manifest["training_budget"]["decision_budget_per_fit"] == protocol.M1314R3_DECISION_BUDGET_PER_FIT
    assert manifest["training_budget"]["development_training_seeds"] == list(protocol.M1314R3_DEVELOPMENT_TRAINING_SEEDS)
    assert manifest["training_budget"]["confirmation_training_seeds"] == list(protocol.M1314R3_CONFIRMATION_TRAINING_SEEDS)
    assert manifest["evaluation_budget"]["replay_requirement"] == protocol.M1314R3_REPLAY_REQUIREMENT
    assert manifest["evaluation_budget"]["cross_arm_winner_selection"] == "forbidden"
    assert manifest["causal_comparison_gate"]["no_cross_arm_selection"] is True
    assert "ecosystem_gym/experiments/m1314r3.py" in manifest["source_hashes"]
    assert manifest["future_command_names"] == list(protocol.M1314R3_COMMAND_NAMES)
    assert len(manifest["future_commands"]) == 8
    assert all(command.startswith("uv run python -m ecosystem_gym") for command in manifest["future_commands"])
    assert manifest["authorization"] == protocol.M1314R3_NO_RUN_AUTHORIZATION


def test_manifest_write_and_verify_roundtrip_when_protocol_roundtrips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = _load_protocol_module()
    _stub_source_hashes(monkeypatch, protocol)
    output = tmp_path / "manifest.json"
    written = protocol.write_m1314r3_policy_manifest(output)
    assert protocol.verify_m1314r3_policy_manifest(output) == written


def test_m1314r3_ledger_is_strictly_one_way_and_only_fit_stages_are_resumable(tmp_path: Path) -> None:
    support = _load_support_module()
    ledger = support.M1314R3StageLedger(tmp_path / "ledger.json", protocol_fingerprint="protocol")
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
    protocol = _load_protocol_module()
    support = _load_support_module()
    _stub_source_hashes(monkeypatch, protocol)
    manifest_fingerprint = "m1314r3-manifest"
    stage = "development_preflight"
    expected_conditions = tuple(protocol.M1314R3_CONDITIONS["development_fit"])
    expected_counts = {
        partition: len(_partition_seeds(protocol)[partition])
        for partition in protocol.M1314R3_PREFLIGHT_TARGETS[stage]["scan_partitions"]
    }

    passing = support.seal_m1314r3_preflight_report(
        _stage_report(protocol, stage, manifest_fingerprint=manifest_fingerprint, passes=True),
        expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    )
    assert passing["coverage_passes"] is True
    assert passing["failing_rows"] == []
    assert support.verify_m1314r3_preflight_report(
        passing,
        expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    ) == passing

    failing = support.seal_m1314r3_preflight_report(
        _stage_report(protocol, stage, manifest_fingerprint=manifest_fingerprint, passes=False),
        expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
    )
    assert failing["coverage_passes"] is False
    assert len(failing["failing_rows"]) == 8
    assert support.verify_m1314r3_preflight_report(
        failing,
        expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
        expected_manifest_fingerprint=manifest_fingerprint,
        expected_stage=stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_counts,
        require_pass=False,
    ) == failing
    with pytest.raises(support.M1314R3PreflightVerificationError):
        support.verify_m1314r3_preflight_report(
            failing,
            expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
            expected_manifest_fingerprint=manifest_fingerprint,
            expected_stage=stage,
            expected_conditions=expected_conditions,
            expected_episode_counts=expected_counts,
        )
    tampered = dict(passing)
    tampered["content_hash"] = "0" * 64
    with pytest.raises(support.M1314R3PreflightVerificationError):
        support.verify_m1314r3_preflight_report(
            tampered,
            expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
            expected_manifest_fingerprint=manifest_fingerprint,
            expected_stage=stage,
            expected_conditions=expected_conditions,
            expected_episode_counts=expected_counts,
        )


def test_m1314r3_replay_cli_wrapper_matches_runner_signature_without_environment_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_future_runner_module()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"m1314r3_policy_label": "modular_anchor_shield_candidate"}) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    policy = tmp_path / "policy.json"
    events: list[tuple[str, object]] = []

    _patch_if_present(
        monkeypatch,
        module,
        "verify_m1314r3_policy_manifest",
        lambda path: events.append(("verify_manifest", Path(path))) or {"protocol_fingerprint": "ok"},
    )
    _patch_if_present(monkeypatch, module, "m139_config", lambda: events.append(("config", None)) or {"cfg": True})
    _patch_if_present(
        monkeypatch,
        module,
        "load_modular_q_policy_m1314r3",
        lambda path, *, config, expected_protocol_fingerprint: events.append(
            ("load_policy", (Path(path), config, expected_protocol_fingerprint))
        )
        or {"loaded_policy": True},
    )
    _patch_if_present(monkeypatch, module, "ShieldConfig", lambda *, enabled: {"enabled": enabled})
    _patch_if_present(
        monkeypatch,
        module,
        "ShieldedModularQPolicy",
        lambda *, learner, shield: {"learner": learner, "shield": shield},
    )
    _patch_if_present(
        monkeypatch,
        module,
        "ShieldedModularQPolicyM1314R3",
        lambda *, learner, shield: {"learner": learner, "shield": shield},
    )
    _patch_if_present(
        monkeypatch,
        module,
        "replay_m139_trace",
        lambda trace_path, policy_obj: events.append(("replay_trace", (Path(trace_path), policy_obj)))
        or {"replay_pass": True, "steps": 1},
    )
    _patch_if_present(
        monkeypatch,
        module,
        "EcosystemEnv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("environment construction is forbidden")),
    )

    cli.main([
        "m1314r3-replay",
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
    writer = _future_export(module, "write_m1314r3_development_preflight_report")
    runner_name = "m1314r3_development_preflight"
    if not hasattr(module, runner_name):
        pytest.skip("future M13.14-r3 preflight runner is not available yet")

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
    protocol = _load_protocol_module()
    support = _load_support_module()
    module = _load_future_runner_module()
    runner = _future_export(module, "m1314r3_development")

    manifest = tmp_path / "manifest.json"
    preflight = tmp_path / "preflight.json"
    manifest.write_text("{}", encoding="utf-8")
    preflight.write_text("{}", encoding="utf-8")
    events: list[str] = []

    def verify_manifest(_path):
        events.append("verify_manifest")
        return {"protocol_fingerprint": protocol.m1314r3_protocol_fingerprint()}

    def fail_preflight(*_args, **_kwargs):
        events.append("verify_preflight")
        raise support.M1314R3PreflightVerificationError("blocked before fit")

    class ForbiddenLedger:
        def __init__(self, *_args, **_kwargs):
            events.append("ledger_init")

        def open(self, *_args, **_kwargs):
            events.append("ledger_open")
            raise AssertionError("ledger fit stage should not open before preflight passes")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fit/eval/environment work should not start before preflight verification")

    _patch_if_present(monkeypatch, module, "verify_m1314r3_policy_manifest", verify_manifest)
    _patch_if_present(monkeypatch, module, "verify_m1314r3_preflight_report", fail_preflight)
    _patch_if_present(monkeypatch, module, "M1314R3StageLedger", ForbiddenLedger)
    for name in (
        "m10_scan_coverage",
        "fit_modular_q_policy_m1314r3",
        "evaluate_m1314r3_policy",
        "replay_m1314r3_trace",
        "_fit_declared_arm",
        "_evaluate_declared_stage",
        "_serialize_policy_artifact",
    ):
        _patch_if_present(monkeypatch, module, name, forbidden)

    with pytest.raises(support.M1314R3PreflightVerificationError):
        _call_supported(
            runner,
            manifest_path=manifest,
            preflight_report_path=preflight,
            artifact_dir=tmp_path / "artifacts",
        )
    assert events == ["verify_manifest", "verify_preflight"]


def test_later_stage_tampered_rehashed_report_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    protocol = _load_protocol_module()
    _stub_source_hashes(monkeypatch, protocol)
    module = _load_future_runner_module()
    report_path = tmp_path / "confirmation.json"
    expected_manifest = "expected-manifest"
    report = module._seal_stage_report(
        {
            "schema_version": module.M1314R3_CONFIRMATION_REPORT_SCHEMA_VERSION,
            "stage": "confirmation",
            "protocol_fingerprint": protocol.m1314r3_protocol_fingerprint(),
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
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    seed = protocol.M1314R3_DEVELOPMENT_TRAINING_SEEDS[0]
    expected_env_seeds = protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS
    conditions = protocol.M1314R3_CONDITIONS["development_check"]
    candidate_arm = "modular_anchor_shield_candidate"
    control_arm = "monolithic_anchor_shield_control"
    evidence: list[dict[str, object]] = []

    for condition in conditions:
        for env_seed in expected_env_seeds:
            evidence.append(
                _m1314r3_evidence_row(
                    arm=candidate_arm,
                    training_seed=seed,
                    condition=condition,
                    env_seed=env_seed,
                    recovery_required=condition in {"event_relocation", "compound"},
                    recovery_complete=True,
                )
            )
            evidence.append(
                _m1314r3_evidence_row(
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


def test_audit_stage_expects_candidate_only_policy_rows_plus_separate_diagnostics() -> None:
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    policy_rows = (
        len(protocol.M1314R3_CONFIRMATION_TRAINING_SEEDS)
        * len(protocol.M1314R3_CONDITIONS["audit"])
        * len(protocol.M1314R3_AUDIT_SEEDS)
    )
    diagnostic_rows = 2 * len(protocol.M1314R3_CONDITIONS["audit"]) * len(protocol.M1314R3_AUDIT_SEEDS)
    assert policy_rows == 640
    assert module._expected_stage_row_count("audit") == policy_rows + diagnostic_rows


def test_audit_gate_uses_only_candidate_replicas_without_control_comparisons() -> None:
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    evidence = [
        _m1314r3_evidence_row(
            arm="modular_anchor_shield_candidate",
            training_seed=training_seed,
            condition=condition,
            env_seed=env_seed,
            recovery_required=condition in {"event_relocation", "compound"},
            recovery_complete=True,
        )
        for training_seed in protocol.M1314R3_CONFIRMATION_TRAINING_SEEDS
        for condition in protocol.M1314R3_CONDITIONS["audit"]
        for env_seed in protocol.M1314R3_AUDIT_SEEDS
    ]
    gate = module.m1314r3_gate(
        evidence,
        training_seeds=protocol.M1314R3_CONFIRMATION_TRAINING_SEEDS,
        conditions=protocol.M1314R3_CONDITIONS["audit"],
        expected_env_seeds=protocol.M1314R3_AUDIT_SEEDS,
    )
    assert gate["passes"] is True
    assert len(gate["replicas"]) == len(protocol.M1314R3_CONFIRMATION_TRAINING_SEEDS)
    for replica in gate["replicas"]:
        assert set(replica["arms"]) == {"modular_anchor_shield_candidate"}
        assert replica.get("comparisons", {}) == {}


def test_relocation_recovery_required_all_20_episodes_is_enforced() -> None:
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    rows = [
        _m1314r3_evidence_row(
            arm="modular_anchor_shield_candidate",
            training_seed=protocol.M1314R3_DEVELOPMENT_TRAINING_SEEDS[0],
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


def test_future_stage_gate_rejects_legacy_teacher_or_artifact_paths() -> None:
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    evidence = [
        _m1314r3_evidence_row(
            arm="modular_anchor_shield_candidate",
            training_seed=protocol.M1314R3_DEVELOPMENT_TRAINING_SEEDS[0],
            condition=condition,
            env_seed=env_seed,
            teacher_dataset_path="artifacts/m1314r2/teacher.npz",
            artifact_sha256="legacy-artifact",
        )
        for condition in protocol.M1314R3_CONDITIONS["development_check"]
        for env_seed in protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS
    ]
    gate = module._hard_gate_for_arm(
        evidence,
        arm="modular_anchor_shield_candidate",
        training_seed=protocol.M1314R3_DEVELOPMENT_TRAINING_SEEDS[0],
        conditions=protocol.M1314R3_CONDITIONS["development_check"],
        expected_env_seeds=protocol.M1314R3_DEVELOPMENT_CHECK_SEEDS,
    )
    assert gate["passes"] is False
    assert gate["legacy_teacher_or_artifact_reuse_detected"] is True


def test_random_trace_replay_regression_uses_noncanonical_seed_and_strict_replay(tmp_path: Path) -> None:
    module = _load_future_runner_module()
    _future_export(module, "_evaluate_one")
    if not hasattr(module, "SeededRandomM139Policy"):
        pytest.skip("future M13.14-r3 random-policy replay helpers are not available yet")

    synthetic_seed = 17
    policy = module.SeededRandomM139Policy(synthetic_seed)
    trace_dir = tmp_path / "traces"
    conditions = {
        "synthetic_old_probe": {
            "layout_id": "m139_probe_northeast",
            "food_variant": "orange",
            "toy_variant": "ball",
            "camera_control": "scan_v2",
            "initial_scan_sector": "north",
        }
    }
    spec = {
        "label": "synthetic_old_mask_random",
        "arm": "mask_random",
        "role": "mask_random",
        "training_seed": synthetic_seed,
        "artifact": {"sha256": "pcg64"},
        "evaluation_spec": {
            "shield": {"enabled": False},
            "policy_fingerprint": f"pcg64-{synthetic_seed}",
            "weight_fingerprint": f"pcg64-{synthetic_seed}",
            "parameter_bytes_sha256": f"pcg64-{synthetic_seed}",
            "weight_bytes_sha256": f"pcg64-{synthetic_seed}",
        },
        "policy": policy,
    }

    rows, count = module._evaluate_one(
        spec,
        conditions=conditions,
        seeds=(synthetic_seed,),
        trace_dir=trace_dir,
    )
    assert count == 1
    assert len(rows) == 1

    trace = trace_dir / "synthetic_old_mask_random" / "synthetic_old_probe" / f"seed-{synthetic_seed}.jsonl"
    header_line = trace.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(header_line)
    assert f"pcg64-{synthetic_seed}" in header_line
    assert header["seed"] == synthetic_seed
    replay = (
        module.replay_m139_random_trace(trace, module.SeededRandomM139Policy(synthetic_seed))
        if hasattr(module, "replay_m139_random_trace")
        else module.replay_m139_trace(trace, module.SeededRandomM139Policy(synthetic_seed))
    )
    assert replay.steps > 0
    assert isinstance(replay.task_success, bool)


def test_learned_artifact_trace_and_replay_regression_uses_real_r3_roundtrip_and_rejects_tampered_header(
    tmp_path: Path,
) -> None:
    protocol = _load_protocol_module()
    module = _load_future_runner_module()
    _future_export(module, "_evaluate_one")

    manifest_path = tmp_path / "m1314r3-manifest.json"
    policy_path = tmp_path / "m1314r3-learned-artifact.json"
    protocol.write_m1314r3_policy_manifest(manifest_path)

    learner = module.ModularQPolicy(
        arm="modular_anchor_shield_candidate",
        seed=113,
        config=module.m139_config(),
        protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
    )
    artifact = module.save_modular_q_policy_m1314r3(policy_path, learner)
    loaded = module.load_modular_q_policy_m1314r3(
        policy_path,
        config=module.m139_config(),
        expected_protocol_fingerprint=protocol.m1314r3_protocol_fingerprint(),
    )
    assert loaded.parameter_fingerprint() == learner.parameter_fingerprint()
    assert artifact["policy_fingerprint"] == loaded.parameter_fingerprint()

    shield = module.ShieldConfig(enabled=True)
    spec = {
        "label": "synthetic_old_learned_candidate",
        "arm": "modular_anchor_shield_candidate",
        "role": "candidate",
        "training_seed": 113,
        "artifact": artifact,
        "evaluation_spec": module.modular_q_evaluation_spec(loaded, shield=shield),
        "policy": module.ShieldedModularQPolicy(learner=loaded, shield=shield),
    }
    conditions = {
        "synthetic_old_probe": {
            "layout_id": "m139_probe_northeast",
            "food_variant": "orange",
            "toy_variant": "ball",
            "camera_control": "scan_v2",
            "initial_scan_sector": "north",
        }
    }

    rows, count = module._evaluate_one(
        spec,
        conditions=conditions,
        seeds=(29,),
        trace_dir=tmp_path / "m1314r3-traces",
    )
    assert count == 1
    assert len(rows) == 1

    trace = tmp_path / "m1314r3-traces" / "synthetic_old_learned_candidate" / "synthetic_old_probe" / "seed-29.jsonl"
    header_line = trace.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(header_line)
    policy_fingerprint = loaded.parameter_fingerprint()
    assert policy_fingerprint in header_line
    assert header["m139_policy_fingerprint"] == policy_fingerprint
    assert header["seed"] == 29

    replay = module.replay_m1314r3_trace(
        manifest_path=manifest_path,
        trace_path=trace,
        policy_path=policy_path,
    )
    assert replay.steps > 0
    assert isinstance(replay.task_success, bool)

    tampered_trace = tmp_path / "m1314r3-tampered-trace.jsonl"
    tampered_header = dict(header)
    tampered_header["m139_policy_fingerprint"] = "0" * 64
    lines = trace.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(tampered_header, sort_keys=True, separators=(",", ":"))
    tampered_trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="policy fingerprint mismatch"):
        module.replay_m1314r3_trace(
            manifest_path=manifest_path,
            trace_path=tampered_trace,
            policy_path=policy_path,
        )
