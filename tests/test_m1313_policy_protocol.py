from __future__ import annotations

import json

import pytest

from ecosystem_gym.__main__ import main
from ecosystem_gym.experiments import cli, m1313
from ecosystem_gym.experiments.m1313 import m1313_gate
from ecosystem_gym.experiments.m1313_support import M1313SplitLedger
from ecosystem_gym.maintenance.policy_protocol import (
    M1313_AUDIT_SEEDS,
    M1313_CONFIRMATION_EVALUATION_SEEDS,
    M1313_CONFIRMATION_FIT_SEEDS,
    M1313_CONDITIONS,
    M1313_DEVELOPMENT_CHECK_SEEDS,
    M1313_DEVELOPMENT_FIT_SEEDS,
    m1313_protocol_fingerprint,
    policy_family_manifest,
    verify_policy_family_manifest,
)
from ecosystem_gym.tasks import LAYOUTS


def _points(layout):
    return (layout.agent_xy, layout.food_low, layout.food_high, layout.toy_xy, layout.rest_xy)


def test_m1313_partitions_and_all_five_layout_families_are_fresh() -> None:
    partitions = (
        M1313_DEVELOPMENT_FIT_SEEDS,
        M1313_DEVELOPMENT_CHECK_SEEDS,
        M1313_CONFIRMATION_FIT_SEEDS,
        M1313_CONFIRMATION_EVALUATION_SEEDS,
        M1313_AUDIT_SEEDS,
    )
    assert M1313_DEVELOPMENT_FIT_SEEDS == tuple(range(6200, 6220))
    assert M1313_DEVELOPMENT_CHECK_SEEDS == tuple(range(6220, 6240))
    assert all(set(left).isdisjoint(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :])

    expected = {
        str(condition["layout_id"])
        for conditions in M1313_CONDITIONS.values()
        for condition in conditions.values()
    }
    names = {name for name in LAYOUTS if name.startswith("m1313_")}
    assert names == expected and len(names) == 20
    current = {point for name in names for point in _points(LAYOUTS[name])}
    previous = {point for name, layout in LAYOUTS.items() if not name.startswith("m1313_") for point in _points(layout)}
    assert len(current) == 100
    assert current.isdisjoint(previous)


def test_manifest_freezes_three_one_factor_families_budgets_gates_and_commands() -> None:
    manifest = policy_family_manifest()
    assert manifest["protocol_fingerprint"] == m1313_protocol_fingerprint()
    assert manifest["status"] == "frozen-and-unopened"
    assert manifest["public_boundary"]["features"] == 30
    assert manifest["public_boundary"]["extra_commitment_fields_are_unencoded"] is True
    families = manifest["families"]
    assert set(families) == {"urgency_commitment", "short_horizon_model", "shielded_learned"}
    assert families["urgency_commitment"]["only_changed_factor"] == "goal commitment and hysteretic continuation"
    assert families["short_horizon_model"]["only_changed_factor"] == "lookahead depth 4 versus depth 1"
    assert families["shielded_learned"]["actor"]["candidate_and_control_actor_bytes_identical"] is True
    assert manifest["development_episode_budget"]["total_check_episodes"] == 1120
    assert manifest["hard_gate_per_candidate_condition"]["full_objective_at_least"] == 18
    assert manifest["paired_family_gate"]["cross_family_selection"].startswith("forbidden")
    assert len(manifest["future_commands"]) == 4
    assert all(command.startswith("uv run python -m ecosystem_gym") for command in manifest["future_commands"])
    assert "No fit" in manifest["authorization"]


def test_manifest_cli_is_registered_and_runs_without_an_environment(tmp_path, capsys) -> None:
    output = tmp_path / "manifest.json"
    main(["maintenance-policy-manifest", "--output", str(output)])
    printed = json.loads(capsys.readouterr().out)
    payload = verify_policy_family_manifest(output)
    assert printed == {
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "status": "frozen-and-unopened",
    }


def test_m1313_ledger_is_strictly_one_way_and_only_fit_is_resumable(tmp_path) -> None:
    ledger = M1313SplitLedger(tmp_path / "ledger.json", protocol_fingerprint="protocol")
    assert ledger.open("development_fit")["opened_partitions"] == ["development_fit"]
    assert ledger.open("development_fit", resumable_fit=True)["opened_partitions"] == ["development_fit"]
    with pytest.raises(ValueError):
        ledger.open("confirmation_fit")
    ledger.open("development_check")
    ledger.open("confirmation_fit")
    ledger.open("confirmation_evaluation")
    assert ledger.open("audit")["opened_partitions"] == [
        "development_fit",
        "development_check",
        "confirmation_fit",
        "confirmation_evaluation",
        "audit",
    ]
    with pytest.raises(ValueError):
        ledger.open("audit")


def _gate_rows(
    family: str,
    seed: int,
    actor: str | None = None,
    env_seeds: tuple[int, ...] = M1313_DEVELOPMENT_CHECK_SEEDS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ("candidate", "control"):
        for condition in M1313_CONDITIONS["development_check"]:
            for env_seed in env_seeds:
                rows.append(
                    {
                        "family": family,
                        "role": role,
                        "training_seed": seed,
                        "condition": condition,
                        "env_seed": env_seed,
                        "survived": role == "candidate",
                        "maintenance_complete": role == "candidate",
                        "full_objective_success": role == "candidate",
                        "decision_safe_fraction": 0.95 if role == "candidate" else 0.70,
                        "duration_safe_fraction": 0.95 if role == "candidate" else 0.70,
                        "recovery_required": condition in {"event_relocation", "compound"},
                        "recovery_complete": role == "candidate",
                        "unsafe_wait_decisions": 0,
                        "wait_decisions": 1,
                        "conformance_violations": 0,
                        "actor_fingerprint": actor,
                    }
                )
    return rows


def test_gate_keeps_families_independent_and_requires_identical_shield_actor_bytes() -> None:
    evidence = _gate_rows("urgency_commitment", 0)
    evidence += _gate_rows("short_horizon_model", 0)
    for seed in (20261331, 20261332, 20261333, 20261334):
        evidence += _gate_rows("shielded_learned", seed, actor=f"actor-{seed}")
    gate = m1313_gate(evidence, {"urgency_commitment", "short_horizon_model", "shielded_learned"})
    assert gate["promoted_families"] == [
        "shielded_learned",
        "short_horizon_model",
        "urgency_commitment",
    ]
    assert gate["cross_family_selection_performed"] is False

    for row in evidence:
        if row["family"] == "shielded_learned" and row["training_seed"] == 20261334 and row["role"] == "control":
            row["actor_fingerprint"] = "different"
    gate = m1313_gate(evidence, {"shielded_learned"})
    assert gate["promoted_families"] == []
    assert gate["families"]["shielded_learned"]["replicas"][-1]["actor_bytes_identical"] is False

    confirmation = []
    for seed in range(20261335, 20261343):
        confirmation += _gate_rows(
            "shielded_learned",
            seed,
            actor=f"confirm-{seed}",
            env_seeds=M1313_CONFIRMATION_EVALUATION_SEEDS,
        )
    assert m1313_gate(confirmation, {"shielded_learned"})["promoted_families"] == ["shielded_learned"]


def test_future_experiment_clis_are_registered_without_running(monkeypatch, tmp_path, capsys) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []

    def fake(stage):
        def run(output, **kwargs):
            calls.append((stage, output, kwargs))
            return {"stage": stage, "strict_replay_count": 0, "gate": {}}

        return run

    monkeypatch.setattr(cli, "write_m1313_development_report", fake("development"))
    monkeypatch.setattr(cli, "write_m1313_confirmation_report", fake("confirmation"))
    monkeypatch.setattr(cli, "write_m1313_audit_report", fake("audit"))
    manifest = tmp_path / "manifest.json"
    development = tmp_path / "development.json"
    confirmation = tmp_path / "confirmation.json"
    output = tmp_path / "output.json"
    cli.main(["m1313-development", "--manifest", str(manifest), "--output", str(output)])
    cli.main([
        "m1313-confirmation",
        "--manifest",
        str(manifest),
        "--development-report",
        str(development),
        "--output",
        str(output),
    ])
    cli.main([
        "m1313-audit",
        "--manifest",
        str(manifest),
        "--confirmation-report",
        str(confirmation),
        "--output",
        str(output),
    ])
    capsys.readouterr()
    assert [stage for stage, _, _ in calls] == ["development", "confirmation", "audit"]
    assert calls[0][2] == {"manifest_path": manifest}
    assert calls[1][2] == {"manifest_path": manifest, "development_report_path": development}
    assert calls[2][2] == {"manifest_path": manifest, "confirmation_report_path": confirmation}


def test_existing_report_blocks_runner_before_any_experiment_work(monkeypatch, tmp_path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("already here")
    called = False

    def forbidden(**_):
        nonlocal called
        called = True
        raise AssertionError("runner should not be reached")

    monkeypatch.setattr(m1313, "m1313_development", forbidden)
    with pytest.raises(FileExistsError):
        m1313.write_m1313_development_report(output, manifest_path=tmp_path / "manifest.json")
    assert called is False
