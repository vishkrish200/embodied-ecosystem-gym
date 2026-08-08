from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ecosystem_gym.experiments import cli, m1310
from ecosystem_gym.experiments.m13 import M13Macro
from ecosystem_gym.experiments.m1310 import (
    M1310_AUDIT_SEEDS,
    M1310_CONFIRMATION_EVALUATION_SEEDS,
    M1310_CONFIRMATION_FIT_SEEDS,
    M1310_SCREEN_FIT_CONDITIONS,
    M1310_SCREEN_FIT_SEEDS,
    M1310_SCREEN_PROBE_CONDITIONS,
    M1310_SCREEN_PROBE_SEEDS,
    M1310_SCREEN_TRAINING_SEEDS,
    CompactM1310QPolicy,
    fit_m1310_imitation,
    load_m1310_policy,
    load_m1310_teacher_dataset,
    m1310_epsilon,
    m1310_policy_fingerprint,
    m1310_protocol_fingerprint,
    m1310_screen_jobs,
    write_m1310_policy,
)
from ecosystem_gym.experiments.m1310_support import (
    SplitAlreadyOpenedError,
    SplitDependencyError,
    SplitOpenLedger,
    m1310_screen_promotes,
)
from ecosystem_gym.tasks import LAYOUTS


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
        "conformance_violations": 0,
        "replay_violations": 0,
    }


def _matrix(**overrides: object) -> list[dict[str, object]]:
    return [
        _summary(
            training_seed=training_seed,
            condition=condition,
            env_seed=env_seed,
            **overrides,
        )
        for training_seed in M1310_SCREEN_TRAINING_SEEDS
        for condition in M1310_SCREEN_PROBE_CONDITIONS
        for env_seed in M1310_SCREEN_PROBE_SEEDS
    ]


def _synthetic_dataset(path: Path) -> None:
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(len(M13Macro)), 32)
    features = np.zeros((labels.size, 30), dtype=np.float32)
    features[np.arange(labels.size), labels] = 1.0
    features += rng.normal(0.0, 0.01, features.shape).astype(np.float32)
    masks = np.ones((labels.size, len(M13Macro)), dtype=np.bool_)
    np.savez_compressed(path, features=features, masks=masks, labels=labels)


def test_m1310_splits_jobs_and_epsilon_are_exact() -> None:
    split_sets = (
        M1310_SCREEN_FIT_SEEDS,
        M1310_SCREEN_PROBE_SEEDS,
        M1310_CONFIRMATION_FIT_SEEDS,
        M1310_CONFIRMATION_EVALUATION_SEEDS,
        M1310_AUDIT_SEEDS,
    )
    assert [len(rows) for rows in split_sets] == [20, 8, 40, 20, 20]
    assert len(set().union(*map(set, split_sets))) == sum(map(len, split_sets))
    assert len(m1310_screen_jobs()) == len(set(m1310_screen_jobs())) == 6
    assert m1310_epsilon(0) == pytest.approx(0.20)
    assert m1310_epsilon(1_999) == pytest.approx(0.05)
    assert m1310_epsilon(9_000, stage="confirmation") == pytest.approx(0.05)
    with pytest.raises(ValueError):
        m1310_epsilon(-1)


def test_m1310_layouts_are_fresh_and_geometry_unique() -> None:
    names = [name for name in LAYOUTS if name.startswith("m1310_")]
    assert len(names) == 20
    geometries = {
        (
            layout.agent_xy,
            layout.food_low,
            layout.food_high,
            layout.toy_xy,
            layout.rest_xy,
        )
        for name in names
        for layout in (LAYOUTS[name],)
    }
    points = {
        point
        for name in names
        for point in (
            LAYOUTS[name].agent_xy,
            LAYOUTS[name].food_low,
            LAYOUTS[name].food_high,
            LAYOUTS[name].toy_xy,
            LAYOUTS[name].rest_xy,
        )
    }
    prior_points = {
        point
        for name, layout in LAYOUTS.items()
        if name.startswith(("m10_", "m13")) and not name.startswith("m1310_")
        for point in (
            layout.agent_xy,
            layout.food_low,
            layout.food_high,
            layout.toy_xy,
            layout.rest_xy,
        )
    }
    assert len(geometries) == 20
    assert len(points) == 100
    assert points.isdisjoint(prior_points)
    assert max(abs(value) for point in points for value in point) == pytest.approx(0.831)


def test_m1310_ledger_is_ordered_one_shot_and_fit_only_resumable(tmp_path: Path) -> None:
    ledger = SplitOpenLedger(tmp_path / "ledger.json", protocol_fingerprint="protocol")
    assert ledger.ensure_screen_fit_open()["opened_partitions"] == ["screen_fit"]
    assert ledger.ensure_screen_fit_open()["opened_partitions"] == ["screen_fit"]
    with pytest.raises(SplitDependencyError):
        ledger.open_partition("confirmation_fit")
    assert ledger.open_partition("screen_probe")["opened_partitions"] == [
        "screen_fit",
        "screen_probe",
    ]
    with pytest.raises(SplitAlreadyOpenedError):
        ledger.open_partition("screen_probe")
    with pytest.raises(Exception, match="unavailable"):
        ledger.ensure_screen_fit_open()


def test_m1310_teacher_dataset_has_only_public_arrays_and_enforces_mask(tmp_path: Path) -> None:
    dataset = tmp_path / "teacher.npz"
    _synthetic_dataset(dataset)
    features, masks, labels = load_m1310_teacher_dataset(dataset)
    assert features.shape == (256, 30)
    assert masks.shape == (256, 8)
    assert labels.shape == (256,)
    with np.load(dataset, allow_pickle=False) as payload:
        assert set(payload.files) == {"features", "masks", "labels"}
    masks[0, labels[0]] = False
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, features=features, masks=masks, labels=labels)
    with pytest.raises(ValueError, match="ineligible"):
        load_m1310_teacher_dataset(invalid)


def test_m1310_imitation_arms_start_identically_and_fit_identically(tmp_path: Path) -> None:
    dataset = tmp_path / "teacher.npz"
    _synthetic_dataset(dataset)
    candidate = CompactM1310QPolicy(m1310_arm="imitation_warmstart_rl", seed=M1310_SCREEN_TRAINING_SEEDS[0])
    guardrail = CompactM1310QPolicy(m1310_arm="imitation_only_guardrail", seed=M1310_SCREEN_TRAINING_SEEDS[0])
    assert candidate.base_parameter_hash == guardrail.base_parameter_hash
    candidate_metrics = fit_m1310_imitation(candidate, dataset_path=dataset)
    guardrail_metrics = fit_m1310_imitation(guardrail, dataset_path=dataset)
    assert candidate_metrics["passes"] is guardrail_metrics["passes"] is True
    assert candidate.post_imitation_parameter_hash == guardrail.post_imitation_parameter_hash
    assert all(
        np.array_equal(candidate.online.params[key], guardrail.online.params[key])
        for key in candidate.online.params
    )


def test_m1310_policy_artifact_binds_external_arm_dataset_and_kernel(tmp_path: Path) -> None:
    policy = CompactM1310QPolicy(m1310_arm="random_init_rl_control", seed=M1310_SCREEN_TRAINING_SEEDS[0])
    artifact = write_m1310_policy(tmp_path / "policy.json", policy, dataset_sha256="dataset-hash")
    restored = load_m1310_policy(artifact["path"], expected_dataset_sha256="dataset-hash")
    assert restored.m1310_arm == "random_init_rl_control"
    assert m1310_policy_fingerprint(restored) == artifact["policy_fingerprint"]
    with pytest.raises(ValueError, match="dataset"):
        load_m1310_policy(artifact["path"], expected_dataset_sha256="wrong")


def test_m1310_screen_gate_requires_control_gain_retention_and_both_replicas() -> None:
    candidate = _matrix()
    control = _matrix(full=False, safe=0.75, unsafe_waits=2)
    guardrail = _matrix()
    random = _matrix(survived=False, maintenance=False, full=False, safe=0.20)
    gate = m1310_screen_promotes(
        candidate,
        control,
        guardrail,
        random,
        training_seeds=M1310_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M1310_SCREEN_PROBE_CONDITIONS),
        env_seeds=M1310_SCREEN_PROBE_SEEDS,
    )
    assert gate["passes"] is True
    broken = deepcopy(candidate)
    for row in broken:
        if row["training_seed"] == M1310_SCREEN_TRAINING_SEEDS[1]:
            row["full_gate_success"] = False
    assert (
        m1310_screen_promotes(
            broken,
            control,
            guardrail,
            random,
            training_seeds=M1310_SCREEN_TRAINING_SEEDS,
            conditions=tuple(M1310_SCREEN_PROBE_CONDITIONS),
            env_seeds=M1310_SCREEN_PROBE_SEEDS,
        )["passes"]
        is False
    )


def test_m1310_protocol_fingerprint_binds_conditions_and_sources() -> None:
    first = m1310_protocol_fingerprint()
    second = m1310_protocol_fingerprint()
    assert first == second and len(first) == 64
    assert all(
        str(row["layout_id"]).startswith("m1310_screen_") for row in M1310_SCREEN_FIT_CONDITIONS.values()
    )


def test_m1310_cli_dispatches_screen_without_opening_it(monkeypatch, tmp_path: Path, capsys) -> None:
    called: dict[str, Path] = {}

    def fake_writer(path: Path) -> dict[str, object]:
        called["path"] = path
        return {"screen": {"passes": False, "status": "reject"}}

    monkeypatch.setattr(cli, "write_m1310_screen_report", fake_writer)
    output = tmp_path / "screen.json"
    cli.main(["m1310-screen", "--output", str(output)])
    assert called == {"path": output}
    assert json.loads(capsys.readouterr().out)["status"] == "reject"


def test_m1310_screen_refuses_non_six_worker_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="worker count"):
        m1310.m1310_screen(
            artifact_dir=tmp_path / "artifacts",
            ledger_path=tmp_path / "ledger.json",
            workers=5,
        )
    assert not (tmp_path / "ledger.json").exists()
