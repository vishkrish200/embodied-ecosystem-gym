from __future__ import annotations

import numpy as np
import pytest

from ecosystem_gym.experiments import cli, m1312
from ecosystem_gym.experiments.m1312 import M1312_CHECK_SEEDS, M1312_FIT_CONDITIONS, M1312_FIT_SEEDS, M1312_LEARNING_RATES, M1312_TRAINING_SEEDS, _assert_layouts, _evaluate, _gate, _train_ppo, m1312_protocol_fingerprint
from ecosystem_gym.experiments.m1312_support import M1312DevelopmentLedger
from ecosystem_gym.maintenance.ppo_r2 import CorrectedMaskedPPOPolicy
from ecosystem_gym.experiments.m139 import m139_config


def test_m1312_splits_ledgers_layouts_and_initializations_are_fresh(tmp_path) -> None:
    assert M1312_FIT_SEEDS == tuple(range(6100, 6120))
    assert M1312_CHECK_SEEDS == tuple(range(6120, 6140))
    assert set(M1312_FIT_SEEDS).isdisjoint(M1312_CHECK_SEEDS)
    assert M1312_TRAINING_SEEDS == (20261321, 20261322, 20261323, 20261324)
    _assert_layouts()
    ledger = M1312DevelopmentLedger(tmp_path / "ledger.json", protocol_fingerprint=m1312_protocol_fingerprint())
    assert ledger.open("development_fit")["opened_partitions"] == ["development_fit"]
    assert ledger.open("development_fit", resumable_fit=True)["opened_partitions"] == ["development_fit"]
    assert ledger.open("development_check")["opened_partitions"] == ["development_fit", "development_check"]
    with pytest.raises(ValueError):
        ledger.open("development_check")


def test_m1312_changes_only_the_learning_rate(monkeypatch) -> None:
    seen: list[float] = []
    original = CorrectedMaskedPPOPolicy.update
    monkeypatch.setattr(m1312, "M1312_PPO_ROLLOUT", 32)

    def wrapped(self, rollout, **kwargs):
        seen.append(float(kwargs["learning_rate"]))
        return original(self, rollout, **kwargs)

    monkeypatch.setattr(CorrectedMaskedPPOPolicy, "update", wrapped)
    baseline, _ = _train_ppo("ppo_lr_3e4_baseline", 7, decision_budget=10**9, max_episodes=2)
    baseline_initial = CorrectedMaskedPPOPolicy(seed=7, config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
    candidate_initial = CorrectedMaskedPPOPolicy(seed=7, config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
    assert baseline_initial.fingerprint() == candidate_initial.fingerprint()
    assert baseline.fingerprint() != baseline_initial.fingerprint()
    assert seen and set(seen) == {M1312_LEARNING_RATES["ppo_lr_3e4_baseline"]}
    seen.clear()
    _train_ppo("ppo_lr_1e3_candidate", 7, decision_budget=10**9, max_episodes=2)
    assert seen and set(seen) == {M1312_LEARNING_RATES["ppo_lr_1e3_candidate"]}


def test_m1312_gate_requires_every_candidate_replica() -> None:
    rows = []
    for arm in M1312_LEARNING_RATES:
        for seed in M1312_TRAINING_SEEDS:
            rows.append({"arm": arm, "training_seed": seed, "full_objective_success": arm == "ppo_lr_1e3_candidate"})
    assert _gate(rows)["passes"] is True
    rows[-1]["full_objective_success"] = False
    assert _gate(rows)["passes"] is False


def test_m1312_trace_strictly_replays(tmp_path) -> None:
    policy = CorrectedMaskedPPOPolicy(seed=19, config=m139_config(), protocol_fingerprint=m1312_protocol_fingerprint())
    rows, replay_count = _evaluate(policy, arm="ppo_lr_3e4_baseline", seed=19, conditions={"persistent_reference": M1312_FIT_CONDITIONS["persistent_reference"]}, seeds=(M1312_FIT_SEEDS[0],), trace_dir=tmp_path)
    assert len(rows) == replay_count == 1


def test_m1312_cli_is_registered_without_running(monkeypatch, tmp_path) -> None:
    called = {}

    def fake(path):
        called["path"] = path
        return {"label": "x", "elapsed_seconds": 0.0, "strict_replay_count": 0}

    monkeypatch.setattr(cli, "write_m1312_fit_smoke_report", fake)
    cli.main(["m1312-fit-smoke", "--output", str(tmp_path / "report.json")])
    assert "path" in called
