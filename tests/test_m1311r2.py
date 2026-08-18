from __future__ import annotations

import numpy as np
import pytest

from ecosystem_gym.experiments import cli
from ecosystem_gym.experiments import m1311r2
from ecosystem_gym.experiments.m1311r2 import CorrectedMaskedPPOPolicy, M1311R2_CHECK_SEEDS, M1311R2_FIT_SEEDS, _assert_layouts, _evaluate, _train_ppo, m1311r2_protocol_fingerprint
from ecosystem_gym.experiments.m1311r2_support import R2DevelopmentLedger
from ecosystem_gym.experiments.m139 import m139_config


def test_r2_splits_ledgers_and_layouts_are_fresh(tmp_path) -> None:
    assert len(M1311R2_FIT_SEEDS) == len(M1311R2_CHECK_SEEDS) == 20
    assert set(M1311R2_FIT_SEEDS).isdisjoint(M1311R2_CHECK_SEEDS)
    assert set(range(6080, 6100)).isdisjoint(set(M1311R2_FIT_SEEDS) | set(M1311R2_CHECK_SEEDS))
    _assert_layouts()
    ledger = R2DevelopmentLedger(tmp_path / "ledger.json", protocol_fingerprint=m1311r2_protocol_fingerprint())
    assert ledger.open("development_fit")["opened_partitions"] == ["development_fit"]
    assert ledger.open("development_fit", resumable_fit=True)["opened_partitions"] == ["development_fit"]
    assert ledger.open("development_check")["opened_partitions"] == ["development_fit", "development_check"]
    with pytest.raises(ValueError):
        ledger.open("development_check")


def test_rollout_accumulates_across_terminal_episodes(monkeypatch) -> None:
    seen: list[np.ndarray] = []
    original = CorrectedMaskedPPOPolicy.update
    monkeypatch.setattr(m1311r2, "M1311R2_PPO_ROLLOUT", 32)
    def wrapped(self, rollout, **kwargs):
        seen.append(rollout.done.copy())
        return original(self, rollout, **kwargs)
    monkeypatch.setattr(CorrectedMaskedPPOPolicy, "update", wrapped)
    _, training = _train_ppo(2, decision_budget=10**9, max_episodes=6)
    assert training["update_batches"] >= 2
    assert any(len(done) == 32 and np.count_nonzero(done) >= 2 for done in seen)


def test_r2_artifacts_are_deterministic_and_fit_trace_replays(tmp_path) -> None:
    first = CorrectedMaskedPPOPolicy(seed=19, config=m139_config(), protocol_fingerprint="test")
    second = CorrectedMaskedPPOPolicy(seed=19, config=m139_config(), protocol_fingerprint="test")
    assert first.fingerprint() == second.fingerprint()
    rows, replay_count = _evaluate(first, arm="corrected_masked_semimarkov_ppo_candidate", seed=19, conditions={"persistent_reference": m1311r2.M1311R2_FIT_CONDITIONS["persistent_reference"]}, seeds=(M1311R2_FIT_SEEDS[0],), trace_dir=tmp_path)
    assert len(rows) == replay_count == 1


def test_r2_cli_is_registered_without_running(monkeypatch, tmp_path) -> None:
    called = {}
    def fake(path):
        called["path"] = path
        return {"label": "x", "elapsed_seconds": 0.0, "strict_replay_count": 0}
    monkeypatch.setattr(cli, "write_m1311r2_fit_smoke_report", fake)
    cli.main(["m1311r2-fit-smoke", "--output", str(tmp_path / "report.json")])
    assert "path" in called
