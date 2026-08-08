from __future__ import annotations

from ecosystem_gym.experiments import cli
from ecosystem_gym.experiments.m1311 import M1311_CHECK_SEEDS, M1311_FIT_SEEDS, M1311_TRAINING_SEEDS, _assert_layouts, m1311_protocol_fingerprint
from ecosystem_gym.experiments.m1311_support import DevelopmentLedger


def test_m1311_splits_are_fresh_and_ledger_is_one_way(tmp_path) -> None:
    assert len(M1311_FIT_SEEDS)==len(M1311_CHECK_SEEDS)==20 and set(M1311_FIT_SEEDS).isdisjoint(M1311_CHECK_SEEDS) and len(M1311_TRAINING_SEEDS)==4
    _assert_layouts(); ledger=DevelopmentLedger(tmp_path/"ledger.json",protocol_fingerprint=m1311_protocol_fingerprint()); assert ledger.open("development_fit")["opened_partitions"]==["development_fit"]; assert ledger.open("development_fit",resumable_fit=True)["opened_partitions"]==["development_fit"]; assert ledger.open("development_check")["opened_partitions"]==["development_fit","development_check"]


def test_m1311_cli_is_registered_without_running(monkeypatch,tmp_path) -> None:
    called={}
    def fake(path):
        called["path"] = path
        return {"label":"x","elapsed_seconds":0.,"strict_replay_count":0}
    monkeypatch.setattr(cli,"write_m1311_fit_smoke_report",fake)
    cli.main(["m1311-fit-smoke","--output",str(tmp_path/"report.json")])
    assert "path" in called
