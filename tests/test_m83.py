from __future__ import annotations

import json

from ecosystem_gym.m82 import M81_FROZEN_POLICY_FINGERPRINT, M81_FROZEN_PROTOCOL_FINGERPRINT
from ecosystem_gym.m83 import M83_CONDITIONS, M83_DIAGNOSTIC_SEEDS, m83_diagnostics, write_m83_report


def test_m83_is_a_frozen_one_factor_diagnostic_matrix(tmp_path) -> None:
    report = write_m83_report(tmp_path / "m83.json")
    assert json.loads((tmp_path / "m83.json").read_text(encoding="utf-8")) == report
    assert tuple(report["diagnostic_seeds"]) == M83_DIAGNOSTIC_SEEDS
    assert report["m81_freeze"]["m83_training_episodes"] == 0
    assert report["m81_freeze"]["observed_protocol_fingerprint"] == M81_FROZEN_PROTOCOL_FINGERPRINT
    assert report["m81_freeze"]["observed_policy_fingerprint"] == M81_FROZEN_POLICY_FINGERPRINT
    assert set(report["results"]) == set(M83_CONDITIONS)
    for condition, metrics in report["results"].items():
        assert metrics["state_oracle_ceiling"]["successes"] == len(M83_DIAGNOSTIC_SEEDS)
        initial = report["initial_frame_diagnostics"][condition]
        for policy in initial.values():
            assert policy["episodes"] == len(M83_DIAGNOSTIC_SEEDS)
            assert sum(policy["first_action_counts"].values()) == len(M83_DIAGNOSTIC_SEEDS)


def test_m83_rejects_custom_seeds() -> None:
    try:
        m83_diagnostics(seeds=(300,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("custom M8.3 diagnostic seeds must be rejected")
