from __future__ import annotations

from ecosystem_gym.m9 import M9_AUDIT_SEEDS, M9_TRAIN_SEEDS, M9_VALIDATION_SEEDS, m9_scan_coverage
from ecosystem_gym.m91 import M91_CONDITIONS, M91_SEEDS, m91_diagnosis


def test_m91_uses_fresh_seeds_and_coverage_valid_rows() -> None:
    assert set(M91_SEEDS).isdisjoint(M9_TRAIN_SEEDS)
    assert set(M91_SEEDS).isdisjoint(M9_VALIDATION_SEEDS)
    assert set(M91_SEEDS).isdisjoint(M9_AUDIT_SEEDS)
    coverage = m9_scan_coverage(M91_CONDITIONS, seeds=M91_SEEDS[:2])
    assert all(row["passes"] for row in coverage.values())


def test_m91_rejects_custom_diagnosis_seeds() -> None:
    try:
        m91_diagnosis(seeds=(1_601,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("M9.1 must reject custom diagnosis seeds")
