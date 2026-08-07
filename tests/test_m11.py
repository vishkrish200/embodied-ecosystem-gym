from ecosystem_gym.experiments.m10 import M10_VALIDATION_CONDITIONS
from ecosystem_gym.experiments.m11 import (
    M11_FROZEN_M9_POLICY_FINGERPRINT,
    M11_SAFE_DRIVE_BANDS,
    m11_baseline,
    m11_protocol_fingerprint,
    run_m11_episode,
)
from ecosystem_gym.experiments.m9 import fit_m9_policy, m9_policy_fingerprint


def test_m11_pins_the_m9_policy_and_safe_bands() -> None:
    assert m9_policy_fingerprint(fit_m9_policy()) == M11_FROZEN_M9_POLICY_FINGERPRINT
    assert M11_SAFE_DRIVE_BANDS == {"satiety_min": 0.15, "energy_min": 0.15, "boredom_max": 0.90}
    assert len(m11_protocol_fingerprint()) == 64


def test_m11_episode_records_an_inspectable_failure() -> None:
    episode = run_m11_episode(
        fit_m9_policy(),
        seed=1_800,
        condition="event_relocation",
        controls=M10_VALIDATION_CONDITIONS["event_relocation"],
    )
    assert episode.steps
    assert all("target_observation" in step and "drives_after" in step for step in episode.steps)
    assert episode.first_failure is not None
    assert episode.forced_recovery["required"]


def test_m11_rejects_custom_seeds() -> None:
    try:
        m11_baseline(seeds=(1_800,))
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("M11 baseline must reject custom seeds")
