from __future__ import annotations

import json

import numpy as np
import pytest

from ecosystem_gym.experiments.m139 import m139_config
from ecosystem_gym.maintenance.ppo_r2 import ACTIONS, CorrectedMaskedPPOPolicy, Rollout, masked_probabilities, semimarkov_gae


def _rollout(policy: CorrectedMaskedPPOPolicy, *, actions: tuple[int, ...] = (0, 1), rewards: tuple[float, ...] = (1.0, -1.0)) -> Rollout:
    features = np.zeros((len(actions), 30), np.float32)
    features[:, :3] = np.asarray(((1.0, .4, .2), (.2, 1.0, .5)), np.float32)[:len(actions)]
    masks = np.ones((len(actions), ACTIONS), np.bool_)
    old_log_probs = np.asarray([np.log(policy.action_distribution(feature, mask)[0][action]) for feature, mask, action in zip(features, masks, actions)], np.float32)
    return Rollout(features, np.asarray(actions), masks, old_log_probs, np.asarray(rewards, np.float32), np.ones(len(actions), np.float32), np.ones(len(actions), np.bool_), np.zeros(len(actions), np.float32), np.zeros(len(actions), np.float32))


def test_positive_and_negative_advantages_move_selected_probabilities_in_right_direction() -> None:
    policy = CorrectedMaskedPPOPolicy(seed=11, config=m139_config())
    rollout = _rollout(policy)
    before = [policy.action_distribution(row, mask)[0][action] for row, mask, action in zip(rollout.features, rollout.masks, rollout.actions)]
    policy.update(rollout, epochs=1, learning_rate=1e-3, max_grad_norm=10.0)
    after = [policy.action_distribution(row, mask)[0][action] for row, mask, action in zip(rollout.features, rollout.masks, rollout.actions)]
    assert after[0] > before[0]
    assert after[1] < before[1]


def test_analytic_actor_entropy_critic_and_trunk_gradients_match_central_difference() -> None:
    policy = CorrectedMaskedPPOPolicy(seed=3, config=m139_config())
    rollout = _rollout(policy)
    metrics, gradients = policy.objective_and_gradients(rollout, advantages=np.asarray([.7, -.3], np.float32), returns=np.asarray([.2, -.4], np.float32))
    assert np.isfinite(metrics["total_loss"])
    epsilon = 1e-3
    for key, index in (("ba", 0), ("wa", (0, 0)), ("bv", 0), ("wv", (0, 0)), ("w2", (0, 0)), ("w1", (0, 0))):
        original = float(policy.params[key][index])
        policy.params[key][index] = original + epsilon
        upper = policy.objective_and_gradients(rollout, advantages=np.asarray([.7, -.3], np.float32), returns=np.asarray([.2, -.4], np.float32))[0]["total_loss"]
        policy.params[key][index] = original - epsilon
        lower = policy.objective_and_gradients(rollout, advantages=np.asarray([.7, -.3], np.float32), returns=np.asarray([.2, -.4], np.float32))[0]["total_loss"]
        policy.params[key][index] = original
        assert float(gradients[key][index]) == pytest.approx((upper - lower) / (2 * epsilon), abs=4e-3)


def test_ppo_clipping_and_terminal_boundary_gae() -> None:
    policy = CorrectedMaskedPPOPolicy(seed=4, config=m139_config())
    rollout = _rollout(policy)
    rollout.old_log_probs = rollout.old_log_probs - np.log(1.4)
    high, _ = policy.objective_and_gradients(rollout, advantages=np.asarray([1.0, 1.0], np.float32), returns=np.zeros(2, np.float32))
    rollout.old_log_probs = rollout.old_log_probs + np.log(1.4) + np.log(.6)
    low, _ = policy.objective_and_gradients(rollout, advantages=np.asarray([-1.0, -1.0], np.float32), returns=np.zeros(2, np.float32))
    assert high["clip_fraction"] == 1.0 and low["clip_fraction"] == 1.0
    advantages, _ = semimarkov_gae(np.asarray([1., 2., 100.]), np.zeros(3), np.zeros(3), np.ones(3), np.asarray([False, True, True]))
    assert advantages[1] == pytest.approx(2.0) and advantages[0] == pytest.approx(1 + .99 * .95 * 2)
    assert advantages[2] == pytest.approx(100.0)


def test_one_action_mask_is_finite_and_artifact_rejects_tampering(tmp_path) -> None:
    mask = np.zeros((1, ACTIONS), np.bool_)
    mask[0, 3] = True
    probabilities = masked_probabilities(np.arange(ACTIONS, dtype=np.float32)[None, :], mask)
    assert probabilities[0, 3] == 1.0
    policy = CorrectedMaskedPPOPolicy(seed=7, config=m139_config(), protocol_fingerprint="r2-test")
    artifact = policy.save(tmp_path / "policy.json")
    assert CorrectedMaskedPPOPolicy.load(artifact["path"], config=m139_config(), protocol_fingerprint="r2-test").fingerprint() == policy.fingerprint()
    value = json.loads((tmp_path / "policy.json").read_text())
    value["network"]["w1"][0][0] = float("nan")
    (tmp_path / "bad.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        CorrectedMaskedPPOPolicy.load(tmp_path / "bad.json", config=m139_config(), protocol_fingerprint="r2-test")
    with pytest.raises(ValueError):
        CorrectedMaskedPPOPolicy.load(artifact["path"], config=m139_config(), protocol_fingerprint="wrong")
