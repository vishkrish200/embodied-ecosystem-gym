from __future__ import annotations

import numpy as np

from ecosystem_gym.experiments.m139 import m139_config
from ecosystem_gym.maintenance.ppo import MaskedPPOPolicy, Rollout, masked_probabilities, semimarkov_gae


def test_masked_probabilities_and_one_action_gradient_are_finite() -> None:
    probabilities=masked_probabilities(np.asarray([[10.,-3.,4.,2.,1.,0.,-1.,7.]]),np.asarray([[False,False,True,False,False,False,False,False]]))
    assert probabilities[0,2] == 1. and probabilities.sum() == 1. and np.count_nonzero(probabilities)==1
    policy=MaskedPPOPolicy(seed=1,config=m139_config()); rollout=Rollout(np.zeros((2,30),np.float32),np.asarray([2,2]),np.tile(np.asarray([False,False,True,False,False,False,False,False]),(2,1)),np.zeros(2,np.float32),np.ones(2,np.float32),np.ones(2,np.float32),np.asarray([False,True]),np.zeros(2,np.float32),np.zeros(2,np.float32)); assert np.isfinite(policy.update(rollout)["actor_loss"])


def test_variable_duration_gae_and_artifact_determinism(tmp_path) -> None:
    advantage,returns=semimarkov_gae(np.asarray([1.,2.]),np.asarray([.5,.25]),np.asarray([.25,0.]),np.asarray([2.,1.]),np.asarray([False,True]))
    delta1=2.-.25; delta0=1.+.99**2*.25-.5; np.testing.assert_allclose(advantage[1],delta1,atol=1e-6); np.testing.assert_allclose(advantage[0],delta0+.99**2*.95*delta1,atol=1e-6); assert np.allclose(returns,advantage+np.asarray([.5,.25]))
    first=MaskedPPOPolicy(seed=7,config=m139_config(),protocol_fingerprint="test"); second=MaskedPPOPolicy(seed=7,config=m139_config(),protocol_fingerprint="test"); assert first.fingerprint()==second.fingerprint(); artifact=first.save(tmp_path/"p.json"); assert MaskedPPOPolicy.load(artifact["path"],config=m139_config(),protocol_fingerprint="test").fingerprint()==first.fingerprint()
