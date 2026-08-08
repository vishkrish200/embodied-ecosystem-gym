from __future__ import annotations

import numpy as np

from ecosystem_gym.actions import ActionKind, ActionOutcome
from ecosystem_gym.experiments.m13 import M13Macro, M13Memory, compile_macro as frozen_compile
from ecosystem_gym.experiments.m132 import encode_features as frozen_features
from ecosystem_gym.experiments.m133 import advance_m133_memory
from ecosystem_gym.experiments.m134 import m134_mask
from ecosystem_gym.experiments.m139 import M139RewardState, m139_config, m139_learning_reward
from ecosystem_gym.maintenance.contract import MaintenanceMacro, MaintenanceMemory, advance_memory, compile_macro, eligible_macro_mask, encode_features
from ecosystem_gym.maintenance.reward import MaintenanceRewardState, duration_discount, learning_reward


def _observation(rng: np.random.Generator) -> dict[str, object]:
    return {"agent_xy":rng.uniform(-.8,.8,2).astype(np.float32),"food_xy":rng.uniform(-.8,.8,2).astype(np.float32),"toy_xy":rng.uniform(-.8,.8,2).astype(np.float32),"rest_xy":rng.uniform(-.8,.8,2).astype(np.float32),"drives":rng.uniform(0,1,3).astype(np.float32),"holding_food":int(rng.integers(2)),"prior_outcome":int(rng.integers(len(ActionOutcome)))}


def test_contract_matches_frozen_public_boundary_for_thousand_generated_cases() -> None:
    rng=np.random.default_rng(1311); config=m139_config()
    for _ in range(1_000):
        observation=_observation(rng); old=M13Memory(feed_count=int(rng.integers(4)),play_count=int(rng.integers(4)),rest_count=int(rng.integers(3)),food_cooldown_bucket=float(rng.uniform(0,5)),last_macro=None,pending_recovery=bool(rng.integers(2))); new=MaintenanceMemory(old.feed_count,old.play_count,old.rest_count,old.food_cooldown_bucket,None,old.pending_recovery)
        assert np.array_equal(encode_features(observation,new),frozen_features(observation,old))
        assert np.array_equal(eligible_macro_mask(observation,config),m134_mask(observation,config))
        macro=MaintenanceMacro(int(rng.integers(8))); current,frozen=compile_macro(macro,observation,config),frozen_compile(M13Macro(int(macro)),observation,config)
        assert current["kind"] == frozen["kind"] and np.array_equal(current["target"],frozen["target"]) and np.array_equal(current["duration"],frozen["duration"])


def test_contract_memory_and_reward_recompose_exactly_for_thousand_transitions() -> None:
    rng=np.random.default_rng(13112); config=m139_config()
    for _ in range(1_000):
        before=_observation(rng); after=_observation(rng); after["prior_outcome"]=int(rng.integers(len(ActionOutcome))); macro=MaintenanceMacro(int(rng.integers(8))); action=compile_macro(macro,before,config); info={"outcome":list(ActionOutcome)[int(after["prior_outcome"])].value,"disturbance":"food_relocated" if macro is MaintenanceMacro.PICK_UP and rng.integers(2) else None,"feed_cycles":int(rng.integers(4)),"play_cycles":int(rng.integers(4)),"rest_cycles":int(rng.integers(3)),"survived":bool(rng.integers(2))}
        old,new=M13Memory(),MaintenanceMemory(); advance_m133_memory(old,observation_before=before,macro=M13Macro(int(macro)),action=action,observation_after=after,config=config); advance_memory(new,observation_before=before,macro=macro,action=action,observation_after=after,config=config)
        assert (old.feed_count,old.play_count,old.rest_count,old.food_cooldown_bucket,old.last_macro,old.pending_recovery)==(new.feed_count,new.play_count,new.rest_count,new.food_cooldown_bucket,MaintenanceMacro(int(new.last_macro)) if new.last_macro is not None else None,new.pending_recovery)
        old_state,new_state=M139RewardState(True),MaintenanceRewardState(True); frozen,_=m139_learning_reward(arm="public_potential_candidate",environment_reward=3.,state=old_state,observation_before=before,observation_after=after,action=action,info=info,terminated=False,truncated=False,macro=M13Macro(int(macro))); actual,components=learning_reward(environment_reward=3.,state=new_state,observation_before=before,observation_after=after,action=action,info=info,terminated=False,truncated=False)
        np.testing.assert_allclose(actual,frozen,rtol=1e-7,atol=1e-7)
        assert components["candidate_recomposed"] == components["learning_reward"]


def test_duration_discount_rejects_bad_values() -> None:
    for value in (0.,-1.,float("nan"),float("inf")):
        try: duration_discount(value)
        except ValueError: pass
        else: raise AssertionError("invalid duration was accepted")
