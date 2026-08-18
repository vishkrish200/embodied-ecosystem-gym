"""M13.11 development-only masked semi-Markov PPO versus random-init DQN.

This is deliberately not a screen harness: it owns only the 6000–6039
development ledger and refuses to define confirmation or audit partitions.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..env import EcosystemEnv
from ..maintenance import MaintenanceMacro, MaintenanceRewardState, advance_memory, compile_macro, eligible_macro_mask, encode_features, learning_reward
from ..maintenance.ppo import MaskedPPOPolicy, Rollout
from ..tasks import LAYOUTS
from .m10 import m10_scan_coverage
from .m13 import M13Macro, _options
from .m132 import M132_REPLAY_WARMUP, _MLP
from .m133 import m133_epsilon
from .m139 import (CompactM139QPolicy, M139RewardState, _DurationReplay, _duration, m139_config,
                   m139_learning_reward, m139_policy_fingerprint, replay_m139_trace, run_m139_episode)
from .m1311_support import DevelopmentLedger, content_hash

for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

M1311_PROTOCOL_VERSION = "m13.11-development-ppo-r1"
M1311_FIT_SEEDS = tuple(range(6000,6020)); M1311_CHECK_SEEDS = tuple(range(6020,6040))
M1311_TRAINING_SEEDS = (20261311,20261312,20261313,20261314)
M1311_DECISION_BUDGET = 400_000; M1311_PPO_ROLLOUT = 2048; M1311_PPO_EPOCHS = 4
M1311_CANONICAL_LEDGER_PATH = Path(__file__).resolve().parents[2] / "artifacts/m1311/split-open-ledger.json"
Arm = Literal["masked_semimarkov_ppo_candidate","random_init_duration_dqn_control"]


def _controls(prefix: str) -> dict[str, dict[str, Any]]:
    return {
      "persistent_reference":{"layout_id":f"{prefix}_northeast","food_variant":"orange","toy_variant":"ball","camera_control":"scan_v2","initial_scan_sector":"north"},
      "renewal_and_morphology":{"layout_id":f"{prefix}_southwest","food_variant":"purple","food_shape_variant":"capsule","toy_variant":"cube","agent_shape_variant":"capsule","camera_control":"scan_v2","initial_scan_sector":"south"},
      "event_relocation":{"layout_id":f"{prefix}_northwest","food_variant":"blue","food_shape_variant":"box","toy_variant":"capsule","lighting_variant":"dim","camera_control":"scan_v2","initial_scan_sector":"east","event_relocation_on_first_pickup":True},
      "compound":{"layout_id":f"{prefix}_southeast","food_variant":"red","food_shape_variant":"capsule","toy_variant":"cube","agent_shape_variant":"box","dynamics_variant":"grippy","blocked_distractor":True,"distractor_xy":[-0.04,0.04],"camera_control":"scan_v2","initial_scan_sector":"west","event_relocation_on_first_pickup":True},
    }

M1311_FIT_CONDITIONS, M1311_CHECK_CONDITIONS = _controls("m1311_fit"), _controls("m1311_check")


def _source_hashes() -> dict[str,str]:
    root=Path(__file__).resolve().parents[2]
    package=root / "ecosystem_gym"
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),package/"maintenance/contract.py",package/"maintenance/reward.py",package/"maintenance/ppo.py",package/"tasks.py")}


def m1311_protocol_fingerprint() -> str:
    payload={"version":M1311_PROTOCOL_VERSION,"splits":{"development_fit":M1311_FIT_SEEDS,"development_check":M1311_CHECK_SEEDS},"conditions":{"fit":M1311_FIT_CONDITIONS,"check":M1311_CHECK_CONDITIONS},"training_seeds":M1311_TRAINING_SEEDS,"decision_budget":M1311_DECISION_BUDGET,"ppo":{"network":[30,64,64,8,1],"rollout":M1311_PPO_ROLLOUT,"epochs":M1311_PPO_EPOCHS,"lr":3e-4,"clip":.2,"gae_lambda":.95,"gamma":.99},"dqn":"M13.10 random-init duration-aware DQN schedule","workers":"spawn one numeric thread","sources":_source_hashes()}
    return content_hash(payload)


def _assert_layouts() -> None:
    names=[name for name in LAYOUTS if name.startswith("m1311_")]
    if len(names)!=8 or any(name not in LAYOUTS for controls in (M1311_FIT_CONDITIONS,M1311_CHECK_CONDITIONS) for name in [str(row["layout_id"]) for row in controls.values()]): raise ValueError("M13.11 layouts are incomplete")
    points=lambda layout:(layout.agent_xy,layout.food_low,layout.food_high,layout.toy_xy,layout.rest_xy)
    current={point for name in names for point in points(LAYOUTS[name])}; previous={point for name,layout in LAYOUTS.items() if not name.startswith("m1311_") for point in points(layout)}
    if len(current)!=40 or current & previous: raise ValueError("M13.11 geometry is not fresh")


def _train_ppo(seed: int, *, decision_budget: int, max_episodes: int|None=None) -> tuple[MaskedPPOPolicy,dict[str,Any]]:
    policy=MaskedPPOPolicy(seed=seed,config=m139_config(),protocol_fingerprint=m1311_protocol_fingerprint()); rng=np.random.Generator(np.random.PCG64(seed ^ 0x50504F)); env=EcosystemEnv(policy.config); decisions=episodes=updates=0; losses=[]; rows: list[tuple[Any,...]]=[]; items=tuple(M1311_FIT_CONDITIONS.items())
    try:
      while decisions < decision_budget and (max_episodes is None or episodes < max_episodes):
        condition,controls=items[episodes%len(items)]; env_seed=M1311_FIT_SEEDS[(episodes//len(items))%len(M1311_FIT_SEEDS)]; observation,_=env.reset(seed=env_seed,options=_options(controls)); memory=policy.reset(); reward_state=MaintenanceRewardState(required_recovery=bool(controls.get("event_relocation_on_first_pickup"))); episodes+=1
        while True:
          features=encode_features(observation,memory); mask=eligible_macro_mask(observation,policy.config); action_index,logp,value=policy.sample(features,mask,rng); macro=MaintenanceMacro(action_index); action=compile_macro(macro,observation,policy.config); next_observation,environment_reward,terminated,truncated,info=env.step(action); reward,_=learning_reward(environment_reward=environment_reward,state=reward_state,observation_before=observation,observation_after=next_observation,action=action,info=info,terminated=terminated,truncated=truncated); advance_memory(memory,observation_before=observation,macro=macro,action=action,observation_after=next_observation,config=policy.config); next_features=encode_features(next_observation,memory); rows.append((features,action_index,mask,logp,reward,float(np.asarray(action["duration"]).item()),terminated or truncated,value,policy.value(next_features))); decisions+=1; observation=next_observation
          if len(rows)>=M1311_PPO_ROLLOUT or terminated or truncated:
            block=rows; rows=[]; rollout=Rollout(np.asarray([r[0] for r in block],np.float32),np.asarray([r[1] for r in block],np.int64),np.asarray([r[2] for r in block],np.bool_),np.asarray([r[3] for r in block],np.float32),np.asarray([r[4] for r in block],np.float32),np.asarray([r[5] for r in block],np.float32),np.asarray([r[6] for r in block],np.bool_),np.asarray([r[7] for r in block],np.float32),np.asarray([r[8] for r in block],np.float32)); metrics=policy.update(rollout,epochs=M1311_PPO_EPOCHS); updates+=1; losses.append(metrics)
          if terminated or truncated: break
    finally: env.close()
    return policy,{"decisions":decisions,"episodes":episodes,"updates":updates,"last_update":losses[-1] if losses else {},"overshoot":max(0,decisions-decision_budget)}


def _train_dqn(seed: int, *, decision_budget: int, max_episodes: int|None=None) -> tuple[CompactM139QPolicy,dict[str,Any]]:
    policy=CompactM139QPolicy(arm="public_potential_candidate",seed=seed,stage="fit_smoke"); rng=np.random.default_rng(seed); policy.online=_MLP(rng); policy.target=policy.online.copy(); policy.update_count=0; replay=_DurationReplay(); env=EcosystemEnv(policy.config); decisions=episodes=0; losses=[]; items=tuple(M1311_FIT_CONDITIONS.items())
    try:
      while decisions < decision_budget and (max_episodes is None or episodes < max_episodes):
        _,controls=items[episodes%len(items)]; env_seed=M1311_FIT_SEEDS[(episodes//len(items))%len(M1311_FIT_SEEDS)]; observation,_=env.reset(seed=env_seed,options=_options(controls)); memory=policy.reset(); state=M139RewardState(required_recovery=bool(controls.get("event_relocation_on_first_pickup"))); epsilon=m133_epsilon(episodes); episodes+=1
        while True:
          features,mask=policy.features(observation,memory),policy.mask(observation); macro=M13Macro(int(rng.choice(np.flatnonzero(mask)))) if rng.random()<epsilon else policy.choose(observation,memory); action=compile_macro(MaintenanceMacro(int(macro)),observation,policy.config); next_observation,env_reward,terminated,truncated,info=env.step(action); reward,_=m139_learning_reward(arm="public_potential_candidate",environment_reward=env_reward,state=state,observation_before=observation,observation_after=next_observation,action=action,info=info,terminated=terminated,truncated=truncated,macro=macro); policy.observe(memory,observation_before=observation,macro=macro,action=action,observation_after=next_observation); replay.add(features,int(macro),reward,policy.features(next_observation,memory),policy.mask(next_observation),terminated or truncated,_duration(action)); decisions+=1
          if replay.size>=M132_REPLAY_WARMUP and decisions%policy.update_every==0: losses.append(policy._update(replay,rng))
          observation=next_observation
          if terminated or truncated: break
    finally: env.close()
    return policy,{"decisions":decisions,"episodes":episodes,"updates":policy.update_count,"mean_huber_loss":float(np.mean(losses)) if losses else 0.,"overshoot":max(0,decisions-decision_budget)}


def _worker(arm: Arm, seed: int, decision_budget: int, max_episodes: int|None, artifact_dir: str) -> dict[str,Any]:
    started=time.perf_counter(); policy,training=_train_ppo(seed,decision_budget=decision_budget,max_episodes=max_episodes) if arm=="masked_semimarkov_ppo_candidate" else _train_dqn(seed,decision_budget=decision_budget,max_episodes=max_episodes); target=Path(artifact_dir)/f"seed-{seed}"/f"{arm}.json"; artifact=policy.save(target) if isinstance(policy,MaskedPPOPolicy) else _save_dqn(policy,target); return {"arm":arm,"seed":seed,"pid":os.getpid(),"elapsed_seconds":time.perf_counter()-started,"training":training,"artifact":artifact}


def _save_dqn(policy: CompactM139QPolicy,path: Path) -> dict[str,str]:
    path.parent.mkdir(parents=True,exist_ok=True); payload={"schema_version":"m1311-dqn-control-v1","seed":policy.seed,"network":{k:v.tolist() for k,v in policy.online.params.items()},"fingerprint":m139_policy_fingerprint(policy)}; path.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n"); return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"policy_fingerprint":payload["fingerprint"]}


def _load_dqn(path: str|Path) -> CompactM139QPolicy:
    payload=json.loads(Path(path).read_text()); policy=CompactM139QPolicy(arm="public_potential_candidate",seed=int(payload["seed"]),stage="fit_smoke")
    for key,values in payload["network"].items(): policy.online.params[key]=np.asarray(values,dtype=np.float32)
    policy.target=policy.online.copy()
    if payload.get("fingerprint")!=m139_policy_fingerprint(policy): raise ValueError("DQN artifact fingerprint mismatch")
    return policy


def _wave(*, artifact_dir: Path, budget: int, max_episodes: int|None, workers: int|None) -> list[dict[str,Any]]:
    worker_count=workers or min(8,max(1,os.cpu_count() or 1)); context=multiprocessing.get_context("spawn"); jobs=[(arm,seed) for arm in ("masked_semimarkov_ppo_candidate","random_init_duration_dqn_control") for seed in M1311_TRAINING_SEEDS]; results=[]
    with ProcessPoolExecutor(max_workers=min(worker_count,len(jobs)),mp_context=context) as executor:
      futures=[executor.submit(_worker,arm,seed,budget,max_episodes,str(artifact_dir)) for arm,seed in jobs]
      for future in as_completed(futures): results.append(future.result())
    return results


def _evaluate(policy: Any, *, arm: Arm, seed: int, conditions: dict[str,dict[str,Any]], seeds: tuple[int,...], trace_dir: Path) -> tuple[list[dict[str,Any]],int]:
    rows=[]; replayed=0
    for condition,controls in conditions.items():
      for env_seed in seeds:
        trace=trace_dir/f"seed-{seed}"/arm/condition/f"seed-{env_seed}.jsonl"; fingerprint=m139_policy_fingerprint(policy) if isinstance(policy,CompactM139QPolicy) else None; episode=run_m139_episode(policy,arm="public_potential_candidate",training_seed=seed,seed=env_seed,condition=condition,controls=controls,trace_path=trace,policy_label=f"m1311/{arm}/{seed}",policy_fingerprint=fingerprint); replay_m139_trace(trace,policy); replayed+=1; rows.append({"training_seed":seed,"arm":arm,"condition":condition,"env_seed":env_seed,"survived":episode.survived,"maintenance_complete":episode.maintenance_complete,"full_objective_success":episode.full_gate_success,"decision_safe_fraction":episode.decision_safe_fraction,"duration_safe_fraction":episode.duration_safe_fraction,"recovery_required":episode.recovery_required,"recovery_complete":episode.recovery_complete,"unsafe_wait_fraction":episode.unsafe_wait_fraction,"trace_sha256":hashlib.sha256(trace.read_bytes()).hexdigest()})
    return rows,replayed


def m1311_fit_smoke(*, artifact_dir: str|Path, workers: int|None=None) -> dict[str,Any]:
    root=Path(artifact_dir); ledger=DevelopmentLedger(root/"split-open-ledger.non-protocol.json",protocol_fingerprint=m1311_protocol_fingerprint()); ledger.open("development_fit",resumable_fit=True); started=time.perf_counter(); workers_used=min(workers or min(8,max(1,os.cpu_count() or 1)),8); results=_wave(artifact_dir=root/"policies",budget=10**9,max_episodes=32,workers=workers_used); evaluated=[]; replays=0
    for row in results:
      policy=MaskedPPOPolicy.load(row["artifact"]["path"],config=m139_config(),protocol_fingerprint=m1311_protocol_fingerprint()) if row["arm"]=="masked_semimarkov_ppo_candidate" else _load_dqn(row["artifact"]["path"]); evidence,count=_evaluate(policy,arm=row["arm"],seed=int(row["seed"]),conditions=M1311_FIT_CONDITIONS,seeds=(M1311_FIT_SEEDS[0],),trace_dir=root/"traces"); evaluated.extend(evidence); replays+=count
    return {"label":"M13.11 development-fit smoke only; non-promotional","elapsed_seconds":time.perf_counter()-started,"workers":workers_used,"thread_settings":{key:os.environ[key] for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")},"jobs":results,"strict_replay_count":replays,"evaluation":evaluated,"split_ledger":ledger.snapshot(),"canonical_protocol_ledger_accessed":False}


def m1311_development(*, artifact_dir: str|Path, ledger_path: str|Path=M1311_CANONICAL_LEDGER_PATH, workers: int|None=None) -> dict[str,Any]:
    _assert_layouts(); root=Path(artifact_dir); ledger=DevelopmentLedger(ledger_path,protocol_fingerprint=m1311_protocol_fingerprint()); ledger.open("development_fit",resumable_fit=True); coverage=m10_scan_coverage(M1311_FIT_CONDITIONS,seeds=M1311_FIT_SEEDS); started=time.perf_counter(); jobs=_wave(artifact_dir=root/"policies",budget=M1311_DECISION_BUDGET,max_episodes=None,workers=workers); ledger.open("development_check"); evidence=[]; replays=0
    for row in jobs:
      policy=MaskedPPOPolicy.load(row["artifact"]["path"],config=m139_config(),protocol_fingerprint=m1311_protocol_fingerprint()) if row["arm"]=="masked_semimarkov_ppo_candidate" else _load_dqn(row["artifact"]["path"]); rows,count=_evaluate(policy,arm=row["arm"],seed=int(row["seed"]),conditions=M1311_CHECK_CONDITIONS,seeds=M1311_CHECK_SEEDS,trace_dir=root/"traces"); evidence.extend(rows); replays+=count
    return {"label":"M13.11 development-only, non-promotional","elapsed_seconds":time.perf_counter()-started,"protocol_fingerprint":m1311_protocol_fingerprint(),"coverage":coverage,"jobs":jobs,"strict_replay_count":replays,"episode_evidence":evidence,"split_ledger":ledger.snapshot(),"unopened_partitions":["screen","confirmation","audit"],"limits":["No screen, confirmation, audit, M14, or M15 result is authorized."]}


def _write_report(output: str|Path, report: dict[str,Any]) -> dict[str,Any]:
    path=Path(output)
    if path.exists(): raise FileExistsError(f"M13.11 output already exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report


def write_m1311_fit_smoke_report(output: str|Path) -> dict[str,Any]:
    destination=Path(output); return _write_report(destination,m1311_fit_smoke(artifact_dir=destination.parent/f"{destination.stem}-artifacts"))


def write_m1311_development_report(output: str|Path) -> dict[str,Any]:
    destination=Path(output); return _write_report(destination,m1311_development(artifact_dir=destination.parent/f"{destination.stem}-artifacts"))
