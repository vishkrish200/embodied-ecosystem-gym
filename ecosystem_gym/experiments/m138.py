"""M13.8: bounded maintenance-aligned reward and fresh two-stage study.

The policy boundary is inherited unchanged from M13.6.  Authoritative task
counters are used only in this module's reward/evaluation sidecar and never in
features, masks, policy memory, or action selection.
"""

from __future__ import annotations

import os

# Spawned fits must not multiply numerical-library worker threads.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..env import EcosystemEnv
from ..trajectory import ReplayResult, SCHEMA_VERSION, _json_value, replay_and_validate
from .m10 import m10_scan_coverage
from .m13 import (
    M13_GAMMA,
    M13Macro,
    M13Memory,
    ScriptedM13Oracle,
    _file_sha256,
    _memory_snapshot,
    _options,
    compile_macro,
)
from .m132 import (
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP,
    M132_TARGET_UPDATE_EVERY,
    _MLP,
    encode_features,
)
from .m133 import advance_m133_memory, m133_config, m133_epsilon
from .m134 import _MaskedReplay, _masked_argmax, m134_mask
from .m138_support import (
    SplitOpenLedger,
    assert_compact_report,
    m138_confirmation_passes,
    m138_content_hash,
    m138_screen_promotes,
    seal_screen_report,
    verify_m138_screen_report,
)


M138_PROTOCOL_VERSION = "m13.8-bounded-maintenance-reward-r1"
M138_SCREEN_FIT_SEEDS = tuple(range(4_800, 4_820))
M138_SCREEN_PROBE_SEEDS = tuple(range(4_820, 4_828))
M138_CONFIRMATION_FIT_SEEDS = tuple(range(4_900, 4_940))
M138_CONFIRMATION_EVALUATION_SEEDS = tuple(range(5_000, 5_020))
M138_AUDIT_SEEDS = tuple(range(5_100, 5_120))
M138_SCREEN_TRAINING_SEEDS = (20_260_901, 20_260_902)
M138_CONFIRMATION_TRAINING_SEEDS = tuple(range(20_260_903, 20_260_911))
M138_SCREEN_EPISODES = 4_000
M138_CONFIRMATION_EPISODES = 12_000
M138_UPDATE_EVERY = 4
M138_TARGET_UPDATE_EVERY = M132_TARGET_UPDATE_EVERY
M138_WORKERS = 8
M138_HORIZON = 200
M138_CANONICAL_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "m138" / "split-open-ledger.json"
)

Arm = Literal["candidate", "legacy_control"]
Stage = Literal["screen", "confirmation"]


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


def _conditions(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        "persistent_reference": _controls(
            f"{prefix}_northeast",
            food_variant="orange",
            toy_variant="ball",
            initial_scan_sector="north",
        ),
        "renewal_and_morphology": _controls(
            f"{prefix}_southwest",
            food_variant="purple",
            food_shape_variant="capsule",
            toy_variant="cube",
            agent_shape_variant="capsule",
            initial_scan_sector="south",
        ),
        "event_relocation": _controls(
            f"{prefix}_northwest",
            food_variant="blue",
            food_shape_variant="box",
            toy_variant="capsule",
            lighting_variant="dim",
            initial_scan_sector="east",
            event_relocation_on_first_pickup=True,
        ),
        "compound": _controls(
            f"{prefix}_southeast",
            food_variant="red",
            food_shape_variant="capsule",
            toy_variant="cube",
            agent_shape_variant="box",
            dynamics_variant="grippy",
            blocked_distractor=True,
            distractor_xy=[-0.04, 0.04],
            initial_scan_sector="west",
            event_relocation_on_first_pickup=True,
        ),
    }


M138_SCREEN_FIT_CONDITIONS = _conditions("m138_screen")
M138_SCREEN_PROBE_CONDITIONS = _conditions("m138_probe")
M138_CONFIRMATION_FIT_CONDITIONS = _conditions("m138_confirm_fit")
M138_CONFIRMATION_EVALUATION_CONDITIONS = _conditions("m138_confirm_eval")
M138_AUDIT_CONDITIONS = _conditions("m138_audit")


def m138_config():
    """M13.3 mechanics/reward with only the declared 200-step correction."""

    return replace(m133_config(), max_episode_steps=M138_HORIZON)


def m138_safe(drives: np.ndarray) -> bool:
    values = np.asarray(drives, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("public drive vector must have shape (3,)")
    return bool(
        values[0] > np.float32(0.15)
        and values[1] > np.float32(0.15)
        and values[2] < np.float32(0.90)
    )


def m138_safe_error(drives: np.ndarray) -> float:
    values = np.asarray(drives, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("public drive vector must have shape (3,)")
    satiety = float(np.clip((0.30 - values[0]) / 0.15, 0.0, 1.0))
    energy = float(np.clip((0.30 - values[1]) / 0.15, 0.0, 1.0))
    boredom = float(np.clip((values[2] - 0.70) / 0.20, 0.0, 1.0))
    return max(satiety * satiety, energy * energy, boredom * boredom)


def _duration(action: dict[str, np.ndarray | int]) -> float:
    duration = float(np.asarray(action["duration"], dtype=np.float32).item())
    if duration <= 0.0:
        raise ValueError("macro duration must be positive")
    return duration


@dataclass(slots=True)
class M138RewardState:
    """Privileged reward/evaluation state, never passed to policy inference."""

    required_recovery: bool
    feed_cycles: int = 0
    play_cycles: int = 0
    rest_cycles: int = 0
    decision_steps: int = 0
    decision_safe_steps: int = 0
    total_duration: float = 0.0
    duration_safe_seconds: float = 0.0
    relocation_seen: bool = False
    stale_pickup: bool = False
    recovery_complete: bool = False
    recovery_rewarded: bool = False

    @property
    def decision_safe_fraction(self) -> float:
        return self.decision_safe_steps / self.decision_steps if self.decision_steps else 0.0

    @property
    def duration_safe_fraction(self) -> float:
        return self.duration_safe_seconds / self.total_duration if self.total_duration else 0.0


def _reward_state_snapshot(state: M138RewardState) -> dict[str, Any]:
    return asdict(state) | {
        "decision_safe_fraction": state.decision_safe_fraction,
        "duration_safe_fraction": state.duration_safe_fraction,
    }


def _outcome_value(info: dict[str, Any]) -> str:
    return str(info["outcome"])


def m138_learning_reward(
    *,
    arm: Arm,
    environment_reward: float,
    state: M138RewardState,
    observation_before: dict[str, Any],
    observation_after: dict[str, Any],
    action: dict[str, np.ndarray | int],
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    macro: M13Macro | None = None,
) -> tuple[float, dict[str, float | bool | int]]:
    """Update the sidecar and return the exact scalar used by Double-DQN."""

    duration = _duration(action)
    drives_before = np.asarray(observation_before["drives"], dtype=np.float64)
    drives_after = np.asarray(observation_after["drives"], dtype=np.float64)
    error_before = m138_safe_error(drives_before)
    error_after = m138_safe_error(drives_after)
    safety_integral = duration * (error_before + error_after) / 2.0

    # Final-transition occupancy is incorporated before the terminal gate.
    post_safe = m138_safe(drives_after)
    state.decision_steps += 1
    state.decision_safe_steps += int(post_safe)
    state.total_duration += duration
    state.duration_safe_seconds += duration * int(post_safe)

    outcome = _outcome_value(info)
    action_kind = ActionKind(int(np.asarray(action["kind"]).item())) if "kind" in action else None
    forced_stale = (
        action_kind is ActionKind.PICK_UP
        and outcome == ActionOutcome.BLOCKED.value
        and info.get("disturbance") == "food_relocated"
    )
    ordinary_blocked = outcome == ActionOutcome.BLOCKED.value and not forced_stale
    invalid = outcome not in {ActionOutcome.SUCCESS.value, ActionOutcome.BLOCKED.value}
    if forced_stale:
        state.relocation_seen = True
        state.stale_pickup = True

    old_feed, old_play, old_rest = state.feed_cycles, state.play_cycles, state.rest_cycles
    next_feed = int(info.get("feed_cycles", old_feed))
    next_play = int(info.get("play_cycles", old_play))
    next_rest = int(info.get("rest_cycles", old_rest))
    feed_delta = max(0, min(next_feed, 3) - min(old_feed, 3))
    play_delta = max(0, min(next_play, 3) - min(old_play, 3))
    rest_delta = max(0, min(next_rest, 2) - min(old_rest, 2))
    state.feed_cycles, state.play_cycles, state.rest_cycles = next_feed, next_play, next_rest

    successful_consume = (
        action_kind is ActionKind.CONSUME
        and outcome == ActionOutcome.SUCCESS.value
    )
    if state.required_recovery and state.stale_pickup and successful_consume:
        state.recovery_complete = True
    recovery_milestone = state.required_recovery and state.recovery_complete and not state.recovery_rewarded
    if recovery_milestone:
        state.recovery_rewarded = True

    episode_end = bool(terminated or truncated)
    survived = bool(info.get("survived", False))
    full_terminal_success = bool(
        episode_end
        and truncated
        and not terminated
        and state.decision_steps == M138_HORIZON
        and survived
        and state.feed_cycles >= 3
        and state.play_cycles >= 3
        and state.rest_cycles >= 2
        and state.decision_safe_fraction >= 0.85
        and state.duration_safe_fraction >= 0.85
        and (not state.required_recovery or state.recovery_complete)
    )
    terminal_failure = bool(episode_end and not full_terminal_success)

    components: dict[str, float | bool | int] = {
        "duration": duration,
        "safe_error_before": error_before,
        "safe_error_after": error_after,
        "safety_integral": safety_integral,
        "duration_cost": -0.01 * duration,
        "safety_cost": -0.02 * safety_integral,
        "invalid": invalid,
        "invalid_cost": -0.10 * int(invalid),
        "ordinary_blocked": ordinary_blocked,
        "blocked_cost": -0.10 * int(ordinary_blocked),
        "forced_stale_exempt": forced_stale,
        "feed_milestone_delta": feed_delta,
        "play_milestone_delta": play_delta,
        "rest_milestone_delta": rest_delta,
        "feed_milestone_reward": 0.25 * feed_delta,
        "play_milestone_reward": 0.25 * play_delta,
        "rest_milestone_reward": 0.375 * rest_delta,
        "recovery_milestone": recovery_milestone,
        "recovery_reward": 0.25 * int(recovery_milestone),
        "post_safe": post_safe,
        "full_terminal_success": full_terminal_success,
        "terminal_success_reward": 5.0 * int(full_terminal_success),
        "terminal_failure": terminal_failure,
        "terminal_failure_cost": -2.0 * int(terminal_failure),
    }
    candidate = float(
        float(components["duration_cost"])
        + float(components["safety_cost"])
        + float(components["invalid_cost"])
        + float(components["blocked_cost"])
        + float(components["feed_milestone_reward"])
        + float(components["play_milestone_reward"])
        + float(components["rest_milestone_reward"])
        + float(components["recovery_reward"])
        + float(components["terminal_success_reward"])
        + float(components["terminal_failure_cost"])
    )
    components["candidate_recomposed"] = candidate
    components["environment_reward"] = float(environment_reward)
    learning_reward = candidate if arm == "candidate" else float(environment_reward)
    if arm not in {"candidate", "legacy_control"}:
        raise ValueError(f"unknown M13.8 arm {arm!r}")
    components["learning_reward"] = learning_reward
    return learning_reward, components


def m138_reward_components(**kwargs: Any) -> dict[str, float | bool | int]:
    """Return the exact component ledger while applying one sidecar transition."""

    return m138_learning_reward(**kwargs)[1]


class CompactM138QPolicy:
    """Frozen M13.6 dense masked Double-DQN; only its training scalar varies."""

    include_drives = use_memory = True
    mask_mode = "complementary"
    update_every = M138_UPDATE_EVERY
    target_update_every = M138_TARGET_UPDATE_EVERY

    def __init__(self, *, arm: Arm, seed: int, stage: Stage) -> None:
        if arm not in {"candidate", "legacy_control"}:
            raise ValueError(f"unknown M13.8 arm {arm!r}")
        if stage not in {"screen", "confirmation"}:
            raise ValueError(f"unknown M13.8 stage {stage!r}")
        self.arm, self.seed, self.stage = arm, int(seed), stage
        self.config = m138_config()
        self.online = _MLP(np.random.default_rng(self.seed))
        self.target = self.online.copy()
        self.update_count = 0
        self.initial_parameter_hash = m138_parameter_fingerprint(self)

    def reset(self) -> M13Memory:
        return M13Memory()

    def features(self, observation: dict[str, Any], memory: M13Memory) -> np.ndarray:
        return encode_features(observation, memory, include_drives=True, use_memory=True)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(_masked_argmax(self.online.predict(self.features(observation, memory)), self.mask(observation)))

    def observe(
        self,
        memory: M13Memory,
        *,
        observation_before: dict[str, Any],
        macro: M13Macro,
        action: dict[str, np.ndarray | int],
        observation_after: dict[str, Any],
    ) -> None:
        advance_m133_memory(
            memory,
            observation_before=observation_before,
            macro=macro,
            action=action,
            observation_after=observation_after,
            config=self.config,
        )

    def _update(self, replay: _MaskedReplay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, next_masks, done = replay.sample(rng)
        selected = np.argmax(np.where(next_masks, self.online.predict(next_features), -np.inf), axis=1)
        bootstrap = self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        targets = rewards + M13_GAMMA * (~done) * bootstrap
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % self.target_update_every == 0:
            self.target = self.online.copy()
        return loss

    def train(self, *, episodes: int | None = None) -> dict[str, float]:
        expected = M138_SCREEN_EPISODES if self.stage == "screen" else M138_CONFIRMATION_EPISODES
        if episodes is None:
            episodes = expected
        if episodes != expected:
            raise ValueError(f"M13.8 {self.stage} training budget is frozen at {expected}")
        conditions = M138_SCREEN_FIT_CONDITIONS if self.stage == "screen" else M138_CONFIRMATION_FIT_CONDITIONS
        seeds = M138_SCREEN_FIT_SEEDS if self.stage == "screen" else M138_CONFIRMATION_FIT_SEEDS
        items = tuple(conditions.items())
        rng, replay = np.random.default_rng(self.seed), _MaskedReplay()
        self.online, self.target, self.update_count = _MLP(rng), None, 0
        self.target = self.online.copy()
        self.initial_parameter_hash = m138_parameter_fingerprint(self)
        decisions, losses = 0, []
        env = EcosystemEnv(self.config)
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                env_seed = seeds[(episode // len(items)) % len(seeds)]
                observation, _ = env.reset(seed=env_seed, options=_options(controls))
                memory = self.reset()
                reward_state = M138RewardState(required_recovery=bool(controls.get("event_relocation_on_first_pickup")))
                epsilon = m133_epsilon(episode)
                for _ in range(self.config.max_episode_steps):
                    features, mask = self.features(observation, memory), self.mask(observation)
                    if rng.random() < epsilon:
                        macro = M13Macro(int(rng.choice(np.flatnonzero(mask))))
                    else:
                        macro = self.choose(observation, memory)
                    action = compile_macro(macro, observation, self.config)
                    next_observation, env_reward, terminated, truncated, info = env.step(action)
                    learning_reward, _ = m138_learning_reward(
                        arm=self.arm,
                        environment_reward=env_reward,
                        state=reward_state,
                        observation_before=observation,
                        observation_after=next_observation,
                        action=action,
                        info=info,
                        terminated=terminated,
                        truncated=truncated,
                        macro=macro,
                    )
                    self.observe(
                        memory,
                        observation_before=observation,
                        macro=macro,
                        action=action,
                        observation_after=next_observation,
                    )
                    replay.add(
                        features,
                        int(macro),
                        learning_reward,
                        self.features(next_observation, memory),
                        self.mask(next_observation),
                        terminated or truncated,
                    )
                    decisions += 1
                    if replay.size >= M132_REPLAY_WARMUP and decisions % self.update_every == 0:
                        losses.append(self._update(replay, rng))
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()
        return {
            "episodes": float(episodes),
            "decisions": float(decisions),
            "updates": float(self.update_count),
            "target_copies": float(self.update_count // self.target_update_every),
            "mean_huber_loss": float(np.mean(losses)) if losses else 0.0,
        }


class SeededRandomM138Policy:
    """Matched mask-aware random policy; its single PCG64 stream is seed s."""

    include_drives, use_memory, mask_mode = True, True, "complementary"

    def __init__(self, seed: int) -> None:
        self.seed, self.config = int(seed), m138_config()
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> M13Memory:
        return M13Memory()

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        del memory
        return M13Macro(int(self.rng.choice(np.flatnonzero(self.mask(observation)))))

    def observe(self, memory: M13Memory, **kwargs: Any) -> None:
        advance_m133_memory(memory, config=self.config, **kwargs)


class _SpecialistPolicy:
    include_drives, use_memory, mask_mode = True, True, "complementary"

    def __init__(self, kind: Literal["feed_only", "play_only", "rest_wait"]) -> None:
        self.kind, self.config = kind, m138_config()

    def reset(self) -> M13Memory:
        return M13Memory()

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        mask = self.mask(observation)
        if self.kind == "feed_only":
            if bool(observation["holding_food"]):
                preferred = M13Macro.CONSUME
            elif memory.food_cooldown_bucket < int(self.config.food_respawn_seconds):
                preferred = M13Macro.WAIT
            elif bool(mask[int(M13Macro.PICK_UP)]):
                preferred = M13Macro.PICK_UP
            else:
                preferred = M13Macro.GO_FOOD
        elif self.kind == "play_only":
            preferred = M13Macro.PLAY if bool(mask[int(M13Macro.PLAY)]) else M13Macro.GO_TOY
        else:
            energy = float(np.asarray(observation["drives"], dtype=np.float32)[1])
            if energy <= 0.45:
                preferred = M13Macro.REST if bool(mask[int(M13Macro.REST)]) else M13Macro.GO_REST
            else:
                preferred = M13Macro.WAIT
        if bool(mask[int(preferred)]):
            return preferred
        return M13Macro(int(np.flatnonzero(mask)[0]))

    def observe(self, memory: M13Memory, **kwargs: Any) -> None:
        advance_m133_memory(memory, config=self.config, **kwargs)


def _source_hashes() -> dict[str, str]:
    experiment_dir = Path(__file__).resolve().parent
    package_dir = experiment_dir.parent
    repository_dir = package_dir.parent
    paths: dict[str, Path] = {
        f"experiments/{name}": experiment_dir / name
        for name in (
            "m138.py",
            "m134.py",
            "m133.py",
            "m132.py",
            "m131.py",
            "m13.py",
            "m10.py",
        )
    }
    support = experiment_dir / "m138_support.py"
    if support.exists():
        paths["experiments/m138_support.py"] = support
    paths.update(
        {
            name: package_dir / name
            for name in ("actions.py", "config.py", "drives.py", "env.py", "tasks.py", "trajectory.py")
        }
    )
    protocol_doc = repository_dir / "docs" / "M13_8_PROTOCOL.md"
    if protocol_doc.exists():
        paths["docs/M13_8_PROTOCOL.md"] = protocol_doc
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.8 fingerprint sources missing: {missing}")
    return {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(paths.items())}


def m138_protocol_fingerprint() -> str:
    """Fingerprint the frozen reward, boundary, splits, budgets, and sources."""

    payload = {
        "version": M138_PROTOCOL_VERSION,
        "config": asdict(m138_config()),
        "reward": {
            "safe_error": [0.30, 0.30, 0.70, 0.15, 0.15, 0.90],
            "duration_cost": -0.01,
            "safety_cost": -0.02,
            "invalid_cost": -0.10,
            "blocked_cost": -0.10,
            "cycle_rewards": [0.25, 0.25, 0.375],
            "cycle_caps": [3, 3, 2],
            "recovery": 0.25,
            "terminal_success": 5.0,
            "terminal_failure": -2.0,
        },
        "splits": {
            "screen_fit": M138_SCREEN_FIT_SEEDS,
            "screen_probe": M138_SCREEN_PROBE_SEEDS,
            "confirmation_fit": M138_CONFIRMATION_FIT_SEEDS,
            "confirmation_evaluation": M138_CONFIRMATION_EVALUATION_SEEDS,
            "audit": M138_AUDIT_SEEDS,
        },
        "conditions": {
            "screen_fit": M138_SCREEN_FIT_CONDITIONS,
            "screen_probe": M138_SCREEN_PROBE_CONDITIONS,
            "confirmation_fit": M138_CONFIRMATION_FIT_CONDITIONS,
            "confirmation_evaluation": M138_CONFIRMATION_EVALUATION_CONDITIONS,
            "audit": M138_AUDIT_CONDITIONS,
        },
        "training": {
            "screen_seeds": M138_SCREEN_TRAINING_SEEDS,
            "confirmation_seeds": M138_CONFIRMATION_TRAINING_SEEDS,
            "screen_episodes": M138_SCREEN_EPISODES,
            "confirmation_episodes": M138_CONFIRMATION_EPISODES,
            "feature_dim": M132_FEATURE_DIM,
            "replay_capacity": M132_REPLAY_CAPACITY,
            "replay_warmup": M132_REPLAY_WARMUP,
            "batch_size": M132_BATCH_SIZE,
            "gamma": M13_GAMMA,
            "update_every": M138_UPDATE_EVERY,
            "target_update_every": M138_TARGET_UPDATE_EVERY,
            "workers": M138_WORKERS,
        },
        "specialists": ["feed_only", "play_only", "rest_wait"],
        "random_seed_derivation": "single PCG64 stream seeded exactly by paired training seed",
        "sources": _source_hashes(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def m138_policy_fingerprint(policy: CompactM138QPolicy) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "protocol_version": M138_PROTOCOL_VERSION,
                "arm": policy.arm,
                "seed": policy.seed,
                "stage": policy.stage,
                "feature_dim": M132_FEATURE_DIM,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for key in sorted(policy.online.params):
        digest.update(key.encode())
        digest.update(np.asarray(policy.online.params[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def m138_parameter_fingerprint(policy: CompactM138QPolicy) -> str:
    digest = hashlib.sha256()
    for key in sorted(policy.online.params):
        digest.update(key.encode())
        digest.update(np.asarray(policy.online.params[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def write_m138_policy(path: str | Path, policy: CompactM138QPolicy) -> dict[str, str]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"M13.8 policy artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "m138-policy-v1",
        "protocol_fingerprint": m138_protocol_fingerprint(),
        "arm": policy.arm,
        "seed": policy.seed,
        "stage": policy.stage,
        "initial_parameter_hash": policy.initial_parameter_hash,
        "network": {key: value.tolist() for key, value in policy.online.params.items()},
    }
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "path": str(output.resolve()),
        "sha256": _file_sha256(output),
        "policy_fingerprint": m138_policy_fingerprint(policy),
    }


def load_m138_policy(path: str | Path) -> CompactM138QPolicy:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m138-policy-v1":
        raise ValueError("unsupported M13.8 policy schema")
    if payload.get("protocol_fingerprint") != m138_protocol_fingerprint():
        raise ValueError("M13.8 policy artifact does not match the frozen protocol")
    policy = CompactM138QPolicy(
        arm=payload["arm"],
        seed=int(payload["seed"]),
        stage=payload["stage"],
    )
    if payload.get("initial_parameter_hash") != policy.initial_parameter_hash:
        raise ValueError("M13.8 policy artifact has an inconsistent initial parameter hash")
    network = payload.get("network", {})
    if set(network) != set(policy.online.params):
        raise ValueError("M13.8 policy artifact has an invalid parameter set")
    for key, values in network.items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != policy.online.params[key].shape or not np.all(np.isfinite(array)):
            raise ValueError(f"M13.8 policy parameter {key!r} is invalid")
        policy.online.params[key] = array
    policy.target = policy.online.copy()
    return policy


class BalancedM138Oracle(ScriptedM13Oracle):
    """Inherited public scripted ceiling with the M13.3 public memory update."""

    def __init__(self) -> None:
        self.config = m138_config()

    def observe(self, memory: M13Memory, **kwargs: Any) -> None:
        advance_m133_memory(memory, config=self.config, **kwargs)


@dataclass(frozen=True, slots=True)
class M138EpisodeResult:
    training_seed: int
    seed: int
    condition: str
    survived: bool
    maintenance_complete: bool
    full_gate_success: bool
    terminal_cause: str
    feed_cycles: int
    play_cycles: int
    rest_cycles: int
    decision_steps: int
    decision_safe_fraction: float
    duration_safe_fraction: float
    total_duration: float
    recovery_required: bool
    stale_pickup: bool
    recovery_complete: bool
    environment_return: float
    learning_return: float
    interventions: dict[str, int]
    macro_counts: dict[str, int]

    def compact_dict(self) -> dict[str, Any]:
        return asdict(self)

    def gate_dict(self, *, replay_violations: int = 0) -> dict[str, Any]:
        return {
            "training_seed": self.training_seed,
            "condition": self.condition,
            "env_seed": self.seed,
            "survived": self.survived,
            "maintenance_complete": self.maintenance_complete,
            "decision_safe_fraction": self.decision_safe_fraction,
            "duration_safe_fraction": self.duration_safe_fraction,
            "recovery_required": self.recovery_required,
            "recovery_complete": self.recovery_complete,
            "conformance_violations": int(sum(self.interventions.values())),
            "replay_violations": int(replay_violations),
        }


def _json_close(actual: Any, expected: Any, *, path: str = "value") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{path} keys differ")
        for key in actual:
            _json_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"{path} lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _json_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
        if not np.isclose(float(actual), float(expected), rtol=1e-7, atol=1e-9):
            raise ValueError(f"{path} differs: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{path} differs: {actual!r} != {expected!r}")


def _policy_mask(policy: Any, observation: dict[str, Any]) -> np.ndarray:
    if hasattr(policy, "mask"):
        return np.asarray(policy.mask(observation), dtype=np.bool_)
    return m134_mask(observation, m138_config(), "complementary")


def _trace_header(
    *,
    episode_id: str,
    seed: int,
    controls: dict[str, Any],
    arm: Arm,
    training_seed: int,
    policy_label: str,
    policy_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "episode_metadata",
        "episode_id": episode_id,
        "seed": seed,
        "reset_options": _json_value(_options(controls)),
        "observation_mode": "state_oracle",
        "config": asdict(m138_config()),
        "m138_protocol_fingerprint": m138_protocol_fingerprint(),
        "m138_policy_fingerprint": policy_fingerprint,
        "m138_policy_label": policy_label,
        "m138_arm": arm,
        "m138_training_seed": training_seed,
        "m138_required_recovery": bool(controls.get("event_relocation_on_first_pickup")),
    }


def run_m138_episode(
    policy: Any,
    *,
    arm: Arm,
    training_seed: int,
    seed: int,
    condition: str,
    controls: dict[str, Any],
    trace_path: str | Path | None = None,
    policy_label: str,
    policy_fingerprint: str | None,
) -> M138EpisodeResult:
    """Evaluate one policy without retaining transition records in RAM."""

    config = getattr(policy, "config", m138_config())
    if config != m138_config():
        raise ValueError("M13.8 evaluation requires the exact frozen 200-step config")
    env = EcosystemEnv(config)
    trace_handle = None
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        reward_state = M138RewardState(
            required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
        )
        interventions = {
            "feature_shape_violation": 0,
            "nonfinite_q_violation": 0,
            "mask_shape_violation": 0,
            "mask_violation": 0,
            "redundant_go": 0,
            "reward_recomposition_violation": 0,
        }
        macro_counts = {macro.name: 0 for macro in M13Macro}
        environment_return = learning_return = 0.0
        trace = Path(trace_path) if trace_path is not None else None
        episode_id = f"m138-{policy_label.replace('/', '-')}-{condition}-seed-{seed}"
        if trace is not None:
            if trace.exists():
                raise FileExistsError(f"M13.8 trace already exists: {trace}")
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace.open("x", encoding="utf-8")
            trace_handle.write(
                json.dumps(
                    _trace_header(
                        episode_id=episode_id,
                        seed=seed,
                        controls=controls,
                        arm=arm,
                        training_seed=training_seed,
                        policy_label=policy_label,
                        policy_fingerprint=policy_fingerprint,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        last_info: dict[str, Any] | None = None
        for step in range(1, config.max_episode_steps + 1):
            policy_observation = _json_value(observation)
            memory_before = _memory_snapshot(memory)
            reward_state_before = _reward_state_snapshot(reward_state)
            mask = _policy_mask(policy, observation)
            if mask.shape != (len(M13Macro),) or not np.any(mask):
                interventions["mask_shape_violation"] += 1
                raise ValueError("M13.8 public mask is malformed")
            features = None
            q_values = None
            if isinstance(policy, CompactM138QPolicy):
                features = policy.features(observation, memory)
                if features.shape != (M132_FEATURE_DIM,):
                    interventions["feature_shape_violation"] += 1
                q_values = np.asarray(policy.online.predict(features), dtype=np.float32)
                if q_values.shape != (len(M13Macro),) or not np.all(np.isfinite(q_values)):
                    interventions["nonfinite_q_violation"] += 1
            rng_state = (
                _json_value(policy.rng.bit_generator.state)
                if isinstance(policy, SeededRandomM138Policy)
                else None
            )
            macro = policy.choose(observation, memory)
            if not bool(mask[int(macro)]):
                interventions["mask_violation"] += 1
            if macro in {M13Macro.GO_FOOD, M13Macro.GO_TOY, M13Macro.GO_REST} and not bool(mask[int(macro)]):
                interventions["redundant_go"] += 1
            macro_counts[macro.name] += 1
            action = compile_macro(macro, observation, config)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            learning_reward, components = m138_learning_reward(
                arm=arm,
                environment_reward=env_reward,
                state=reward_state,
                observation_before=observation,
                observation_after=next_observation,
                action=action,
                info=info,
                terminated=terminated,
                truncated=truncated,
                macro=macro,
            )
            expected_learning = (
                float(components["candidate_recomposed"])
                if arm == "candidate"
                else float(env_reward)
            )
            if not np.isclose(learning_reward, expected_learning, rtol=1e-7, atol=1e-9):
                interventions["reward_recomposition_violation"] += 1
            policy.observe(
                memory,
                observation_before=observation,
                macro=macro,
                action=action,
                observation_after=next_observation,
            )
            environment_return += float(env_reward)
            learning_return += float(learning_reward)
            if trace_handle is not None:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "step": step,
                    "seed": seed,
                    "observation_mode": "state_oracle",
                    "policy_observation": policy_observation,
                    "features": None if features is None else features.tolist(),
                    "q_values": None if q_values is None else q_values.tolist(),
                    "mask": mask.astype(bool).tolist(),
                    "eligible_macros": [item.name for item in M13Macro if bool(mask[int(item)])],
                    "memory_before": memory_before,
                    "memory_after": _memory_snapshot(memory),
                    "reward_state_before": reward_state_before,
                    "reward_state_after": _reward_state_snapshot(reward_state),
                    "macro": macro.name,
                    "action": _json_value(action),
                    "random_rng_state_before": rng_state,
                    "reward": float(env_reward),
                    "learning_reward": float(learning_reward),
                    "reward_components": _json_value(components),
                    "outcome": info["outcome"],
                    "task_success": bool(info["task_success"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "environment_version": str(info["environment_version"]),
                    "disturbance": info.get("disturbance"),
                    "post_disturbance_completion": bool(info.get("post_disturbance_completion", False)),
                    "resource_event": info.get("resource_event"),
                    "feed_cycles": int(info.get("feed_cycles", 0)),
                    "play_cycles": int(info.get("play_cycles", 0)),
                    "rest_cycles": int(info.get("rest_cycles", 0)),
                    "food_available": bool(info.get("food_available", True)),
                    "camera_sector": info.get("camera_sector"),
                    "survived": bool(info.get("survived", False)),
                    "maintenance_complete_legacy": bool(info.get("maintenance_complete", False)),
                    "observation": _json_value(next_observation),
                }
                trace_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            observation, last_info = next_observation, info
            if terminated or truncated:
                if trace_handle is not None:
                    trace_handle.flush()
                    os.fsync(trace_handle.fileno())
                survived = bool(info.get("survived", False))
                maintenance = bool(
                    truncated
                    and not terminated
                    and step == M138_HORIZON
                    and survived
                    and reward_state.feed_cycles >= 3
                    and reward_state.play_cycles >= 3
                    and reward_state.rest_cycles >= 2
                )
                full_gate = bool(components["full_terminal_success"])
                drives = np.asarray(observation["drives"], dtype=np.float32)
                terminal_cause = (
                    "survived_horizon"
                    if truncated and survived
                    else "energy_depleted"
                    if drives[1] <= 0.0
                    else "satiety_depleted"
                    if drives[0] <= 0.0
                    else "terminated"
                )
                return M138EpisodeResult(
                    training_seed=training_seed,
                    seed=seed,
                    condition=condition,
                    survived=survived,
                    maintenance_complete=maintenance,
                    full_gate_success=full_gate,
                    terminal_cause=terminal_cause,
                    feed_cycles=reward_state.feed_cycles,
                    play_cycles=reward_state.play_cycles,
                    rest_cycles=reward_state.rest_cycles,
                    decision_steps=reward_state.decision_steps,
                    decision_safe_fraction=reward_state.decision_safe_fraction,
                    duration_safe_fraction=reward_state.duration_safe_fraction,
                    total_duration=reward_state.total_duration,
                    recovery_required=reward_state.required_recovery,
                    stale_pickup=reward_state.stale_pickup,
                    recovery_complete=reward_state.recovery_complete,
                    environment_return=environment_return,
                    learning_return=learning_return,
                    interventions=interventions,
                    macro_counts=macro_counts,
                )
        raise AssertionError(f"M13.8 episode did not terminate; last info={last_info!r}")
    finally:
        if trace_handle is not None:
            trace_handle.close()
        env.close()


def replay_m138_trace(
    path: str | Path,
    policy: Any,
) -> ReplayResult:
    """Strictly replay environment, inference, memory, and reward sidecar."""

    source = Path(path)
    generic = replay_and_validate(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("record_type") != "episode_metadata" or not steps:
        raise ValueError("M13.8 trace lacks metadata or steps")
    if header.get("m138_protocol_fingerprint") != m138_protocol_fingerprint():
        raise ValueError("M13.8 trace protocol fingerprint mismatch")
    reset_requires_recovery = bool(
        header.get("reset_options", {}).get("event_relocation_on_first_pickup", False)
    )
    if bool(header.get("m138_required_recovery")) != reset_requires_recovery:
        raise ValueError("M13.8 trace recovery requirement disagrees with reset metadata")
    expected_policy_fingerprint = header.get("m138_policy_fingerprint")
    if isinstance(policy, CompactM138QPolicy):
        if expected_policy_fingerprint != m138_policy_fingerprint(policy):
            raise ValueError("M13.8 learned trace policy fingerprint mismatch")
        if header.get("m138_arm") != policy.arm:
            raise ValueError("M13.8 learned trace arm mismatch")
    elif isinstance(policy, SeededRandomM138Policy):
        if expected_policy_fingerprint != f"pcg64-{policy.seed}":
            raise ValueError("M13.8 random trace policy fingerprint mismatch")
    elif expected_policy_fingerprint is not None:
        raise ValueError("M13.8 deterministic trace unexpectedly names a policy artifact")
    arm: Arm = header["m138_arm"]
    env = EcosystemEnv(m138_config())
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        reward_state = M138RewardState(required_recovery=bool(header["m138_required_recovery"]))
        for expected in steps:
            step = int(expected["step"])
            if _json_value(observation) != expected["policy_observation"]:
                raise ValueError(f"M13.8 policy observation mismatch at step {step}")
            if _memory_snapshot(memory) != expected["memory_before"]:
                raise ValueError(f"M13.8 memory-before mismatch at step {step}")
            _json_close(_reward_state_snapshot(reward_state), expected["reward_state_before"], path="reward_state_before")
            mask = _policy_mask(policy, observation)
            if mask.astype(bool).tolist() != expected["mask"]:
                raise ValueError(f"M13.8 mask mismatch at step {step}")
            if isinstance(policy, CompactM138QPolicy):
                features = policy.features(observation, memory)
                q_values = np.asarray(policy.online.predict(features), dtype=np.float32)
                if features.tolist() != expected["features"]:
                    raise ValueError(f"M13.8 feature mismatch at step {step}")
                if not np.array_equal(q_values, np.asarray(expected["q_values"], dtype=np.float32)):
                    raise ValueError(f"M13.8 Q-value mismatch at step {step}")
            elif isinstance(policy, SeededRandomM138Policy):
                state = expected.get("random_rng_state_before")
                if state is None:
                    raise ValueError(f"M13.8 random trace lacks RNG state at step {step}")
                policy.rng.bit_generator.state = state
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, m138_config())
            if not bool(mask[int(macro)]) or macro.name != expected["macro"]:
                raise ValueError(f"M13.8 constrained macro mismatch at step {step}")
            if _json_value(action) != expected["action"]:
                raise ValueError(f"M13.8 compiled action mismatch at step {step}")
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            learning_reward, components = m138_learning_reward(
                arm=arm,
                environment_reward=env_reward,
                state=reward_state,
                observation_before=observation,
                observation_after=next_observation,
                action=action,
                info=info,
                terminated=terminated,
                truncated=truncated,
                macro=macro,
            )
            if not np.isclose(learning_reward, float(expected["learning_reward"]), rtol=1e-7, atol=1e-9):
                raise ValueError(f"M13.8 learning reward mismatch at step {step}")
            _json_close(_json_value(components), expected["reward_components"], path="reward_components")
            _json_close(_reward_state_snapshot(reward_state), expected["reward_state_after"], path="reward_state_after")
            policy.observe(
                memory,
                observation_before=observation,
                macro=macro,
                action=action,
                observation_after=next_observation,
            )
            if _memory_snapshot(memory) != expected["memory_after"]:
                raise ValueError(f"M13.8 memory-after mismatch at step {step}")
            observation = next_observation
        return generic
    finally:
        env.close()


def replay_m138_random_trace(path: str | Path, policy: SeededRandomM138Policy) -> ReplayResult:
    """Typed public wrapper retained for random-policy conformance tests."""

    return replay_m138_trace(path, policy)


def _aggregate_episode_results(episodes: list[M138EpisodeResult]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate empty M13.8 episode results")
    decision_safe = np.asarray([row.decision_safe_fraction for row in episodes], dtype=np.float64)
    duration_safe = np.asarray([row.duration_safe_fraction for row in episodes], dtype=np.float64)
    learning_returns = np.asarray([row.learning_return for row in episodes], dtype=np.float64)
    return {
        "episodes": len(episodes),
        "survivals": sum(row.survived for row in episodes),
        "maintenance_complete": sum(row.maintenance_complete for row in episodes),
        "full_gate_success": sum(row.full_gate_success for row in episodes),
        "mean_decision_safe_fraction": float(np.mean(decision_safe)),
        "mean_duration_safe_fraction": float(np.mean(duration_safe)),
        "decision_safe_quantiles": {
            "q05": float(np.quantile(decision_safe, 0.05)),
            "q50": float(np.quantile(decision_safe, 0.50)),
            "q95": float(np.quantile(decision_safe, 0.95)),
        },
        "duration_safe_quantiles": {
            "q05": float(np.quantile(duration_safe, 0.05)),
            "q50": float(np.quantile(duration_safe, 0.50)),
            "q95": float(np.quantile(duration_safe, 0.95)),
        },
        "cycles": {
            "minimum_feed": min(row.feed_cycles for row in episodes),
            "minimum_play": min(row.play_cycles for row in episodes),
            "minimum_rest": min(row.rest_cycles for row in episodes),
            "mean_feed": float(np.mean([row.feed_cycles for row in episodes])),
            "mean_play": float(np.mean([row.play_cycles for row in episodes])),
            "mean_rest": float(np.mean([row.rest_cycles for row in episodes])),
        },
        "recovery": {
            "required": sum(row.recovery_required for row in episodes),
            "complete": sum(row.recovery_complete for row in episodes),
        },
        "mean_learning_return": float(np.mean(learning_returns)),
        "interventions": {
            key: sum(row.interventions.get(key, 0) for row in episodes)
            for key in sorted({key for row in episodes for key in row.interventions})
        },
        "macro_counts": {
            macro.name: sum(row.macro_counts.get(macro.name, 0) for row in episodes)
            for macro in M13Macro
        },
    }


def _evaluate_policy(
    policy: Any,
    *,
    arm: Arm,
    training_seed: int,
    conditions: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
    label: str,
    policy_fingerprint: str | None,
    replay_mode: Literal["learned", "random", "deterministic", "generic"],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    by_condition: dict[str, list[M138EpisodeResult]] = {}
    for condition, controls in conditions.items():
        condition_rows: list[M138EpisodeResult] = []
        for seed in seeds:
            trace = trace_dir / label / condition / f"seed-{seed}.jsonl"
            episode = run_m138_episode(
                policy,
                arm=arm,
                training_seed=training_seed,
                seed=seed,
                condition=condition,
                controls=controls,
                trace_path=trace,
                policy_label=label,
                policy_fingerprint=policy_fingerprint,
            )
            replay_violations = 0
            replay_error = None
            try:
                if replay_mode == "learned":
                    if not isinstance(policy, CompactM138QPolicy):
                        raise TypeError("learned replay requires CompactM138QPolicy")
                    replay_m138_trace(trace, policy)
                elif replay_mode == "random":
                    replay_m138_trace(trace, SeededRandomM138Policy(training_seed))
                elif replay_mode == "deterministic":
                    replay_m138_trace(trace, policy)
                else:
                    replay_and_validate(trace)
            except Exception as exc:  # preserve opened-split evidence as a rejecting result
                replay_violations = 1
                replay_error = f"{type(exc).__name__}: {exc}"
            summaries.append(episode.gate_dict(replay_violations=replay_violations))
            condition_rows.append(episode)
            manifest.append(
                {
                    "policy": label,
                    "condition": condition,
                    "env_seed": seed,
                    "path": str(trace.resolve()),
                    "sha256": _file_sha256(trace),
                    "replay_pass": replay_violations == 0,
                    "replay_error": replay_error,
                }
            )
        by_condition[condition] = condition_rows
    aggregates = {condition: _aggregate_episode_results(rows) for condition, rows in by_condition.items()}
    return summaries, aggregates, manifest


def _train_worker(
    arm: Arm,
    seed: int,
    stage: Stage,
) -> tuple[Arm, int, CompactM138QPolicy, dict[str, float]]:
    started = time.perf_counter()
    policy = CompactM138QPolicy(arm=arm, seed=seed, stage=stage)
    training = policy.train()
    training["elapsed_seconds"] = time.perf_counter() - started
    return arm, seed, policy, training


def _train_paired(
    *,
    stage: Stage,
    seeds: tuple[int, ...],
    workers: int,
) -> tuple[
    dict[tuple[Arm, int], CompactM138QPolicy],
    dict[tuple[Arm, int], dict[str, float]],
    dict[str, Any],
]:
    if workers != M138_WORKERS:
        raise ValueError(f"M13.8 worker count is frozen at {M138_WORKERS}")
    jobs: list[tuple[Arm, int]] = [
        (arm, seed)
        for seed in seeds
        for arm in ("candidate", "legacy_control")
    ]
    policies: dict[tuple[Arm, int], CompactM138QPolicy] = {}
    training: dict[tuple[Arm, int], dict[str, float]] = {}
    completion_order: list[str] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(_train_worker, arm, seed, stage): (arm, seed)
            for arm, seed in jobs
        }
        for future in as_completed(futures):
            arm, seed, policy, diagnostics = future.result()
            policies[(arm, seed)] = policy
            training[(arm, seed)] = diagnostics
            completion_order.append(f"{arm}:{seed}")
    for seed in seeds:
        candidate = policies[("candidate", seed)]
        control = policies[("legacy_control", seed)]
        if candidate.initial_parameter_hash != control.initial_parameter_hash:
            raise ValueError(f"M13.8 paired seed {seed} did not start from identical parameters")
        if candidate.config != control.config:
            raise ValueError(f"M13.8 paired seed {seed} used different configs")
    execution = {
        "workers": workers,
        "start_method": "spawn",
        "worker_threads": 1,
        "submission_order": [f"{arm}:{seed}" for arm, seed in jobs],
        "completion_order": completion_order,
    }
    return policies, training, execution


def _serialize_all_policies(
    policies: dict[tuple[Arm, int], CompactM138QPolicy],
    *,
    seeds: tuple[int, ...],
    policy_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[tuple[Arm, int], CompactM138QPolicy]]:
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    restored: dict[tuple[Arm, int], CompactM138QPolicy] = {}
    for seed in seeds:
        artifacts[str(seed)] = {}
        for arm in ("candidate", "legacy_control"):
            policy = policies[(arm, seed)]
            artifact = write_m138_policy(policy_dir / f"seed-{seed}" / f"{arm}.json", policy)
            loaded = load_m138_policy(artifact["path"])
            if m138_policy_fingerprint(loaded) != artifact["policy_fingerprint"]:
                raise ValueError(f"M13.8 serialized {arm}:{seed} fingerprint mismatch")
            artifacts[str(seed)][arm] = artifact
            restored[(arm, seed)] = loaded
    expected = len(seeds) * 2
    if len(restored) != expected or sum(len(row) for row in artifacts.values()) != expected:
        raise ValueError("M13.8 did not serialize every learned final policy")
    return artifacts, restored


def _preflight(
    *,
    artifact_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = m10_scan_coverage(M138_SCREEN_FIT_CONDITIONS, seeds=M138_SCREEN_FIT_SEEDS)
    trace_dir = artifact_dir / "traces" / "preflight"
    manifest: list[dict[str, Any]] = []
    ceiling_summaries, ceiling_aggregates, ceiling_manifest = _evaluate_policy(
        BalancedM138Oracle(),
        arm="candidate",
        training_seed=0,
        conditions=M138_SCREEN_FIT_CONDITIONS,
        seeds=M138_SCREEN_FIT_SEEDS,
        trace_dir=trace_dir,
        label="balanced_public_oracle",
        policy_fingerprint=None,
        replay_mode="deterministic",
    )
    manifest.extend(ceiling_manifest)
    specialists: dict[str, Any] = {}
    specialist_pass = True
    for kind in ("feed_only", "play_only", "rest_wait"):
        _, aggregates, specialist_manifest = _evaluate_policy(
            _SpecialistPolicy(kind),
            arm="candidate",
            training_seed=0,
            conditions=M138_SCREEN_FIT_CONDITIONS,
            seeds=M138_SCREEN_FIT_SEEDS,
            trace_dir=trace_dir,
            label=f"specialist_{kind}",
            policy_fingerprint=None,
            replay_mode="deterministic",
        )
        manifest.extend(specialist_manifest)
        gaps = {
            condition: float(ceiling_aggregates[condition]["mean_learning_return"])
            - float(aggregates[condition]["mean_learning_return"])
            for condition in M138_SCREEN_FIT_CONDITIONS
        }
        passes = all(value >= 2.0 for value in gaps.values())
        specialist_pass = specialist_pass and passes
        specialists[kind] = {"aggregates": aggregates, "return_gap_from_balanced": gaps, "passes": passes}
    coverage_pass = all(bool(row["passes"]) for row in coverage.values())
    ceiling_pass = all(
        int(row["episodes"]) == len(M138_SCREEN_FIT_SEEDS)
        and int(row["full_gate_success"]) == len(M138_SCREEN_FIT_SEEDS)
        and all(int(value) == 0 for value in row["interventions"].values())
        for row in ceiling_aggregates.values()
    )
    replay_pass = all(bool(row["replay_pass"]) for row in manifest)
    preflight = {
        "passes": coverage_pass and ceiling_pass and specialist_pass and replay_pass,
        "coverage_pass": coverage_pass,
        "ceiling_pass": ceiling_pass,
        "specialist_return_pass": specialist_pass,
        "replay_pass": replay_pass,
        "coverage": coverage,
        "balanced_public_oracle": {
            "aggregates": ceiling_aggregates,
            "compact_episode_summaries": ceiling_summaries,
        },
        "specialists": specialists,
    }
    return preflight, manifest


def _report_hash(report: dict[str, Any]) -> str:
    return m138_content_hash({key: value for key, value in report.items() if key != "content_hash"})


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"M13.8 report already exists: {path}")
    assert_compact_report(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def m138_screen(
    *,
    artifact_dir: str | Path,
    ledger_path: str | Path = M138_CANONICAL_LEDGER_PATH,
    workers: int = M138_WORKERS,
) -> dict[str, Any]:
    """Run the frozen preflight, paired fits, and one-shot screen probe."""

    if workers != M138_WORKERS:
        raise ValueError(f"M13.8 worker count is frozen at {M138_WORKERS}")
    protocol_fingerprint = m138_protocol_fingerprint()  # before touching a split
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.8 screen artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.open_partition("screen_fit")  # before the first screen-fit reset
    artifacts.mkdir(parents=True)
    preflight, manifest = _preflight(artifact_dir=artifacts)
    if not bool(preflight["passes"]):
        report = {
            "schema_version": "m13.8-screen-preflight-reject-r1",
            "stage": "screen",
            "protocol_version": M138_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_ledger": ledger.snapshot(),
            "preflight": preflight,
            "trace_manifest": manifest,
            "screen": {"status": "reject", "passes": False, "phase": "preflight"},
            "limits": [
                "The one-shot probe was not opened.",
                "No confirmation or audit data was touched.",
            ],
        }
        report["content_hash"] = _report_hash(report)
        return report

    policies, training, execution = _train_paired(
        stage="screen",
        seeds=M138_SCREEN_TRAINING_SEEDS,
        workers=workers,
    )
    policy_artifacts, restored = _serialize_all_policies(
        policies,
        seeds=M138_SCREEN_TRAINING_SEEDS,
        policy_dir=artifacts / "policies",
    )
    # Global phase boundary: every learned final policy exists and reloads
    # before the one-shot probe is marked open or reset.
    ledger.open_partition("screen_probe")
    episode_summaries: dict[str, list[dict[str, Any]]] = {
        "candidate": [],
        "legacy_control": [],
        "random": [],
    }
    aggregates: dict[str, Any] = {}
    for seed in M138_SCREEN_TRAINING_SEEDS:
        aggregates[str(seed)] = {}
        for arm in ("candidate", "legacy_control"):
            policy = restored[(arm, seed)]
            rows, arm_aggregates, traces = _evaluate_policy(
                policy,
                arm=arm,
                training_seed=seed,
                conditions=M138_SCREEN_PROBE_CONDITIONS,
                seeds=M138_SCREEN_PROBE_SEEDS,
                trace_dir=artifacts / "traces" / "probe",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m138_policy_fingerprint(policy),
                replay_mode="learned",
            )
            episode_summaries[arm].extend(rows)
            aggregates[str(seed)][arm] = arm_aggregates
            manifest.extend(traces)
        random_policy = SeededRandomM138Policy(seed)
        rows, random_aggregates, traces = _evaluate_policy(
            random_policy,
            arm="candidate",
            training_seed=seed,
            conditions=M138_SCREEN_PROBE_CONDITIONS,
            seeds=M138_SCREEN_PROBE_SEEDS,
            trace_dir=artifacts / "traces" / "probe",
            label=f"seed-{seed}/random",
            policy_fingerprint=f"pcg64-{seed}",
            replay_mode="random",
        )
        episode_summaries["random"].extend(rows)
        aggregates[str(seed)]["random"] = random_aggregates
        manifest.extend(traces)
    promotion = m138_screen_promotes(
        episode_summaries["candidate"],
        episode_summaries["legacy_control"],
        episode_summaries["random"],
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
    )
    training_report = {
        str(seed): {
            arm: training[(arm, seed)]
            | {"initial_parameter_hash": policies[(arm, seed)].initial_parameter_hash}
            for arm in ("candidate", "legacy_control")
        }
        for seed in M138_SCREEN_TRAINING_SEEDS
    }
    base = {
        "stage": "screen",
        "protocol_version": M138_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "split_ledger": ledger.snapshot(),
        "preflight": preflight,
        "execution": execution,
        "training": training_report,
        "policy_artifacts": policy_artifacts,
        "episode_summaries": episode_summaries,
        "aggregates": aggregates,
        "trace_manifest": manifest,
        "screen": {
            "status": "promote" if bool(promotion["passes"]) else "reject",
            "passes": bool(promotion["passes"]),
            "phase": "probe",
        },
        "limits": [
            "A screen promotion authorizes confirmation; it is not an improvement claim.",
            "No confirmation evaluation or audit data is present.",
        ],
    }
    return seal_screen_report(
        base,
        expected_protocol_fingerprint=protocol_fingerprint,
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
    )


def write_m138_screen_report(
    path: str | Path,
    *,
    ledger_path: str | Path = M138_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.8 screen report or artifact target already exists")
    report = m138_screen(artifact_dir=artifacts, ledger_path=ledger_path)
    _write_json_exclusive(output, report)
    return report


def _load_screen_report(screen_report: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str | None, str]:
    if isinstance(screen_report, dict):
        payload = screen_report
        path = None
        sha256 = m138_content_hash(payload)
    else:
        source = Path(screen_report)
        payload = json.loads(source.read_text(encoding="utf-8"))
        path = str(source.resolve())
        sha256 = _file_sha256(source)
    if not isinstance(payload, dict):
        raise ValueError("M13.8 screen report must contain a JSON object")
    return payload, path, sha256


def _verify_screen_policy_artifacts(report: dict[str, Any]) -> None:
    artifacts = report.get("policy_artifacts")
    expected_seeds = {str(seed) for seed in M138_SCREEN_TRAINING_SEEDS}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_seeds:
        raise ValueError("M13.8 screen report has an incomplete policy artifact matrix")
    for seed in M138_SCREEN_TRAINING_SEEDS:
        arms = artifacts[str(seed)]
        if not isinstance(arms, dict) or set(arms) != {"candidate", "legacy_control"}:
            raise ValueError(f"M13.8 screen seed {seed} has incomplete policy artifacts")
        for arm in ("candidate", "legacy_control"):
            metadata = arms[arm]
            if not isinstance(metadata, dict):
                raise ValueError(f"M13.8 screen {arm}:{seed} artifact metadata is invalid")
            source = Path(str(metadata.get("path", "")))
            if not source.is_file() or _file_sha256(source) != metadata.get("sha256"):
                raise ValueError(f"M13.8 screen {arm}:{seed} policy file hash mismatch")
            policy = load_m138_policy(source)
            if policy.arm != arm or policy.seed != seed or policy.stage != "screen":
                raise ValueError(f"M13.8 screen {arm}:{seed} policy identity mismatch")
            if m138_policy_fingerprint(policy) != metadata.get("policy_fingerprint"):
                raise ValueError(f"M13.8 screen {arm}:{seed} policy fingerprint mismatch")


def m138_confirm(
    *,
    screen_report: str | Path | dict[str, Any],
    artifact_dir: str | Path,
    ledger_path: str | Path = M138_CANONICAL_LEDGER_PATH,
    workers: int = M138_WORKERS,
) -> dict[str, Any]:
    """Run confirmation only after recomputing an authentic screen promotion."""

    if workers != M138_WORKERS:
        raise ValueError(f"M13.8 worker count is frozen at {M138_WORKERS}")
    protocol_fingerprint = m138_protocol_fingerprint()
    screen_payload, screen_path, screen_sha256 = _load_screen_report(screen_report)
    trusted_promotion = verify_m138_screen_report(
        screen_payload,
        expected_protocol_fingerprint=protocol_fingerprint,
        training_seeds=M138_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M138_SCREEN_PROBE_CONDITIONS),
        env_seeds=M138_SCREEN_PROBE_SEEDS,
        require_promotion=True,
    )
    _verify_screen_policy_artifacts(screen_payload)
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.8 confirmation artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    # Verification above is deliberately complete before the first 4900+ reset.
    ledger.open_partition("confirmation_fit")
    artifacts.mkdir(parents=True)
    policies, training, execution = _train_paired(
        stage="confirmation",
        seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        workers=workers,
    )
    policy_artifacts, restored = _serialize_all_policies(
        policies,
        seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        policy_dir=artifacts / "policies",
    )
    # All sixteen learned final policies reload cleanly before evaluation opens.
    ledger.open_partition("confirmation_evaluation")
    episode_summaries: dict[str, list[dict[str, Any]]] = {
        "candidate": [],
        "legacy_control": [],
        "random": [],
    }
    aggregates: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for seed in M138_CONFIRMATION_TRAINING_SEEDS:
        aggregates[str(seed)] = {}
        for arm in ("candidate", "legacy_control"):
            policy = restored[(arm, seed)]
            rows, arm_aggregates, traces = _evaluate_policy(
                policy,
                arm=arm,
                training_seed=seed,
                conditions=M138_CONFIRMATION_EVALUATION_CONDITIONS,
                seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
                trace_dir=artifacts / "traces" / "evaluation",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m138_policy_fingerprint(policy),
                replay_mode="learned",
            )
            episode_summaries[arm].extend(rows)
            aggregates[str(seed)][arm] = arm_aggregates
            manifest.extend(traces)
        random_policy = SeededRandomM138Policy(seed)
        rows, random_aggregates, traces = _evaluate_policy(
            random_policy,
            arm="candidate",
            training_seed=seed,
            conditions=M138_CONFIRMATION_EVALUATION_CONDITIONS,
            seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
            trace_dir=artifacts / "traces" / "evaluation",
            label=f"seed-{seed}/random",
            policy_fingerprint=f"pcg64-{seed}",
            replay_mode="random",
        )
        episode_summaries["random"].extend(rows)
        aggregates[str(seed)]["random"] = random_aggregates
        manifest.extend(traces)
    gate = m138_confirmation_passes(
        episode_summaries["candidate"],
        episode_summaries["legacy_control"],
        episode_summaries["random"],
        training_seeds=M138_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M138_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M138_CONFIRMATION_EVALUATION_SEEDS,
    )
    training_report = {
        str(seed): {
            arm: training[(arm, seed)]
            | {"initial_parameter_hash": policies[(arm, seed)].initial_parameter_hash}
            for arm in ("candidate", "legacy_control")
        }
        for seed in M138_CONFIRMATION_TRAINING_SEEDS
    }
    report: dict[str, Any] = {
        "schema_version": "m13.8-confirmation-report-r1",
        "stage": "confirmation",
        "protocol_version": M138_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "screen_authorization": {
            "path": screen_path,
            "file_sha256": screen_sha256,
            "content_hash": screen_payload["content_hash"],
            "trusted_promotion": trusted_promotion,
        },
        "split_ledger": ledger.snapshot(),
        "execution": execution,
        "training": training_report,
        "policy_artifacts": policy_artifacts,
        "episode_summaries": episode_summaries,
        "aggregates": aggregates,
        "trace_manifest": manifest,
        "confirmation_gate": gate,
        "confirmation": {
            "status": "pass" if bool(gate["passes"]) else "fail",
            "passes": bool(gate["passes"]),
            "robust_baseline_pass": bool(gate["robust_baseline_pass"]),
            "reward_improvement_supported": bool(gate["passes"]),
            "paired_margin_wins": int(gate["paired_margin_wins"]),
        },
        "limits": [
            "The audit split remains sealed and is not scored here.",
            "A causal reward-improvement claim requires both the 8/8 hard gate and 7/8 paired-margin test.",
            "M13.8 is not causally comparable to historical 160-step M13.6 absolute scores.",
        ],
    }
    report["content_hash"] = _report_hash(report)
    assert_compact_report(report)
    return report


def write_m138_confirmation_report(
    path: str | Path,
    *,
    screen_report: str | Path,
    ledger_path: str | Path = M138_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.8 confirmation report or artifact target already exists")
    report = m138_confirm(
        screen_report=screen_report,
        artifact_dir=artifacts,
        ledger_path=ledger_path,
    )
    _write_json_exclusive(output, report)
    return report
