"""M13.9: semi-Markov public-drive-potential reward study.

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

import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..env import EcosystemEnv
from ..tasks import LAYOUTS
from ..trajectory import SCHEMA_VERSION, ReplayResult, _json_value, replay_and_validate
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
    _MLP,
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_REPLAY_CAPACITY,
    M132_REPLAY_WARMUP,
    M132_TARGET_UPDATE_EVERY,
    encode_features,
)
from .m133 import advance_m133_memory, m133_config, m133_epsilon
from .m134 import _masked_argmax, m134_mask
from .m139_support import (
    M139_LEDGER_SCHEMA_VERSION,
    M139_PARTITION_DEPENDENCIES,
    M139_SCREEN_REPORT_SCHEMA_VERSION,
    SplitOpenLedger,
    assert_compact_report,
    compact_episode_summary,
    m139_confirmation_passes,
    m139_content_hash,
    m139_screen_promotes,
    seal_screen_report,
    verify_m139_screen_report,
)

M139_PROTOCOL_VERSION = "m13.9-public-drive-potential-r1"
M139_SCREEN_FIT_SEEDS = tuple(range(5_200, 5_220))
M139_SCREEN_PROBE_SEEDS = tuple(range(5_220, 5_228))
M139_CONFIRMATION_FIT_SEEDS = tuple(range(5_300, 5_340))
M139_CONFIRMATION_EVALUATION_SEEDS = tuple(range(5_400, 5_420))
M139_AUDIT_SEEDS = tuple(range(5_500, 5_520))
M139_SCREEN_TRAINING_SEEDS = (20_260_911, 20_260_912)
M139_CONFIRMATION_TRAINING_SEEDS = tuple(range(20_260_913, 20_260_921))
M139_SCREEN_EPISODES = 2_000
M139_CONFIRMATION_EPISODES = 12_000
M139_UPDATE_EVERY = 4
M139_TARGET_UPDATE_EVERY = M132_TARGET_UPDATE_EVERY
M139_WORKERS = 6
M139_HORIZON = 200
M139_CANONICAL_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "m139" / "split-open-ledger.json"
)

Arm = Literal["public_potential_candidate", "static_cost_control", "legacy_guardrail"]
Stage = Literal["screen", "confirmation", "fit_smoke"]
M139_SCREEN_ARMS: tuple[Arm, ...] = (
    "public_potential_candidate",
    "static_cost_control",
    "legacy_guardrail",
)
M139_CONFIRMATION_ARMS: tuple[Arm, ...] = (
    "public_potential_candidate",
    "static_cost_control",
)
M139_SMOKE_EPISODES_PER_FIT = 16


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


M139_SCREEN_FIT_CONDITIONS = _conditions("m139_screen")
M139_SCREEN_PROBE_CONDITIONS = _conditions("m139_probe")
M139_CONFIRMATION_FIT_CONDITIONS = _conditions("m139_confirm_fit")
M139_CONFIRMATION_EVALUATION_CONDITIONS = _conditions("m139_confirm_eval")
M139_AUDIT_CONDITIONS = _conditions("m139_audit")


def m139_config():
    """M13.3 mechanics/reward with only the declared 200-step correction."""

    return replace(m133_config(), max_episode_steps=M139_HORIZON)


def m139_safe(drives: np.ndarray) -> bool:
    values = np.asarray(drives, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("public drive vector must have shape (3,)")
    return bool(
        values[0] > np.float32(0.15) and values[1] > np.float32(0.15) and values[2] < np.float32(0.90)
    )


def m139_safe_error(drives: np.ndarray) -> float:
    values = np.asarray(drives, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("public drive vector must have shape (3,)")
    satiety = float(np.clip((0.30 - values[0]) / 0.15, 0.0, 1.0))
    energy = float(np.clip((0.30 - values[1]) / 0.15, 0.0, 1.0))
    boredom = float(np.clip((values[2] - 0.70) / 0.20, 0.0, 1.0))
    return max(satiety * satiety, energy * energy, boredom * boredom)


def m139_potential(drives: np.ndarray) -> float:
    """Return the frozen public potential ``Phi(x) = -E(x)``."""

    return -m139_safe_error(drives)


def m139_duration_discount(duration: float) -> float:
    """Return the coherent semi-Markov discount for one compiled macro."""

    value = float(duration)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("macro duration must be finite and positive")
    return float(M13_GAMMA**value)


def m139_backup_targets(
    rewards: np.ndarray,
    durations: np.ndarray,
    done: np.ndarray,
    bootstrap: np.ndarray,
) -> np.ndarray:
    """Compute the shared duration-aware Double-DQN target for every arm."""

    rewards_array = np.asarray(rewards, dtype=np.float32)
    durations_array = np.asarray(durations, dtype=np.float32)
    done_array = np.asarray(done, dtype=np.bool_)
    bootstrap_array = np.asarray(bootstrap, dtype=np.float32)
    if not (rewards_array.shape == durations_array.shape == done_array.shape == bootstrap_array.shape):
        raise ValueError("semi-Markov backup arrays must have identical shapes")
    if not np.all(np.isfinite(durations_array)) or np.any(durations_array <= 0.0):
        raise ValueError("semi-Markov backup durations must be finite and positive")
    discounts = np.power(np.float32(M13_GAMMA), durations_array, dtype=np.float32)
    return rewards_array + discounts * (~done_array) * bootstrap_array


def _duration(action: dict[str, np.ndarray | int]) -> float:
    duration = float(np.asarray(action["duration"], dtype=np.float32).item())
    if duration <= 0.0:
        raise ValueError("macro duration must be positive")
    return duration


@dataclass(slots=True)
class M139RewardState:
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


def _reward_state_snapshot(state: M139RewardState) -> dict[str, Any]:
    return asdict(state) | {
        "decision_safe_fraction": state.decision_safe_fraction,
        "duration_safe_fraction": state.duration_safe_fraction,
    }


def _outcome_value(info: dict[str, Any]) -> str:
    return str(info["outcome"])


def m139_learning_reward(
    *,
    arm: Arm,
    environment_reward: float,
    state: M139RewardState,
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
    error_before = m139_safe_error(drives_before)
    error_after = m139_safe_error(drives_after)
    safety_integral = duration * (error_before + error_after) / 2.0
    phi_before = -error_before
    phi_after = -error_after
    duration_discount = m139_duration_discount(duration)
    potential_reward = duration_discount * phi_after - phi_before

    # Final-transition occupancy is incorporated before the terminal gate.
    post_safe = m139_safe(drives_after)
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

    successful_consume = action_kind is ActionKind.CONSUME and outcome == ActionOutcome.SUCCESS.value
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
        and state.decision_steps == M139_HORIZON
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
        "phi_before": phi_before,
        "phi_after": phi_after,
        "duration_discount": duration_discount,
        "potential_reward": potential_reward,
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
    static_cost = float(
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
    candidate = float(static_cost - float(components["safety_cost"]) + potential_reward)
    components["static_cost_recomposed"] = static_cost
    components["candidate_recomposed"] = candidate
    components["environment_reward"] = float(environment_reward)
    if arm == "public_potential_candidate":
        learning_reward = candidate
    elif arm == "static_cost_control":
        learning_reward = static_cost
    elif arm == "legacy_guardrail":
        learning_reward = float(environment_reward)
    else:
        raise ValueError(f"unknown M13.9 arm {arm!r}")
    components["learning_reward"] = learning_reward
    return learning_reward, components


def m139_reward_components(**kwargs: Any) -> dict[str, float | bool | int]:
    """Return the exact component ledger while applying one sidecar transition."""

    return m139_learning_reward(**kwargs)[1]


class _DurationReplay:
    """Frozen FIFO replay with the compiler duration beside each transition."""

    def __init__(self) -> None:
        self.features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.actions = np.empty(M132_REPLAY_CAPACITY, dtype=np.int64)
        self.rewards = np.empty(M132_REPLAY_CAPACITY, dtype=np.float32)
        self.next_features = np.empty((M132_REPLAY_CAPACITY, M132_FEATURE_DIM), dtype=np.float32)
        self.next_masks = np.empty((M132_REPLAY_CAPACITY, len(M13Macro)), dtype=np.bool_)
        self.done = np.empty(M132_REPLAY_CAPACITY, dtype=np.bool_)
        self.durations = np.empty(M132_REPLAY_CAPACITY, dtype=np.float32)
        self.size = self.cursor = 0

    def add(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        duration: float,
    ) -> None:
        value = float(duration)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("replay duration must be finite and positive")
        index = self.cursor
        self.features[index] = features
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_features[index] = next_features
        self.next_masks[index] = next_mask
        self.done[index] = done
        self.durations[index] = value
        self.cursor = (index + 1) % M132_REPLAY_CAPACITY
        self.size = min(self.size + 1, M132_REPLAY_CAPACITY)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.choice(self.size, size=M132_BATCH_SIZE, replace=False)
        return (
            self.features[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_features[indices],
            self.next_masks[indices],
            self.done[indices],
            self.durations[indices],
        )


class CompactM139QPolicy:
    """Frozen M13.6 dense masked Double-DQN; only its training scalar varies."""

    include_drives = use_memory = True
    mask_mode = "complementary"
    update_every = M139_UPDATE_EVERY
    target_update_every = M139_TARGET_UPDATE_EVERY

    def __init__(self, *, arm: Arm, seed: int, stage: Stage) -> None:
        if arm not in M139_SCREEN_ARMS:
            raise ValueError(f"unknown M13.9 arm {arm!r}")
        if stage not in {"screen", "confirmation", "fit_smoke"}:
            raise ValueError(f"unknown M13.9 stage {stage!r}")
        self.arm, self.seed, self.stage = arm, int(seed), stage
        self.config = m139_config()
        self.online = _MLP(np.random.default_rng(self.seed))
        self.target = self.online.copy()
        self.update_count = 0
        self.initial_parameter_hash = m139_parameter_fingerprint(self)

    def reset(self) -> M13Memory:
        return M13Memory()

    def features(self, observation: dict[str, Any], memory: M13Memory) -> np.ndarray:
        return encode_features(observation, memory, include_drives=True, use_memory=True)

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        return m134_mask(observation, self.config, "complementary")

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(
            _masked_argmax(self.online.predict(self.features(observation, memory)), self.mask(observation))
        )

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

    def _update(self, replay: _DurationReplay, rng: np.random.Generator) -> float:
        features, actions, rewards, next_features, next_masks, done, durations = replay.sample(rng)
        selected = np.argmax(np.where(next_masks, self.online.predict(next_features), -np.inf), axis=1)
        bootstrap = self.target.predict(next_features)[np.arange(M132_BATCH_SIZE), selected]
        targets = m139_backup_targets(rewards, durations, done, bootstrap)
        loss = self.online.update(features, actions, targets.astype(np.float32))
        self.update_count += 1
        if self.update_count % self.target_update_every == 0:
            self.target = self.online.copy()
        return loss

    def _train(self, *, episodes: int) -> dict[str, float]:
        screen_like = self.stage in {"screen", "fit_smoke"}
        conditions = M139_SCREEN_FIT_CONDITIONS if screen_like else M139_CONFIRMATION_FIT_CONDITIONS
        seeds = M139_SCREEN_FIT_SEEDS if screen_like else M139_CONFIRMATION_FIT_SEEDS
        items = tuple(conditions.items())
        rng, replay = np.random.default_rng(self.seed), _DurationReplay()
        self.online, self.target, self.update_count = _MLP(rng), None, 0
        self.target = self.online.copy()
        self.initial_parameter_hash = m139_parameter_fingerprint(self)
        decisions, losses = 0, []
        env = EcosystemEnv(self.config)
        try:
            for episode in range(episodes):
                _, controls = items[episode % len(items)]
                env_seed = seeds[(episode // len(items)) % len(seeds)]
                observation, _ = env.reset(seed=env_seed, options=_options(controls))
                memory = self.reset()
                reward_state = M139RewardState(
                    required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
                )
                epsilon = m133_epsilon(episode)
                for _ in range(self.config.max_episode_steps):
                    features, mask = self.features(observation, memory), self.mask(observation)
                    if rng.random() < epsilon:
                        macro = M13Macro(int(rng.choice(np.flatnonzero(mask))))
                    else:
                        macro = self.choose(observation, memory)
                    action = compile_macro(macro, observation, self.config)
                    next_observation, env_reward, terminated, truncated, info = env.step(action)
                    learning_reward, _ = m139_learning_reward(
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
                        _duration(action),
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

    def train(self, *, episodes: int | None = None) -> dict[str, float]:
        if self.stage == "fit_smoke":
            raise ValueError("fit-smoke policies cannot enter frozen protocol training")
        expected = M139_SCREEN_EPISODES if self.stage == "screen" else M139_CONFIRMATION_EPISODES
        if episodes is None:
            episodes = expected
        if episodes != expected:
            raise ValueError(f"M13.9 {self.stage} training budget is frozen at {expected}")
        return self._train(episodes=episodes)

    def train_smoke(self, *, episodes: int) -> dict[str, float]:
        if self.stage != "fit_smoke" or not 1 <= episodes <= 32:
            raise ValueError("M13.9 smoke requires 1..32 screen-fit episodes")
        return self._train(episodes=episodes)


class SeededRandomM139Policy:
    """Matched mask-aware random policy; its single PCG64 stream is seed s."""

    include_drives, use_memory, mask_mode = True, True, "complementary"

    def __init__(self, seed: int) -> None:
        self.seed, self.config = int(seed), m139_config()
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

    def __init__(self, kind: Literal["feed_only", "play_only", "rest_wait", "quota_then_wait"]) -> None:
        self.kind, self.config = kind, m139_config()

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
        elif self.kind == "rest_wait":
            energy = float(np.asarray(observation["drives"], dtype=np.float32)[1])
            if energy <= 0.45:
                preferred = M13Macro.REST if bool(mask[int(M13Macro.REST)]) else M13Macro.GO_REST
            else:
                preferred = M13Macro.WAIT
        else:
            if memory.feed_count >= 3 and memory.play_count >= 3 and memory.rest_count >= 2:
                preferred = M13Macro.WAIT
            else:
                preferred = BalancedM139Oracle().choose(observation, memory)
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
            "m139.py",
            "m134.py",
            "m133.py",
            "m132.py",
            "m131.py",
            "m13.py",
            "m10.py",
        )
    }
    support = experiment_dir / "m139_support.py"
    if support.exists():
        paths["experiments/m139_support.py"] = support
    paths.update(
        {
            name: package_dir / name
            for name in ("actions.py", "config.py", "drives.py", "env.py", "tasks.py", "trajectory.py")
        }
    )
    protocol_doc = repository_dir / "docs" / "M13_9_PROTOCOL.md"
    if protocol_doc.exists():
        paths["docs/M13_9_PROTOCOL.md"] = protocol_doc
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.9 fingerprint sources missing: {missing}")
    return {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(paths.items())}


def m139_protocol_fingerprint() -> str:
    """Fingerprint the frozen reward, boundary, splits, budgets, and sources."""

    condition_sets = (
        M139_SCREEN_FIT_CONDITIONS,
        M139_SCREEN_PROBE_CONDITIONS,
        M139_CONFIRMATION_FIT_CONDITIONS,
        M139_CONFIRMATION_EVALUATION_CONDITIONS,
        M139_AUDIT_CONDITIONS,
    )
    layout_ids = tuple(str(row["layout_id"]) for conditions in condition_sets for row in conditions.values())
    payload = {
        "version": M139_PROTOCOL_VERSION,
        "config": asdict(m139_config()),
        "reward": {
            "safe_error": [0.30, 0.30, 0.70, 0.15, 0.15, 0.90],
            "duration_cost": -0.01,
            "static_safety_cost": -0.02,
            "potential_scale": -1.0,
            "potential_discount": "0.99**compiled_duration",
            "invalid_cost": -0.10,
            "blocked_cost": -0.10,
            "cycle_rewards": [0.25, 0.25, 0.375],
            "cycle_caps": [3, 3, 2],
            "recovery": 0.25,
            "terminal_success": 5.0,
            "terminal_failure": -2.0,
        },
        "splits": {
            "screen_fit": M139_SCREEN_FIT_SEEDS,
            "screen_probe": M139_SCREEN_PROBE_SEEDS,
            "confirmation_fit": M139_CONFIRMATION_FIT_SEEDS,
            "confirmation_evaluation": M139_CONFIRMATION_EVALUATION_SEEDS,
            "audit": M139_AUDIT_SEEDS,
        },
        "conditions": {
            "screen_fit": M139_SCREEN_FIT_CONDITIONS,
            "screen_probe": M139_SCREEN_PROBE_CONDITIONS,
            "confirmation_fit": M139_CONFIRMATION_FIT_CONDITIONS,
            "confirmation_evaluation": M139_CONFIRMATION_EVALUATION_CONDITIONS,
            "audit": M139_AUDIT_CONDITIONS,
        },
        "layouts": {layout_id: asdict(LAYOUTS[layout_id]) for layout_id in layout_ids},
        "ledger": {
            "schema": M139_LEDGER_SCHEMA_VERSION,
            "dependencies": M139_PARTITION_DEPENDENCIES,
            "screen_report_schema": M139_SCREEN_REPORT_SCHEMA_VERSION,
        },
        "training": {
            "screen_seeds": M139_SCREEN_TRAINING_SEEDS,
            "confirmation_seeds": M139_CONFIRMATION_TRAINING_SEEDS,
            "screen_episodes": M139_SCREEN_EPISODES,
            "confirmation_episodes": M139_CONFIRMATION_EPISODES,
            "feature_dim": M132_FEATURE_DIM,
            "replay_capacity": M132_REPLAY_CAPACITY,
            "replay_warmup": M132_REPLAY_WARMUP,
            "batch_size": M132_BATCH_SIZE,
            "gamma": M13_GAMMA,
            "update_every": M139_UPDATE_EVERY,
            "target_update_every": M139_TARGET_UPDATE_EVERY,
            "workers": M139_WORKERS,
            "worker_threads": 1,
            "backup": "reward + not_done * 0.99**compiled_duration * masked_double_dqn",
        },
        "screen_arms": M139_SCREEN_ARMS,
        "confirmation_arms": M139_CONFIRMATION_ARMS,
        "specialists": ["feed_only", "play_only", "rest_wait", "quota_then_wait"],
        "random_seed_derivation": "single PCG64 stream seeded exactly by paired training seed",
        "sources": _source_hashes(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def m139_policy_fingerprint(policy: CompactM139QPolicy) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "protocol_version": M139_PROTOCOL_VERSION,
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


def m139_parameter_fingerprint(policy: CompactM139QPolicy) -> str:
    digest = hashlib.sha256()
    for key in sorted(policy.online.params):
        digest.update(key.encode())
        digest.update(np.asarray(policy.online.params[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def write_m139_policy(path: str | Path, policy: CompactM139QPolicy) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "m139-policy-v1",
        "protocol_fingerprint": m139_protocol_fingerprint(),
        "arm": policy.arm,
        "seed": policy.seed,
        "stage": policy.stage,
        "initial_parameter_hash": policy.initial_parameter_hash,
        "network": {key: value.tolist() for key, value in policy.online.params.items()},
        "policy_fingerprint": m139_policy_fingerprint(policy),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(output.resolve()),
        "sha256": _file_sha256(output),
        "policy_fingerprint": m139_policy_fingerprint(policy),
    }


def load_m139_policy(path: str | Path) -> CompactM139QPolicy:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m139-policy-v1":
        raise ValueError("unsupported M13.9 policy schema")
    if payload.get("protocol_fingerprint") != m139_protocol_fingerprint():
        raise ValueError("M13.9 policy artifact does not match the frozen protocol")
    policy = CompactM139QPolicy(
        arm=payload["arm"],
        seed=int(payload["seed"]),
        stage=payload["stage"],
    )
    if payload.get("initial_parameter_hash") != policy.initial_parameter_hash:
        raise ValueError("M13.9 policy artifact has an inconsistent initial parameter hash")
    network = payload.get("network", {})
    if set(network) != set(policy.online.params):
        raise ValueError("M13.9 policy artifact has an invalid parameter set")
    for key, values in network.items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != policy.online.params[key].shape or not np.all(np.isfinite(array)):
            raise ValueError(f"M13.9 policy parameter {key!r} is invalid")
        policy.online.params[key] = array
    policy.target = policy.online.copy()
    if payload.get("policy_fingerprint") != m139_policy_fingerprint(policy):
        raise ValueError("M13.9 policy artifact fingerprint mismatch")
    return policy


class BalancedM139Oracle(ScriptedM13Oracle):
    """Inherited public scripted ceiling with the M13.3 public memory update."""

    def __init__(self) -> None:
        self.config = m139_config()

    def observe(self, memory: M13Memory, **kwargs: Any) -> None:
        advance_m133_memory(memory, config=self.config, **kwargs)


@dataclass(frozen=True, slots=True)
class M139EpisodeResult:
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
    wait_decisions: int
    unsafe_wait_decisions: int
    unsafe_wait_fraction: float
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
            "full_gate_success": self.full_gate_success,
            "decision_safe_fraction": self.decision_safe_fraction,
            "duration_safe_fraction": self.duration_safe_fraction,
            "recovery_required": self.recovery_required,
            "recovery_complete": self.recovery_complete,
            "decision_steps": self.decision_steps,
            "unsafe_wait_fraction": self.unsafe_wait_fraction,
            "wait_decisions": self.wait_decisions,
            "unsafe_wait_decisions": self.unsafe_wait_decisions,
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
    return m134_mask(observation, m139_config(), "complementary")


def _trace_header(
    *,
    episode_id: str,
    seed: int,
    controls: dict[str, Any],
    arm: Arm,
    training_seed: int,
    condition: str,
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
        "config": asdict(m139_config()),
        "m139_protocol_fingerprint": m139_protocol_fingerprint(),
        "m139_policy_fingerprint": policy_fingerprint,
        "m139_policy_label": policy_label,
        "m139_arm": arm,
        "m139_training_seed": training_seed,
        "m139_condition": condition,
        "m139_required_recovery": bool(controls.get("event_relocation_on_first_pickup")),
    }


def run_m139_episode(
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
) -> M139EpisodeResult:
    """Evaluate one policy without retaining transition records in RAM."""

    config = getattr(policy, "config", m139_config())
    if config != m139_config():
        raise ValueError("M13.9 evaluation requires the exact frozen 200-step config")
    env = EcosystemEnv(config)
    trace_handle = None
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        reward_state = M139RewardState(
            required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
        )
        interventions = {
            "feature_shape_violation": 0,
            "nonfinite_q_violation": 0,
            "mask_shape_violation": 0,
            "mask_violation": 0,
            "redundant_go": 0,
            "reward_recomposition_violation": 0,
            "duration_discount_violation": 0,
        }
        macro_counts = {macro.name: 0 for macro in M13Macro}
        wait_decisions = unsafe_wait_decisions = 0
        environment_return = learning_return = 0.0
        trace = Path(trace_path) if trace_path is not None else None
        episode_id = f"m139-{policy_label.replace('/', '-')}-{condition}-seed-{seed}"
        if trace is not None:
            if trace.exists():
                raise FileExistsError(f"M13.9 trace already exists: {trace}")
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
                        condition=condition,
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
                raise ValueError("M13.9 public mask is malformed")
            features = None
            q_values = None
            if isinstance(policy, CompactM139QPolicy):
                features = policy.features(observation, memory)
                if features.shape != (M132_FEATURE_DIM,):
                    interventions["feature_shape_violation"] += 1
                q_values = np.asarray(policy.online.predict(features), dtype=np.float32)
                if q_values.shape != (len(M13Macro),) or not np.all(np.isfinite(q_values)):
                    interventions["nonfinite_q_violation"] += 1
            rng_state = (
                _json_value(policy.rng.bit_generator.state)
                if isinstance(policy, SeededRandomM139Policy)
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
            learning_reward, components = m139_learning_reward(
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
            expected_learning = {
                "public_potential_candidate": float(components["candidate_recomposed"]),
                "static_cost_control": float(components["static_cost_recomposed"]),
                "legacy_guardrail": float(env_reward),
            }[arm]
            if not np.isclose(learning_reward, expected_learning, rtol=1e-7, atol=1e-9):
                interventions["reward_recomposition_violation"] += 1
            if not np.isclose(
                float(components["duration_discount"]),
                m139_duration_discount(_duration(action)),
                rtol=1e-7,
                atol=1e-9,
            ):
                interventions["duration_discount_violation"] += 1
            policy.observe(
                memory,
                observation_before=observation,
                macro=macro,
                action=action,
                observation_after=next_observation,
            )
            environment_return += float(env_reward)
            learning_return += float(learning_reward)
            if macro is M13Macro.WAIT:
                wait_decisions += 1
                unsafe_wait_decisions += int(float(components["safe_error_after"]) > 0.0)
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
                    and step == M139_HORIZON
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
                return M139EpisodeResult(
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
                    wait_decisions=wait_decisions,
                    unsafe_wait_decisions=unsafe_wait_decisions,
                    unsafe_wait_fraction=(unsafe_wait_decisions / wait_decisions if wait_decisions else 0.0),
                    interventions=interventions,
                    macro_counts=macro_counts,
                )
        raise AssertionError(f"M13.9 episode did not terminate; last info={last_info!r}")
    finally:
        if trace_handle is not None:
            trace_handle.close()
        env.close()


def replay_m139_trace(
    path: str | Path,
    policy: Any,
) -> ReplayResult:
    """Strictly replay environment, inference, memory, and reward sidecar."""

    source = Path(path)
    generic = replay_and_validate(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    if header.get("record_type") != "episode_metadata" or not steps:
        raise ValueError("M13.9 trace lacks metadata or steps")
    if header.get("m139_protocol_fingerprint") != m139_protocol_fingerprint():
        raise ValueError("M13.9 trace protocol fingerprint mismatch")
    reset_requires_recovery = bool(
        header.get("reset_options", {}).get("event_relocation_on_first_pickup", False)
    )
    if bool(header.get("m139_required_recovery")) != reset_requires_recovery:
        raise ValueError("M13.9 trace recovery requirement disagrees with reset metadata")
    expected_policy_fingerprint = header.get("m139_policy_fingerprint")
    if isinstance(policy, CompactM139QPolicy):
        if expected_policy_fingerprint != m139_policy_fingerprint(policy):
            raise ValueError("M13.9 learned trace policy fingerprint mismatch")
        if header.get("m139_arm") != policy.arm:
            raise ValueError("M13.9 learned trace arm mismatch")
    elif isinstance(policy, SeededRandomM139Policy):
        if expected_policy_fingerprint != f"pcg64-{policy.seed}":
            raise ValueError("M13.9 random trace policy fingerprint mismatch")
    elif expected_policy_fingerprint is not None:
        raise ValueError("M13.9 deterministic trace unexpectedly names a policy artifact")
    arm: Arm = header["m139_arm"]
    env = EcosystemEnv(m139_config())
    try:
        observation, _ = env.reset(seed=int(header["seed"]), options=dict(header["reset_options"]))
        memory = policy.reset()
        reward_state = M139RewardState(required_recovery=bool(header["m139_required_recovery"]))
        for expected in steps:
            step = int(expected["step"])
            if _json_value(observation) != expected["policy_observation"]:
                raise ValueError(f"M13.9 policy observation mismatch at step {step}")
            if _memory_snapshot(memory) != expected["memory_before"]:
                raise ValueError(f"M13.9 memory-before mismatch at step {step}")
            _json_close(
                _reward_state_snapshot(reward_state),
                expected["reward_state_before"],
                path="reward_state_before",
            )
            mask = _policy_mask(policy, observation)
            if mask.astype(bool).tolist() != expected["mask"]:
                raise ValueError(f"M13.9 mask mismatch at step {step}")
            if isinstance(policy, CompactM139QPolicy):
                features = policy.features(observation, memory)
                q_values = np.asarray(policy.online.predict(features), dtype=np.float32)
                if features.tolist() != expected["features"]:
                    raise ValueError(f"M13.9 feature mismatch at step {step}")
                if not np.array_equal(q_values, np.asarray(expected["q_values"], dtype=np.float32)):
                    raise ValueError(f"M13.9 Q-value mismatch at step {step}")
            elif isinstance(policy, SeededRandomM139Policy):
                state = expected.get("random_rng_state_before")
                if state is None:
                    raise ValueError(f"M13.9 random trace lacks RNG state at step {step}")
                policy.rng.bit_generator.state = state
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, m139_config())
            if not bool(mask[int(macro)]) or macro.name != expected["macro"]:
                raise ValueError(f"M13.9 constrained macro mismatch at step {step}")
            if _json_value(action) != expected["action"]:
                raise ValueError(f"M13.9 compiled action mismatch at step {step}")
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            learning_reward, components = m139_learning_reward(
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
                raise ValueError(f"M13.9 learning reward mismatch at step {step}")
            _json_close(_json_value(components), expected["reward_components"], path="reward_components")
            _json_close(
                _reward_state_snapshot(reward_state),
                expected["reward_state_after"],
                path="reward_state_after",
            )
            policy.observe(
                memory,
                observation_before=observation,
                macro=macro,
                action=action,
                observation_after=next_observation,
            )
            if _memory_snapshot(memory) != expected["memory_after"]:
                raise ValueError(f"M13.9 memory-after mismatch at step {step}")
            observation = next_observation
        return generic
    finally:
        env.close()


def replay_m139_random_trace(path: str | Path, policy: SeededRandomM139Policy) -> ReplayResult:
    """Typed public wrapper retained for random-policy conformance tests."""

    return replay_m139_trace(path, policy)


def _compact_summary_from_trace(
    path: str | Path,
    policy: CompactM139QPolicy | SeededRandomM139Policy,
) -> dict[str, object]:
    """Strictly replay a trace and derive the exact gate evidence from it."""

    source = Path(path)
    replay_m139_trace(source, policy)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    header, steps = rows[0], rows[1:]
    final = steps[-1]
    reward_state = final["reward_state_after"]
    wait_rows = [row for row in steps if row["macro"] == M13Macro.WAIT.name]
    unsafe_waits = sum(float(row["reward_components"]["safe_error_after"]) > 0.0 for row in wait_rows)
    survived = bool(final["survived"])
    maintenance = bool(
        final["truncated"]
        and not final["terminated"]
        and int(final["step"]) == M139_HORIZON
        and survived
        and int(reward_state["feed_cycles"]) >= 3
        and int(reward_state["play_cycles"]) >= 3
        and int(reward_state["rest_cycles"]) >= 2
    )
    return compact_episode_summary(
        {
            "training_seed": int(header["m139_training_seed"]),
            "condition": str(header["m139_condition"]),
            "env_seed": int(header["seed"]),
            "survived": survived,
            "maintenance_complete": maintenance,
            "full_gate_success": bool(final["reward_components"]["full_terminal_success"]),
            "decision_safe_fraction": float(reward_state["decision_safe_fraction"]),
            "duration_safe_fraction": float(reward_state["duration_safe_fraction"]),
            "recovery_required": bool(header["m139_required_recovery"]),
            "recovery_complete": bool(reward_state["recovery_complete"]),
            "decision_steps": int(reward_state["decision_steps"]),
            "wait_decisions": len(wait_rows),
            "unsafe_wait_decisions": int(unsafe_waits),
            "unsafe_wait_fraction": unsafe_waits / len(wait_rows) if wait_rows else 0.0,
            "conformance_violations": 0,
            "replay_violations": 0,
        }
    )


def _verify_report_trace_evidence(
    report: dict[str, Any],
    *,
    learned_policies: dict[tuple[Arm, int], CompactM139QPolicy],
    arms: tuple[str, ...],
    training_seeds: tuple[int, ...],
    conditions: dict[str, dict[str, Any]],
    env_seeds: tuple[int, ...],
) -> None:
    """Bind sealed gate summaries to hashed, replayed traces and frozen axes."""

    manifest = report.get("trace_manifest")
    evidence = report.get("episode_summaries")
    if not isinstance(manifest, list) or not isinstance(evidence, dict) or set(evidence) != set(arms):
        raise ValueError("M13.9 report has incomplete trace evidence")
    expected: dict[tuple[str, int, str, int], dict[str, object]] = {}
    for arm in arms:
        rows = evidence.get(arm)
        if not isinstance(rows, list):
            raise TypeError(f"M13.9 {arm} evidence is not a list")
        for row in rows:
            compact = compact_episode_summary(row)
            key = (
                arm,
                int(compact["training_seed"]),
                str(compact["condition"]),
                int(compact["env_seed"]),
            )
            if key in expected:
                raise ValueError(f"M13.9 report duplicates trace evidence {key!r}")
            expected[key] = compact
    required = {
        (arm, training_seed, condition, env_seed)
        for arm in arms
        for training_seed in training_seeds
        for condition in conditions
        for env_seed in env_seeds
    }
    if set(expected) != required:
        raise ValueError("M13.9 report gate evidence does not match the frozen score matrix")

    seen: set[tuple[str, int, str, int]] = set()
    for arm, training_seed, condition, env_seed in sorted(required):
        label = f"seed-{training_seed}/{arm}"
        matches = [
            row
            for row in manifest
            if isinstance(row, dict)
            and row.get("policy") == label
            and row.get("condition") == condition
            and row.get("env_seed") == env_seed
        ]
        if len(matches) != 1:
            raise ValueError(
                f"M13.9 trace manifest mismatch for {(arm, training_seed, condition, env_seed)!r}"
            )
        metadata = matches[0]
        source = Path(str(metadata.get("path", "")))
        if not source.is_file() or _file_sha256(source) != metadata.get("sha256"):
            raise ValueError(f"M13.9 trace hash mismatch for {(arm, training_seed, condition, env_seed)!r}")
        header = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        if header.get("reset_options") != _json_value(_options(conditions[condition])):
            raise ValueError("M13.9 trace reset options do not match the frozen condition")
        if arm == "random":
            policy: CompactM139QPolicy | SeededRandomM139Policy = SeededRandomM139Policy(training_seed)
        else:
            policy = learned_policies[(arm, training_seed)]  # type: ignore[index]
        derived = _compact_summary_from_trace(source, policy)
        key = (arm, training_seed, condition, env_seed)
        if derived != expected[key]:
            raise ValueError(f"M13.9 gate summary is not derived from trace {key!r}")
        seen.add(key)
    if seen != required:
        raise ValueError("M13.9 trace evidence verification was incomplete")


def _aggregate_episode_results(episodes: list[M139EpisodeResult]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate empty M13.9 episode results")
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
        "wait_decisions": sum(row.wait_decisions for row in episodes),
        "unsafe_wait_decisions": sum(row.unsafe_wait_decisions for row in episodes),
        "unsafe_wait_fraction": (
            sum(row.unsafe_wait_decisions for row in episodes) / sum(row.wait_decisions for row in episodes)
            if sum(row.wait_decisions for row in episodes)
            else 0.0
        ),
        "interventions": {
            key: sum(row.interventions.get(key, 0) for row in episodes)
            for key in sorted({key for row in episodes for key in row.interventions})
        },
        "macro_counts": {
            macro.name: sum(row.macro_counts.get(macro.name, 0) for row in episodes) for macro in M13Macro
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
    by_condition: dict[str, list[M139EpisodeResult]] = {}
    for condition, controls in conditions.items():
        condition_rows: list[M139EpisodeResult] = []
        for seed in seeds:
            trace = trace_dir / label / condition / f"seed-{seed}.jsonl"
            episode = run_m139_episode(
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
                    if not isinstance(policy, CompactM139QPolicy):
                        raise TypeError("learned replay requires CompactM139QPolicy")
                    replay_m139_trace(trace, policy)
                elif replay_mode == "random":
                    replay_m139_trace(trace, SeededRandomM139Policy(training_seed))
                elif replay_mode == "deterministic":
                    replay_m139_trace(trace, policy)
                else:
                    replay_and_validate(trace)
            except Exception as exc:  # noqa: BLE001 - preserve opened-split rejection evidence
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
) -> tuple[Arm, int, CompactM139QPolicy, dict[str, float]]:
    started = time.perf_counter()
    policy = CompactM139QPolicy(arm=arm, seed=seed, stage=stage)
    training = policy.train()
    training["elapsed_seconds"] = time.perf_counter() - started
    return arm, seed, policy, training


def _smoke_worker(
    arm: Arm,
    seed: int,
) -> tuple[Arm, int, CompactM139QPolicy, dict[str, float]]:
    started = time.perf_counter()
    policy = CompactM139QPolicy(arm=arm, seed=seed, stage="fit_smoke")
    training = policy.train_smoke(episodes=M139_SMOKE_EPISODES_PER_FIT)
    training["elapsed_seconds"] = time.perf_counter() - started
    return arm, seed, policy, training


def _train_smoke_wave() -> tuple[
    dict[tuple[Arm, int], CompactM139QPolicy],
    dict[tuple[Arm, int], dict[str, float]],
    dict[str, Any],
]:
    jobs = m139_screen_jobs()
    policies: dict[tuple[Arm, int], CompactM139QPolicy] = {}
    training: dict[tuple[Arm, int], dict[str, float]] = {}
    completion_order: list[str] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=M139_WORKERS, mp_context=context) as executor:
        futures = {executor.submit(_smoke_worker, arm, seed): (arm, seed) for arm, seed in jobs}
        for future in as_completed(futures):
            arm, seed, policy, diagnostics = future.result()
            policies[(arm, seed)] = policy
            training[(arm, seed)] = diagnostics
            completion_order.append(f"{arm}:{seed}")
    for seed in M139_SCREEN_TRAINING_SEEDS:
        replicas = [policies[(arm, seed)] for arm in M139_SCREEN_ARMS]
        if len({policy.initial_parameter_hash for policy in replicas}) != 1:
            raise ValueError(f"M13.9 smoke seed {seed} did not start byte-identically")
    return (
        policies,
        training,
        {
            "workers": M139_WORKERS,
            "start_method": "spawn",
            "worker_threads": 1,
            "submission_order": [f"{arm}:{seed}" for arm, seed in jobs],
            "completion_order": completion_order,
        },
    )


def m139_screen_jobs() -> list[tuple[Arm, int]]:
    """Return the frozen six-job submission wave in deterministic order."""

    return [(arm, seed) for seed in M139_SCREEN_TRAINING_SEEDS for arm in M139_SCREEN_ARMS]


def _train_paired(
    *,
    stage: Stage,
    seeds: tuple[int, ...],
    workers: int,
) -> tuple[
    dict[tuple[Arm, int], CompactM139QPolicy],
    dict[tuple[Arm, int], dict[str, float]],
    dict[str, Any],
]:
    if workers != M139_WORKERS:
        raise ValueError(f"M13.9 worker count is frozen at {M139_WORKERS}")
    arms = M139_SCREEN_ARMS if stage == "screen" else M139_CONFIRMATION_ARMS
    jobs: list[tuple[Arm, int]] = (
        m139_screen_jobs()
        if stage == "screen" and seeds == M139_SCREEN_TRAINING_SEEDS
        else [(arm, seed) for seed in seeds for arm in arms]
    )
    policies: dict[tuple[Arm, int], CompactM139QPolicy] = {}
    training: dict[tuple[Arm, int], dict[str, float]] = {}
    completion_order: list[str] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_train_worker, arm, seed, stage): (arm, seed) for arm, seed in jobs}
        for future in as_completed(futures):
            arm, seed, policy, diagnostics = future.result()
            policies[(arm, seed)] = policy
            training[(arm, seed)] = diagnostics
            completion_order.append(f"{arm}:{seed}")
    for seed in seeds:
        replicas = [policies[(arm, seed)] for arm in arms]
        if len({policy.initial_parameter_hash for policy in replicas}) != 1:
            raise ValueError(f"M13.9 paired seed {seed} did not start from identical parameters")
        if any(policy.config != replicas[0].config for policy in replicas[1:]):
            raise ValueError(f"M13.9 paired seed {seed} used different configs")
    execution = {
        "workers": workers,
        "start_method": "spawn",
        "worker_threads": 1,
        "submission_order": [f"{arm}:{seed}" for arm, seed in jobs],
        "completion_order": completion_order,
    }
    return policies, training, execution


def _serialize_all_policies(
    policies: dict[tuple[Arm, int], CompactM139QPolicy],
    *,
    seeds: tuple[int, ...],
    policy_dir: Path,
    arms: tuple[Arm, ...] = M139_SCREEN_ARMS,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[tuple[Arm, int], CompactM139QPolicy]]:
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    restored: dict[tuple[Arm, int], CompactM139QPolicy] = {}
    for seed in seeds:
        artifacts[str(seed)] = {}
        for arm in arms:
            policy = policies[(arm, seed)]
            artifact = write_m139_policy(policy_dir / f"seed-{seed}" / f"{arm}.json", policy)
            loaded = load_m139_policy(artifact["path"])
            if m139_policy_fingerprint(loaded) != artifact["policy_fingerprint"]:
                raise ValueError(f"M13.9 serialized {arm}:{seed} fingerprint mismatch")
            artifacts[str(seed)][arm] = artifact
            restored[(arm, seed)] = loaded
    expected = len(seeds) * len(arms)
    if len(restored) != expected or sum(len(row) for row in artifacts.values()) != expected:
        raise ValueError("M13.9 did not serialize every learned final policy")
    return artifacts, restored


def _potential_preflight() -> dict[str, Any]:
    """Cheap deterministic proof of signs, bounds, and telescoping identity."""

    comfort = np.asarray((0.60, 0.60, 0.30), dtype=np.float64)
    unsafe = np.asarray((0.15, 0.60, 0.30), dtype=np.float64)
    wait_comfort = m139_duration_discount(1.0) * m139_potential(comfort) - m139_potential(comfort)
    wait_worse = m139_duration_discount(1.0) * m139_potential(unsafe) - m139_potential(comfort)
    restore = m139_duration_discount(1.0) * m139_potential(comfort) - m139_potential(unsafe)
    rng = np.random.default_rng(20_260_909)
    max_residual = 0.0
    cases = 0
    trajectories: list[tuple[np.ndarray, np.ndarray]] = [
        (np.stack((unsafe, comfort, unsafe)), np.asarray((1.0, 1.0))),
        (np.stack((comfort, unsafe, comfort)), np.asarray((0.1, 1.0))),
    ]
    for _ in range(128):
        transitions = int(rng.integers(2, 10))
        trajectories.append(
            (
                rng.uniform(0.0, 1.0, size=(transitions + 1, 3)),
                rng.uniform(0.1, 2.0, size=transitions),
            )
        )
    for drives, durations in trajectories:
        elapsed = 0.0
        discounted_sum = 0.0
        for index, duration in enumerate(durations):
            term = m139_duration_discount(float(duration)) * m139_potential(
                drives[index + 1]
            ) - m139_potential(drives[index])
            discounted_sum += (M13_GAMMA**elapsed) * term
            elapsed += float(duration)
        endpoint = (M13_GAMMA**elapsed) * m139_potential(drives[-1]) - m139_potential(drives[0])
        max_residual = max(max_residual, abs(discounted_sum - endpoint))
        cases += 1
    passes = (
        np.isclose(wait_comfort, 0.0, atol=1e-12)
        and wait_worse < 0.0
        and restore > 0.0
        and max_residual <= 1e-10
    )
    return {
        "passes": bool(passes),
        "comfort_wait_potential": float(wait_comfort),
        "worsening_wait_potential": float(wait_worse),
        "restoration_potential": float(restore),
        "telescoping_cases": cases,
        "max_telescoping_residual": float(max_residual),
    }


def _preflight(
    *,
    artifact_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = m10_scan_coverage(M139_SCREEN_FIT_CONDITIONS, seeds=M139_SCREEN_FIT_SEEDS)
    trace_dir = artifact_dir / "traces" / "preflight"
    manifest: list[dict[str, Any]] = []
    ceiling_summaries, ceiling_aggregates, ceiling_manifest = _evaluate_policy(
        BalancedM139Oracle(),
        arm="public_potential_candidate",
        training_seed=0,
        conditions=M139_SCREEN_FIT_CONDITIONS,
        seeds=M139_SCREEN_FIT_SEEDS,
        trace_dir=trace_dir,
        label="balanced_public_oracle",
        policy_fingerprint=None,
        replay_mode="deterministic",
    )
    manifest.extend(ceiling_manifest)
    specialists: dict[str, Any] = {}
    specialist_pass = True
    for kind in ("feed_only", "play_only", "rest_wait", "quota_then_wait"):
        _, aggregates, specialist_manifest = _evaluate_policy(
            _SpecialistPolicy(kind),
            arm="public_potential_candidate",
            training_seed=0,
            conditions=M139_SCREEN_FIT_CONDITIONS,
            seeds=M139_SCREEN_FIT_SEEDS,
            trace_dir=trace_dir,
            label=f"specialist_{kind}",
            policy_fingerprint=None,
            replay_mode="deterministic",
        )
        manifest.extend(specialist_manifest)
        gaps = {
            condition: float(ceiling_aggregates[condition]["mean_learning_return"])
            - float(aggregates[condition]["mean_learning_return"])
            for condition in M139_SCREEN_FIT_CONDITIONS
        }
        passes = all(value >= 2.0 for value in gaps.values())
        specialist_pass = specialist_pass and passes
        specialists[kind] = {"aggregates": aggregates, "return_gap_from_balanced": gaps, "passes": passes}
    coverage_pass = all(bool(row["passes"]) for row in coverage.values())
    ceiling_pass = all(
        int(row["episodes"]) == len(M139_SCREEN_FIT_SEEDS)
        and int(row["full_gate_success"]) == len(M139_SCREEN_FIT_SEEDS)
        and all(int(value) == 0 for value in row["interventions"].values())
        for row in ceiling_aggregates.values()
    )
    replay_pass = all(bool(row["replay_pass"]) for row in manifest)
    potential = _potential_preflight()
    preflight = {
        "passes": coverage_pass
        and ceiling_pass
        and specialist_pass
        and replay_pass
        and bool(potential["passes"]),
        "coverage_pass": coverage_pass,
        "ceiling_pass": ceiling_pass,
        "specialist_return_pass": specialist_pass,
        "replay_pass": replay_pass,
        "potential_pass": bool(potential["passes"]),
        "potential_diagnostics": potential,
        "coverage": coverage,
        "balanced_public_oracle": {
            "aggregates": ceiling_aggregates,
            "compact_episode_summaries": ceiling_summaries,
        },
        "specialists": specialists,
    }
    return preflight, manifest


def _coverage_ceiling_gate(
    *,
    conditions: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    trace_dir: Path,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = m10_scan_coverage(conditions, seeds=seeds)
    summaries, aggregates, manifest = _evaluate_policy(
        BalancedM139Oracle(),
        arm="public_potential_candidate",
        training_seed=0,
        conditions=conditions,
        seeds=seeds,
        trace_dir=trace_dir,
        label=label,
        policy_fingerprint=None,
        replay_mode="deterministic",
    )
    coverage_pass = all(bool(row["passes"]) for row in coverage.values())
    ceiling_pass = all(
        int(row["episodes"]) == len(seeds)
        and int(row["full_gate_success"]) == len(seeds)
        and all(int(value) == 0 for value in row["interventions"].values())
        for row in aggregates.values()
    )
    replay_pass = all(bool(row["replay_pass"]) for row in manifest)
    return {
        "passes": coverage_pass and ceiling_pass and replay_pass,
        "coverage_pass": coverage_pass,
        "ceiling_pass": ceiling_pass,
        "replay_pass": replay_pass,
        "coverage": coverage,
        "balanced_public_oracle": {
            "aggregates": aggregates,
            "compact_episode_summaries": summaries,
        },
    }, manifest


def _report_hash(report: dict[str, Any]) -> str:
    return m139_content_hash({key: value for key, value in report.items() if key != "content_hash"})


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"M13.9 report already exists: {path}")
    assert_compact_report(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def m139_fit_smoke(
    *,
    artifact_dir: str | Path,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a fixed, fit-only plumbing smoke that can never open held-out data."""

    started = time.perf_counter()
    protocol_fingerprint = m139_protocol_fingerprint()
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.9 smoke artifact directory already exists: {artifacts}")
    potential = _potential_preflight()
    if not bool(potential["passes"]):
        raise ValueError("M13.9 potential preflight failed before smoke fitting")
    artifacts.mkdir(parents=True)
    smoke_ledger_path = (
        Path(ledger_path) if ledger_path is not None else artifacts / "split-open-ledger.non-protocol.json"
    )
    ledger = SplitOpenLedger(smoke_ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.ensure_screen_fit_open()
    policies, training, execution = _train_smoke_wave()
    policy_artifacts, restored = _serialize_all_policies(
        policies,
        seeds=M139_SCREEN_TRAINING_SEEDS,
        policy_dir=artifacts / "non_protocol_policies",
        arms=M139_SCREEN_ARMS,
    )
    manifest: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    smoke_seed = M139_SCREEN_FIT_SEEDS[0]
    smoke_conditions = {name: controls for name, controls in M139_SCREEN_FIT_CONDITIONS.items()}
    for seed in M139_SCREEN_TRAINING_SEEDS:
        evaluations[str(seed)] = {}
        for arm in M139_SCREEN_ARMS:
            policy = restored[(arm, seed)]
            rows, aggregates, traces = _evaluate_policy(
                policy,
                arm=arm,
                training_seed=seed,
                conditions=smoke_conditions,
                seeds=(smoke_seed,),
                trace_dir=artifacts / "non_protocol_traces",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m139_policy_fingerprint(policy),
                replay_mode="learned",
            )
            if any(int(row["replay_violations"]) for row in rows):
                raise ValueError(f"M13.9 smoke strict replay failed for {arm}:{seed}")
            evaluations[str(seed)][arm] = aggregates
            manifest.extend(traces)
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema_version": "m13.9-fit-smoke-non-protocol-r1",
        "protocol_version": M139_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "label": "NON-PROTOCOL / NON-PROMOTIONAL FIT-ONLY SMOKE",
        "promotional": False,
        "eligible_policy_artifacts": False,
        "ledger_scope": "smoke-only-non-protocol",
        "canonical_protocol_ledger_accessed": False,
        "opened_partitions": list(ledger.snapshot()["opened_partitions"]),
        "forbidden_partitions_opened": any(
            name in ledger.snapshot()["opened_partitions"]
            for name in ("screen_probe", "confirmation_fit", "confirmation_evaluation", "audit")
        ),
        "fit_data": {
            "partition": "screen_fit",
            "environment_seed": smoke_seed,
            "layouts": [controls["layout_id"] for controls in smoke_conditions.values()],
        },
        "budget": {
            "episodes_per_fit": M139_SMOKE_EPISODES_PER_FIT,
            "fits": len(M139_SCREEN_TRAINING_SEEDS) * len(M139_SCREEN_ARMS),
            "total_training_episodes": M139_SMOKE_EPISODES_PER_FIT
            * len(M139_SCREEN_TRAINING_SEEDS)
            * len(M139_SCREEN_ARMS),
            "evaluation_episodes": len(M139_SCREEN_TRAINING_SEEDS)
            * len(M139_SCREEN_ARMS)
            * len(smoke_conditions),
        },
        "execution": execution,
        "potential_diagnostics": potential,
        "training": {
            str(seed): {
                arm: training[(arm, seed)]
                | {"initial_parameter_hash": policies[(arm, seed)].initial_parameter_hash}
                for arm in M139_SCREEN_ARMS
            }
            for seed in M139_SCREEN_TRAINING_SEEDS
        },
        "policy_artifacts": policy_artifacts,
        "evaluations": evaluations,
        "trace_manifest": manifest,
        "replay_pass": all(bool(row["replay_pass"]) for row in manifest),
        "elapsed_seconds": elapsed,
        "next_action": "Run the exact m139-screen only after reviewing this plumbing smoke; do not infer policy quality.",
    }
    report["content_hash"] = _report_hash(report)
    assert_compact_report(report)
    return report


def write_m139_fit_smoke_report(
    path: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.9 fit-smoke report or artifact target already exists")
    report = m139_fit_smoke(artifact_dir=artifacts, ledger_path=ledger_path)
    _write_json_exclusive(output, report)
    return report


def m139_screen(
    *,
    artifact_dir: str | Path,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
    workers: int = M139_WORKERS,
) -> dict[str, Any]:
    """Run the frozen preflight, paired fits, and one-shot screen probe."""

    if workers != M139_WORKERS:
        raise ValueError(f"M13.9 worker count is frozen at {M139_WORKERS}")
    protocol_fingerprint = m139_protocol_fingerprint()  # before touching a split
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.9 screen artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.ensure_screen_fit_open()  # smoke may have already marked this fitting-only partition
    artifacts.mkdir(parents=True)
    preflight, manifest = _preflight(artifact_dir=artifacts)
    if not bool(preflight["passes"]):
        report = {
            "schema_version": "m13.9-screen-preflight-reject-r1",
            "stage": "screen",
            "protocol_version": M139_PROTOCOL_VERSION,
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
        seeds=M139_SCREEN_TRAINING_SEEDS,
        workers=workers,
    )
    policy_artifacts, restored = _serialize_all_policies(
        policies,
        seeds=M139_SCREEN_TRAINING_SEEDS,
        policy_dir=artifacts / "policies",
    )
    # Global phase boundary: every learned final policy exists and reloads
    # before the one-shot probe is marked open or reset.
    ledger.open_partition("screen_probe")
    probe_gate, probe_manifest = _coverage_ceiling_gate(
        conditions=M139_SCREEN_PROBE_CONDITIONS,
        seeds=M139_SCREEN_PROBE_SEEDS,
        trace_dir=artifacts / "traces" / "probe_gate",
        label="balanced_public_oracle",
    )
    manifest.extend(probe_manifest)
    if not bool(probe_gate["passes"]):
        report = {
            "schema_version": "m13.9-screen-probe-mechanics-reject-r1",
            "stage": "screen",
            "protocol_version": M139_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_ledger": ledger.snapshot(),
            "preflight": preflight,
            "probe_gate": probe_gate,
            "execution": execution,
            "policy_artifacts": policy_artifacts,
            "trace_manifest": manifest,
            "screen": {"status": "reject", "passes": False, "phase": "probe_mechanics"},
            "limits": ["No learned probe score was produced.", "Confirmation and audit remain sealed."],
        }
        report["content_hash"] = _report_hash(report)
        return report
    episode_summaries: dict[str, list[dict[str, Any]]] = {
        "public_potential_candidate": [],
        "static_cost_control": [],
        "legacy_guardrail": [],
        "random": [],
    }
    aggregates: dict[str, Any] = {}
    for seed in M139_SCREEN_TRAINING_SEEDS:
        aggregates[str(seed)] = {}
        for arm in M139_SCREEN_ARMS:
            policy = restored[(arm, seed)]
            rows, arm_aggregates, traces = _evaluate_policy(
                policy,
                arm=arm,
                training_seed=seed,
                conditions=M139_SCREEN_PROBE_CONDITIONS,
                seeds=M139_SCREEN_PROBE_SEEDS,
                trace_dir=artifacts / "traces" / "probe",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m139_policy_fingerprint(policy),
                replay_mode="learned",
            )
            episode_summaries[arm].extend(rows)
            aggregates[str(seed)][arm] = arm_aggregates
            manifest.extend(traces)
        random_policy = SeededRandomM139Policy(seed)
        rows, random_aggregates, traces = _evaluate_policy(
            random_policy,
            arm="public_potential_candidate",
            training_seed=seed,
            conditions=M139_SCREEN_PROBE_CONDITIONS,
            seeds=M139_SCREEN_PROBE_SEEDS,
            trace_dir=artifacts / "traces" / "probe",
            label=f"seed-{seed}/random",
            policy_fingerprint=f"pcg64-{seed}",
            replay_mode="random",
        )
        episode_summaries["random"].extend(rows)
        aggregates[str(seed)]["random"] = random_aggregates
        manifest.extend(traces)
    promotion = m139_screen_promotes(
        episode_summaries["public_potential_candidate"],
        episode_summaries["static_cost_control"],
        episode_summaries["legacy_guardrail"],
        episode_summaries["random"],
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
        env_seeds=M139_SCREEN_PROBE_SEEDS,
    )
    training_report = {
        str(seed): {
            arm: training[(arm, seed)]
            | {"initial_parameter_hash": policies[(arm, seed)].initial_parameter_hash}
            for arm in M139_SCREEN_ARMS
        }
        for seed in M139_SCREEN_TRAINING_SEEDS
    }
    base = {
        "stage": "screen",
        "protocol_version": M139_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "split_ledger": ledger.snapshot(),
        "preflight": preflight,
        "probe_gate": probe_gate,
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
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
        env_seeds=M139_SCREEN_PROBE_SEEDS,
    )


def write_m139_screen_report(
    path: str | Path,
    *,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.9 screen report or artifact target already exists")
    report = m139_screen(artifact_dir=artifacts, ledger_path=ledger_path)
    _write_json_exclusive(output, report)
    return report


def _load_screen_report(screen_report: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str | None, str]:
    if isinstance(screen_report, dict):
        payload = screen_report
        path = None
        sha256 = m139_content_hash(payload)
    else:
        source = Path(screen_report)
        payload = json.loads(source.read_text(encoding="utf-8"))
        path = str(source.resolve())
        sha256 = _file_sha256(source)
    if not isinstance(payload, dict):
        raise TypeError("M13.9 screen report must contain a JSON object")
    return payload, path, sha256


def _verify_screen_policy_artifacts(
    report: dict[str, Any],
) -> dict[tuple[Arm, int], CompactM139QPolicy]:
    artifacts = report.get("policy_artifacts")
    expected_seeds = {str(seed) for seed in M139_SCREEN_TRAINING_SEEDS}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_seeds:
        raise ValueError("M13.9 screen report has an incomplete policy artifact matrix")
    restored: dict[tuple[Arm, int], CompactM139QPolicy] = {}
    for seed in M139_SCREEN_TRAINING_SEEDS:
        arms = artifacts[str(seed)]
        if not isinstance(arms, dict) or set(arms) != set(M139_SCREEN_ARMS):
            raise ValueError(f"M13.9 screen seed {seed} has incomplete policy artifacts")
        for arm in M139_SCREEN_ARMS:
            metadata = arms[arm]
            if not isinstance(metadata, dict):
                raise TypeError(f"M13.9 screen {arm}:{seed} artifact metadata is invalid")
            source = Path(str(metadata.get("path", "")))
            if not source.is_file() or _file_sha256(source) != metadata.get("sha256"):
                raise ValueError(f"M13.9 screen {arm}:{seed} policy file hash mismatch")
            policy = load_m139_policy(source)
            if policy.arm != arm or policy.seed != seed or policy.stage != "screen":
                raise ValueError(f"M13.9 screen {arm}:{seed} policy identity mismatch")
            if m139_policy_fingerprint(policy) != metadata.get("policy_fingerprint"):
                raise ValueError(f"M13.9 screen {arm}:{seed} policy fingerprint mismatch")
            restored[(arm, seed)] = policy
    return restored


def m139_confirm(
    *,
    screen_report: str | Path | dict[str, Any],
    artifact_dir: str | Path,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
    workers: int = M139_WORKERS,
) -> dict[str, Any]:
    """Run confirmation only after recomputing an authentic screen promotion."""

    if workers != M139_WORKERS:
        raise ValueError(f"M13.9 worker count is frozen at {M139_WORKERS}")
    protocol_fingerprint = m139_protocol_fingerprint()
    screen_payload, screen_path, screen_sha256 = _load_screen_report(screen_report)
    trusted_promotion = verify_m139_screen_report(
        screen_payload,
        expected_protocol_fingerprint=protocol_fingerprint,
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
        env_seeds=M139_SCREEN_PROBE_SEEDS,
        require_promotion=True,
    )
    restored_screen = _verify_screen_policy_artifacts(screen_payload)
    _verify_report_trace_evidence(
        screen_payload,
        learned_policies=restored_screen,
        arms=(*M139_SCREEN_ARMS, "random"),
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=M139_SCREEN_PROBE_CONDITIONS,
        env_seeds=M139_SCREEN_PROBE_SEEDS,
    )
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.9 confirmation artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    # Verification above is deliberately complete before the first 5300+ reset.
    ledger.open_partition("confirmation_fit")
    artifacts.mkdir(parents=True)
    policies, training, execution = _train_paired(
        stage="confirmation",
        seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        workers=workers,
    )
    policy_artifacts, restored = _serialize_all_policies(
        policies,
        seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        policy_dir=artifacts / "policies",
        arms=M139_CONFIRMATION_ARMS,
    )
    # All sixteen learned final policies reload cleanly before evaluation opens.
    ledger.open_partition("confirmation_evaluation")
    evaluation_gate, evaluation_gate_manifest = _coverage_ceiling_gate(
        conditions=M139_CONFIRMATION_EVALUATION_CONDITIONS,
        seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
        trace_dir=artifacts / "traces" / "evaluation_gate",
        label="balanced_public_oracle",
    )
    if not bool(evaluation_gate["passes"]):
        report = {
            "schema_version": "m13.9-confirmation-mechanics-reject-r1",
            "stage": "confirmation",
            "protocol_version": M139_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "screen_authorization": {
                "path": screen_path,
                "file_sha256": screen_sha256,
                "content_hash": screen_payload["content_hash"],
                "trusted_promotion": trusted_promotion,
            },
            "split_ledger": ledger.snapshot(),
            "execution": execution,
            "training": {
                str(seed): {arm: training[(arm, seed)] for arm in M139_CONFIRMATION_ARMS}
                for seed in M139_CONFIRMATION_TRAINING_SEEDS
            },
            "policy_artifacts": policy_artifacts,
            "evaluation_gate": evaluation_gate,
            "trace_manifest": evaluation_gate_manifest,
            "confirmation": {"status": "fail", "passes": False, "phase": "evaluation_mechanics"},
            "limits": ["No learned confirmation score was produced.", "Audit remains sealed."],
        }
        report["content_hash"] = _report_hash(report)
        return report
    episode_summaries: dict[str, list[dict[str, Any]]] = {
        "public_potential_candidate": [],
        "static_cost_control": [],
        "random": [],
    }
    aggregates: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = list(evaluation_gate_manifest)
    for seed in M139_CONFIRMATION_TRAINING_SEEDS:
        aggregates[str(seed)] = {}
        for arm in M139_CONFIRMATION_ARMS:
            policy = restored[(arm, seed)]
            rows, arm_aggregates, traces = _evaluate_policy(
                policy,
                arm=arm,
                training_seed=seed,
                conditions=M139_CONFIRMATION_EVALUATION_CONDITIONS,
                seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
                trace_dir=artifacts / "traces" / "evaluation",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m139_policy_fingerprint(policy),
                replay_mode="learned",
            )
            episode_summaries[arm].extend(rows)
            aggregates[str(seed)][arm] = arm_aggregates
            manifest.extend(traces)
        random_policy = SeededRandomM139Policy(seed)
        rows, random_aggregates, traces = _evaluate_policy(
            random_policy,
            arm="public_potential_candidate",
            training_seed=seed,
            conditions=M139_CONFIRMATION_EVALUATION_CONDITIONS,
            seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
            trace_dir=artifacts / "traces" / "evaluation",
            label=f"seed-{seed}/random",
            policy_fingerprint=f"pcg64-{seed}",
            replay_mode="random",
        )
        episode_summaries["random"].extend(rows)
        aggregates[str(seed)]["random"] = random_aggregates
        manifest.extend(traces)
    gate = m139_confirmation_passes(
        episode_summaries["public_potential_candidate"],
        episode_summaries["static_cost_control"],
        episode_summaries["random"],
        training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M139_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
    )
    training_report = {
        str(seed): {
            arm: training[(arm, seed)]
            | {"initial_parameter_hash": policies[(arm, seed)].initial_parameter_hash}
            for arm in M139_CONFIRMATION_ARMS
        }
        for seed in M139_CONFIRMATION_TRAINING_SEEDS
    }
    report: dict[str, Any] = {
        "schema_version": "m13.9-confirmation-report-r1",
        "stage": "confirmation",
        "protocol_version": M139_PROTOCOL_VERSION,
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
        "evaluation_gate": evaluation_gate,
        "episode_summaries": episode_summaries,
        "aggregates": aggregates,
        "trace_manifest": manifest,
        "confirmation_gate": gate,
        "confirmation": {
            "status": (
                "robust_and_causal"
                if bool(gate["passes"])
                else "robust_noncausal"
                if bool(gate["hard_gate_pass"])
                else "fail"
            ),
            "passes": bool(gate["passes"]),
            "hard_gate_pass": bool(gate["hard_gate_pass"]),
            "paired_margin_pass": int(gate["paired_margin_wins"]) >= 7,
            "robust_baseline_pass": bool(gate["robust_baseline_pass"]),
            "potential_improvement_supported": bool(gate["passes"]),
            "paired_margin_wins": int(gate["paired_margin_wins"]),
        },
        "limits": [
            "The audit split remains sealed and is not scored here.",
            "A causal reward-improvement claim requires both the 8/8 hard gate and 7/8 paired-margin test.",
            "M13.9 is not causally comparable to historical per-decision-discount scores.",
        ],
    }
    report["content_hash"] = _report_hash(report)
    assert_compact_report(report)
    return report


def write_m139_confirmation_report(
    path: str | Path,
    *,
    screen_report: str | Path,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.9 confirmation report or artifact target already exists")
    report = m139_confirm(
        screen_report=screen_report,
        artifact_dir=artifacts,
        ledger_path=ledger_path,
    )
    _write_json_exclusive(output, report)
    return report


def _verify_confirmation_policy_artifacts(
    report: dict[str, Any],
) -> dict[tuple[Arm, int], CompactM139QPolicy]:
    artifacts = report.get("policy_artifacts")
    expected_seeds = {str(seed) for seed in M139_CONFIRMATION_TRAINING_SEEDS}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_seeds:
        raise ValueError("M13.9 confirmation report has an incomplete policy artifact matrix")
    restored: dict[tuple[Arm, int], CompactM139QPolicy] = {}
    for seed in M139_CONFIRMATION_TRAINING_SEEDS:
        arms = artifacts[str(seed)]
        if not isinstance(arms, dict) or set(arms) != set(M139_CONFIRMATION_ARMS):
            raise ValueError(f"M13.9 confirmation seed {seed} has incomplete policy artifacts")
        for arm in M139_CONFIRMATION_ARMS:
            metadata = arms[arm]
            if not isinstance(metadata, dict):
                raise TypeError(f"M13.9 confirmation {arm}:{seed} metadata is invalid")
            source = Path(str(metadata.get("path", "")))
            if not source.is_file() or _file_sha256(source) != metadata.get("sha256"):
                raise ValueError(f"M13.9 confirmation {arm}:{seed} policy file hash mismatch")
            policy = load_m139_policy(source)
            if policy.arm != arm or policy.seed != seed or policy.stage != "confirmation":
                raise ValueError(f"M13.9 confirmation {arm}:{seed} policy identity mismatch")
            if m139_policy_fingerprint(policy) != metadata.get("policy_fingerprint"):
                raise ValueError(f"M13.9 confirmation {arm}:{seed} fingerprint mismatch")
            restored[(arm, seed)] = policy
    return restored


def _verify_confirmation_screen_authorization(report: dict[str, Any]) -> None:
    authorization = report.get("screen_authorization")
    if not isinstance(authorization, dict):
        raise TypeError("M13.9 confirmation lacks screen authorization")
    source_value = authorization.get("path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("M13.9 audit requires a durable originating screen report")
    source = Path(source_value)
    if not source.is_file() or _file_sha256(source) != authorization.get("file_sha256"):
        raise ValueError("M13.9 originating screen report file hash mismatch")
    screen = json.loads(source.read_text(encoding="utf-8"))
    if screen.get("content_hash") != authorization.get("content_hash"):
        raise ValueError("M13.9 originating screen content hash mismatch")
    trusted = verify_m139_screen_report(
        screen,
        expected_protocol_fingerprint=m139_protocol_fingerprint(),
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M139_SCREEN_PROBE_CONDITIONS),
        env_seeds=M139_SCREEN_PROBE_SEEDS,
        require_promotion=True,
    )
    if authorization.get("trusted_promotion") != trusted:
        raise ValueError("M13.9 stored screen authorization disagrees with the source report")
    restored = _verify_screen_policy_artifacts(screen)
    _verify_report_trace_evidence(
        screen,
        learned_policies=restored,
        arms=(*M139_SCREEN_ARMS, "random"),
        training_seeds=M139_SCREEN_TRAINING_SEEDS,
        conditions=M139_SCREEN_PROBE_CONDITIONS,
        env_seeds=M139_SCREEN_PROBE_SEEDS,
    )


def _verify_confirmation_report(
    confirmation_report: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], str | None, str, dict[tuple[Arm, int], CompactM139QPolicy]]:
    payload, path, sha256 = _load_screen_report(confirmation_report)
    if (
        payload.get("schema_version") != "m13.9-confirmation-report-r1"
        or payload.get("stage") != "confirmation"
    ):
        raise ValueError("M13.9 audit requires the frozen confirmation report schema")
    if payload.get("protocol_fingerprint") != m139_protocol_fingerprint():
        raise ValueError("M13.9 confirmation protocol fingerprint mismatch")
    content_hash = payload.get("content_hash")
    if not isinstance(content_hash, str) or content_hash != _report_hash(payload):
        raise ValueError("M13.9 confirmation report content hash mismatch")
    _verify_confirmation_screen_authorization(payload)
    evidence = payload.get("episode_summaries")
    if not isinstance(evidence, dict) or set(evidence) != {
        "public_potential_candidate",
        "static_cost_control",
        "random",
    }:
        raise ValueError("M13.9 confirmation evidence matrix is incomplete")
    trusted = m139_confirmation_passes(
        evidence["public_potential_candidate"],
        evidence["static_cost_control"],
        evidence["random"],
        training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        conditions=tuple(M139_CONFIRMATION_EVALUATION_CONDITIONS),
        env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
    )
    if not bool(trusted["passes"]) or payload.get("confirmation_gate") != trusted:
        raise ValueError("M13.9 confirmation report does not authorize audit")
    restored = _verify_confirmation_policy_artifacts(payload)
    _verify_report_trace_evidence(
        payload,
        learned_policies=restored,
        arms=(*M139_CONFIRMATION_ARMS, "random"),
        training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
        conditions=M139_CONFIRMATION_EVALUATION_CONDITIONS,
        env_seeds=M139_CONFIRMATION_EVALUATION_SEEDS,
    )
    return payload, path, sha256, restored


def m139_audit(
    *,
    confirmation_report: str | Path | dict[str, Any],
    artifact_dir: str | Path,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    """Open and score the sealed audit once, without fitting any policy."""

    protocol_fingerprint = m139_protocol_fingerprint()
    confirmation, confirmation_path, confirmation_sha256, restored = _verify_confirmation_report(
        confirmation_report
    )
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.9 audit artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.open_partition("audit")
    artifacts.mkdir(parents=True)
    audit_gate, manifest = _coverage_ceiling_gate(
        conditions=M139_AUDIT_CONDITIONS,
        seeds=M139_AUDIT_SEEDS,
        trace_dir=artifacts / "traces" / "audit_gate",
        label="balanced_public_oracle",
    )
    episode_summaries: dict[str, list[dict[str, Any]]] = {
        "public_potential_candidate": [],
        "static_cost_control": [],
        "random": [],
    }
    aggregates: dict[str, Any] = {}
    if bool(audit_gate["passes"]):
        for seed in M139_CONFIRMATION_TRAINING_SEEDS:
            aggregates[str(seed)] = {}
            for arm in M139_CONFIRMATION_ARMS:
                policy = restored[(arm, seed)]
                rows, arm_aggregates, traces = _evaluate_policy(
                    policy,
                    arm=arm,
                    training_seed=seed,
                    conditions=M139_AUDIT_CONDITIONS,
                    seeds=M139_AUDIT_SEEDS,
                    trace_dir=artifacts / "traces" / "audit",
                    label=f"seed-{seed}/{arm}",
                    policy_fingerprint=m139_policy_fingerprint(policy),
                    replay_mode="learned",
                )
                episode_summaries[arm].extend(rows)
                aggregates[str(seed)][arm] = arm_aggregates
                manifest.extend(traces)
            random_policy = SeededRandomM139Policy(seed)
            rows, random_aggregates, traces = _evaluate_policy(
                random_policy,
                arm="public_potential_candidate",
                training_seed=seed,
                conditions=M139_AUDIT_CONDITIONS,
                seeds=M139_AUDIT_SEEDS,
                trace_dir=artifacts / "traces" / "audit",
                label=f"seed-{seed}/random",
                policy_fingerprint=f"pcg64-{seed}",
                replay_mode="random",
            )
            episode_summaries["random"].extend(rows)
            aggregates[str(seed)]["random"] = random_aggregates
            manifest.extend(traces)
        gate = m139_confirmation_passes(
            episode_summaries["public_potential_candidate"],
            episode_summaries["static_cost_control"],
            episode_summaries["random"],
            training_seeds=M139_CONFIRMATION_TRAINING_SEEDS,
            conditions=tuple(M139_AUDIT_CONDITIONS),
            env_seeds=M139_AUDIT_SEEDS,
        )
    else:
        gate = {"passes": False, "phase": "audit_mechanics"}
    report: dict[str, Any] = {
        "schema_version": "m13.9-audit-report-r1",
        "stage": "audit",
        "protocol_version": M139_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "confirmation_authorization": {
            "path": confirmation_path,
            "file_sha256": confirmation_sha256,
            "content_hash": confirmation["content_hash"],
        },
        "split_ledger": ledger.snapshot(),
        "audit_gate": audit_gate,
        "episode_summaries": episode_summaries,
        "aggregates": aggregates,
        "trace_manifest": manifest,
        "audit_result": {
            "status": "pass" if bool(gate["passes"]) else "fail",
            "passes": bool(gate["passes"]),
        },
        "audit_confirmation_gate": gate,
        "limits": ["No audit retry, replacement seed, coefficient change, or retraining is authorized."],
    }
    report["content_hash"] = _report_hash(report)
    assert_compact_report(report)
    return report


def write_m139_audit_report(
    path: str | Path,
    *,
    confirmation_report: str | Path,
    ledger_path: str | Path = M139_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.9 audit report or artifact target already exists")
    report = m139_audit(
        confirmation_report=confirmation_report,
        artifact_dir=artifacts,
        ledger_path=ledger_path,
    )
    _write_json_exclusive(output, report)
    return report
