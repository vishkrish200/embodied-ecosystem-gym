"""M13 reward-led state-oracle RL baseline for persistent maintenance.

The policy boundary in this module is deliberately narrow: actions are chosen
from observations and serializable policy-owned memory only.  Evaluation may
inspect ``info`` for metrics, but no policy method accepts an environment or an
``info`` mapping.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .actions import ActionKind, ActionOutcome
from .config import EcosystemConfig
from .env import EcosystemEnv
from .m10 import m10_config, m10_scan_coverage
from .m8 import wilson_interval
from .policies import skill_action


M13_PROTOCOL_VERSION = "m13-state-oracle-persistent-rl-r1"
M13_TASK_ID = "persistent_maintenance"
M13_DEVELOPMENT_SEEDS = tuple(range(2_000, 2_040))
M13_VALIDATION_SEEDS = tuple(range(2_100, 2_120))
M13_AUDIT_SEEDS = tuple(range(2_200, 2_220))
M13_TRAINING_EPISODES = 12_000
M13_TRAINING_SEED = 20_260_806
M13_ALPHA = 0.20
M13_GAMMA = 0.99
M13_EPSILON_FINAL = 0.05
M13_EPSILON_DECAY_EPISODES = 9_000
M13_SAFE_DRIVE_BANDS = {"satiety_min": 0.15, "energy_min": 0.15, "boredom_max": 0.90}
M13_VECTOR_EDGES = (-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8)
M13_DRIVE_EDGES = (0.0, 0.15, 0.45, 0.60, 0.90, 1.0)


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


M13_DEVELOPMENT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": _controls("m13_dev_northeast", food_variant="orange", toy_variant="ball", initial_scan_sector="north"),
    "renewal_and_morphology": _controls("m13_dev_southwest", food_variant="purple", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="capsule", initial_scan_sector="south"),
    "event_relocation": _controls("m13_dev_northwest", food_variant="blue", food_shape_variant="box", toy_variant="capsule", lighting_variant="dim", initial_scan_sector="east", event_relocation_on_first_pickup=True),
    "compound": _controls("m13_dev_southeast", food_variant="red", food_shape_variant="capsule", toy_variant="cube", agent_shape_variant="box", dynamics_variant="grippy", blocked_distractor=True, distractor_xy=[-0.04, 0.04], initial_scan_sector="west", event_relocation_on_first_pickup=True),
}
M13_VALIDATION_CONDITIONS: dict[str, dict[str, Any]] = {
    name: {**controls, "layout_id": f"m13_validation_{controls['layout_id'].removeprefix('m13_dev_')}"}
    for name, controls in M13_DEVELOPMENT_CONDITIONS.items()
}
M13_AUDIT_CONDITIONS: dict[str, dict[str, Any]] = {
    name: {**controls, "layout_id": f"m13_audit_{controls['layout_id'].removeprefix('m13_dev_')}"}
    for name, controls in M13_DEVELOPMENT_CONDITIONS.items()
}


def m13_config() -> EcosystemConfig:
    """The unchanged M10 persistent reward/mechanics configuration."""

    return m10_config(observation_mode="state_oracle")


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M13_TASK_ID, **controls}


class M13Macro(IntEnum):
    GO_FOOD = 0
    PICK_UP = 1
    CONSUME = 2
    GO_TOY = 3
    PLAY = 4
    GO_REST = 5
    REST = 6
    WAIT = 7


@dataclass(slots=True)
class M13Memory:
    feed_count: int = 0
    play_count: int = 0
    rest_count: int = 0
    food_cooldown_bucket: int = 5
    last_macro: M13Macro | None = None
    pending_recovery: bool = False


def _outcome(observation: dict[str, Any]) -> ActionOutcome:
    return list(ActionOutcome)[int(observation["prior_outcome"])]


def _duration(action: dict[str, np.ndarray | int]) -> float:
    return float(np.asarray(action["duration"], dtype=np.float32).item())


def _memory_snapshot(memory: M13Memory) -> dict[str, object]:
    return {
        "feed_count": memory.feed_count,
        "play_count": memory.play_count,
        "rest_count": memory.rest_count,
        "food_cooldown_bucket": memory.food_cooldown_bucket,
        "last_macro": memory.last_macro.name if memory.last_macro is not None else "START",
        "pending_recovery": memory.pending_recovery,
    }


def _clear_memory(memory: M13Memory) -> None:
    memory.feed_count = 0
    memory.play_count = 0
    memory.rest_count = 0
    memory.food_cooldown_bucket = 5
    memory.last_macro = None
    memory.pending_recovery = False


def advance_memory(
    memory: M13Memory,
    *,
    observation_before: dict[str, Any],
    macro: M13Macro,
    action: dict[str, np.ndarray | int],
    observation_after: dict[str, Any],
) -> None:
    """Update only from policy-visible observations plus the chosen action."""

    result = _outcome(observation_after)
    drives_before = np.asarray(observation_before["drives"], dtype=np.float32)
    if macro is M13Macro.CONSUME and result is ActionOutcome.SUCCESS:
        memory.feed_count = min(3, memory.feed_count + 1)
        memory.food_cooldown_bucket = 0
        memory.pending_recovery = False
    elif memory.food_cooldown_bucket < 5:
        memory.food_cooldown_bucket = min(5, memory.food_cooldown_bucket + int(np.ceil(_duration(action))))
    if macro is M13Macro.PLAY and result is ActionOutcome.SUCCESS and float(drives_before[2]) >= 0.60:
        memory.play_count = min(3, memory.play_count + 1)
    if macro is M13Macro.REST and result is ActionOutcome.SUCCESS and float(drives_before[1]) <= 0.45:
        memory.rest_count = min(2, memory.rest_count + 1)
    if macro is M13Macro.PICK_UP and result is ActionOutcome.BLOCKED:
        memory.pending_recovery = True
    memory.last_macro = macro


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    clipped = float(np.clip(value, edges[0], edges[-1]))
    return min(len(edges) - 2, max(0, int(np.searchsorted(edges, clipped, side="right") - 1)))


M13State = tuple[int, ...]


def encode_state(observation: dict[str, Any], memory: M13Memory, *, include_drives: bool = True) -> M13State:
    """Fixed M13 encoder; no task/reset/info/private values can enter here."""

    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    target_buckets = tuple(
        _bucket(float(component), M13_VECTOR_EDGES)
        for target_key in ("food_xy", "toy_xy", "rest_xy")
        for component in (np.asarray(observation[target_key], dtype=np.float32) - agent)
    )
    drives = np.asarray(observation["drives"], dtype=np.float32)
    drive_buckets = tuple(_bucket(float(value), M13_DRIVE_EDGES) for value in drives) if include_drives else (0, 0, 0)
    return (
        *target_buckets,
        *drive_buckets,
        int(observation["holding_food"]),
        int(observation["prior_outcome"]),
        memory.feed_count,
        memory.play_count,
        memory.rest_count,
        memory.food_cooldown_bucket,
        len(M13Macro) if memory.last_macro is None else int(memory.last_macro),
        int(memory.pending_recovery),
    )


def compile_macro(macro: M13Macro, observation: dict[str, Any], config: EcosystemConfig) -> dict[str, np.ndarray | int]:
    """Map the selected discrete macro to one existing guarded typed skill."""

    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    target_by_macro = {
        M13Macro.GO_FOOD: "food_xy",
        M13Macro.GO_TOY: "toy_xy",
        M13Macro.GO_REST: "rest_xy",
    }
    if macro in target_by_macro:
        target = np.asarray(observation[target_by_macro[macro]], dtype=np.float32)
        duration = max(0.1, float(np.linalg.norm(target - agent)) / config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_TO, target, duration)
    action_by_macro = {
        M13Macro.PICK_UP: ActionKind.PICK_UP,
        M13Macro.CONSUME: ActionKind.CONSUME,
        M13Macro.PLAY: ActionKind.RUN_AROUND,
        M13Macro.REST: ActionKind.REST,
        M13Macro.WAIT: ActionKind.IDLE,
    }
    duration = 0.1 if macro in {M13Macro.PICK_UP, M13Macro.CONSUME} else 1.0
    return skill_action(action_by_macro[macro], np.zeros(2, dtype=np.float32), duration)


class M13Policy(Protocol):
    include_drives: bool
    use_memory: bool

    def reset(self) -> M13Memory: ...

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro: ...

    def observe(
        self,
        memory: M13Memory,
        *,
        observation_before: dict[str, Any],
        macro: M13Macro,
        action: dict[str, np.ndarray | int],
        observation_after: dict[str, Any],
    ) -> None: ...


class TabularM13QPolicy:
    """Reward-led tabular Q policy with no environment handle or teacher data."""

    def __init__(self, *, include_drives: bool = True, use_memory: bool = True) -> None:
        self.include_drives = include_drives
        self.use_memory = use_memory
        self.q_values: dict[M13State, np.ndarray] = {}

    def reset(self) -> M13Memory:
        return M13Memory()

    def _state(self, observation: dict[str, Any], memory: M13Memory) -> M13State:
        active_memory = memory if self.use_memory else M13Memory()
        return encode_state(observation, active_memory, include_drives=self.include_drives)

    def _q(self, state: M13State) -> np.ndarray:
        return self.q_values.setdefault(state, np.zeros(len(M13Macro), dtype=np.float64))

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        return M13Macro(int(np.argmax(self._q(self._state(observation, memory)))))

    def observe(
        self,
        memory: M13Memory,
        *,
        observation_before: dict[str, Any],
        macro: M13Macro,
        action: dict[str, np.ndarray | int],
        observation_after: dict[str, Any],
    ) -> None:
        advance_memory(
            memory,
            observation_before=observation_before,
            macro=macro,
            action=action,
            observation_after=observation_after,
        )
        if not self.use_memory:
            _clear_memory(memory)

    def train(self, *, episodes: int = M13_TRAINING_EPISODES, seed: int = M13_TRAINING_SEED) -> None:
        if episodes != M13_TRAINING_EPISODES:
            raise ValueError("M13 training budget is frozen; use M13_TRAINING_EPISODES")
        rng = np.random.default_rng(seed)
        condition_items = tuple(M13_DEVELOPMENT_CONDITIONS.items())
        env = EcosystemEnv(m13_config())
        try:
            for episode in range(episodes):
                condition, controls = condition_items[episode % len(condition_items)]
                del condition
                episode_seed = M13_DEVELOPMENT_SEEDS[(episode // len(condition_items)) % len(M13_DEVELOPMENT_SEEDS)]
                observation, _ = env.reset(seed=episode_seed, options=_options(controls))
                memory = self.reset()
                epsilon = max(M13_EPSILON_FINAL, 1.0 - episode / M13_EPSILON_DECAY_EPISODES)
                for _ in range(env.config.max_episode_steps):
                    state = self._state(observation, memory)
                    macro = M13Macro(int(rng.integers(len(M13Macro)))) if rng.random() < epsilon else self.choose(observation, memory)
                    action = compile_macro(macro, observation, env.config)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    self.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
                    next_state = self._state(next_observation, memory)
                    target = reward if terminated or truncated else reward + M13_GAMMA * float(np.max(self._q(next_state)))
                    self._q(state)[int(macro)] += M13_ALPHA * (target - self._q(state)[int(macro)])
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()


class RandomM13Policy:
    include_drives = True
    use_memory = False

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> M13Memory:
        return M13Memory()

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        del observation, memory
        return M13Macro(int(self.rng.integers(len(M13Macro))))

    def observe(self, memory: M13Memory, **_: Any) -> None:
        _clear_memory(memory)


class ScriptedM13Oracle:
    """Public-state mechanics ceiling, never a learned or equal-input result."""

    include_drives = True
    use_memory = True

    def reset(self) -> M13Memory:
        return M13Memory()

    def choose(self, observation: dict[str, Any], memory: M13Memory) -> M13Macro:
        drives = np.asarray(observation["drives"], dtype=np.float32)
        agent = np.asarray(observation["agent_xy"], dtype=np.float32)
        if bool(observation["holding_food"]):
            return M13Macro.CONSUME
        if float(drives[1]) <= 0.45:
            return M13Macro.REST if float(np.linalg.norm(np.asarray(observation["rest_xy"]) - agent)) <= m13_config().rest_interaction_radius else M13Macro.GO_REST
        if memory.food_cooldown_bucket >= 5 and float(drives[0]) <= 0.55:
            return M13Macro.PICK_UP if float(np.linalg.norm(np.asarray(observation["food_xy"]) - agent)) <= m13_config().pickup_radius else M13Macro.GO_FOOD
        if float(drives[2]) >= 0.60:
            return M13Macro.PLAY if float(np.linalg.norm(np.asarray(observation["toy_xy"]) - agent)) <= m13_config().toy_interaction_radius else M13Macro.GO_TOY
        return M13Macro.WAIT

    def observe(self, memory: M13Memory, **kwargs: Any) -> None:
        advance_memory(memory, **kwargs)


def m13_policy_fingerprint(policy: TabularM13QPolicy) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"include_drives": policy.include_drives, "use_memory": policy.use_memory}, sort_keys=True).encode())
    for state, values in sorted(policy.q_values.items()):
        digest.update(np.asarray(state, dtype=np.int16).tobytes())
        digest.update(np.asarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def m13_protocol_fingerprint() -> str:
    payload = {
        "version": M13_PROTOCOL_VERSION,
        "config": asdict(m13_config()),
        "development_seeds": M13_DEVELOPMENT_SEEDS,
        "validation_seeds": M13_VALIDATION_SEEDS,
        "audit_seeds": M13_AUDIT_SEEDS,
        "development_conditions": M13_DEVELOPMENT_CONDITIONS,
        "validation_conditions": M13_VALIDATION_CONDITIONS,
        "audit_conditions": M13_AUDIT_CONDITIONS,
        "vector_edges": M13_VECTOR_EDGES,
        "drive_edges": M13_DRIVE_EDGES,
        "training": {"episodes": M13_TRAINING_EPISODES, "seed": M13_TRAINING_SEED, "alpha": M13_ALPHA, "gamma": M13_GAMMA, "epsilon_final": M13_EPSILON_FINAL, "epsilon_decay_episodes": M13_EPSILON_DECAY_EPISODES},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe(drives: np.ndarray) -> bool:
    return bool(drives[0] > 0.15 and drives[1] > 0.15 and drives[2] < 0.90)


@dataclass(frozen=True, slots=True)
class M13Episode:
    seed: int
    condition: str
    survived: bool
    maintenance_complete: bool
    terminal_cause: str
    feed_cycles: int
    play_cycles: int
    rest_cycles: int
    safe_drive_fraction: float
    forced_recovery: dict[str, object]
    interventions: dict[str, int]
    steps: tuple[dict[str, object], ...]


def run_m13_episode(
    policy: M13Policy, *, seed: int, condition: str, controls: dict[str, Any], env: EcosystemEnv | None = None
) -> M13Episode:
    owns_env = env is None
    if env is None:
        env = EcosystemEnv(m13_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        records: list[dict[str, object]] = []
        safe_steps = 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        interventions: Counter[str] = Counter()
        for step in range(1, env.config.max_episode_steps + 1):
            memory_before = _memory_snapshot(memory)
            encoded_state = encode_state(observation, memory if policy.use_memory else M13Memory(), include_drives=policy.include_drives)
            macro = policy.choose(observation, memory)
            action = compile_macro(macro, observation, env.config)
            next_observation, reward, terminated, truncated, info = env.step(action)
            policy.observe(memory, observation_before=observation, macro=macro, action=action, observation_after=next_observation)
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen = relocation_seen or relocation_now
            stale_pickup = stale_pickup or (relocation_now and info["outcome"] == ActionOutcome.BLOCKED.value)
            consumed_after_relocation = consumed_after_relocation or (relocation_seen and macro is M13Macro.CONSUME and info["outcome"] == ActionOutcome.SUCCESS.value)
            if relocation_now:
                interventions["food_relocated"] += 1
            if info["resource_event"] is not None:
                interventions[str(info["resource_event"])] += 1
            if info["outcome"] == ActionOutcome.BLOCKED.value:
                interventions["blocked_action"] += 1
            safe_steps += int(_safe(np.asarray(next_observation["drives"], dtype=np.float32)))
            records.append({
                "step": step,
                "policy_observation": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in observation.items()},
                "encoded_state": list(encoded_state),
                "memory_before": memory_before,
                "memory_after": _memory_snapshot(memory),
                "macro": macro.name,
                "action": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in action.items()},
                "reward": float(reward),
                "outcome": info["outcome"],
                "disturbance": info["disturbance"],
                "resource_event": info["resource_event"],
                "feed_cycles": int(info["feed_cycles"]),
                "play_cycles": int(info["play_cycles"]),
                "rest_cycles": int(info["rest_cycles"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            observation = next_observation
            if terminated or truncated:
                drives = np.asarray(observation["drives"], dtype=np.float32)
                terminal_cause = "survived_horizon" if truncated and bool(info["survived"]) else "energy_depleted" if drives[1] <= 0.0 else "satiety_depleted" if drives[0] <= 0.0 else "terminated"
                return M13Episode(
                    seed=seed, condition=condition, survived=bool(info["survived"]), maintenance_complete=bool(info["maintenance_complete"]), terminal_cause=terminal_cause,
                    feed_cycles=int(info["feed_cycles"]), play_cycles=int(info["play_cycles"]), rest_cycles=int(info["rest_cycles"]), safe_drive_fraction=safe_steps / step,
                    forced_recovery={"required": bool(controls.get("event_relocation_on_first_pickup")), "stale_pickup": stale_pickup, "consumed_after_relocation": consumed_after_relocation, "complete": stale_pickup and consumed_after_relocation},
                    interventions=dict(interventions), steps=tuple(records),
                )
        raise AssertionError("M13 episode did not terminate")
    finally:
        if owns_env:
            env.close()


def _aggregate(episodes: list[M13Episode]) -> dict[str, object]:
    required = [item for item in episodes if bool(item.forced_recovery["required"])]
    return {
        "episodes": len(episodes),
        "survivals": sum(item.survived for item in episodes),
        "survival_rate": float(np.mean([item.survived for item in episodes])),
        "survival_wilson_95": list(wilson_interval(sum(item.survived for item in episodes), len(episodes))),
        "completed_maintenance_episodes": sum(item.maintenance_complete for item in episodes),
        "maintenance_wilson_95": list(wilson_interval(sum(item.maintenance_complete for item in episodes), len(episodes))),
        "completed_cycles": {"minimum_feed": min(item.feed_cycles for item in episodes), "minimum_play": min(item.play_cycles for item in episodes), "minimum_rest": min(item.rest_cycles for item in episodes), "mean_feed": float(np.mean([item.feed_cycles for item in episodes])), "mean_play": float(np.mean([item.play_cycles for item in episodes])), "mean_rest": float(np.mean([item.rest_cycles for item in episodes]))},
        "mean_time_inside_safe_drive_bands": float(np.mean([item.safe_drive_fraction for item in episodes])),
        "forced_recovery_chains": {"required_episodes": len(required), "stale_pickups": sum(bool(item.forced_recovery["stale_pickup"]) for item in required), "completed": sum(bool(item.forced_recovery["complete"]) for item in required)},
        "terminal_causes": dict(sorted(Counter(item.terminal_cause for item in episodes).items())),
        "episodes_detail": [asdict(item) for item in episodes],
    }


def _evaluate(policy: M13Policy, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    env = EcosystemEnv(m13_config())
    try:
        return {name: _aggregate([run_m13_episode(policy, seed=seed, condition=name, controls=controls, env=env) for seed in seeds]) for name, controls in conditions.items()}
    finally:
        env.close()


def _condition_gates(results: dict[str, dict[str, object]]) -> dict[str, dict[str, bool]]:
    gates: dict[str, dict[str, bool]] = {}
    for name, row in results.items():
        recovery_required = bool(M13_VALIDATION_CONDITIONS[name].get("event_relocation_on_first_pickup"))
        gates[name] = {
            "survival_pass": int(row["survivals"]) >= 18,
            "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 18,
            "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= 0.85,
            "recovery_pass": (not recovery_required) or int(row["forced_recovery_chains"]["completed"]) >= 18,  # type: ignore[index]
        }
    return gates


def _ceiling_passes(results: dict[str, dict[str, object]], *, seeds: tuple[int, ...]) -> bool:
    return all(
        int(row["survivals"]) == len(seeds)
        and int(row["completed_maintenance_episodes"]) == len(seeds)
        and int(row["completed_cycles"]["minimum_feed"]) >= 3  # type: ignore[index]
        and int(row["completed_cycles"]["minimum_play"]) >= 3  # type: ignore[index]
        and int(row["completed_cycles"]["minimum_rest"]) >= 2  # type: ignore[index]
        for row in results.values()
    )


def _flatten(results: dict[str, dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(bool(episode[field])) for condition in M13_VALIDATION_CONDITIONS for episode in results[condition]["episodes_detail"]], dtype=np.float64)  # type: ignore[index]


def m13_validation(*, training_episodes: int = M13_TRAINING_EPISODES, seeds: tuple[int, ...] = M13_VALIDATION_SEEDS) -> dict[str, object]:
    if training_episodes != M13_TRAINING_EPISODES or seeds != M13_VALIDATION_SEEDS:
        raise ValueError("M13 validation budget and seeds are frozen")
    coverage = m10_scan_coverage(M13_VALIDATION_CONDITIONS, seeds=seeds)
    if not all(bool(row["passes"]) for row in coverage.values()):
        raise RuntimeError("M13 validation has unobservable public targets")
    ceiling = _evaluate(ScriptedM13Oracle(), M13_VALIDATION_CONDITIONS, seeds)
    if not _ceiling_passes(ceiling, seeds=seeds):
        raise RuntimeError("M13 persistent-maintenance mechanics ceiling failed")
    full = TabularM13QPolicy()
    no_drive = TabularM13QPolicy(include_drives=False)
    no_memory = TabularM13QPolicy(use_memory=False)
    full.train()
    no_drive.train(seed=M13_TRAINING_SEED + 1)
    no_memory.train(seed=M13_TRAINING_SEED + 2)
    results = {"full_state_oracle_q": _evaluate(full, M13_VALIDATION_CONDITIONS, seeds), "no_drive_q": _evaluate(no_drive, M13_VALIDATION_CONDITIONS, seeds), "no_memory_q": _evaluate(no_memory, M13_VALIDATION_CONDITIONS, seeds), "macro_random": _evaluate(RandomM13Policy(M13_TRAINING_SEED + 3), M13_VALIDATION_CONDITIONS, seeds)}
    gates = _condition_gates(results["full_state_oracle_q"])
    full_maintenance = _flatten(results["full_state_oracle_q"], "maintenance_complete")
    full_survival = _flatten(results["full_state_oracle_q"], "survived")
    comparisons = {name: {"maintenance_delta": float(np.mean(full_maintenance - _flatten(rows, "maintenance_complete"))), "survival_delta": float(np.mean(full_survival - _flatten(rows, "survived")))} for name, rows in results.items() if name != "full_state_oracle_q"}
    passes = all(all(row.values()) for row in gates.values()) and all(float(comparisons[name]["maintenance_delta"]) >= 0.10 for name in comparisons) and float(comparisons["macro_random"]["survival_delta"]) >= 0.20
    return {"schema_version": "0.13", "protocol_version": M13_PROTOCOL_VERSION, "protocol_fingerprint": m13_protocol_fingerprint(), "policy_boundary": ["agent_xy", "food_xy", "toy_xy", "rest_xy", "drives", "holding_food", "prior_outcome", "policy_owned_memory"], "training": {"episodes": training_episodes, "seed": M13_TRAINING_SEED, "full_policy_fingerprint": m13_policy_fingerprint(full), "no_drive_fingerprint": m13_policy_fingerprint(no_drive), "no_memory_fingerprint": m13_policy_fingerprint(no_memory)}, "coverage": coverage, "state_oracle_scripted_ceiling": ceiling, "results": results, "condition_gates": gates, "comparisons": comparisons, "gate": {"passes": passes}, "limits": ["M13 is a reward-trained state-oracle macro baseline, not an RGB result.", "The policy never receives task IDs, reset options, info, reward decomposition, private environment state, or M12 labels.", "This command is validation only; a sealed audit requires a separately frozen policy artifact and is intentionally not opened here."]}


def write_m13_report(path: str | Path) -> dict[str, object]:
    report = m13_validation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
