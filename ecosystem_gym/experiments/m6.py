"""M6 learned state-oracle drive arbitration over the existing M4 skill API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..config import EcosystemConfig
from ..env import EcosystemEnv
from .m4 import HELDOUT_CONDITIONS, M4_EVALUATION_SEEDS, TRAIN_CONDITIONS, DriveAwareOraclePolicy
from ..policies import skill_action
from ..tasks import get_task
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


class DriveMacroAction(IntEnum):
    PURSUE_FOOD = 0
    PURSUE_PLAY = 1
    IDLE = 2


DriveState = tuple[bool, bool, bool, int, int, int, bool, bool]


def _drive_bin(value: float) -> int:
    return int(np.digitize(value, (0.30, 0.60)))


def observable_drive_state(
    observation: dict[str, Any], config: EcosystemConfig, *, played_toy: bool, include_drives: bool = True
) -> DriveState:
    """The complete learned-policy input boundary: public observation + local memory only."""

    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    food = np.asarray(observation["food_xy"], dtype=np.float32)
    toy = np.asarray(observation["toy_xy"], dtype=np.float32)
    drives = np.asarray(observation["drives"], dtype=np.float32)
    drive_bins = tuple(_drive_bin(float(value)) for value in drives) if include_drives else (0, 0, 0)
    blocked = int(observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.BLOCKED)
    return (
        bool(observation["holding_food"]),
        float(np.linalg.norm(food - agent)) <= config.pickup_radius,
        float(np.linalg.norm(toy - agent)) <= config.toy_interaction_radius,
        *drive_bins,
        blocked,
        played_toy,
    )


def _macro_to_skill(action: DriveMacroAction, observation: dict[str, Any], config: EcosystemConfig) -> dict[str, np.ndarray | int]:
    """Ground a selected macro action through public observations and config only."""

    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    food = np.asarray(observation["food_xy"], dtype=np.float32)
    toy = np.asarray(observation["toy_xy"], dtype=np.float32)
    if action is DriveMacroAction.PURSUE_FOOD:
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        if float(np.linalg.norm(food - agent)) <= config.pickup_radius:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        target = food
        return skill_action(
            ActionKind.WALK_TO,
            target,
            max(0.1, float(np.linalg.norm(target - agent)) / config.walk_speed_per_second),
        )
    if action is DriveMacroAction.PURSUE_PLAY:
        if float(np.linalg.norm(toy - agent)) <= config.toy_interaction_radius:
            return skill_action(ActionKind.RUN_AROUND, toy, 0.5)
        return skill_action(
            ActionKind.WALK_TO,
            toy,
            max(0.1, float(np.linalg.norm(toy - agent)) / config.walk_speed_per_second),
        )
    return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)


class MacroPolicy(Protocol):
    def choose(self, state: DriveState) -> DriveMacroAction: ...


class TabularDriveQPolicy:
    """A learned macro selector; it receives an encoded tuple, never an environment."""

    def __init__(self, *, include_drives: bool = True) -> None:
        self.include_drives = include_drives
        self.q_values: dict[DriveState, np.ndarray] = {}

    def _q(self, state: DriveState) -> np.ndarray:
        return self.q_values.setdefault(state, np.zeros(len(DriveMacroAction), dtype=np.float64))

    def choose(self, state: DriveState) -> DriveMacroAction:
        return DriveMacroAction(int(np.argmax(self._q(state))))

    def train(self, *, episodes: int = 1_200, seed: int = 20260803, alpha: float = 0.30, discount: float = 0.95) -> None:
        """Train one shared policy on M4 train tasks, layouts, and conditions only."""

        rng = np.random.default_rng(seed)
        task_ids = ("play_when_bored", "competing_drives")
        for episode in range(episodes):
            task_id = task_ids[episode % len(task_ids)]
            task = get_task(task_id)
            condition = TRAIN_CONDITIONS[(episode // len(task_ids)) % len(TRAIN_CONDITIONS)]
            env = EcosystemEnv()
            observation, _ = env.reset(
                seed=seed + episode,
                options={"task_id": task_id, "layout_id": task.train_layout_ids[0], **condition},
            )
            played_toy = False
            try:
                for _ in range(env.config.max_episode_steps):
                    state = observable_drive_state(
                        observation, env.config, played_toy=played_toy, include_drives=self.include_drives
                    )
                    epsilon = max(0.03, 1.0 - episode / (episodes * 0.75))
                    action = (
                        DriveMacroAction(int(rng.integers(len(DriveMacroAction))))
                        if rng.random() < epsilon
                        else self.choose(state)
                    )
                    gym_action = _macro_to_skill(action, observation, env.config)
                    next_observation, reward, terminated, truncated, _ = env.step(gym_action)
                    if (
                        action is DriveMacroAction.PURSUE_PLAY
                        and int(gym_action["kind"]) == int(ActionKind.RUN_AROUND)
                        and int(next_observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.SUCCESS)
                    ):
                        played_toy = True
                    next_state = observable_drive_state(
                        next_observation, env.config, played_toy=played_toy, include_drives=self.include_drives
                    )
                    target = reward if terminated or truncated else reward + discount * float(np.max(self._q(next_state)))
                    self._q(state)[int(action)] += alpha * (target - self._q(state)[int(action)])
                    observation = next_observation
                    if terminated or truncated:
                        break
            finally:
                env.close()


class RandomDriveMacroPolicy:
    """Weak baseline over precisely the same public macro-action surface."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def choose(self, state: DriveState) -> DriveMacroAction:
        del state
        return DriveMacroAction(int(self.rng.integers(len(DriveMacroAction))))


class _OracleMacroPolicy:
    """Explicitly task-labeled, scripted upper bound; never called learned."""

    def __init__(self, task_id: str) -> None:
        self.policy = DriveAwareOraclePolicy(task_id)

    def action(self, observation: dict[str, Any], env: EcosystemEnv) -> dict[str, np.ndarray | int]:
        return self.policy.act(observation, env)


@dataclass(frozen=True, slots=True)
class M6Episode:
    task_id: str
    layout_id: str
    seed: int
    condition: dict[str, str]
    task_success: bool
    survived: bool
    total_reward: float
    steps: int
    boundary_contacts: int
    play_before_food: bool


def _run_episode(
    policy: MacroPolicy | _OracleMacroPolicy, *, task_id: str, layout_id: str, seed: int, condition: dict[str, str]
) -> M6Episode:
    env = EcosystemEnv()
    try:
        observation, _ = env.reset(seed=seed, options={"task_id": task_id, "layout_id": layout_id, **condition})
        played_toy = False
        total_reward = 0.0
        for step in range(1, env.config.max_episode_steps + 1):
            if isinstance(policy, _OracleMacroPolicy):
                gym_action = policy.action(observation, env)
                macro_action: DriveMacroAction | None = None
            else:
                state = observable_drive_state(
                    observation, env.config, played_toy=played_toy, include_drives=getattr(policy, "include_drives", True)
                )
                macro_action = policy.choose(state)
                gym_action = _macro_to_skill(macro_action, observation, env.config)
            observation, reward, terminated, truncated, info = env.step(gym_action)
            total_reward += reward
            if (
                macro_action is DriveMacroAction.PURSUE_PLAY
                and int(gym_action["kind"]) == int(ActionKind.RUN_AROUND)
                and int(observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.SUCCESS)
            ):
                played_toy = True
            if terminated or truncated:
                return M6Episode(
                    task_id, layout_id, seed, condition, bool(info["task_success"]), bool(info["survived"]), total_reward,
                    step, int(info["boundary_contacts"]), bool(info["toy_played"]),
                )
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _aggregate(episodes: list[M6Episode]) -> dict[str, float | int]:
    return {
        "episodes": len(episodes),
        "successes": sum(item.task_success for item in episodes),
        "task_success_rate": float(np.mean([item.task_success for item in episodes])),
        "survival_rate": float(np.mean([item.survived for item in episodes])),
        "mean_total_reward": float(np.mean([item.total_reward for item in episodes])),
        "boundary_contacts": sum(item.boundary_contacts for item in episodes),
        "play_before_food_episodes": sum(item.play_before_food for item in episodes),
    }


def _condition_name(condition: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in condition.items())


def _evaluate(
    policy: MacroPolicy | _OracleMacroPolicy,
    *,
    task_id: str,
    layouts: tuple[str, ...],
    conditions: tuple[dict[str, str], ...],
    seeds: tuple[int, ...],
) -> dict[str, object]:
    episodes = [
        _run_episode(policy, task_id=task_id, layout_id=layout, seed=seed, condition=condition)
        for layout in layouts
        for condition in conditions
        for seed in seeds
    ]
    result: dict[str, object] = _aggregate(episodes)
    result["by_condition"] = {
        _condition_name(condition): _aggregate([item for item in episodes if item.condition == condition])
        for condition in conditions
    }
    return result


def _matrix_results(policy: MacroPolicy | _OracleMacroPolicy, *, task_id: str, seeds: tuple[int, ...]) -> dict[str, object]:
    task = get_task(task_id)
    return {
        "train": _evaluate(policy, task_id=task_id, layouts=task.train_layout_ids, conditions=TRAIN_CONDITIONS, seeds=seeds),
        "heldout": _evaluate(policy, task_id=task_id, layouts=task.heldout_layout_ids, conditions=HELDOUT_CONDITIONS, seeds=seeds),
    }


def _probe_observation(*, satiety: float, boredom: float) -> dict[str, Any]:
    return {
        "agent_xy": np.zeros(2, dtype=np.float32),
        "food_xy": np.asarray((0.5, 0.0), dtype=np.float32),
        "toy_xy": np.asarray((-0.5, 0.0), dtype=np.float32),
        "drives": np.asarray((satiety, 1.0, boredom), dtype=np.float32),
        "holding_food": 0,
        "prior_outcome": list(ActionOutcome).index(ActionOutcome.SUCCESS),
    }


def arbitration_probe(policy: MacroPolicy, config: EcosystemConfig | None = None) -> dict[str, object]:
    """Test the same geometry with only drives changed; task labels never enter."""

    active_config = config or EcosystemConfig()
    food_state = observable_drive_state(
        _probe_observation(satiety=0.20, boredom=0.85), active_config, played_toy=False,
        include_drives=getattr(policy, "include_drives", True),
    )
    play_state = observable_drive_state(
        _probe_observation(satiety=0.75, boredom=0.85), active_config, played_toy=False,
        include_drives=getattr(policy, "include_drives", True),
    )
    food_action = policy.choose(food_state)
    play_action = policy.choose(play_state)
    return {
        "food_first_action": food_action.name,
        "play_first_action": play_action.name,
        "passed": food_action is DriveMacroAction.PURSUE_FOOD and play_action is DriveMacroAction.PURSUE_PLAY,
    }


def _gate_passed(
    *,
    learned_rates: list[float],
    learned_survival_rates: list[float],
    condition_rates: list[float],
    no_drive_rates: list[float],
    random_rates: list[float],
    learned_returns: list[float],
    random_returns: list[float],
    arbitration_passed: bool,
    competing_play_before_food_episodes: int,
) -> bool:
    """Keep every published M6 comparison as an enforceable release condition."""

    return (
        min(learned_rates) >= 0.95
        and min(learned_survival_rates) >= 0.95
        and min(condition_rates) >= 0.90
        and float(np.mean(learned_rates)) > float(np.mean(no_drive_rates))
        and float(np.mean(learned_rates)) > float(np.mean(random_rates))
        and float(np.mean(learned_returns)) > float(np.mean(random_returns))
        and arbitration_passed
        and competing_play_before_food_episodes == 0
    )


def m6_benchmark(*, training_episodes: int = 1_200, seeds: tuple[int, ...] = M4_EVALUATION_SEEDS) -> dict[str, object]:
    """Run learned and baseline policies on one fixed M4 train/held-out matrix."""

    learned = TabularDriveQPolicy()
    no_drive = TabularDriveQPolicy(include_drives=False)
    learned.train(episodes=training_episodes)
    no_drive.train(episodes=training_episodes)
    results: dict[str, dict[str, dict[str, object]]] = {}
    for task_id in ("play_when_bored", "competing_drives"):
        results[task_id] = {
            "learned_state_oracle_q": _matrix_results(learned, task_id=task_id, seeds=seeds),
            "no_drive_state_oracle_q": _matrix_results(no_drive, task_id=task_id, seeds=seeds),
            "state_oracle_macro_random": _matrix_results(RandomDriveMacroPolicy(seed=900 + len(task_id)), task_id=task_id, seeds=seeds),
            "scripted_drive_oracle": _matrix_results(_OracleMacroPolicy(task_id), task_id=task_id, seeds=seeds),
        }
    learned_heldout = [results[task]["learned_state_oracle_q"]["heldout"] for task in results]
    no_drive_heldout = [results[task]["no_drive_state_oracle_q"]["heldout"] for task in results]
    random_heldout = [results[task]["state_oracle_macro_random"]["heldout"] for task in results]
    learned_rates = [float(item["task_success_rate"]) for item in learned_heldout]
    learned_survival_rates = [float(item["survival_rate"]) for item in learned_heldout]
    no_drive_rates = [float(item["task_success_rate"]) for item in no_drive_heldout]
    random_rates = [float(item["task_success_rate"]) for item in random_heldout]
    learned_returns = [float(item["mean_total_reward"]) for item in learned_heldout]
    random_returns = [float(item["mean_total_reward"]) for item in random_heldout]
    condition_rates = [
        float(condition["task_success_rate"])
        for split in learned_heldout
        for condition in split["by_condition"].values()  # type: ignore[index,union-attr]
    ]
    learned_probe = arbitration_probe(learned)
    return {
        "schema_version": "0.6",
        "environment_version": "0.4.0",
        "training": {"algorithm": "tabular_q_learning", "episodes": training_episodes, "seed": 20260803},
        "evaluation_seeds": list(seeds),
        "policy_input": ["agent_xy", "food_xy", "toy_xy", "drives", "holding_food", "prior_outcome"],
        "conditions": {"train": list(TRAIN_CONDITIONS), "heldout": list(HELDOUT_CONDITIONS)},
        "results": results,
        "arbitration_probe": {"learned": learned_probe, "no_drive": arbitration_probe(no_drive)},
        "gate": {
            "minimum_task_success_rate": 0.95,
            "minimum_survival_rate": 0.95,
            "minimum_condition_success_rate": 0.90,
            "learned_beats_no_drive_mean": float(np.mean(learned_rates)) > float(np.mean(no_drive_rates)),
            "learned_beats_macro_random_mean": float(np.mean(learned_rates)) > float(np.mean(random_rates)),
            "learned_beats_macro_random_mean_return": float(np.mean(learned_returns)) > float(np.mean(random_returns)),
            "no_successful_competing_play_before_food": results["competing_drives"]["learned_state_oracle_q"]["heldout"]["play_before_food_episodes"] == 0,
            "passed": _gate_passed(
                learned_rates=learned_rates,
                learned_survival_rates=learned_survival_rates,
                condition_rates=condition_rates,
                no_drive_rates=no_drive_rates,
                random_rates=random_rates,
                learned_returns=learned_returns,
                random_returns=random_returns,
                arbitration_passed=bool(learned_probe["passed"]),
                competing_play_before_food_episodes=results["competing_drives"]["learned_state_oracle_q"]["heldout"]["play_before_food_episodes"],
            ),
        },
        "limits": [
            "This is a state-oracle macro controller, not an RGB or torque-control result.",
            "Policy selection sees only the named observation fields and one memory bit reconstructed from prior_outcome.",
            "The scripted drive oracle is an upper bound, not a learned result.",
        ],
    }


def write_m6_report(path: str | Path, **kwargs: object) -> dict[str, object]:
    report = m6_benchmark(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m6_viewer_demo(
    trace_path: str | Path, *, task_id: str = "competing_drives", seed: int = 7, training_episodes: int = 1_200
) -> ReplayResult:
    """Record then replay one fixed held-out M6 rollout through ViewerSession."""

    policy = TabularDriveQPolicy()
    policy.train(episodes=training_episodes)
    task = get_task(task_id)
    with ViewerSession(trace_path=trace_path, episode_id="m6-learned-demo") as session:
        observation, _ = session.reset(
            seed=seed,
            options={"task_id": task_id, "layout_id": task.heldout_layout_ids[0], **HELDOUT_CONDITIONS[-1]},
        )
        played_toy = False
        for _ in range(session.env.config.max_episode_steps):
            state = observable_drive_state(observation, session.env.config, played_toy=played_toy)
            action = policy.choose(state)
            gym_action = _macro_to_skill(action, observation, session.env.config)
            observation, _, terminated, truncated, _ = session.step(gym_action)
            if (
                action is DriveMacroAction.PURSUE_PLAY
                and int(gym_action["kind"]) == int(ActionKind.RUN_AROUND)
                and int(observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.SUCCESS)
            ):
                played_toy = True
            if terminated or truncated:
                break
        return session.replay(trace_path)
