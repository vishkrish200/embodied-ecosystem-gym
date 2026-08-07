"""M7 learned RGB macro arbitration over the M4 food-and-toy tasks.

The learned selector consumes RGB-derived local object evidence, drives, held
state, outcome state, and one local memory bit.  The visual adapter is fixed
and inspectable; its offsets are grounded through the public relative-walk
skill so it never needs an oracle global pose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..config import EcosystemConfig
from ..env import EcosystemEnv
from .m4 import HELDOUT_CONDITIONS, M4_EVALUATION_SEEDS, TRAIN_CONDITIONS, DriveAwareOraclePolicy
from .m6 import DriveMacroAction
from ..perception import detection_to_relative_xy
from ..policies import skill_action
from ..tasks import get_task
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


RgbDriveState = tuple[bool, bool, bool, bool, bool, int, int, int, bool, bool, bool, bool]
M7_MAX_MACRO_STEPS = 12
M7_TRAIN_LAYOUT_IDS = ("m4_train_play", "m4_train_competing")


def _rgb_config() -> EcosystemConfig:
    """Bound exploration without changing the M4 layouts, conditions, or seeds."""

    return EcosystemConfig(observation_mode="rgb", max_episode_steps=M7_MAX_MACRO_STEPS)


@dataclass(frozen=True, slots=True)
class RgbObjectDetections:
    """Food/toy relative positions recovered from policy-visible RGB only."""

    food: np.ndarray | None
    toy: np.ndarray | None


def _relative_detection(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return None
    height, width = mask.shape
    detection = np.asarray(
        [1.0, float(np.mean(xs)) / max(width - 1, 1) * 2.0 - 1.0, float(np.mean(ys)) / max(height - 1, 1) * 2.0 - 1.0],
        dtype=np.float32,
    )
    return detection_to_relative_xy(detection)


def rgb_object_detections(rgb: np.ndarray) -> RgbObjectDetections:
    """Detect saturated food and the stable yellow toy without simulator state.

    Food is defined as a saturated object other than the rendered cyan agent or
    yellow toy.  That keeps blue/purple held-out food on the same code path as
    red/orange training food.  The detector is a fixed visual adapter, not a
    learned perception result.
    """

    image = np.asarray(rgb, dtype=np.uint8)
    red, green, blue = (image[..., channel] for channel in range(3))
    # These hue-ratio bands remain valid after MuJoCo lighting attenuates the
    # source RGBA values.  The toy's yellow family is deliberately separated
    # from orange food by its tighter red/green ratio.
    toy = (red > 140) & (green > 110) & (blue < 130)
    warm_food = (red > 70) & (blue < 100) & (red > green * 1.35)
    # A blue food object can be partially shadowed, where its green channel
    # rises above the brighter cyan agent's usual band.  Its red channel stays
    # distinctly higher, which keeps the two components separable.
    bright_blue_food = (blue > 170) & (blue > green * 1.90) & (green < 115) & (red < 90)
    shadowed_blue_food = (red > 65) & (red < 110) & (blue > 180) & (blue > green * 1.30) & (green < 180)
    blue_food = bright_blue_food | shadowed_blue_food
    purple_food = (red > 80) & (blue > 120) & (green < 100)
    food = (warm_food | blue_food | purple_food) & ~toy
    return RgbObjectDetections(food=_relative_detection(food), toy=_relative_detection(toy))


def _drive_bin(value: float) -> int:
    return int(np.digitize(value, (0.30, 0.60, 0.80)))


def observable_rgb_drive_state(
    observation: dict[str, Any],
    config: EcosystemConfig,
    *,
    played_toy: bool,
    pending_food_pickup: bool,
    pending_toy_play: bool,
    include_drives: bool = True,
) -> RgbDriveState:
    """Complete learned-selector input: RGB observation fields plus local memory."""

    detections = rgb_object_detections(np.asarray(observation["rgb"], dtype=np.uint8))
    drives = np.asarray(observation["drives"], dtype=np.float32)
    drive_bins = tuple(_drive_bin(float(value)) for value in drives) if include_drives else (0, 0, 0)
    food_near = detections.food is not None and float(np.linalg.norm(detections.food)) <= config.pickup_radius
    toy_near = detections.toy is not None and float(np.linalg.norm(detections.toy)) <= config.toy_interaction_radius
    blocked = int(observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.BLOCKED)
    return (
        bool(observation["holding_food"]),
        detections.food is not None,
        detections.toy is not None,
        food_near,
        toy_near,
        *drive_bins,
        blocked,
        played_toy,
        pending_food_pickup,
        pending_toy_play,
    )


def rgb_macro_to_skill(
    action: DriveMacroAction,
    observation: dict[str, Any],
    config: EcosystemConfig,
    *,
    pending_food_pickup: bool,
    pending_toy_play: bool,
) -> dict[str, np.ndarray | int]:
    """Ground a macro choice into a public skill using only RGB-local offsets."""

    detections = rgb_object_detections(np.asarray(observation["rgb"], dtype=np.uint8))
    if action is DriveMacroAction.PURSUE_FOOD:
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        if pending_food_pickup:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        if detections.food is None:
            return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)
        distance = float(np.linalg.norm(detections.food))
        if distance <= config.pickup_radius:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        return skill_action(
            ActionKind.WALK_RELATIVE,
            detections.food,
            max(0.1, distance / config.walk_speed_per_second),
        )
    if action is DriveMacroAction.PURSUE_PLAY:
        if pending_toy_play:
            return skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 0.5)
        if detections.toy is None:
            return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)
        distance = float(np.linalg.norm(detections.toy))
        if distance <= config.toy_interaction_radius:
            return skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 0.5)
        return skill_action(
            ActionKind.WALK_RELATIVE,
            detections.toy,
            max(0.1, distance / config.walk_speed_per_second),
        )
    return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)


def _update_memory(
    *,
    action: DriveMacroAction,
    gym_action: dict[str, np.ndarray | int],
    next_observation: dict[str, Any],
    config: EcosystemConfig,
    played_toy: bool,
    pending_food_pickup: bool,
    pending_toy_play: bool,
) -> tuple[bool, bool, bool]:
    """Update only policy-owned memory from its action and the next RGB outcome."""

    succeeded = int(next_observation["prior_outcome"]) == list(ActionOutcome).index(ActionOutcome.SUCCESS)
    kind = int(gym_action["kind"])
    detections = rgb_object_detections(np.asarray(next_observation["rgb"], dtype=np.uint8))
    if action is DriveMacroAction.PURSUE_FOOD:
        pending_toy_play = False
        if kind == int(ActionKind.WALK_RELATIVE):
            pending_food_pickup = succeeded and (
                detections.food is None or float(np.linalg.norm(detections.food)) <= config.pickup_radius
            )
        elif kind == int(ActionKind.PICK_UP):
            pending_food_pickup = False
    elif action is DriveMacroAction.PURSUE_PLAY:
        pending_food_pickup = False
        if kind == int(ActionKind.WALK_RELATIVE):
            pending_toy_play = succeeded and (
                detections.toy is None or float(np.linalg.norm(detections.toy)) <= config.toy_interaction_radius
            )
        elif kind == int(ActionKind.RUN_AROUND):
            pending_toy_play = False
            played_toy = played_toy or succeeded
    return played_toy, pending_food_pickup, pending_toy_play


class RgbMacroPolicy(Protocol):
    def choose(self, state: RgbDriveState) -> DriveMacroAction: ...


class TabularRgbDriveBCPolicy:
    """A learned RGB macro selector fitted from public-observation labels."""

    def __init__(self, *, include_drives: bool = True) -> None:
        self.include_drives = include_drives
        self.q_values: dict[tuple[bool | int, ...], np.ndarray] = {}

    def _features(self, state: RgbDriveState) -> tuple[bool | int, ...]:
        """Retain decision signals, not appearance-specific detector details."""

        if self.include_drives:
            return state[0], state[5], state[6], state[7]
        return (state[0],)

    def _q(self, state: RgbDriveState) -> np.ndarray:
        return self.q_values.setdefault(self._features(state), np.zeros(len(DriveMacroAction), dtype=np.float64))

    def choose(self, state: RgbDriveState) -> DriveMacroAction:
        if state[0] or state[-2]:
            return DriveMacroAction.PURSUE_FOOD
        if state[-1]:
            return DriveMacroAction.PURSUE_PLAY
        return DriveMacroAction(int(np.argmax(self._q(state))))

    @staticmethod
    def _guided_action(task_id: str, observation: dict[str, Any]) -> DriveMacroAction:
        """Training-task teacher that supplies offline macro labels only."""

        if bool(observation["holding_food"]):
            return DriveMacroAction.PURSUE_FOOD
        if task_id == "play_when_bored":
            return DriveMacroAction.PURSUE_PLAY
        drives = np.asarray(observation["drives"], dtype=np.float32)
        return DriveMacroAction.PURSUE_FOOD if drives[0] <= drives[2] else DriveMacroAction.PURSUE_PLAY

    def train(self, *, episodes: int = 400, seed: int = 20260804) -> None:
        """Fit a tabular RGB macro classifier from public drive-teacher labels."""

        task_ids = ("play_when_bored", "competing_drives")
        for episode in range(episodes):
            pair_index = episode // len(task_ids)
            task_id = task_ids[episode % len(task_ids)]
            condition = TRAIN_CONDITIONS[pair_index % len(TRAIN_CONDITIONS)]
            layout_id = "m4_train_play" if task_id == "play_when_bored" else "m4_train_competing"
            env = EcosystemEnv(_rgb_config())
            try:
                observation, _ = env.reset(
                    seed=seed + pair_index,
                    options={"task_id": task_id, "layout_id": layout_id, **condition},
                )
                state = observable_rgb_drive_state(
                    observation,
                    env.config,
                    played_toy=False,
                    pending_food_pickup=False,
                    pending_toy_play=False,
                    include_drives=self.include_drives,
                )
                self._q(state)[int(self._guided_action(task_id, observation))] += 1.0
            finally:
                env.close()


class RandomRgbMacroPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def choose(self, state: RgbDriveState) -> DriveMacroAction:
        del state
        return DriveMacroAction(int(self.rng.integers(len(DriveMacroAction))))


@dataclass(frozen=True, slots=True)
class M7Episode:
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
    first_macro_action: str


def _run_episode(
    policy: RgbMacroPolicy, *, task_id: str, layout_id: str, seed: int, condition: dict[str, str]
) -> M7Episode:
    env = EcosystemEnv(_rgb_config())
    try:
        observation, _ = env.reset(seed=seed, options={"task_id": task_id, "layout_id": layout_id, **condition})
        played_toy = False
        pending_food_pickup = False
        pending_toy_play = False
        total_reward = 0.0
        first_macro_action = DriveMacroAction.IDLE.name
        for step in range(1, env.config.max_episode_steps + 1):
            state = observable_rgb_drive_state(
                observation,
                env.config,
                played_toy=played_toy,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
                include_drives=getattr(policy, "include_drives", True),
            )
            action = policy.choose(state)
            if step == 1:
                first_macro_action = action.name
            gym_action = rgb_macro_to_skill(
                action,
                observation,
                env.config,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
            )
            observation, reward, terminated, truncated, info = env.step(gym_action)
            total_reward += reward
            played_toy, pending_food_pickup, pending_toy_play = _update_memory(
                action=action,
                gym_action=gym_action,
                next_observation=observation,
                config=env.config,
                played_toy=played_toy,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
            )
            if terminated or truncated:
                return M7Episode(
                    task_id, layout_id, seed, condition, bool(info["task_success"]), bool(info["survived"]), total_reward,
                    step, int(info["boundary_contacts"]), bool(info["toy_played"]), first_macro_action,
                )
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _aggregate(episodes: list[M7Episode]) -> dict[str, float | int]:
    return {
        "episodes": len(episodes),
        "successes": sum(item.task_success for item in episodes),
        "task_success_rate": float(np.mean([item.task_success for item in episodes])),
        "survival_rate": float(np.mean([item.survived for item in episodes])),
        "mean_total_reward": float(np.mean([item.total_reward for item in episodes])),
        "boundary_contacts": sum(item.boundary_contacts for item in episodes),
        "play_before_food_episodes": sum(item.play_before_food for item in episodes),
        "first_macro_actions": {
            action.name: sum(item.first_macro_action == action.name for item in episodes)
            for action in DriveMacroAction
        },
    }


def _condition_name(condition: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in condition.items())


def _evaluate(
    policy: RgbMacroPolicy,
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


def _matrix_results(policy: RgbMacroPolicy, *, task_id: str, seeds: tuple[int, ...]) -> dict[str, object]:
    task = get_task(task_id)
    return {
        "train": _evaluate(policy, task_id=task_id, layouts=task.train_layout_ids, conditions=TRAIN_CONDITIONS, seeds=seeds),
        "visual_heldout": _evaluate(
            policy, task_id=task_id, layouts=task.train_layout_ids, conditions=HELDOUT_CONDITIONS, seeds=seeds
        ),
        "spatial_heldout_diagnostic": _evaluate(
            policy, task_id=task_id, layouts=task.heldout_layout_ids, conditions=HELDOUT_CONDITIONS, seeds=seeds
        ),
    }


def rgb_arbitration_probe(policy: RgbMacroPolicy) -> dict[str, object]:
    """Hold an RGB frame fixed and vary only the visible drive values."""

    env = EcosystemEnv(_rgb_config())
    try:
        observation, _ = env.reset(
            seed=7,
            options={"task_id": "competing_drives", "layout_id": "m4_train_competing", **TRAIN_CONDITIONS[0]},
        )
        food_observation = {**observation, "drives": np.asarray((0.20, 1.0, 0.85), dtype=np.float32)}
        play_observation = {**observation, "drives": np.asarray((0.75, 1.0, 0.85), dtype=np.float32)}
        food_action = policy.choose(
            observable_rgb_drive_state(
                food_observation, env.config, played_toy=False, pending_food_pickup=False, pending_toy_play=False
            )
        )
        play_action = policy.choose(
            observable_rgb_drive_state(
                play_observation, env.config, played_toy=False, pending_food_pickup=False, pending_toy_play=False
            )
        )
        return {
            "food_first_action": food_action.name,
            "play_first_action": play_action.name,
            "passed": food_action is DriveMacroAction.PURSUE_FOOD and play_action is DriveMacroAction.PURSUE_PLAY,
        }
    finally:
        env.close()


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
    return (
        min(learned_rates) >= 0.90
        and min(learned_survival_rates) >= 0.90
        and min(condition_rates) >= 0.80
        and float(np.mean(learned_rates)) > float(np.mean(no_drive_rates))
        and float(np.mean(learned_rates)) > float(np.mean(random_rates))
        and float(np.mean(learned_returns)) > float(np.mean(random_returns))
        and arbitration_passed
        and competing_play_before_food_episodes == 0
    )


def m7_benchmark(*, training_episodes: int = 400, seeds: tuple[int, ...] = M4_EVALUATION_SEEDS) -> dict[str, object]:
    """Train/evaluate RGB arbitration on the unchanged M4 task matrix."""

    learned = TabularRgbDriveBCPolicy()
    no_drive = TabularRgbDriveBCPolicy(include_drives=False)
    learned.train(episodes=training_episodes)
    no_drive.train(episodes=training_episodes)
    results: dict[str, dict[str, dict[str, object]]] = {}
    for task_id in ("play_when_bored", "competing_drives"):
        results[task_id] = {
            "learned_rgb_drive_bc": _matrix_results(learned, task_id=task_id, seeds=seeds),
            "no_drive_rgb_bc": _matrix_results(no_drive, task_id=task_id, seeds=seeds),
            "rgb_macro_random": _matrix_results(RandomRgbMacroPolicy(seed=1_700 + len(task_id)), task_id=task_id, seeds=seeds),
        }
    learned_heldout = [results[task]["learned_rgb_drive_bc"]["visual_heldout"] for task in results]
    no_drive_heldout = [results[task]["no_drive_rgb_bc"]["visual_heldout"] for task in results]
    random_heldout = [results[task]["rgb_macro_random"]["visual_heldout"] for task in results]
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
    probe = rgb_arbitration_probe(learned)
    competing_play_before_food = results["competing_drives"]["learned_rgb_drive_bc"]["visual_heldout"]["play_before_food_episodes"]
    return {
        "schema_version": "0.7",
        "environment_version": "0.4.0",
        "training": {
            "algorithm": "tabular_behavior_cloning_from_task-teacher macro labels",
            "episodes": training_episodes,
            "seed": 20260804,
        },
        "policy_horizon_steps": M7_MAX_MACRO_STEPS,
        "evaluation_seeds": list(seeds),
        "policy_input": ["rgb", "drives", "holding_food", "prior_outcome"],
        "visual_adapter": "fixed saturated-food and yellow-toy local geometry from RGB",
        "conditions": {"train": list(TRAIN_CONDITIONS), "heldout": list(HELDOUT_CONDITIONS)},
        "results": results,
        "arbitration_probe": {"learned": probe, "no_drive": rgb_arbitration_probe(no_drive)},
        "gate": {
            "minimum_task_success_rate": 0.90,
            "minimum_survival_rate": 0.90,
            "minimum_condition_success_rate": 0.80,
            "learned_beats_no_drive_mean": float(np.mean(learned_rates)) > float(np.mean(no_drive_rates)),
            "learned_beats_macro_random_mean": float(np.mean(learned_rates)) > float(np.mean(random_rates)),
            "learned_beats_macro_random_mean_return": float(np.mean(learned_returns)) > float(np.mean(random_returns)),
            "no_successful_competing_play_before_food": competing_play_before_food == 0,
            "passed": _gate_passed(
                learned_rates=learned_rates,
                learned_survival_rates=learned_survival_rates,
                condition_rates=condition_rates,
                no_drive_rates=no_drive_rates,
                random_rates=random_rates,
                learned_returns=learned_returns,
                random_returns=random_returns,
                arbitration_passed=bool(probe["passed"]),
                competing_play_before_food_episodes=competing_play_before_food,
            ),
        },
        "limits": [
            "This is learned RGB macro arbitration trained from task-teacher macro labels, with public drives at inference and a fixed visual-geometry adapter; it is not end-to-end learned perception or reinforcement learning.",
            "The policy uses the additive relative-walk skill; it never receives oracle object coordinates or a global pose.",
            "The gate varies M4 object appearance and dynamics on each task's seen spawn layout; held-out spatial layouts are diagnostic because RGB-relative control has no learned global relocalization yet.",
            "The evaluation keeps M4's fixed center camera and does not claim camera-offset robustness.",
        ],
    }


def write_m7_report(path: str | Path, **kwargs: object) -> dict[str, object]:
    report = m7_benchmark(**kwargs)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m7_viewer_demo(
    trace_path: str | Path, *, task_id: str = "competing_drives", seed: int = 7, training_episodes: int = 400
) -> ReplayResult:
    """Record/replay one fixed held-out M7 RGB rollout through the thin viewer."""

    policy = TabularRgbDriveBCPolicy()
    policy.train(episodes=training_episodes)
    task = get_task(task_id)
    rgb_config = _rgb_config()
    with ViewerSession(EcosystemEnv(rgb_config, render_mode="rgb_array"), trace_path=trace_path, episode_id="m7-rgb-demo") as session:
        observation, _ = session.reset(
            seed=seed,
            options={"task_id": task_id, "layout_id": task.heldout_layout_ids[0], **HELDOUT_CONDITIONS[-1]},
        )
        played_toy = False
        pending_food_pickup = False
        pending_toy_play = False
        for _ in range(session.env.config.max_episode_steps):
            state = observable_rgb_drive_state(
                observation,
                session.env.config,
                played_toy=played_toy,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
            )
            action = policy.choose(state)
            gym_action = rgb_macro_to_skill(
                action,
                observation,
                session.env.config,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
            )
            observation, _, terminated, truncated, _ = session.step(gym_action)
            played_toy, pending_food_pickup, pending_toy_play = _update_memory(
                action=action,
                gym_action=gym_action,
                next_observation=observation,
                config=session.env.config,
                played_toy=played_toy,
                pending_food_pickup=pending_food_pickup,
                pending_toy_play=pending_toy_play,
            )
            if terminated or truncated:
                break
        return session.replay(trace_path)
