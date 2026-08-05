"""M8.1: a frozen, paired feed-forward versus recurrent RGB evaluation.

M8 remains unchanged.  M8.1 adds two small NumPy behavior-cloning
challengers and treats the state policy as a privileged ceiling.  The learned
policies receive pixels at action time; the recurrent policy additionally
receives memory containing only its previous action and observed outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionKind, ActionOutcome
from .config import EcosystemConfig
from .env import EcosystemEnv
from .m8 import FixedRgbScanRecoveryPolicy, M8_LAYOUT_ID, M8_TASK_ID, rgb_food_components, wilson_interval
from .policies import skill_action
from .trajectory import ReplayResult
from .viewer import ViewerSession


M81_PROTOCOL_VERSION = "m81-rgb-memory-bc-v1"
M81_TRAIN_SEEDS = tuple(range(100, 120))
M81_TEST_SEEDS = tuple(range(20))
M81_EVALUATION_SEEDS = M81_TEST_SEEDS
M81_PAIRED_WIN_MARGIN = 0.10

# These four conditions, their order, and the test seeds define the paired
# recurrence comparison.  The landmark shift is diagnostic-only.
M81_CONDITIONS: dict[str, dict[str, Any]] = {
    "reference": {"camera_control": "scan", "initial_scan_sector": "north"},
    "occlusion": {"camera_control": "scan", "initial_scan_sector": "east"},
    "blocked_distractor": {
        "camera_control": "scan",
        "initial_scan_sector": "north",
        "blocked_distractor": True,
        "distractor_xy": [0.0, 0.32],
    },
    "relocation": {"camera_control": "scan", "initial_scan_sector": "north", "disturbance_step": 1},
}
M81_DIAGNOSTIC_CONDITIONS: dict[str, dict[str, Any]] = {
    "landmark_geometry": {
        "camera_control": "scan",
        "initial_scan_sector": "north",
        "geometry_variant": "m81_landmark",
    }
}

_CLASSES = (
    ActionKind.SCAN,
    ActionKind.WALK_RELATIVE,
    ActionKind.PICK_UP_RELATIVE,
    ActionKind.CONSUME,
)
_OUTCOME_COUNT = len(ActionOutcome)


def m81_config(*, observation_mode: str = "rgb") -> EcosystemConfig:
    return EcosystemConfig(observation_mode=observation_mode, max_episode_steps=24)  # type: ignore[arg-type]


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, "layout_id": M8_LAYOUT_ID, "food_variant": "red", **controls}


def _rgb_projection(rgb: np.ndarray) -> np.ndarray:
    """A fixed compact projection; fitted weights, rather than thresholds, choose actions."""

    image = np.asarray(rgb, dtype=np.float32) / 255.0
    # 64x64 protocol frames become an 8x8x3 raw-pixel block projection.
    height, width = image.shape[:2]
    pooled = image[: height // 8 * 8, : width // 8 * 8].reshape(8, height // 8, 8, width // 8, 3).mean((1, 3))
    components = rgb_food_components(np.asarray(rgb, dtype=np.uint8))
    nearest = components[0] if components else np.zeros(2, dtype=np.float32)
    return np.concatenate(
        [pooled.reshape(-1), nearest, np.asarray([bool(components), min(len(components), 2), 1.0], dtype=np.float32)]
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ActionOutcomeMemory:
    previous_action: int = -1
    previous_outcome: int = 0


class _LinearRgbBC:
    def __init__(self, weights: np.ndarray, *, recurrent: bool) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.recurrent = recurrent

    @staticmethod
    def _temporal(memory: ActionOutcomeMemory) -> np.ndarray:
        action = np.zeros(len(_CLASSES) + 1, dtype=np.float32)
        action[0 if memory.previous_action < 0 else _CLASSES.index(ActionKind(memory.previous_action)) + 1] = 1.0
        outcome = np.zeros(_OUTCOME_COUNT, dtype=np.float32)
        outcome[memory.previous_outcome] = 1.0
        return np.concatenate([action, outcome])

    def _features(self, rgb: np.ndarray, memory: ActionOutcomeMemory | None = None) -> np.ndarray:
        visual = _rgb_projection(rgb)
        if not self.recurrent:
            return visual
        if memory is None:
            raise TypeError("recurrent policy requires ActionOutcomeMemory")
        return np.concatenate([visual, self._temporal(memory)])

    def _action(self, rgb: np.ndarray, memory: ActionOutcomeMemory | None = None) -> dict[str, np.ndarray | int]:
        kind = _CLASSES[int(np.argmax(self._features(rgb, memory) @ self.weights))]
        components = rgb_food_components(np.asarray(rgb, dtype=np.uint8))
        target = components[0] if components else np.zeros(2, dtype=np.float32)
        distance = float(np.linalg.norm(target))
        duration = max(0.1, distance / m81_config().walk_speed_per_second) if kind is ActionKind.WALK_RELATIVE else 0.1
        return skill_action(kind, target, duration)


class FeedForwardRgbBCPolicy(_LinearRgbBC):
    """Behavior-cloned action classifier whose act boundary is RGB only."""

    def __init__(self, weights: np.ndarray) -> None:
        super().__init__(weights, recurrent=False)

    def act(self, rgb: np.ndarray) -> dict[str, np.ndarray | int]:
        return self._action(rgb)


class RecurrentRgbBCPolicy(_LinearRgbBC):
    """The same classifier plus previous-action/outcome one-hot memory."""

    def __init__(self, weights: np.ndarray) -> None:
        super().__init__(weights, recurrent=True)

    def reset(self) -> ActionOutcomeMemory:
        return ActionOutcomeMemory()

    def act(self, rgb: np.ndarray, memory: ActionOutcomeMemory) -> dict[str, np.ndarray | int]:
        return self._action(rgb, memory)

    @staticmethod
    def remember(
        memory: ActionOutcomeMemory, action: dict[str, np.ndarray | int], outcome: int
    ) -> ActionOutcomeMemory:
        del memory
        return ActionOutcomeMemory(int(action["kind"]), int(outcome))


def _collect_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual: list[np.ndarray] = []
    temporal: list[np.ndarray] = []
    labels: list[int] = []
    for controls in M81_CONDITIONS.values():
        for seed in M81_TRAIN_SEEDS:
            env = EcosystemEnv(m81_config())
            try:
                observation, _ = env.reset(seed=seed, options=_options(controls))
                teacher = FixedRgbScanRecoveryPolicy()
                teacher_memory = teacher.reset()
                policy_memory = ActionOutcomeMemory()
                for _ in range(env.config.max_episode_steps):
                    rgb = np.asarray(observation["rgb"], dtype=np.uint8)
                    action = teacher.act(observation, teacher_memory)
                    kind = ActionKind(int(action["kind"]))
                    visual.append(_rgb_projection(rgb))
                    temporal.append(_LinearRgbBC._temporal(policy_memory))
                    labels.append(_CLASSES.index(kind))
                    observation, _, terminated, truncated, _ = env.step(action)
                    policy_memory = ActionOutcomeMemory(int(kind), int(observation["prior_outcome"]))
                    if terminated or truncated:
                        break
            finally:
                env.close()
    return np.stack(visual), np.stack(temporal), np.asarray(labels, dtype=np.int64)


def fit_m81_policies(*, ridge: float = 1e-3) -> tuple[FeedForwardRgbBCPolicy, RecurrentRgbBCPolicy]:
    visual, temporal, labels = _collect_training_data()
    targets = np.eye(len(_CLASSES), dtype=np.float64)[labels]

    def fit(features: np.ndarray) -> np.ndarray:
        gram = features.T @ features
        return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), features.T @ targets)

    return FeedForwardRgbBCPolicy(fit(visual)), RecurrentRgbBCPolicy(fit(np.concatenate([visual, temporal], axis=1)))


@dataclass(frozen=True, slots=True)
class M81Episode:
    seed: int
    condition: str
    task_success: bool
    total_reward: float
    steps: int


def _run_episode(policy: Any, *, seed: int, condition: str, controls: dict[str, Any]) -> M81Episode:
    env = EcosystemEnv(m81_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset() if isinstance(policy, RecurrentRgbBCPolicy) else None
        total_reward = 0.0
        for step in range(1, env.config.max_episode_steps + 1):
            rgb = np.asarray(observation["rgb"], dtype=np.uint8)
            action = policy.act(rgb, memory) if memory is not None else policy.act(rgb)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if memory is not None:
                memory = policy.remember(memory, action, int(observation["prior_outcome"]))
            if terminated or truncated:
                return M81Episode(seed, condition, bool(info["task_success"]), total_reward, step)
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _run_fixed(seed: int, condition: str, controls: dict[str, Any]) -> M81Episode:
    env = EcosystemEnv(m81_config())
    policy = FixedRgbScanRecoveryPolicy()
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        total_reward = 0.0
        for step in range(1, env.config.max_episode_steps + 1):
            observation, reward, terminated, truncated, info = env.step(policy.act(observation, memory))
            total_reward += reward
            if terminated or truncated:
                return M81Episode(seed, condition, bool(info["task_success"]), total_reward, step)
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _run_oracle(seed: int, condition: str, controls: dict[str, Any]) -> M81Episode:
    env = EcosystemEnv(m81_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        total_reward = 0.0
        for step in range(1, env.config.max_episode_steps + 1):
            delta = np.asarray(observation["food_xy"]) - np.asarray(observation["agent_xy"])
            if bool(observation["holding_food"]):
                action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
            elif float(np.linalg.norm(delta)) <= env.config.pickup_radius:
                action = skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
            else:
                action = skill_action(ActionKind.WALK_TO, observation["food_xy"], max(0.1, float(np.linalg.norm(delta)) / env.config.walk_speed_per_second))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                return M81Episode(seed, condition, bool(info["task_success"]), total_reward, step)
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _aggregate(episodes: list[M81Episode]) -> dict[str, object]:
    successes = sum(episode.task_success for episode in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "task_success_rate": successes / len(episodes),
        "task_success_wilson_95": list(wilson_interval(successes, len(episodes))),
        "mean_total_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([episode.steps for episode in episodes])),
    }


def m81_benchmark(*, test_seeds: tuple[int, ...] = M81_TEST_SEEDS) -> dict[str, object]:
    if test_seeds != M81_TEST_SEEDS:
        raise ValueError("M8.1 test seeds are frozen; use M81_TEST_SEEDS")
    feed_forward, recurrent = fit_m81_policies()
    results: dict[str, dict[str, object]] = {}
    diagnostic_results: dict[str, dict[str, object]] = {}
    paired_ff: list[bool] = []
    paired_rnn: list[bool] = []

    def evaluate(condition: str, controls: dict[str, Any]) -> tuple[dict[str, object], list[M81Episode], list[M81Episode]]:
        fixed = [_run_fixed(seed, condition, controls) for seed in test_seeds]
        ff = [_run_episode(feed_forward, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        rnn = [_run_episode(recurrent, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        oracle = [_run_oracle(seed, condition, controls) for seed in test_seeds]
        return {
            "m8_fixed_rgb_baseline": _aggregate(fixed),
            "feed_forward_rgb_bc": _aggregate(ff),
            "recurrent_rgb_bc": _aggregate(rnn),
            "state_oracle_ceiling": _aggregate(oracle),
        }, ff, rnn

    for condition, controls in M81_CONDITIONS.items():
        results[condition], ff, rnn = evaluate(condition, controls)
        paired_ff.extend(episode.task_success for episode in ff)
        paired_rnn.extend(episode.task_success for episode in rnn)
    for condition, controls in M81_DIAGNOSTIC_CONDITIONS.items():
        diagnostic_results[condition], _, _ = evaluate(condition, controls)

    recurrent_only = sum(r and not f for f, r in zip(paired_ff, paired_rnn, strict=True))
    feed_forward_only = sum(f and not r for f, r in zip(paired_ff, paired_rnn, strict=True))
    pairs = len(paired_ff)
    net_advantage = (recurrent_only - feed_forward_only) / pairs
    return {
        "schema_version": "0.81",
        "protocol_version": M81_PROTOCOL_VERSION,
        "train_seeds": list(M81_TRAIN_SEEDS),
        "test_seeds": list(test_seeds),
        "conditions": M81_CONDITIONS,
        "diagnostic_conditions": M81_DIAGNOSTIC_CONDITIONS,
        "policy_boundary": {
            "feed_forward_rgb_bc": ["rgb"],
            "recurrent_rgb_bc": ["rgb", "previous_action", "previous_outcome"],
        },
        "results": results,
        "diagnostic_results": diagnostic_results,
        "recurrence_verdict": {
            "paired_episodes": pairs,
            "predeclared_net_advantage_margin": M81_PAIRED_WIN_MARGIN,
            "recurrent_only_successes": recurrent_only,
            "feed_forward_only_successes": feed_forward_only,
            "paired_net_advantage": net_advantage,
            "recurrence_win": bool(net_advantage >= M81_PAIRED_WIN_MARGIN),
            "diagnostics_included": False,
        },
        "limits": [
            "Both challengers are small ridge behavior-cloning classifiers over a fixed raw-RGB block projection; this is not an end-to-end representation-learning result.",
            "The fixed M8 RGB component adapter grounds local targets after the learned action choice.",
            "The recurrent challenger has only the immediately previous action and public outcome, not environment state, info, task identity, reset options, or a privileged pose.",
            "The landmark geometry result is diagnostic and is excluded from the predeclared recurrence verdict.",
        ],
    }


def write_m81_report(path: str | Path) -> dict[str, object]:
    report = m81_benchmark()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m81_viewer_demo(trace_path: str | Path, *, seed: int = 7) -> ReplayResult:
    """Replay one learned recurrent relocation episode through the thin viewer."""

    _, policy = fit_m81_policies()
    controls = {**M81_CONDITIONS["relocation"], "initial_scan_sector": "east"}
    with ViewerSession(EcosystemEnv(m81_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m81-recurrent-rgb") as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(np.asarray(observation["rgb"], dtype=np.uint8), memory)
            observation, _, terminated, truncated, _ = session.step(action)
            memory = policy.remember(memory, action, int(observation["prior_outcome"]))
            if terminated or truncated:
                break
        return session.replay(trace_path)
