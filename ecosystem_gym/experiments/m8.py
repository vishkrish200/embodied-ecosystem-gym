"""Frozen M8 protocol for sequential RGB recovery validity measurements.

The RGB controller here is deliberately a fixed colour-component and scan
baseline. It is useful because it exercises the public sequential action
surface, but it is not a learned perception or recovery result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..actions import ActionKind, ActionOutcome
from ..config import EcosystemConfig
from ..env import EcosystemEnv
from ..policies import skill_action
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


M8_PROTOCOL_VERSION = "m8-rgb-recovery-v1"
M8_EVALUATION_SEEDS = tuple(range(20))
M8_TASK_ID = "find_and_eat_perception"
M8_LAYOUT_ID = "m8_protocol"
M8_SCAN_SECTORS = ("north", "east", "south", "west")
M8_CAMERA_HALF_EXTENT = 0.52

M8_CONDITIONS: dict[str, dict[str, Any]] = {
    "reference": {"camera_control": "scan", "initial_scan_sector": "north"},
    # The target is deliberately outside the east camera sector at the initial pose.
    "occlusion": {"camera_control": "scan", "initial_scan_sector": "east"},
    "blocked_distractor": {
        "camera_control": "scan",
        "initial_scan_sector": "north",
        "blocked_distractor": True,
        "distractor_xy": [0.0, 0.32],
    },
    "relocation": {"camera_control": "scan", "initial_scan_sector": "north", "disturbance_step": 1},
    "unseen_geometry": {
        "camera_control": "scan",
        "initial_scan_sector": "north",
        "geometry_variant": "unseen_block",
    },
    "camera_pose": {"camera_control": "scan", "initial_scan_sector": "south"},
}


def m8_config() -> EcosystemConfig:
    return EcosystemConfig(observation_mode="rgb", max_episode_steps=24)


def _food_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    red, green, blue = (image[..., channel] for channel in range(3))
    warm = (red > 70) & (blue < 100) & (red > green * 1.35)
    bright_blue = (blue > 170) & (blue > green * 1.90) & (green < 115) & (red < 90)
    shadowed_blue = (red > 65) & (red < 110) & (blue > 180) & (blue > green * 1.30) & (green < 180)
    purple = (red > 80) & (blue > 120) & (green < 100)
    toy = (red > 140) & (green > 110) & (blue < 130)
    return (warm | bright_blue | shadowed_blue | purple) & ~toy


def _agent_image_centre(rgb: np.ndarray) -> tuple[float, float] | None:
    image = np.asarray(rgb, dtype=np.uint8)
    red, green, blue = (image[..., channel] for channel in range(3))
    agent = (blue > 120) & (green > 70) & (red < 100) & (blue > green * 1.2)
    ys, xs = np.nonzero(agent)
    if len(xs) < 3:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def rgb_food_components(rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return food-like component offsets relative to the agent from RGB only."""

    mask = _food_mask(rgb)
    height, width = mask.shape
    agent_centre = _agent_image_centre(rgb)
    if agent_centre is None:
        return ()
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not seen[next_y, next_x]:
                    seen[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(pixels) < 3:
            continue
        ys, xs = zip(*pixels, strict=True)
        # The agent's cyan body is visible in every narrow camera sector. Using
        # it as the local image origin makes the adapter camera-pose agnostic:
        # no reset-provided heading or global position is required.
        image_offset = np.asarray(
            (
                (float(np.mean(xs)) - agent_centre[0]) / max(width - 1, 1) * 2.0 * M8_CAMERA_HALF_EXTENT,
                -(float(np.mean(ys)) - agent_centre[1]) / max(height - 1, 1) * 2.0 * M8_CAMERA_HALF_EXTENT,
            ),
            dtype=np.float32,
        )
        components.append(image_offset)
    return tuple(sorted(components, key=lambda offset: float(np.linalg.norm(offset))))


@dataclass(slots=True)
class RgbScanMemory:
    """Policy-owned state derived only from the policy's own actions and outcomes."""

    sector: str = "north"
    scans: int = 0
    awaiting_pickup: bool = False
    blocked_attempts: int = 0
    previous_action: ActionKind | None = None


class FixedRgbScanRecoveryPolicy:
    """Fixed RGB component/scan baseline, deliberately never described as learned."""

    def reset(self) -> RgbScanMemory:
        return RgbScanMemory()

    def act(self, observation: dict[str, Any], memory: RgbScanMemory) -> dict[str, np.ndarray | int]:
        """Choose from RGB/public observations and caller-owned memory only."""

        if bool(observation["holding_food"]):
            memory.previous_action = ActionKind.CONSUME
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        prior_outcome = ActionOutcome(list(ActionOutcome)[int(observation["prior_outcome"])])
        if memory.awaiting_pickup:
            memory.awaiting_pickup = False
            memory.previous_action = ActionKind.PICK_UP_RELATIVE
            return skill_action(ActionKind.PICK_UP_RELATIVE, np.zeros(2, dtype=np.float32), 0.1)
        if prior_outcome is ActionOutcome.BLOCKED:
            memory.blocked_attempts += 1
        components = rgb_food_components(np.asarray(observation["rgb"], dtype=np.uint8))
        if not components:
            memory.sector = M8_SCAN_SECTORS[(M8_SCAN_SECTORS.index(memory.sector) + 1) % len(M8_SCAN_SECTORS)]
            memory.scans += 1
            memory.previous_action = ActionKind.SCAN
            return skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1)
        target = components[min(memory.blocked_attempts, len(components) - 1)]
        distance = float(np.linalg.norm(target))
        if distance <= m8_config().pickup_radius:
            memory.previous_action = ActionKind.PICK_UP_RELATIVE
            return skill_action(ActionKind.PICK_UP_RELATIVE, target, 0.1)
        memory.awaiting_pickup = True
        memory.previous_action = ActionKind.WALK_RELATIVE
        return skill_action(ActionKind.WALK_RELATIVE, target, max(0.1, distance / m8_config().walk_speed_per_second))


@dataclass(frozen=True, slots=True)
class M8Episode:
    seed: int
    condition: str
    task_success: bool
    total_reward: float
    steps: int
    scan_actions: int
    blocked_outcomes: int
    disturbance_events: int
    post_disturbance_completion: bool


def wilson_interval(successes: int, episodes: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (proportion + z * z / (2.0 * episodes)) / denominator
    margin = z * np.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes)) / denominator
    return float(centre - margin), float(centre + margin)


def _m8_options(condition: str) -> dict[str, Any]:
    try:
        controls = M8_CONDITIONS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown M8 condition {condition!r}") from exc
    return {"task_id": M8_TASK_ID, "layout_id": M8_LAYOUT_ID, "food_variant": "red", **controls}


def _run_rgb_episode(policy: FixedRgbScanRecoveryPolicy, *, seed: int, condition: str) -> M8Episode:
    env = EcosystemEnv(m8_config())
    try:
        observation, _ = env.reset(seed=seed, options=_m8_options(condition))
        memory = policy.reset()
        total_reward = 0.0
        scans = blocked = disturbances = 0
        for step in range(1, env.config.max_episode_steps + 1):
            action = policy.act(observation, memory)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            scans += int(int(action["kind"]) == int(ActionKind.SCAN))
            blocked += int(info["outcome"] == ActionOutcome.BLOCKED.value)
            disturbances += int(info["disturbance"] is not None)
            if terminated or truncated:
                return M8Episode(
                    seed,
                    condition,
                    bool(info["task_success"]),
                    total_reward,
                    step,
                    scans,
                    blocked,
                    disturbances,
                    bool(info["post_disturbance_completion"]),
                )
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _run_oracle_episode(*, seed: int, condition: str) -> M8Episode:
    """Privileged upper bound. This is a ceiling, not an equal-input baseline."""

    env = EcosystemEnv(m8_config())
    try:
        observation, _ = env.reset(seed=seed, options=_m8_options(condition))
        total_reward = 0.0
        disturbances = 0
        for step in range(1, env.config.max_episode_steps + 1):
            if bool(observation["holding_food"]):
                action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
            elif env._distance_to_food() <= env.config.pickup_radius:
                action = skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
            else:
                target = env._food_xy().copy()
                action = skill_action(
                    ActionKind.WALK_TO,
                    target,
                    max(0.1, float(np.linalg.norm(target - env._agent_xy())) / env.config.walk_speed_per_second),
                )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            disturbances += int(info["disturbance"] is not None)
            if terminated or truncated:
                return M8Episode(
                    seed,
                    condition,
                    bool(info["task_success"]),
                    total_reward,
                    step,
                    0,
                    0,
                    disturbances,
                    bool(info["post_disturbance_completion"]),
                )
        raise AssertionError("environment did not terminate at its declared step limit")
    finally:
        env.close()


def _aggregate(episodes: list[M8Episode]) -> dict[str, object]:
    successes = sum(episode.task_success for episode in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "task_success_rate": successes / len(episodes),
        "task_success_wilson_95": list(wilson_interval(successes, len(episodes))),
        "mean_total_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([episode.steps for episode in episodes])),
        "scan_actions": sum(episode.scan_actions for episode in episodes),
        "blocked_outcomes": sum(episode.blocked_outcomes for episode in episodes),
        "disturbance_events": sum(episode.disturbance_events for episode in episodes),
        "post_disturbance_completions": sum(episode.post_disturbance_completion for episode in episodes),
    }


def m8_benchmark(*, seeds: tuple[int, ...] = M8_EVALUATION_SEEDS) -> dict[str, object]:
    if seeds != M8_EVALUATION_SEEDS:
        raise ValueError("M8 evaluation seeds are frozen; use M8_EVALUATION_SEEDS")
    rgb_policy = FixedRgbScanRecoveryPolicy()
    results: dict[str, dict[str, object]] = {}
    for condition in M8_CONDITIONS:
        rgb_episodes = [_run_rgb_episode(rgb_policy, seed=seed, condition=condition) for seed in seeds]
        ceiling_episodes = [_run_oracle_episode(seed=seed, condition=condition) for seed in seeds]
        results[condition] = {
            "fixed_rgb_scan_recovery_baseline": _aggregate(rgb_episodes),
            "state_oracle_ceiling": _aggregate(ceiling_episodes),
        }
    return {
        "schema_version": "0.8",
        "protocol_version": M8_PROTOCOL_VERSION,
        "environment_version": "0.5.0",
        "task_id": M8_TASK_ID,
        "evaluation_seeds": list(seeds),
        "conditions": M8_CONDITIONS,
        "policy_input": ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"],
        "baselines": {
            "fixed_rgb_scan_recovery_baseline": "fixed colour-component adapter with policy-owned scan/recovery memory; not learned perception or learned recovery",
            "state_oracle_ceiling": "privileged state-oracle controller; upper bound only, not an equal-input comparison",
        },
        "results": results,
        "limits": [
            "M7 colour adaptation and drive-bin behaviour cloning are retained only as prior scaffolding; this report makes no learned-perception or learned-recovery claim.",
            "The blocked distractor is a visible interaction guard under a kinematic skill executor, not an obstacle-planning or contact-physics task.",
            "The unseen geometry condition is a visual geometry shift. It does not establish collision-aware navigation or broad geometry generalization.",
            "Post-disturbance completion is action-agnostic: it records eventual task completion after a recorded relocation, not a preferred action name.",
        ],
    }


def write_m8_report(path: str | Path) -> dict[str, object]:
    report = m8_benchmark()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m8_viewer_demo(trace_path: str | Path, *, seed: int = 7) -> ReplayResult:
    """Record/replay one relocation rollout through the unchanged thin viewer."""

    policy = FixedRgbScanRecoveryPolicy()
    # The replay artifact combines initial camera-sector occlusion with the
    # fixed relocation, so it shows both scan and recovery transitions.
    options = {**_m8_options("relocation"), "initial_scan_sector": "east"}
    with ViewerSession(EcosystemEnv(m8_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m8-rgb-recovery") as session:
        observation, _ = session.reset(seed=seed, options=options)
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
