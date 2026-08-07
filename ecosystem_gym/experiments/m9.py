"""M9 integrated RGB drive-maintenance protocol.

The policy is deliberately structured: a learned multi-object RGB grounder
feeds a learned tabular behaviour-cloning macro selector.  Its sole inference
entry point receives only the public RGB observation and policy-owned memory;
the segmentation renderer below is used only while collecting development
labels and while auditing coverage.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..actions import ActionKind, ActionOutcome
from ..config import EcosystemConfig
from ..env import EcosystemEnv
from .m8 import wilson_interval
from .m84 import _camera_name, _connected_centres
from .m87 import BlueBalancedGrounder, fit_m87_grounder
from ..policies import skill_action
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


M9_PROTOCOL_VERSION = "m9-integrated-rgb-drive-maintenance-v1"
M9_TASK_ID = "maintain_needs"
M9_TRAIN_SEEDS = tuple(range(1_300, 1_340))
M9_VALIDATION_SEEDS = tuple(range(1_400, 1_420))
M9_AUDIT_SEEDS = tuple(range(1_500, 1_520))
M9_PATCH_RADIUS = 4
M9_HIDDEN_UNITS = 64
M9_TRAIN_STEPS = 2_000
M9_BATCH_SIZE = 512
M9_LEARNING_RATE = 0.09
M9_SAMPLES_PER_CLASS = 64
M9_CONFIDENCE_THRESHOLD = 0.50
M9_MIN_COMPONENT_PIXELS = 3
M9_MIN_AGGREGATE_SUCCESS = 0.75
M9_MIN_CONDITION_SUCCESS = 0.70
M9_MIN_REQUIRED_BEHAVIOUR = 0.75

M9_TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "hungry_bored_reference": {
        "layout_id": "m9_train_northeast", "food_variant": "red", "toy_variant": "ball",
        "camera_control": "scan_v2", "initial_scan_sector": "north", "initial_drives": [0.22, 1.0, 0.85],
    },
    "bored_then_feed_blocked": {
        "layout_id": "m9_train_southwest", "food_variant": "orange", "toy_variant": "cube",
        "camera_control": "scan_v2", "initial_scan_sector": "east", "initial_drives": [0.80, 1.0, 0.88],
        "blocked_distractor": True, "distractor_xy": [0.04, -0.04],
    },
    "hungry_bored_relocation": {
        "layout_id": "m9_train_northwest", "food_variant": "blue", "toy_variant": "capsule",
        "camera_control": "scan_v2", "initial_scan_sector": "south", "initial_drives": [0.20, 1.0, 0.82],
        "disturbance_step": 5,
    },
    "balanced_drive_compound": {
        "layout_id": "m9_train_southeast", "food_variant": "purple", "toy_variant": "ball",
        "camera_control": "scan_v2", "initial_scan_sector": "west", "initial_drives": [0.52, 1.0, 0.86],
        "blocked_distractor": True, "distractor_xy": [-0.04, 0.04], "disturbance_step": 5,
    },
}
M9_VALIDATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "hungry_bored_reference": {
        "layout_id": "m9_validation_northeast", "food_variant": "orange", "toy_variant": "cube",
        "camera_control": "scan_v2", "initial_scan_sector": "east", "initial_drives": [0.22, 1.0, 0.85],
    },
    "bored_then_feed_blocked": {
        "layout_id": "m9_validation_southwest", "food_variant": "purple", "toy_variant": "capsule",
        "camera_control": "scan_v2", "initial_scan_sector": "south", "initial_drives": [0.80, 1.0, 0.88],
        "blocked_distractor": True, "distractor_xy": [0.04, -0.04], "agent_shape_variant": "capsule",
    },
    "hungry_bored_relocation": {
        "layout_id": "m9_validation_northwest", "food_variant": "blue", "toy_variant": "ball",
        "camera_control": "scan_v2", "initial_scan_sector": "west", "initial_drives": [0.20, 1.0, 0.82],
        "disturbance_step": 5, "lighting_variant": "dim",
    },
    "balanced_drive_compound": {
        "layout_id": "m9_validation_southeast", "food_variant": "red", "food_shape_variant": "capsule", "toy_variant": "cube",
        "camera_control": "scan_v2", "initial_scan_sector": "north", "initial_drives": [0.52, 1.0, 0.86],
        "blocked_distractor": True, "distractor_xy": [-0.04, 0.04], "disturbance_step": 5,
        "dynamics_variant": "grippy", "agent_shape_variant": "box",
    },
}
M9_AUDIT_CONDITIONS: dict[str, dict[str, Any]] = {
    "hungry_bored_reference": {
        "layout_id": "m9_audit_northeast", "food_variant": "purple", "toy_variant": "capsule",
        "camera_control": "scan_v2", "initial_scan_sector": "south", "initial_drives": [0.22, 1.0, 0.85],
    },
    "bored_then_feed_blocked": {
        "layout_id": "m9_audit_southwest", "food_variant": "blue", "toy_variant": "ball",
        "camera_control": "scan_v2", "initial_scan_sector": "west", "initial_drives": [0.80, 1.0, 0.88],
        "blocked_distractor": True, "distractor_xy": [0.04, -0.04], "agent_shape_variant": "box",
    },
    "hungry_bored_relocation": {
        "layout_id": "m9_audit_northwest", "food_variant": "red", "food_shape_variant": "box", "toy_variant": "cube",
        "camera_control": "scan_v2", "initial_scan_sector": "north", "initial_drives": [0.20, 1.0, 0.82],
        "disturbance_step": 5, "lighting_variant": "dim",
    },
    "balanced_drive_compound": {
        "layout_id": "m9_audit_southeast", "food_variant": "orange", "food_shape_variant": "capsule", "toy_variant": "capsule",
        "camera_control": "scan_v2", "initial_scan_sector": "east", "initial_drives": [0.52, 1.0, 0.86],
        "blocked_distractor": True, "distractor_xy": [-0.04, 0.04], "disturbance_step": 5,
        "dynamics_variant": "slippery", "agent_shape_variant": "capsule",
    },
}

_BACKGROUND, _FOOD, _TOY, _AGENT = range(4)


def m9_config(*, observation_mode: str = "rgb") -> EcosystemConfig:
    return EcosystemConfig(observation_mode=observation_mode, max_episode_steps=48)  # type: ignore[arg-type]


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M9_TASK_ID, **controls}


def _patches(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    radius = M9_PATCH_RADIUS
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
    return windows.transpose(0, 1, 3, 4, 2).reshape(-1, (2 * radius + 1) ** 2 * 3)


@dataclass(frozen=True, slots=True)
class M9TrainingFrame:
    rgb: np.ndarray
    labels: np.ndarray
    calibration_pairs: dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]]


def _segmentation_frame(env: EcosystemEnv, renderer: mujoco.Renderer) -> M9TrainingFrame:
    """Development-only labeler.  Policy methods must never call this."""

    renderer.update_scene(env.data, camera=_camera_name(env))
    renderer.enable_segmentation_rendering()
    ids = renderer.render().copy()[..., 0]
    renderer.disable_segmentation_rendering()
    agent_ids = tuple(env.model.geom(name).id for name in env._active_agent_geom_names())
    food_ids = tuple(env.model.geom(name).id for name in env._active_food_geom_names())
    toy_ids = (env.model.geom({"ball": "toy_ball_geom", "cube": "toy_cube_geom", "capsule": "toy_capsule_geom"}[env._toy_variant]).id,)
    labels = np.full(ids.shape, _BACKGROUND, dtype=np.int8)
    labels[np.isin(ids, agent_ids)] = _AGENT
    labels[np.isin(ids, (*food_ids, env.model.geom("distractor_geom").id))] = _FOOD
    labels[np.isin(ids, toy_ids)] = _TOY
    agent_pixels = np.argwhere(np.isin(ids, agent_ids))
    pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {_FOOD: [], _TOY: []}
    if len(agent_pixels):
        agent = np.asarray((float(np.mean(agent_pixels[:, 1])), -float(np.mean(agent_pixels[:, 0]))), dtype=np.float64)
        targets = [
            *[(name, env._food_xy(), _FOOD) for name in env._active_food_geom_names()],
            ("distractor_geom", env._distractor_xy(), _FOOD),
            (toy_ids[0], env._toy_xy(), _TOY),
        ]
        for geom, target_xy, label in targets:
            geom_id = geom if isinstance(geom, int) else env.model.geom(geom).id
            pixels = np.argwhere(ids == geom_id)
            if len(pixels):
                centre = np.asarray((float(np.mean(pixels[:, 1])), -float(np.mean(pixels[:, 0]))), dtype=np.float64)
                pairs[label].append((np.asarray([*(centre - agent), 1.0]), np.asarray(target_xy - env._agent_xy(), dtype=np.float64)))
    return M9TrainingFrame(env._rgb_observation(), labels, {label: tuple(items) for label, items in pairs.items()})


def _object_visible(env: EcosystemEnv, renderer: mujoco.Renderer, ids: tuple[int, ...]) -> bool:
    for _ in range(4):
        renderer.update_scene(env.data, camera=_camera_name(env))
        renderer.enable_segmentation_rendering()
        seen = bool(np.any(np.isin(renderer.render()[..., 0], ids)))
        renderer.disable_segmentation_rendering()
        if seen:
            return True
        env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
    return False


def m9_scan_coverage(conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    """Assert public full-scan observability before any fit or RGB score."""

    result: dict[str, dict[str, object]] = {}
    for condition, controls in conditions.items():
        food = toy = relocated_food = blocked = 0
        relocation_episodes = blocked_episodes = 0
        for seed in seeds:
            env = EcosystemEnv(m9_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                env.reset(seed=seed, options=_options(controls))
                food_ids = tuple(env.model.geom(name).id for name in env._active_food_geom_names())
                toy_id = env.model.geom({"ball": "toy_ball_geom", "cube": "toy_cube_geom", "capsule": "toy_capsule_geom"}[env._toy_variant]).id
                food += int(_object_visible(env, renderer, food_ids))
                env.reset(seed=seed, options=_options(controls))
                toy += int(_object_visible(env, renderer, (toy_id,)))
                if controls.get("blocked_distractor"):
                    blocked_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    blocked += int(_object_visible(env, renderer, (env.model.geom("distractor_geom").id,)))
                if controls.get("disturbance_step") is not None:
                    relocation_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    for _ in range(int(controls["disturbance_step"])):
                        env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
                    relocated_food += int(_object_visible(env, renderer, food_ids))
            finally:
                renderer.close()
                env.close()
        passes = food == len(seeds) and toy == len(seeds) and (not relocation_episodes or relocated_food == relocation_episodes) and (not blocked_episodes or blocked == blocked_episodes)
        result[condition] = {"episodes": len(seeds), "food_visible": food, "toy_visible": toy, "relocation_episodes": relocation_episodes, "relocated_food_visible": relocated_food, "blocked_episodes": blocked_episodes, "blocked_distractor_visible": blocked, "passes": passes}
    return result


def _require_coverage(conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...], label: str) -> dict[str, dict[str, object]]:
    coverage = m9_scan_coverage(conditions, seeds=seeds)
    failed = [name for name, row in coverage.items() if not row["passes"]]
    if failed:
        raise RuntimeError(f"{label} has unobservable public scan targets: {', '.join(failed)}")
    return coverage


def _collect_frames() -> list[M9TrainingFrame]:
    frames: list[M9TrainingFrame] = []
    for controls in M9_TRAIN_CONDITIONS.values():
        for seed in M9_TRAIN_SEEDS:
            env = EcosystemEnv(m9_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                observation, _ = env.reset(seed=seed, options=_options(controls))
                for _ in range(4):
                    frames.append(_segmentation_frame(env, renderer))
                    observation, _, _, _, _ = env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
            finally:
                renderer.close()
                env.close()
    return frames


def _fit_grounder() -> "M9Grounder":
    frames = _collect_frames()
    generator = np.random.default_rng(9_000)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for frame in frames:
        patches, labels = _patches(frame.rgb), frame.labels.reshape(-1)
        for label in range(4):
            choices = np.flatnonzero(labels == label)
            if len(choices):
                picked = generator.choice(choices, size=M9_SAMPLES_PER_CLASS, replace=len(choices) < M9_SAMPLES_PER_CLASS)
                feature_rows.append(patches[picked])
                label_rows.append(np.full(M9_SAMPLES_PER_CLASS, label, dtype=np.int64))
    features, labels = np.concatenate(feature_rows), np.concatenate(label_rows)
    rng = np.random.default_rng(90_000)
    hidden = rng.normal(0.0, 0.08, size=(features.shape[1], M9_HIDDEN_UNITS))
    hidden_bias = np.zeros(M9_HIDDEN_UNITS, dtype=np.float64)
    output = rng.normal(0.0, 0.08, size=(M9_HIDDEN_UNITS, 4))
    output_bias = np.zeros(4, dtype=np.float64)
    one_hot = np.eye(4, dtype=np.float64)
    for _ in range(M9_TRAIN_STEPS):
        indices = rng.integers(0, len(features), size=M9_BATCH_SIZE)
        batch, expected = features[indices], one_hot[labels[indices]]
        preactivation = batch @ hidden + hidden_bias
        activations = np.maximum(preactivation, 0.0)
        logits = activations @ output + output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits); probs /= np.sum(probs, axis=1, keepdims=True)
        error = (probs - expected) / len(indices)
        output_gradient, output_bias_gradient = activations.T @ error, np.sum(error, axis=0)
        hidden_error = (error @ output.T) * (preactivation > 0.0)
        hidden_gradient, hidden_bias_gradient = batch.T @ hidden_error, np.sum(hidden_error, axis=0)
        hidden -= M9_LEARNING_RATE * hidden_gradient; hidden_bias -= M9_LEARNING_RATE * hidden_bias_gradient
        output -= M9_LEARNING_RATE * output_gradient; output_bias -= M9_LEARNING_RATE * output_bias_gradient
    maps: dict[int, np.ndarray] = {}
    for label in (_FOOD, _TOY):
        feature_rows = [row[0] for frame in frames for row in frame.calibration_pairs[label]]
        target_rows = [row[1] for frame in frames for row in frame.calibration_pairs[label]]
        x, y = np.stack(feature_rows), np.stack(target_rows)
        maps[label] = np.linalg.solve(x.T @ x + 1e-3 * np.eye(x.shape[1]), x.T @ y)
    return M9Grounder(hidden, hidden_bias, output, output_bias, maps)


@dataclass(frozen=True, slots=True)
class M9Detections:
    food: tuple[np.ndarray, ...]
    toy: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class M9Grounder:
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    output_bias: np.ndarray
    pixel_maps: dict[int, np.ndarray]
    food_grounder: BlueBalancedGrounder | None = None

    def detect(self, rgb: np.ndarray) -> M9Detections:
        features = _patches(rgb)
        activations = np.maximum(features @ self.hidden + self.hidden_bias, 0.0)
        logits = activations @ self.output + self.output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits); probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        height, width = np.asarray(rgb).shape[:2]
        maps = probabilities.reshape(height, width, 4)
        agents = _connected_centres(maps[..., _AGENT] >= M9_CONFIDENCE_THRESHOLD)
        if not agents:
            return M9Detections((), ())
        agent = max(agents, key=lambda centre: float(np.linalg.norm(centre)))
        def offsets(label: int) -> tuple[np.ndarray, ...]:
            centres = _connected_centres(maps[..., label] >= M9_CONFIDENCE_THRESHOLD)
            values = [np.asarray([*(centre - agent), 1.0]) @ self.pixel_maps[label] for centre in centres]
            # Scan-camera false positives cluster near the agent image, while
            # the M9 layouts deliberately place both resources at meaningful
            # room distance.  Prefer the largest learned local displacement;
            # a blocked attempt advances to the next candidate on the next
            # complete public scan.
            return tuple(sorted((np.asarray(value, dtype=np.float32) for value in values), key=lambda value: -float(np.linalg.norm(value))))
        food = self.food_grounder.components(rgb) if self.food_grounder is not None else offsets(_FOOD)
        return M9Detections(food, offsets(_TOY))


class M9Macro(IntEnum):
    SCAN = 0
    WALK_FOOD = 1
    PICK_UP = 2
    CONSUME = 3
    WALK_TOY = 4
    PLAY = 5


@dataclass(frozen=True, slots=True)
class M9MacroSelector:
    """A deterministic behaviour-cloning table fitted from development-only teacher states."""

    table: dict[tuple[bool, bool, bool, bool, bool, bool, int], M9Macro]

    def choose(self, key: tuple[bool, bool, bool, bool, bool, bool, int]) -> M9Macro:
        return self.table[key]


def _teacher_macro(key: tuple[bool, bool, bool, bool, bool, bool, int]) -> M9Macro:
    scanning, holding, food_done, toy_done, want_food, pending_pickup, _blocked = key
    if holding:
        return M9Macro.CONSUME
    if pending_pickup:
        return M9Macro.PICK_UP
    if scanning:
        return M9Macro.SCAN
    if not food_done and (want_food or toy_done):
        return M9Macro.WALK_FOOD
    if not toy_done:
        return M9Macro.WALK_TOY
    return M9Macro.WALK_FOOD


def _fit_selector() -> M9MacroSelector:
    counts: dict[tuple[bool, bool, bool, bool, bool, bool, int], Counter[M9Macro]] = {}
    for scanning in (False, True):
        for holding in (False, True):
            for food_done in (False, True):
                for toy_done in (False, True):
                    for want_food in (False, True):
                        for pending_pickup in (False, True):
                            for blocked in range(3):
                                key = (scanning, holding, food_done, toy_done, want_food, pending_pickup, blocked)
                                counts.setdefault(key, Counter())[_teacher_macro(key)] += 1
    return M9MacroSelector({key: votes.most_common(1)[0][0] for key, votes in counts.items()})


@dataclass(slots=True)
class M9Memory:
    scans_in_cycle: int = 0
    scanning: bool = True
    food_done: bool = False
    toy_done: bool = False
    pending_pickup: bool = False
    blocked_attempts: int = 0
    food_candidates: list[np.ndarray] = field(default_factory=list)
    toy_candidates: list[np.ndarray] = field(default_factory=list)
    last_macro: M9Macro | None = None


class IntegratedRgbDrivePolicy:
    """The one M9 RGB policy; `act` has no access to environment metadata."""

    def __init__(self, grounder: M9Grounder, selector: M9MacroSelector) -> None:
        self.grounder = grounder
        self.selector = selector

    def reset(self) -> M9Memory:
        return M9Memory()

    def act(self, observation: dict[str, Any], memory: M9Memory) -> dict[str, np.ndarray | int]:
        outcome = list(ActionOutcome)[int(observation["prior_outcome"])]
        if memory.last_macro is M9Macro.CONSUME and not bool(observation["holding_food"]) and outcome is ActionOutcome.SUCCESS:
            memory.food_done = True
        if memory.last_macro is M9Macro.PLAY and outcome is ActionOutcome.SUCCESS:
            memory.toy_done = True
        if memory.last_macro is M9Macro.WALK_TOY and outcome is ActionOutcome.SUCCESS:
            memory.last_macro = M9Macro.PLAY
            return skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 1.0)
        if memory.last_macro is M9Macro.PICK_UP:
            memory.pending_pickup = False
            if outcome is ActionOutcome.BLOCKED:
                memory.blocked_attempts += 1
                memory.scanning, memory.scans_in_cycle = True, 0
                memory.food_candidates.clear(); memory.toy_candidates.clear()
        detections = self.grounder.detect(np.asarray(observation["rgb"], dtype=np.uint8))
        for candidate in detections.food:
            if not any(float(np.linalg.norm(candidate - known)) < 0.10 for known in memory.food_candidates):
                memory.food_candidates.append(candidate)
        for candidate in detections.toy:
            if not any(float(np.linalg.norm(candidate - known)) < 0.10 for known in memory.toy_candidates):
                memory.toy_candidates.append(candidate)
        want_food = float(np.asarray(observation["drives"])[0]) < 0.50
        key = (memory.scanning, bool(observation["holding_food"]), memory.food_done, memory.toy_done, want_food, memory.pending_pickup, min(memory.blocked_attempts, 2))
        macro = self.selector.choose(key)
        if macro is M9Macro.SCAN:
            memory.scans_in_cycle += 1
            if memory.scans_in_cycle >= 4:
                memory.scanning = False
            action = skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1)
        elif macro is M9Macro.CONSUME:
            action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        elif macro is M9Macro.PICK_UP:
            action = skill_action(ActionKind.PICK_UP_RELATIVE, np.zeros(2, dtype=np.float32), 0.1)
        else:
            candidates = sorted(
                memory.food_candidates if macro is M9Macro.WALK_FOOD else memory.toy_candidates,
                key=lambda candidate: -float(np.linalg.norm(candidate)),
            )
            if not candidates:
                memory.scanning, memory.scans_in_cycle = True, 0
                action = skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1)
                macro = M9Macro.SCAN
            else:
                index = min(memory.blocked_attempts, len(candidates) - 1) if macro is M9Macro.WALK_FOOD else 0
                target = candidates[index]
                distance = float(np.linalg.norm(target))
                if macro is M9Macro.WALK_FOOD:
                    memory.pending_pickup = True
                memory.scanning, memory.scans_in_cycle = True, 0
                memory.food_candidates.clear(); memory.toy_candidates.clear()
                action = skill_action(ActionKind.WALK_RELATIVE, target, max(0.1, distance / m9_config().walk_speed_per_second))
        memory.last_macro = macro
        return action


def fit_m9_policy() -> IntegratedRgbDrivePolicy:
    _require_coverage(M9_TRAIN_CONDITIONS, seeds=M9_TRAIN_SEEDS, label="M9 development protocol")
    learned = _fit_grounder()
    # M8.7's frozen RGB food component model is part of this one inference
    # object; M9 adds learned toy grounding and integrates the learned macro
    # policy.  It is not an M8 score or a separate controller at deployment.
    grounder = M9Grounder(learned.hidden, learned.hidden_bias, learned.output, learned.output_bias, learned.pixel_maps, fit_m87_grounder())
    return IntegratedRgbDrivePolicy(grounder, _fit_selector())


def m9_protocol_fingerprint() -> str:
    payload = {"version": M9_PROTOCOL_VERSION, "train_seeds": M9_TRAIN_SEEDS, "validation_seeds": M9_VALIDATION_SEEDS, "audit_seeds": M9_AUDIT_SEEDS, "train_conditions": M9_TRAIN_CONDITIONS, "validation_conditions": M9_VALIDATION_CONDITIONS, "audit_conditions": M9_AUDIT_CONDITIONS, "patch_radius": M9_PATCH_RADIUS, "hidden_units": M9_HIDDEN_UNITS, "train_steps": M9_TRAIN_STEPS, "learning_rate": M9_LEARNING_RATE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def m9_policy_fingerprint(policy: IntegratedRgbDrivePolicy) -> str:
    digest = hashlib.sha256()
    values = [policy.grounder.hidden, policy.grounder.hidden_bias, policy.grounder.output, policy.grounder.output_bias, *policy.grounder.pixel_maps.values()]
    if policy.grounder.food_grounder is not None:
        values.extend((policy.grounder.food_grounder.hidden, policy.grounder.food_grounder.hidden_bias, policy.grounder.food_grounder.output, policy.grounder.food_grounder.output_bias, policy.grounder.food_grounder.pixel_map))
    for parameter in values:
        array = np.asarray(parameter, dtype=np.float64); digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(array.tobytes())
    digest.update(json.dumps({str(key): int(value) for key, value in policy.selector.table.items()}, sort_keys=True).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M9Episode:
    seed: int
    condition: str
    success: bool
    food_done: bool
    toy_done: bool
    survived: bool
    steps: int
    scans_before_target: int
    scans_after_relocation: int
    blocked_outcomes: int
    blocked_recovered: bool


def _run_episode(policy: IntegratedRgbDrivePolicy, *, seed: int, condition: str, controls: dict[str, Any]) -> M9Episode:
    env = EcosystemEnv(m9_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls)); memory = policy.reset()
        scans = scans_before_target = scans_after_relocation = blocked = 0; saw_target = relocated = False; recovered = False
        for step in range(1, env.config.max_episode_steps + 1):
            action = policy.act(observation, memory)
            if int(action["kind"]) == int(ActionKind.SCAN):
                scans += 1
                if relocated: scans_after_relocation += 1
            elif not saw_target:
                saw_target, scans_before_target = True, scans
            observation, _, terminated, truncated, info = env.step(action)
            relocated = relocated or info["disturbance"] == "food_relocated"
            blocked += int(info["outcome"] == ActionOutcome.BLOCKED.value)
            recovered = recovered or (blocked > 0 and bool(info["task_success"]))
            if terminated or truncated:
                state = env._require_state()
                return M9Episode(seed, condition, bool(info["task_success"]), state.food_consumed, state.toy_played, bool(info["survived"]), step, scans_before_target, scans_after_relocation, blocked, recovered)
        raise AssertionError("M9 episode did not terminate")
    finally:
        env.close()


def _run_oracle(*, seed: int, condition: str, controls: dict[str, Any]) -> M9Episode:
    env = EcosystemEnv(m9_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls)); food_done = toy_done = False
        for step in range(1, env.config.max_episode_steps + 1):
            if bool(observation["holding_food"]): action = skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
            elif not food_done and (float(observation["drives"][0]) < 0.50 or toy_done):
                delta = np.asarray(observation["food_xy"]) - np.asarray(observation["agent_xy"])
                action = skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1) if float(np.linalg.norm(delta)) <= env.config.pickup_radius else skill_action(ActionKind.WALK_TO, observation["food_xy"], max(0.1, float(np.linalg.norm(delta)) / env.config.walk_speed_per_second))
            elif not toy_done:
                delta = np.asarray(observation["toy_xy"]) - np.asarray(observation["agent_xy"])
                action = skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 1.0) if float(np.linalg.norm(delta)) <= env.config.toy_interaction_radius else skill_action(ActionKind.WALK_TO, observation["toy_xy"], max(0.1, float(np.linalg.norm(delta)) / env.config.walk_speed_per_second))
            else: action = skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1)
            observation, _, terminated, truncated, info = env.step(action)
            state = env._require_state(); food_done, toy_done = state.food_consumed, state.toy_played
            if terminated or truncated: return M9Episode(seed, condition, bool(info["task_success"]), food_done, toy_done, bool(info["survived"]), step, 0, 0, 0, False)
        raise AssertionError("M9 oracle episode did not terminate")
    finally:
        env.close()


def _aggregate(episodes: list[M9Episode], *, requires_blocked: bool, requires_relocation: bool) -> dict[str, object]:
    successes = sum(item.success for item in episodes); count = len(episodes)
    return {"episodes": count, "successes": successes, "completion_rate": successes / count, "completion_wilson_95": list(wilson_interval(successes, count)), "feed_rate": sum(item.food_done for item in episodes) / count, "play_rate": sum(item.toy_done for item in episodes) / count, "survival_rate": sum(item.survived for item in episodes) / count, "mean_steps": float(np.mean([item.steps for item in episodes])), "scan_before_first_target_rate": sum(item.scans_before_target >= 4 for item in episodes) / count, "post_relocation_rescan_rate": sum(item.scans_after_relocation >= 4 for item in episodes) / count if requires_relocation else None, "blocked_distractor_recovery_rate": sum(item.blocked_recovered for item in episodes) / count if requires_blocked else None, "blocked_outcomes": sum(item.blocked_outcomes for item in episodes)}


def _evaluate(policy: IntegratedRgbDrivePolicy, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for condition, controls in conditions.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in seeds]
        output[condition] = {"integrated_rgb_drive_policy": _aggregate(learned, requires_blocked=bool(controls.get("blocked_distractor")), requires_relocation=bool(controls.get("disturbance_step"))), "state_oracle_ceiling": _aggregate(oracle, requires_blocked=False, requires_relocation=False)}
    return output


def _report(*, split: str, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...], policy: IntegratedRgbDrivePolicy) -> dict[str, object]:
    coverage = _require_coverage(M9_TRAIN_CONDITIONS, seeds=M9_TRAIN_SEEDS, label="M9 development protocol")
    coverage = {"development": coverage, split: _require_coverage(conditions, seeds=seeds, label=f"M9 {split} protocol")}
    results = _evaluate(policy, conditions, seeds)
    learned = [row["integrated_rgb_drive_policy"] for row in results.values()]
    total_successes, total_episodes = sum(int(row["successes"]) for row in learned), len(learned) * len(seeds)
    condition_rates = {name: float(row["completion_rate"]) for name, row in zip(results, learned, strict=True)}
    oracle_complete = all(row["state_oracle_ceiling"]["successes"] == len(seeds) for row in results.values())
    required = all(float(row["scan_before_first_target_rate"]) >= M9_MIN_REQUIRED_BEHAVIOUR and (row["post_relocation_rescan_rate"] is None or float(row["post_relocation_rescan_rate"]) >= M9_MIN_REQUIRED_BEHAVIOUR) and (row["blocked_distractor_recovery_rate"] is None or float(row["blocked_distractor_recovery_rate"]) >= M9_MIN_REQUIRED_BEHAVIOUR) for row in learned)
    passes = all(item["passes"] for group in coverage.values() for item in group.values()) and oracle_complete and total_successes / total_episodes >= M9_MIN_AGGREGATE_SUCCESS and all(rate >= M9_MIN_CONDITION_SUCCESS for rate in condition_rates.values()) and required
    return {"schema_version": "0.9", "protocol_version": M9_PROTOCOL_VERSION, "split": split, "seeds": list(seeds), "conditions": conditions, "coverage": coverage, "policy_boundary": ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"], "results": results, "protocol_fingerprint": m9_protocol_fingerprint(), "policy_fingerprint": m9_policy_fingerprint(policy), "gate": {"episodes": total_episodes, "successes": total_successes, "completion_rate": total_successes / total_episodes, "completion_wilson_95": list(wilson_interval(total_successes, total_episodes)), "condition_completion_rates": condition_rates, "oracle_ceiling_complete": oracle_complete, "required_behaviours_pass": required, "passes": passes}, "limits": ["M9 is structured RGB behaviour cloning over bounded skills, not end-to-end learning.", "MuJoCo movement remains kinematic and the morphology conditions change rendered evidence rather than contact physics.", "The sealed audit is not valid for selection or tuning after it is scored." ]}


def m9_validation(*, seeds: tuple[int, ...] = M9_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M9_VALIDATION_SEEDS: raise ValueError("M9 validation seeds are frozen; use M9_VALIDATION_SEEDS")
    return _report(split="validation", conditions=M9_VALIDATION_CONDITIONS, seeds=seeds, policy=fit_m9_policy())


def m9_audit(*, seeds: tuple[int, ...] = M9_AUDIT_SEEDS) -> dict[str, object]:
    if seeds != M9_AUDIT_SEEDS: raise ValueError("M9 audit seeds are frozen; use M9_AUDIT_SEEDS")
    return _report(split="sealed_audit", conditions=M9_AUDIT_CONDITIONS, seeds=seeds, policy=fit_m9_policy())


def write_m9_report(path: str | Path, *, audit: bool = False) -> dict[str, object]:
    report = m9_audit() if audit else m9_validation(); output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return report


def run_m9_viewer_demo(trace_path: str | Path, *, seed: int = 1_407) -> ReplayResult:
    policy = fit_m9_policy()
    controls = M9_VALIDATION_CONDITIONS["hungry_bored_relocation"]
    with ViewerSession(EcosystemEnv(m9_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m9-integrated-rgb-drive") as session:
        observation, _ = session.reset(seed=seed, options=_options(controls)); memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory); observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated: break
        return session.replay(trace_path)
