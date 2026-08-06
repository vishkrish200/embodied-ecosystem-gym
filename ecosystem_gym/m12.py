"""M12 learned persistent-maintenance successor.

M12 fits one deterministic structured RGB policy from development-only oracle
macro labels on the public M10 observation boundary. The deployed policy sees
only RGB, drives, holding-food, prior outcome, and policy-owned memory.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .actions import ActionKind, ActionOutcome
from .config import EcosystemConfig
from .env import EcosystemEnv
from .m8 import wilson_interval
from .m10 import (
    M10_TRAIN_SEEDS,
    M10_VALIDATION_CONDITIONS,
    M10_VALIDATION_SEEDS,
    m10_config,
    m10_protocol_fingerprint,
    m10_validation,
)
from .m11 import M11_SAFE_DRIVE_BANDS, m11_baseline
from .policies import skill_action
from .trajectory import ReplayResult
from .viewer import ViewerSession


M12_PROTOCOL_VERSION = "m12-learned-persistent-maintenance-successor-v11"
M12_TASK_ID = "persistent_maintenance"
M12_TRAIN_SEEDS = M10_TRAIN_SEEDS
M12_VALIDATION_SEEDS = M10_VALIDATION_SEEDS
M12_PATCH_RADIUS = 4
M12_HIDDEN_UNITS = 64
M12_SELECTOR_HIDDEN_UNITS = 48
M12_TRAIN_STEPS = 2_000
M12_BATCH_SIZE = 512
M12_LEARNING_RATE = 0.09
M12_SAMPLES_PER_CLASS = 64
M12_CONFIDENCE_THRESHOLD = 0.50
M12_MIN_COMPONENT_PIXELS = 8
M12_BOOTSTRAP_DRAWS = 10_000
M12_BOOTSTRAP_SEED = 20_260_806

M12_TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": {
        "layout_id": "m10_train_northeast",
        "food_variant": "orange",
        "toy_variant": "ball",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "renewal_and_morphology": {
        "layout_id": "m10_train_southwest",
        "food_variant": "purple",
        "food_shape_variant": "capsule",
        "toy_variant": "cube",
        "agent_shape_variant": "capsule",
        "camera_control": "scan_v2",
        "initial_scan_sector": "south",
    },
    "event_relocation": {
        "layout_id": "m10_train_southwest",
        "food_variant": "purple",
        "toy_variant": "cube",
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "event_relocation_on_first_pickup": True,
    },
    "compound": {
        "layout_id": "m10_train_northeast",
        "food_variant": "blue",
        "food_shape_variant": "capsule",
        "toy_variant": "cube",
        "agent_shape_variant": "box",
        "dynamics_variant": "grippy",
        "blocked_distractor": True,
        "distractor_xy": [-0.04, 0.04],
        "camera_control": "scan_v2",
        "initial_scan_sector": "north",
        "event_relocation_on_first_pickup": True,
    },
}

M12_VALIDATION_CONDITIONS = M10_VALIDATION_CONDITIONS
M12_POLICY_BOUNDARY = ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"]

_BACKGROUND, _FOOD, _TOY, _REST, _AGENT = range(5)


def _connected_components(mask: np.ndarray, scores: np.ndarray) -> tuple[tuple[np.ndarray, int, float], ...]:
    """Return component geometry and confidence while discarding pixel noise."""

    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[np.ndarray, int, float]] = []
    height, width = mask.shape
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
        if len(pixels) >= M12_MIN_COMPONENT_PIXELS:
            ys, xs = zip(*pixels, strict=True)
            centre = np.asarray((float(np.mean(xs)), -float(np.mean(ys))), dtype=np.float64)
            components.append((centre, len(pixels), float(np.mean(scores[ys, xs]))))
    return tuple(components)


def m12_config(*, observation_mode: str = "rgb") -> EcosystemConfig:
    return m10_config(observation_mode=observation_mode)


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M12_TASK_ID, **controls}


def _camera_name(env: EcosystemEnv) -> str:
    return f"agent_cam_scan_v2_{env._scan_sector}"


def _patches(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    radius = M12_PATCH_RADIUS
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
    return windows.transpose(0, 1, 3, 4, 2).reshape(-1, (2 * radius + 1) ** 2 * 3)


@dataclass(frozen=True, slots=True)
class M12TrainingFrame:
    rgb: np.ndarray
    labels: np.ndarray
    calibration_pairs: dict[int, tuple[tuple[np.ndarray, np.ndarray], ...]]


def _segmentation_frame(env: EcosystemEnv, renderer: mujoco.Renderer) -> M12TrainingFrame:
    renderer.update_scene(env.data, camera=_camera_name(env))
    renderer.enable_segmentation_rendering()
    ids = renderer.render().copy()[..., 0]
    renderer.disable_segmentation_rendering()
    agent_ids = tuple(env.model.geom(name).id for name in env._active_agent_geom_names())
    food_ids = tuple(env.model.geom(name).id for name in env._active_food_geom_names())
    toy_id = env.model.geom({"ball": "toy_ball_geom", "cube": "toy_cube_geom", "capsule": "toy_capsule_geom"}[env._toy_variant]).id
    rest_id = env.model.geom("rest_geom").id
    distractor_id = env.model.geom("distractor_geom").id
    labels = np.full(ids.shape, _BACKGROUND, dtype=np.int8)
    labels[np.isin(ids, agent_ids)] = _AGENT
    labels[np.isin(ids, (*food_ids, distractor_id))] = _FOOD
    labels[np.isin(ids, (toy_id,))] = _TOY
    labels[np.isin(ids, (rest_id,))] = _REST
    agent_pixels = np.argwhere(np.isin(ids, agent_ids))
    pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {_FOOD: [], _TOY: [], _REST: []}
    if len(agent_pixels):
        agent = np.asarray((float(np.mean(agent_pixels[:, 1])), -float(np.mean(agent_pixels[:, 0]))), dtype=np.float64)
        targets = [
            *[(name, env._food_xy(), _FOOD) for name in env._active_food_geom_names()],
            ("distractor_geom", env._distractor_xy(), _FOOD),
            (toy_id, env._toy_xy(), _TOY),
            ("rest_geom", env._rest_xy(), _REST),
        ]
        for geom, target_xy, label in targets:
            geom_id = geom if isinstance(geom, int) else env.model.geom(geom).id
            pixels = np.argwhere(ids == geom_id)
            if len(pixels):
                centre = np.asarray((float(np.mean(pixels[:, 1])), -float(np.mean(pixels[:, 0]))), dtype=np.float64)
                pairs[label].append((np.asarray([*(centre - agent), 1.0]), np.asarray(target_xy - env._agent_xy(), dtype=np.float64)))
    return M12TrainingFrame(env._rgb_observation(), labels, {label: tuple(items) for label, items in pairs.items()})


def _visible_in_public_scan(env: EcosystemEnv, renderer: mujoco.Renderer, geom_ids: tuple[int, ...]) -> bool:
    for _ in range(4):
        renderer.update_scene(env.data, camera=_camera_name(env))
        renderer.enable_segmentation_rendering()
        visible = bool(np.any(np.isin(renderer.render()[..., 0], geom_ids)))
        renderer.disable_segmentation_rendering()
        if visible:
            return True
        env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
    return False


def m12_scan_coverage(
    conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    # Reset applies all public condition variants to one model.  Its renderer
    # is therefore valid across the audit and avoids context churn on macOS.
    env = EcosystemEnv(m12_config())
    renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
    try:
        for name, controls in conditions.items():
            initial_food = replenished_food = toy = rest = relocated_food = distractor = 0
            relocation_episodes = distractor_episodes = 0
            for seed in seeds:
                env.reset(seed=seed, options=_options(controls))
                food_ids = tuple(env.model.geom(item).id for item in env._active_food_geom_names())
                toy_id = env.model.geom({"ball": "toy_ball_geom", "cube": "toy_cube_geom", "capsule": "toy_capsule_geom"}[env._toy_variant]).id
                initial_food += int(_visible_in_public_scan(env, renderer, food_ids))

                env.reset(seed=seed, options=_options(controls))
                toy += int(_visible_in_public_scan(env, renderer, (toy_id,)))
                env.reset(seed=seed, options=_options(controls))
                rest += int(_visible_in_public_scan(env, renderer, (env.model.geom("rest_geom").id,)))

                env.reset(seed=seed, options=_options(controls))
                state = env._require_state()
                state.food_available = False
                state.food_consumed = True
                state.food_respawn_remaining = 0.0
                env._set_food_visible(False)
                _, _, _, _, info = env.step(skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 0.1))
                if info["resource_event"] != "food_replenished":
                    raise AssertionError("M12 replenishment did not emit its public event")
                replenished_food += int(_visible_in_public_scan(env, renderer, food_ids))

                if controls.get("event_relocation_on_first_pickup"):
                    relocation_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    env._set_agent_xy(env._food_xy())
                    _, _, _, _, info = env.step(skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1))
                    if info["disturbance"] != "food_relocated" or info["outcome"] != ActionOutcome.BLOCKED.value:
                        raise AssertionError("M12 event relocation must force a stale failed pickup")
                    relocated_food += int(_visible_in_public_scan(env, renderer, food_ids))

                if controls.get("blocked_distractor"):
                    distractor_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    distractor += int(_visible_in_public_scan(env, renderer, (env.model.geom("distractor_geom").id,)))
            count = len(seeds)
            report[name] = {
                "episodes": count,
                "initial_food_visible": initial_food,
                "replenished_food_visible": replenished_food,
                "toy_visible": toy,
                "rest_visible": rest,
                "relocation_episodes": relocation_episodes,
                "relocated_food_visible": relocated_food,
                "distractor_episodes": distractor_episodes,
                "distractor_visible": distractor,
                "passes": (
                    initial_food == count
                    and replenished_food == count
                    and toy == count
                    and rest == count
                    and (not relocation_episodes or relocated_food == relocation_episodes)
                    and (not distractor_episodes or distractor == distractor_episodes)
                ),
            }
    finally:
        renderer.close()
        env.close()
    return report


def _require_coverage(conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...], label: str) -> dict[str, dict[str, object]]:
    coverage = m12_scan_coverage(conditions, seeds=seeds)
    failed = [name for name, row in coverage.items() if not row["passes"]]
    if failed:
        raise RuntimeError(f"{label} has unobservable public scan targets: {', '.join(failed)}")
    return coverage


def _collect_grounder_frames() -> list[M12TrainingFrame]:
    """Collect development-only scans from reset and post-walk-like poses.

    A camera-relative pixel displacement must remain useful after the policy
    has walked to food, toy, or rest.  Reset-only calibration made the map
    accurate for the first movement but systematically wrong thereafter.
    """

    frames: list[M12TrainingFrame] = []
    env = EcosystemEnv(m12_config())
    renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
    try:
        for controls in M12_TRAIN_CONDITIONS.values():
            for seed in M12_TRAIN_SEEDS:
                env.reset(seed=seed, options=_options(controls))
                poses = (
                    None,
                    env._food_xy() + np.asarray((-0.20, -0.20), dtype=np.float32),
                    env._toy_xy() + np.asarray((0.20, -0.20), dtype=np.float32),
                    env._rest_xy() + np.asarray((-0.20, 0.20), dtype=np.float32),
                )
                for pose in poses:
                    if pose is not None:
                        env._set_agent_xy(env._bounded_xy(pose))
                    for _ in range(4):
                        frames.append(_segmentation_frame(env, renderer))
                        env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
    finally:
        renderer.close()
        env.close()
    return frames


@dataclass(frozen=True, slots=True)
class M12Detections:
    food: tuple[np.ndarray, ...]
    toy: tuple[np.ndarray, ...]
    rest: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class M12Grounder:
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    output_bias: np.ndarray
    pixel_maps: dict[int, np.ndarray]

    def detect(self, rgb: np.ndarray) -> M12Detections:
        features = _patches(rgb)
        activations = np.maximum(features @ self.hidden + self.hidden_bias, 0.0)
        logits = activations @ self.output + self.output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        height, width = np.asarray(rgb).shape[:2]
        maps = probabilities.reshape(height, width, 5)
        agents = _connected_components(maps[..., _AGENT] >= M12_CONFIDENCE_THRESHOLD, maps[..., _AGENT])
        if not agents:
            return M12Detections((), (), ())
        agent = max(agents, key=lambda item: (item[1], item[2]))[0]

        def offsets(label: int) -> tuple[np.ndarray, ...]:
            components = _connected_components(maps[..., label] >= M12_CONFIDENCE_THRESHOLD, maps[..., label])
            values = [
                (np.asarray([*(centre - agent), 1.0]) @ self.pixel_maps[label], pixels * confidence)
                for centre, pixels, confidence in components
            ]
            return tuple(
                np.asarray(value, dtype=np.float32)
                for value, _ in sorted(values, key=lambda item: -item[1])
            )

        return M12Detections(offsets(_FOOD), offsets(_TOY), offsets(_REST))


def _fit_grounder() -> M12Grounder:
    frames = _collect_grounder_frames()
    generator = np.random.default_rng(12_000)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for frame in frames:
        patches, labels = _patches(frame.rgb), frame.labels.reshape(-1)
        for label in range(5):
            choices = np.flatnonzero(labels == label)
            if len(choices):
                picked = generator.choice(choices, size=M12_SAMPLES_PER_CLASS, replace=len(choices) < M12_SAMPLES_PER_CLASS)
                feature_rows.append(patches[picked])
                label_rows.append(np.full(M12_SAMPLES_PER_CLASS, label, dtype=np.int64))
    features, labels = np.concatenate(feature_rows), np.concatenate(label_rows)
    rng = np.random.default_rng(120_000)
    hidden = rng.normal(0.0, 0.08, size=(features.shape[1], M12_HIDDEN_UNITS))
    hidden_bias = np.zeros(M12_HIDDEN_UNITS, dtype=np.float64)
    output = rng.normal(0.0, 0.08, size=(M12_HIDDEN_UNITS, 5))
    output_bias = np.zeros(5, dtype=np.float64)
    one_hot = np.eye(5, dtype=np.float64)
    for _ in range(M12_TRAIN_STEPS):
        indices = rng.integers(0, len(features), size=M12_BATCH_SIZE)
        batch, expected = features[indices], one_hot[labels[indices]]
        preactivation = batch @ hidden + hidden_bias
        activations = np.maximum(preactivation, 0.0)
        logits = activations @ output + output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True)
        error = (probs - expected) / len(indices)
        output_gradient, output_bias_gradient = activations.T @ error, np.sum(error, axis=0)
        hidden_error = (error @ output.T) * (preactivation > 0.0)
        hidden_gradient, hidden_bias_gradient = batch.T @ hidden_error, np.sum(hidden_error, axis=0)
        hidden -= M12_LEARNING_RATE * hidden_gradient
        hidden_bias -= M12_LEARNING_RATE * hidden_bias_gradient
        output -= M12_LEARNING_RATE * output_gradient
        output_bias -= M12_LEARNING_RATE * output_bias_gradient
    maps: dict[int, np.ndarray] = {}
    for label in (_FOOD, _TOY, _REST):
        feature_rows = [row[0] for frame in frames for row in frame.calibration_pairs[label]]
        target_rows = [row[1] for frame in frames for row in frame.calibration_pairs[label]]
        x, y = np.stack(feature_rows), np.stack(target_rows)
        maps[label] = np.linalg.solve(x.T @ x + 1e-3 * np.eye(x.shape[1]), x.T @ y)
    return M12Grounder(hidden, hidden_bias, output, output_bias, maps)


class M12Macro(IntEnum):
    SCAN = 0
    WALK_FOOD = 1
    PICK_UP = 2
    CONSUME = 3
    WALK_TOY = 4
    PLAY = 5
    WALK_REST = 6
    REST = 7
    IDLE = 8


@dataclass(frozen=True, slots=True)
class M12Selector:
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    output_bias: np.ndarray

    def choose(self, features: np.ndarray, *, allowed: tuple[M12Macro, ...] | None = None) -> M12Macro:
        hidden = np.maximum(features @ self.hidden + self.hidden_bias, 0.0)
        logits = hidden @ self.output + self.output_bias
        if allowed is not None:
            allowed_indices = np.asarray([int(macro) for macro in allowed], dtype=np.int64)
            masked = np.full_like(logits, -np.inf)
            masked[allowed_indices] = logits[allowed_indices]
            logits = masked
        return M12Macro(int(np.argmax(logits)))


def _select_macro(required: M12Macro | None, selector: M12Selector, features: np.ndarray) -> M12Macro:
    if required is not None:
        return required
    return selector.choose(
        features,
        allowed=(M12Macro.SCAN, M12Macro.WALK_FOOD, M12Macro.WALK_TOY, M12Macro.WALK_REST, M12Macro.IDLE),
    )


@dataclass(slots=True)
class M12Memory:
    scans_in_cycle: int = 0
    required_scans: int = 4
    scanning: bool = True
    pending_pickup: bool = False
    pending_play: bool = False
    pending_rest: bool = False
    blocked_attempts: int = 0
    feed_done: bool = False
    play_done: bool = False
    rest_done: bool = False
    completed_cycles: int = 0
    food_candidates: list[np.ndarray] = field(default_factory=list)
    toy_candidates: list[np.ndarray] = field(default_factory=list)
    rest_candidates: list[np.ndarray] = field(default_factory=list)
    last_macro: M12Macro | None = None


def _clear_memory(memory: M12Memory) -> None:
    memory.scans_in_cycle = 0
    memory.required_scans = 4
    memory.scanning = True
    memory.pending_pickup = False
    memory.pending_play = False
    memory.pending_rest = False
    memory.blocked_attempts = 0
    memory.feed_done = False
    memory.play_done = False
    memory.rest_done = False
    memory.completed_cycles = 0
    memory.food_candidates.clear()
    memory.toy_candidates.clear()
    memory.rest_candidates.clear()
    memory.last_macro = None


def _reset_no_memory(memory: M12Memory) -> None:
    """Erase persistent state while allowing the current RGB frame to act.

    A no-memory policy may use the observation it has just received, but it
    cannot retain a scan phase, candidate, pending interaction, or completed
    cycle into the next action.  Resetting it to ``scanning=True`` made the
    ablation repeatedly issue 0.1-second scans, which preserved its drives
    without demonstrating any maintenance capability.
    """

    _clear_memory(memory)
    memory.scanning = False


def _begin_scan_cycle(memory: M12Memory, *, views: int = 4) -> None:
    """Forget agent-relative targets before collecting a new four-view scan.

    Candidate offsets are expressed in the agent frame.  They are valid while
    the camera moves between scan sectors, but become invalid as soon as the
    agent walks.  Keeping them across movement made the policy repeatedly walk
    toward a location in its old frame.
    """

    memory.scans_in_cycle = 0
    memory.required_scans = views
    memory.scanning = True
    memory.food_candidates.clear()
    memory.toy_candidates.clear()
    memory.rest_candidates.clear()


def _cycle_complete(memory: M12Memory) -> None:
    if memory.feed_done and memory.play_done and memory.rest_done:
        memory.completed_cycles += 1
        memory.feed_done = False
        memory.play_done = False
        memory.rest_done = False


def _merge_candidates(candidates: list[np.ndarray], detected: tuple[np.ndarray, ...]) -> None:
    for candidate in detected:
        if not any(float(np.linalg.norm(candidate - known)) < 0.10 for known in candidates):
            candidates.append(np.asarray(candidate, dtype=np.float32))


def _candidate_counts(memory: M12Memory) -> tuple[float, float, float]:
    return (
        min(len(memory.food_candidates), 3) / 3.0,
        min(len(memory.toy_candidates), 3) / 3.0,
        min(len(memory.rest_candidates), 2) / 2.0,
    )


def _candidate_target(candidates: list[np.ndarray]) -> np.ndarray | None:
    if not candidates:
        return None
    return np.asarray(max(candidates, key=lambda candidate: float(np.linalg.norm(candidate))), dtype=np.float32)


def _normalized_distance(target: np.ndarray | None) -> float:
    if target is None:
        return 1.0
    return min(float(np.linalg.norm(target)), 1.5) / 1.5


def _feature_vector(observation: dict[str, Any], memory: M12Memory, detections: M12Detections, *, use_drives: bool) -> np.ndarray:
    drives = np.asarray(observation["drives"], dtype=np.float32) if use_drives else np.zeros(3, dtype=np.float32)
    prior = int(observation["prior_outcome"])
    prior_one_hot = np.zeros(len(ActionOutcome), dtype=np.float32)
    prior_one_hot[prior] = 1.0
    last_macro = np.zeros(len(M12Macro) + 1, dtype=np.float32)
    last_macro[0 if memory.last_macro is None else 1 + int(memory.last_macro)] = 1.0
    food_count, toy_count, rest_count = _candidate_counts(memory)
    food_target = _food_target(memory)
    toy_target = _candidate_target(memory.toy_candidates)
    rest_target = _candidate_target(memory.rest_candidates)
    return np.concatenate(
        [
            drives.astype(np.float32),
            np.asarray(
                [
                    float(observation["holding_food"]),
                    float(memory.scanning),
                    min(memory.scans_in_cycle, 4) / 4.0,
                    float(memory.pending_pickup),
                    float(memory.pending_play),
                    float(memory.pending_rest),
                    min(memory.blocked_attempts, 2) / 2.0,
                    float(memory.feed_done),
                    float(memory.play_done),
                    float(memory.rest_done),
                    min(memory.completed_cycles, 3) / 3.0,
                    food_count,
                    toy_count,
                    rest_count,
                    _normalized_distance(food_target),
                    _normalized_distance(toy_target),
                    _normalized_distance(rest_target),
                    float(bool(detections.food)),
                    float(bool(detections.toy)),
                    float(bool(detections.rest)),
                ],
                dtype=np.float32,
            ),
            prior_one_hot,
            last_macro,
        ]
    )


def _mandatory_macro(observation: dict[str, Any], memory: M12Memory) -> M12Macro | None:
    """Return a required public transition before learned need selection.

    Scanning and interaction completion are kinematic invariants, not another
    opportunity for the selector to reconsider a target after it has already
    committed to one.  The learned selector still chooses among food, play,
    and rest when this shell has a fresh public candidate bank.
    """

    if bool(observation["holding_food"]):
        return M12Macro.CONSUME
    if memory.pending_pickup:
        return M12Macro.PICK_UP
    if memory.pending_play:
        return M12Macro.PLAY
    if memory.pending_rest:
        return M12Macro.REST
    if memory.scanning:
        return M12Macro.SCAN
    return None


def _teacher_macro(observation: dict[str, Any], memory: M12Memory, config: EcosystemConfig) -> M12Macro:
    required = _mandatory_macro(observation, memory)
    if required is not None:
        return required
    drives = np.asarray(observation["drives"], dtype=np.float32)
    if float(drives[1]) <= config.rest_cycle_energy_threshold:
        return M12Macro.WALK_REST if memory.rest_candidates else M12Macro.SCAN
    if float(drives[0]) <= 0.55:
        return M12Macro.WALK_FOOD if memory.food_candidates else M12Macro.SCAN
    if float(drives[2]) >= config.play_success_boredom_threshold:
        return M12Macro.WALK_TOY if memory.toy_candidates else M12Macro.SCAN
    return M12Macro.IDLE


def _advance_memory(memory: M12Memory, observation: dict[str, Any]) -> None:
    outcome = list(ActionOutcome)[int(observation["prior_outcome"])]
    if memory.last_macro is M12Macro.CONSUME and not bool(observation["holding_food"]) and outcome is ActionOutcome.SUCCESS:
        memory.feed_done = True
        memory.pending_pickup = False
        memory.blocked_attempts = 0
        _begin_scan_cycle(memory, views=2)
        _cycle_complete(memory)
    elif memory.last_macro is M12Macro.PLAY:
        memory.pending_play = False
        if outcome is ActionOutcome.SUCCESS:
            memory.play_done = True
            _cycle_complete(memory)
        _begin_scan_cycle(memory, views=2)
    elif memory.last_macro is M12Macro.REST:
        memory.pending_rest = False
        if outcome is ActionOutcome.SUCCESS:
            memory.rest_done = True
            _cycle_complete(memory)
        _begin_scan_cycle(memory, views=2)
    elif memory.last_macro is M12Macro.WALK_FOOD and outcome is ActionOutcome.SUCCESS:
        memory.pending_pickup = True
        memory.food_candidates.clear()
        memory.toy_candidates.clear()
        memory.rest_candidates.clear()
    elif memory.last_macro is M12Macro.WALK_TOY and outcome is ActionOutcome.SUCCESS:
        memory.pending_play = True
        memory.food_candidates.clear()
        memory.toy_candidates.clear()
        memory.rest_candidates.clear()
    elif memory.last_macro is M12Macro.WALK_REST and outcome is ActionOutcome.SUCCESS:
        memory.pending_rest = True
        memory.food_candidates.clear()
        memory.toy_candidates.clear()
        memory.rest_candidates.clear()
    elif memory.last_macro is M12Macro.PICK_UP:
        memory.pending_pickup = False
        if outcome is ActionOutcome.BLOCKED:
            memory.blocked_attempts += 1
            _begin_scan_cycle(memory)
        elif outcome is ActionOutcome.SUCCESS:
            memory.blocked_attempts = 0


def _collect_selector_dataset(
    grounder: M12Grounder, *, use_drives: bool, use_memory: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    digest = hashlib.sha256()
    env = EcosystemEnv(m12_config())
    try:
        for condition, controls in M12_TRAIN_CONDITIONS.items():
            for seed in M12_TRAIN_SEEDS:
                observation, _ = env.reset(seed=seed, options=_options(controls))
                memory = M12Memory()
                for _ in range(env.config.max_episode_steps):
                    if not use_memory:
                        _reset_no_memory(memory)
                    _advance_memory(memory, observation)
                    detections = grounder.detect(np.asarray(observation["rgb"], dtype=np.uint8))
                    if memory.scanning or not use_memory:
                        _merge_candidates(memory.food_candidates, detections.food)
                        _merge_candidates(memory.toy_candidates, detections.toy)
                        _merge_candidates(memory.rest_candidates, detections.rest)
                    features = _feature_vector(observation, memory, detections, use_drives=use_drives)
                    required = _mandatory_macro(observation, memory)
                    label = M12Macro(_teacher_macro(observation, memory, env.config))
                    if not use_memory or required is None:
                        rows.append(features)
                        labels.append(int(label))
                        digest.update(condition.encode())
                        digest.update(np.asarray(seed, dtype=np.int64).tobytes())
                        digest.update(features.astype(np.float32).tobytes())
                        digest.update(np.asarray(int(label), dtype=np.int64).tobytes())
                    memory.last_macro = label
                    action = _macro_action(memory.last_macro, memory, env.config)
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
    finally:
        env.close()
    return np.stack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64), digest.hexdigest()


def _fit_selector(
    features: np.ndarray, labels: np.ndarray, *, require_full_decision_set: bool = True
) -> M12Selector:
    class_indices = {macro: np.flatnonzero(labels == int(macro)) for macro in M12Macro}
    decision_macros = (M12Macro.SCAN, M12Macro.WALK_FOOD, M12Macro.WALK_TOY, M12Macro.WALK_REST, M12Macro.IDLE)
    expected = tuple(macro for macro in decision_macros if len(class_indices[macro]))
    if require_full_decision_set and not {M12Macro.WALK_FOOD, M12Macro.WALK_TOY, M12Macro.WALK_REST, M12Macro.IDLE}.issubset(expected):
        missing = [macro.name for macro in (M12Macro.WALK_FOOD, M12Macro.WALK_TOY, M12Macro.WALK_REST, M12Macro.IDLE) if macro not in expected]
        raise RuntimeError(f"M12 development supervision lacks decision macros: {', '.join(missing)}")
    rng = np.random.default_rng(121_200)
    hidden = rng.normal(0.0, 0.08, size=(features.shape[1], M12_SELECTOR_HIDDEN_UNITS))
    hidden_bias = np.zeros(M12_SELECTOR_HIDDEN_UNITS, dtype=np.float64)
    output = rng.normal(0.0, 0.08, size=(M12_SELECTOR_HIDDEN_UNITS, len(M12Macro)))
    output_bias = np.zeros(len(M12Macro), dtype=np.float64)
    one_hot = np.eye(len(M12Macro), dtype=np.float64)
    for _ in range(M12_TRAIN_STEPS):
        macro_ids = np.arange(M12_BATCH_SIZE) % len(expected)
        indices = np.concatenate(
            [rng.choice(class_indices[expected[int(macro_id)]], size=1, replace=True) for macro_id in macro_ids]
        )
        rng.shuffle(indices)
        batch = features[indices].astype(np.float64)
        expected_labels = one_hot[labels[indices]]
        preactivation = batch @ hidden + hidden_bias
        activations = np.maximum(preactivation, 0.0)
        logits = activations @ output + output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True)
        error = (probs - expected_labels) / len(indices)
        output_gradient, output_bias_gradient = activations.T @ error, np.sum(error, axis=0)
        hidden_error = (error @ output.T) * (preactivation > 0.0)
        hidden_gradient, hidden_bias_gradient = batch.T @ hidden_error, np.sum(hidden_error, axis=0)
        hidden -= M12_LEARNING_RATE * hidden_gradient
        hidden_bias -= M12_LEARNING_RATE * hidden_bias_gradient
        output -= M12_LEARNING_RATE * output_gradient
        output_bias -= M12_LEARNING_RATE * output_bias_gradient
    return M12Selector(hidden, hidden_bias, output, output_bias)


def _food_target(memory: M12Memory) -> np.ndarray | None:
    if not memory.food_candidates:
        return None
    if memory.blocked_attempts:
        # A forced relocation is deliberately placed near the agent.  After
        # its stale pickup, rank the fresh public candidates by proximity so
        # the old distant target cannot dominate the recovery scan.
        candidates = sorted(memory.food_candidates, key=lambda candidate: float(np.linalg.norm(candidate)))
        index = min(memory.blocked_attempts - 1, len(candidates) - 1)
    else:
        candidates = sorted(memory.food_candidates, key=lambda candidate: -float(np.linalg.norm(candidate)))
        index = 0
    return np.asarray(candidates[index], dtype=np.float32)


def _macro_action(macro: M12Macro, memory: M12Memory, config: EcosystemConfig) -> dict[str, np.ndarray | int]:
    if macro is M12Macro.SCAN:
        if not memory.scanning:
            _begin_scan_cycle(memory)
        memory.scans_in_cycle += 1
        if memory.scans_in_cycle >= memory.required_scans:
            memory.scanning = False
        return skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1)
    if macro is M12Macro.CONSUME:
        return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
    if macro is M12Macro.PICK_UP:
        return skill_action(ActionKind.PICK_UP_RELATIVE, np.zeros(2, dtype=np.float32), 0.1)
    if macro is M12Macro.PLAY:
        memory.pending_play = False
        return skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 1.0)
    if macro is M12Macro.REST:
        memory.pending_rest = False
        return skill_action(ActionKind.REST, np.zeros(2, dtype=np.float32), 1.0)
    if macro is M12Macro.IDLE:
        return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 1.0)
    if macro is M12Macro.WALK_FOOD:
        target = _food_target(memory)
        if target is None:
            _begin_scan_cycle(memory)
            return _macro_action(M12Macro.SCAN, memory, config)
        duration = max(0.1, float(np.linalg.norm(target)) / config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_RELATIVE, target, duration)
    if macro is M12Macro.WALK_TOY:
        if not memory.toy_candidates:
            _begin_scan_cycle(memory)
            return _macro_action(M12Macro.SCAN, memory, config)
        target = _candidate_target(memory.toy_candidates)
        assert target is not None
        duration = max(0.1, float(np.linalg.norm(target)) / config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_RELATIVE, target, duration)
    if macro is M12Macro.WALK_REST:
        if not memory.rest_candidates:
            _begin_scan_cycle(memory)
            return _macro_action(M12Macro.SCAN, memory, config)
        target = _candidate_target(memory.rest_candidates)
        assert target is not None
        duration = max(0.1, float(np.linalg.norm(target)) / config.walk_speed_per_second)
        return skill_action(ActionKind.WALK_RELATIVE, target, duration)
    raise AssertionError(f"unhandled M12 macro {macro}")


@dataclass(frozen=True, slots=True)
class M12Policy:
    grounder: M12Grounder
    selector: M12Selector
    training_data_fingerprint: str
    config: EcosystemConfig
    use_drives: bool = True
    use_memory: bool = True

    def reset(self) -> M12Memory:
        return M12Memory()

    def act(self, observation: dict[str, Any], memory: M12Memory) -> dict[str, np.ndarray | int]:
        if not self.use_memory:
            _reset_no_memory(memory)
        _advance_memory(memory, observation)
        detections = self.grounder.detect(np.asarray(observation["rgb"], dtype=np.uint8))
        if memory.scanning or not self.use_memory:
            _merge_candidates(memory.food_candidates, detections.food)
            _merge_candidates(memory.toy_candidates, detections.toy)
            _merge_candidates(memory.rest_candidates, detections.rest)
        features = _feature_vector(observation, memory, detections, use_drives=self.use_drives)
        required = _mandatory_macro(observation, memory)
        macro = _select_macro(required, self.selector, features.astype(np.float64))
        action = _macro_action(macro, memory, self.config)
        memory.last_macro = macro
        return action


def fit_m12_policy(
    *,
    use_drives: bool = True,
    use_memory: bool = True,
    verify_coverage: bool = True,
    grounder: M12Grounder | None = None,
) -> M12Policy:
    if verify_coverage:
        _require_coverage(M12_TRAIN_CONDITIONS, seeds=M12_TRAIN_SEEDS, label="M12 development protocol")
    fitted_grounder = grounder or _fit_grounder()
    features, labels, training_data_fingerprint = _collect_selector_dataset(
        fitted_grounder, use_drives=use_drives, use_memory=use_memory
    )
    selector = _fit_selector(features, labels, require_full_decision_set=use_memory)
    return M12Policy(
        grounder=fitted_grounder,
        selector=selector,
        training_data_fingerprint=training_data_fingerprint,
        config=m12_config(),
        use_drives=use_drives,
        use_memory=use_memory,
    )


def m12_protocol_fingerprint() -> str:
    payload = {
        "version": M12_PROTOCOL_VERSION,
        "config": asdict(m12_config()),
        "train_seeds": M12_TRAIN_SEEDS,
        "validation_seeds": M12_VALIDATION_SEEDS,
        "train_conditions": M12_TRAIN_CONDITIONS,
        "validation_conditions": M12_VALIDATION_CONDITIONS,
        "patch_radius": M12_PATCH_RADIUS,
        "hidden_units": M12_HIDDEN_UNITS,
        "selector_hidden_units": M12_SELECTOR_HIDDEN_UNITS,
        "train_steps": M12_TRAIN_STEPS,
        "learning_rate": M12_LEARNING_RATE,
        "confidence_threshold": M12_CONFIDENCE_THRESHOLD,
        "min_component_pixels": M12_MIN_COMPONENT_PIXELS,
        "scan_views_per_cycle": 4,
        "bootstrap_draws": M12_BOOTSTRAP_DRAWS,
        "bootstrap_seed": M12_BOOTSTRAP_SEED,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def m12_policy_fingerprint(policy: M12Policy) -> str:
    digest = hashlib.sha256()
    values = [
        policy.grounder.hidden,
        policy.grounder.hidden_bias,
        policy.grounder.output,
        policy.grounder.output_bias,
        *policy.grounder.pixel_maps.values(),
        policy.selector.hidden,
        policy.selector.hidden_bias,
        policy.selector.output,
        policy.selector.output_bias,
    ]
    for parameter in values:
        array = np.asarray(parameter, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(policy.training_data_fingerprint.encode())
    digest.update(json.dumps(asdict(policy.config), sort_keys=True).encode())
    digest.update(json.dumps({"use_drives": policy.use_drives, "use_memory": policy.use_memory}, sort_keys=True).encode())
    return digest.hexdigest()


def _drive_record(drives: np.ndarray) -> dict[str, float]:
    values = np.asarray(drives, dtype=np.float32)
    return {"satiety": float(values[0]), "energy": float(values[1]), "boredom": float(values[2])}


def _inside_safe_band(drives: np.ndarray) -> bool:
    values = _drive_record(drives)
    return (
        values["satiety"] > M11_SAFE_DRIVE_BANDS["satiety_min"]
        and values["energy"] > M11_SAFE_DRIVE_BANDS["energy_min"]
        and values["boredom"] < M11_SAFE_DRIVE_BANDS["boredom_max"]
    )


def _candidate_snapshot(policy: M12Policy, observation: dict[str, Any]) -> dict[str, object]:
    detections = policy.grounder.detect(np.asarray(observation["rgb"], dtype=np.uint8))
    return {
        "food_offsets": [np.asarray(offset, dtype=np.float32).tolist() for offset in detections.food],
        "toy_offsets": [np.asarray(offset, dtype=np.float32).tolist() for offset in detections.toy],
        "rest_offsets": [np.asarray(offset, dtype=np.float32).tolist() for offset in detections.rest],
        "food_detected": bool(detections.food),
        "toy_detected": bool(detections.toy),
        "rest_detected": bool(detections.rest),
    }


def _first_failure(
    steps: list[dict[str, object]], *, survived: bool, maintenance_complete: bool
) -> dict[str, object] | None:
    if survived and maintenance_complete:
        return None
    relocation = next((row for row in steps if row["disturbance"] == "food_relocated"), None)
    if relocation is not None:
        later = [row for row in steps if int(row["step"]) > int(relocation["step"])]
        if not any(row["action_kind"] == ActionKind.SCAN.name for row in later):
            return {"stage": "recovery", "classification": "no_rescan_after_relocation", "step": relocation["step"]}
        if not any(row["action_kind"] == ActionKind.CONSUME.name and row["outcome"] == "success" for row in later):
            return {"stage": "recovery", "classification": "relocated_food_not_consumed", "step": relocation["step"]}
    energy_breach = next((row for row in steps if float(row["drives_after"]["energy"]) <= M11_SAFE_DRIVE_BANDS["energy_min"]), None)
    if energy_breach is not None and not any(row["action_kind"] == ActionKind.REST.name for row in steps):
        return {"stage": "need_selection", "classification": "energy_need_has_no_rest_action", "step": energy_breach["step"]}
    satiety_breach = next((row for row in steps if float(row["drives_after"]["satiety"]) <= M11_SAFE_DRIVE_BANDS["satiety_min"]), None)
    if satiety_breach is not None and not any(row["action_kind"] == ActionKind.CONSUME.name for row in steps):
        return {"stage": "need_selection", "classification": "satiety_need_has_no_consume_action", "step": satiety_breach["step"]}
    boredom_breach = next((row for row in steps if float(row["drives_after"]["boredom"]) >= M11_SAFE_DRIVE_BANDS["boredom_max"]), None)
    if boredom_breach is not None and not any(row["action_kind"] == ActionKind.RUN_AROUND.name for row in steps):
        return {"stage": "need_selection", "classification": "boredom_need_has_no_play_action", "step": boredom_breach["step"]}
    if not maintenance_complete:
        return {"stage": "maintenance", "classification": "cycle_requirements_not_met", "step": steps[-1]["step"]}
    return {"stage": "terminal", "classification": "survival_failure", "step": steps[-1]["step"]}


@dataclass(frozen=True, slots=True)
class M12Episode:
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
    first_failure: dict[str, object] | None
    steps: tuple[dict[str, object], ...]


def run_m12_episode(
    policy: M12Policy, *, seed: int, condition: str, controls: dict[str, Any], env: EcosystemEnv | None = None
) -> M12Episode:
    owns_env = env is None
    if env is None:
        env = EcosystemEnv(m12_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        records: list[dict[str, object]] = []
        safe_steps = 0
        relocation_seen = stale_pickup = consumed_after_relocation = False
        scans_after_relocation = 0
        interventions: Counter[str] = Counter()
        for step in range(1, env.config.max_episode_steps + 1):
            candidates = _candidate_snapshot(policy, observation)
            drives_before = _drive_record(np.asarray(observation["drives"], dtype=np.float32))
            action = policy.act(observation, memory)
            macro = memory.last_macro.name if memory.last_macro is not None else None
            action_kind = ActionKind(int(action["kind"])).name
            target = np.asarray(action["target"], dtype=np.float32).tolist()
            candidate_rank: int | None = None
            if action_kind == ActionKind.WALK_RELATIVE.name and macro == M12Macro.WALK_FOOD.name:
                candidate_rank = min(memory.blocked_attempts, max(0, len(candidates["food_offsets"]) - 1))
            elif action_kind == ActionKind.WALK_RELATIVE.name and macro == M12Macro.WALK_TOY.name:
                candidate_rank = 0
            elif action_kind == ActionKind.WALK_RELATIVE.name and macro == M12Macro.WALK_REST.name:
                candidate_rank = 0
            observation, _, terminated, truncated, info = env.step(action)
            if relocation_seen and action_kind == ActionKind.SCAN.name:
                scans_after_relocation += 1
            relocation_now = info["disturbance"] == "food_relocated"
            relocation_seen = relocation_seen or relocation_now
            stale_pickup = stale_pickup or (relocation_now and info["outcome"] == ActionOutcome.BLOCKED.value)
            consumed_after_relocation = consumed_after_relocation or (
                relocation_seen and action_kind == ActionKind.CONSUME.name and info["outcome"] == ActionOutcome.SUCCESS.value
            )
            if relocation_now:
                interventions["food_relocated"] += 1
            if info["resource_event"] is not None:
                interventions[str(info["resource_event"])] += 1
            if info["outcome"] == ActionOutcome.BLOCKED.value:
                interventions["blocked_action"] += 1
            safe_steps += int(_inside_safe_band(np.asarray(observation["drives"], dtype=np.float32)))
            records.append(
                {
                    "step": step,
                    "drives_before": drives_before,
                    "drives_after": _drive_record(np.asarray(observation["drives"], dtype=np.float32)),
                    "inside_safe_drive_band": _inside_safe_band(np.asarray(observation["drives"], dtype=np.float32)),
                    "target_observation": candidates,
                    "macro": macro,
                    "action_kind": action_kind,
                    "candidate_rank": candidate_rank,
                    "local_offset": target if action_kind == ActionKind.WALK_RELATIVE.name else None,
                    "outcome": info["outcome"],
                    "disturbance": info["disturbance"],
                    "resource_event": info["resource_event"],
                    "feed_cycles": info["feed_cycles"],
                    "play_cycles": info["play_cycles"],
                    "rest_cycles": info["rest_cycles"],
                    "food_available": info["food_available"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                state = env._require_state()
                survived = bool(info["survived"])
                maintenance_complete = bool(info["maintenance_complete"])
                terminal_cause = (
                    "survived_horizon"
                    if truncated and survived
                    else "energy_depleted"
                    if state.drives.energy <= 0.0
                    else "satiety_depleted"
                    if state.drives.satiety <= 0.0
                    else "terminated"
                )
                recovery = {
                    "required": bool(controls.get("event_relocation_on_first_pickup")),
                    "stale_pickup": stale_pickup,
                    "scans_after_relocation": scans_after_relocation,
                    "consumed_after_relocation": consumed_after_relocation,
                    "complete": stale_pickup and scans_after_relocation >= 4 and consumed_after_relocation,
                }
                return M12Episode(
                    seed=seed,
                    condition=condition,
                    survived=survived,
                    maintenance_complete=maintenance_complete,
                    terminal_cause=terminal_cause,
                    feed_cycles=state.feed_cycles,
                    play_cycles=state.play_cycles,
                    rest_cycles=state.rest_cycles,
                    safe_drive_fraction=safe_steps / step,
                    forced_recovery=recovery,
                    interventions=dict(interventions),
                    first_failure=_first_failure(records, survived=survived, maintenance_complete=maintenance_complete),
                    steps=tuple(records),
                )
        raise AssertionError("M12 episode did not terminate")
    finally:
        if owns_env:
            env.close()


def _episode_record(episode: M12Episode) -> dict[str, object]:
    return {
        "seed": episode.seed,
        "survived": episode.survived,
        "maintenance_complete": episode.maintenance_complete,
        "terminal_cause": episode.terminal_cause,
        "completed_cycles": {"feed": episode.feed_cycles, "play": episode.play_cycles, "rest": episode.rest_cycles},
        "time_inside_safe_drive_bands": episode.safe_drive_fraction,
        "forced_recovery": episode.forced_recovery,
        "interventions": episode.interventions,
        "first_failure": episode.first_failure,
        "steps": list(episode.steps),
    }


def _aggregate(episodes: list[M12Episode]) -> dict[str, object]:
    count = len(episodes)
    curve = [
        {"step": step, "surviving_episodes": sum(item.survived or len(item.steps) >= step for item in episodes)}
        for step in range(1, max(len(item.steps) for item in episodes) + 1)
    ]
    intervention_counts: Counter[str] = Counter()
    failure_classes = Counter(
        str(item.first_failure["classification"]) for item in episodes if item.first_failure is not None
    )
    for item in episodes:
        intervention_counts.update(item.interventions)
    required = [item for item in episodes if bool(item.forced_recovery["required"])]
    return {
        "episodes": count,
        "survivals": sum(item.survived for item in episodes),
        "survival_rate": sum(item.survived for item in episodes) / count,
        "survival_wilson_95": list(wilson_interval(sum(item.survived for item in episodes), count)),
        "survival_curve": curve,
        "completed_maintenance_episodes": sum(item.maintenance_complete for item in episodes),
        "completed_cycles": {
            "minimum_feed": min(item.feed_cycles for item in episodes),
            "minimum_play": min(item.play_cycles for item in episodes),
            "minimum_rest": min(item.rest_cycles for item in episodes),
            "mean_feed": float(np.mean([item.feed_cycles for item in episodes])),
            "mean_play": float(np.mean([item.play_cycles for item in episodes])),
            "mean_rest": float(np.mean([item.rest_cycles for item in episodes])),
        },
        "mean_time_inside_safe_drive_bands": float(np.mean([item.safe_drive_fraction for item in episodes])),
        "forced_recovery_chains": {
            "required_episodes": len(required),
            "stale_pickups": sum(bool(item.forced_recovery["stale_pickup"]) for item in required),
            "full_rescans": sum(int(item.forced_recovery["scans_after_relocation"]) >= 4 for item in required),
            "completed": sum(bool(item.forced_recovery["complete"]) for item in required),
        },
        "interventions": dict(sorted(intervention_counts.items())),
        "terminal_causes": dict(sorted(Counter(item.terminal_cause for item in episodes).items())),
        "first_failure_classifications": dict(sorted(failure_classes.items())),
        "failures_are_inspectable": all(item.first_failure is not None for item in episodes if not (item.survived and item.maintenance_complete)),
        "episodes_detail": [_episode_record(item) for item in episodes],
    }


def _evaluate(policy: M12Policy, conditions: dict[str, dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    env = EcosystemEnv(m12_config())
    try:
        return {
            name: _aggregate(
                [run_m12_episode(policy, seed=seed, condition=name, controls=controls, env=env) for seed in seeds]
            )
            for name, controls in conditions.items()
        }
    finally:
        env.close()


def _paired_bootstrap_interval(current: list[bool], baseline: list[bool]) -> tuple[float, float]:
    deltas = np.asarray(current, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if not len(deltas):
        raise ValueError("paired comparison needs at least one episode")
    generator = np.random.default_rng(M12_BOOTSTRAP_SEED)
    bootstrap_means = generator.choice(deltas, size=(M12_BOOTSTRAP_DRAWS, len(deltas)), replace=True).mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return float(low), float(high)


def _flatten_survivals(results: dict[str, dict[str, object]]) -> list[bool]:
    ordered: list[bool] = []
    for condition in M12_VALIDATION_CONDITIONS:
        ordered.extend(bool(item["survived"]) for item in results[condition]["episodes_detail"])
    return ordered


def _aggregate_counts(results: dict[str, dict[str, object]]) -> tuple[int, int]:
    return (
        sum(int(row["survivals"]) for row in results.values()),
        sum(int(row["completed_maintenance_episodes"]) for row in results.values()),
    )


def _pick_adverse_episode(results: dict[str, dict[str, object]]) -> tuple[str, int]:
    for condition, summary in results.items():
        for episode in summary["episodes_detail"]:
            if not bool(episode["survived"]) or not bool(episode["maintenance_complete"]):
                return condition, int(episode["seed"])
    return next(iter(results)), int(results[next(iter(results))]["episodes_detail"][0]["seed"])


def m12_validation(*, seeds: tuple[int, ...] = M12_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M12_VALIDATION_SEEDS:
        raise ValueError("M12 validation seeds are frozen; use M12_VALIDATION_SEEDS")
    mechanics = m10_validation()
    if not bool(mechanics["gate"]["passes"]):
        raise RuntimeError("M10 mechanics gate failed; M12 policy interpretation is invalid")
    train_coverage = _require_coverage(M12_TRAIN_CONDITIONS, seeds=M12_TRAIN_SEEDS, label="M12 development protocol")
    validation_coverage = _require_coverage(M12_VALIDATION_CONDITIONS, seeds=seeds, label="M12 validation protocol")
    grounder = _fit_grounder()
    full_policy = fit_m12_policy(verify_coverage=False, grounder=grounder)
    no_memory_policy = fit_m12_policy(use_memory=False, verify_coverage=False, grounder=grounder)
    no_drive_policy = fit_m12_policy(use_drives=False, verify_coverage=False, grounder=grounder)
    baseline = m11_baseline(mechanics=mechanics)

    full_results = _evaluate(full_policy, M12_VALIDATION_CONDITIONS, seeds)
    no_memory_results = _evaluate(no_memory_policy, M12_VALIDATION_CONDITIONS, seeds)
    no_drive_results = _evaluate(no_drive_policy, M12_VALIDATION_CONDITIONS, seeds)

    paired_current = _flatten_survivals(full_results)
    paired_baseline = _flatten_survivals(baseline["results"])
    paired_advantage = float(np.mean(np.asarray(paired_current, dtype=np.float64) - np.asarray(paired_baseline, dtype=np.float64)))
    paired_low, paired_high = _paired_bootstrap_interval(paired_current, paired_baseline)

    full_survivals, full_maintenance = _aggregate_counts(full_results)
    no_memory_survivals, no_memory_maintenance = _aggregate_counts(no_memory_results)
    no_drive_survivals, no_drive_maintenance = _aggregate_counts(no_drive_results)

    condition_gate = {
        name: {
            "survival_pass": int(row["survivals"]) >= 15,
            "maintenance_pass": int(row["completed_maintenance_episodes"]) >= 15,
            "safe_drive_pass": float(row["mean_time_inside_safe_drive_bands"]) >= 0.80,
            "recovery_pass": (
                True
                if not M12_VALIDATION_CONDITIONS[name].get("event_relocation_on_first_pickup")
                else int(row["forced_recovery_chains"]["completed"]) >= 15
            ),
        }
        for name, row in full_results.items()
    }
    passes = (
        all(item["passes"] for item in train_coverage.values())
        and all(item["passes"] for item in validation_coverage.values())
        and all(all(gate.values()) for gate in condition_gate.values())
        and paired_advantage >= 0.20
        and paired_low > 0.0
        and full_survivals > no_memory_survivals
        and full_maintenance > no_memory_maintenance
        and full_survivals > no_drive_survivals
        and full_maintenance > no_drive_maintenance
    )
    adverse_condition, adverse_seed = _pick_adverse_episode(full_results)
    return {
        "schema_version": "0.12",
        "protocol_version": M12_PROTOCOL_VERSION,
        "protocol_fingerprint": m12_protocol_fingerprint(),
        "m10_protocol_fingerprint": m10_protocol_fingerprint(),
        "split": "m10_validation_successor",
        "seeds": list(seeds),
        "train_seeds": list(M12_TRAIN_SEEDS),
        "train_conditions": M12_TRAIN_CONDITIONS,
        "conditions": M12_VALIDATION_CONDITIONS,
        "safe_drive_bands": M11_SAFE_DRIVE_BANDS,
        "policy_boundary": M12_POLICY_BOUNDARY,
        "coverage": {"development": train_coverage, "validation": validation_coverage},
        "m10_mechanics_gate": mechanics["gate"],
        "m10_coverage": mechanics["coverage"],
        "m11_baseline_protocol_fingerprint": baseline["protocol_fingerprint"],
        "full_policy": {
            "policy_fingerprint": m12_policy_fingerprint(full_policy),
            "training_data_fingerprint": full_policy.training_data_fingerprint,
            "results": full_results,
        },
        "ablations": {
            "no_memory": {
                "policy_fingerprint": m12_policy_fingerprint(no_memory_policy),
                "training_data_fingerprint": no_memory_policy.training_data_fingerprint,
                "results": no_memory_results,
            },
            "no_drive": {
                "policy_fingerprint": m12_policy_fingerprint(no_drive_policy),
                "training_data_fingerprint": no_drive_policy.training_data_fingerprint,
                "results": no_drive_results,
            },
        },
        "paired_m11_baseline": {
            "policy_fingerprint": baseline["frozen_m9_policy_fingerprint"],
            "results": baseline["results"],
            "paired_episodes": len(paired_current),
            "survival_advantage": paired_advantage,
            "survival_advantage_bootstrap_95": [paired_low, paired_high],
            "bootstrap_draws": M12_BOOTSTRAP_DRAWS,
            "bootstrap_seed": M12_BOOTSTRAP_SEED,
        },
        "gate": {
            "condition_gate": condition_gate,
            "full_policy_survivals": full_survivals,
            "full_policy_completed_maintenance": full_maintenance,
            "no_memory_survivals": no_memory_survivals,
            "no_memory_completed_maintenance": no_memory_maintenance,
            "no_drive_survivals": no_drive_survivals,
            "no_drive_completed_maintenance": no_drive_maintenance,
            "beats_no_memory": full_survivals > no_memory_survivals and full_maintenance > no_memory_maintenance,
            "beats_no_drive": full_survivals > no_drive_survivals and full_maintenance > no_drive_maintenance,
            "paired_survival_margin": paired_advantage,
            "paired_survival_bootstrap_95": [paired_low, paired_high],
            "passes": passes,
        },
        "adverse_trace_recommendation": {"condition": adverse_condition, "seed": adverse_seed},
        "limits": [
            "M12 is a structured RGB policy over typed kinematic skills, not end-to-end control or reinforcement learning.",
            "The macro selector is behaviour cloning from development-only oracle labels, and the deployment boundary remains public RGB plus drives plus policy-owned memory.",
            "M10 validation is frozen and M10 audit remains reserved for M13 rather than M12 selection or retries.",
        ],
    }


def write_m12_report(
    path: str | Path,
    *,
    adverse_trace_path: str | Path | None = None,
    adverse_condition: str | None = None,
    adverse_seed: int | None = None,
) -> dict[str, object]:
    report = m12_validation()
    if adverse_trace_path is not None:
        chosen_condition = adverse_condition or str(report["adverse_trace_recommendation"]["condition"])
        chosen_seed = adverse_seed if adverse_seed is not None else int(report["adverse_trace_recommendation"]["seed"])
        replay = run_m12_viewer_demo(adverse_trace_path, seed=chosen_seed, condition=chosen_condition)
        report["adverse_trace"] = {
            "condition": chosen_condition,
            "seed": chosen_seed,
            "trace_path": str(adverse_trace_path),
            "replay_validated": True,
            "replay_summary": asdict(replay),
        }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m12_viewer_demo(
    trace_path: str | Path,
    *,
    seed: int = 1_800,
    condition: str = "compound",
    use_drives: bool = True,
    use_memory: bool = True,
) -> ReplayResult:
    controls = M12_VALIDATION_CONDITIONS[condition]
    policy = fit_m12_policy(use_drives=use_drives, use_memory=use_memory)
    with ViewerSession(
        EcosystemEnv(m12_config(), render_mode="rgb_array"),
        trace_path=trace_path,
        episode_id="m12-learned-persistent-maintenance",
    ) as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
