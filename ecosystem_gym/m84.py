"""M8.4 learned RGB heatmap grounding with a fixed public recovery shell.

The train-only labeler uses MuJoCo segmentation to supervise a patchwise RGB
classifier and a pixel-to-local-coordinate map.  At inference the grounder
receives RGB only; the controller receives the public observation and its own
memory.  Neither receives segmentation, state coordinates, reset options, or
step info.  M8.2 is excluded from this module's training and validation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .actions import ActionKind, ActionOutcome
from .config import EcosystemConfig
from .env import EcosystemEnv
from .m8 import M8_TASK_ID, wilson_interval
from .policies import skill_action
from .trajectory import ReplayResult
from .viewer import ViewerSession


M84_PROTOCOL_VERSION = "m84-learned-rgb-heatmap-v2"
M84_TRAIN_SEEDS = tuple(range(400, 420))
M84_VALIDATION_SEEDS = tuple(range(500, 520))
M84_PATCH_RADIUS = 2
M84_HIDDEN_UNITS = 32
M84_TRAIN_STEPS = 1_500
M84_BATCH_SIZE = 512
M84_LEARNING_RATE = 0.10
M84_CONFIDENCE_THRESHOLD = 0.50
M84_MIN_COMPONENT_PIXELS = 3
M84_COMPONENT_MATCH_RADIUS = 0.10
M84_MIN_CONDITION_SUCCESS_RATE = 0.75

# These rows deliberately use M8.4-specific layouts and avoid the M8.2 audit
# layouts, seeds, and compound rows.  The blocked row teaches two visual peaks.
M84_TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "reference_red": {"layout_id": "m8_protocol", "food_variant": "red", "camera_control": "scan_v2", "initial_scan_sector": "north"},
    "orange_east": {"layout_id": "m84_train_west", "food_variant": "orange", "camera_control": "scan_v2", "initial_scan_sector": "east"},
    "blue_south": {"layout_id": "m84_train_southeast", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "south"},
    "purple_dim_west": {"layout_id": "m84_train_southwest", "food_variant": "purple", "lighting_variant": "dim", "camera_control": "scan_v2", "initial_scan_sector": "west"},
    "blocked_two_peaks": {
        "layout_id": "m84_train_west",
        "food_variant": "red",
        "camera_control": "scan_v2",
        "initial_scan_sector": "north",
        "blocked_distractor": True,
        "distractor_xy": [-0.22, 0.10],
    },
}
M84_VALIDATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "northwest_orange_east": {"layout_id": "m84_validation_northwest", "food_variant": "orange", "camera_control": "scan_v2", "initial_scan_sector": "east"},
    "southeast_blue_dim_south": {"layout_id": "m84_validation_southeast", "food_variant": "blue", "lighting_variant": "dim", "camera_control": "scan_v2", "initial_scan_sector": "south"},
    "northwest_purple_blocked": {
        "layout_id": "m84_validation_northwest",
        "food_variant": "purple",
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "blocked_distractor": True,
        "distractor_xy": [-0.04, 0.12],
    },
    "southeast_red_relocation": {
        "layout_id": "m84_validation_southeast",
        "food_variant": "red",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "disturbance_step": 1,
    },
}

_BACKGROUND = 0
_TARGET = 1
_AGENT = 2


def m84_config(*, observation_mode: str = "rgb") -> EcosystemConfig:
    return EcosystemConfig(observation_mode=observation_mode, max_episode_steps=24)  # type: ignore[arg-type]


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def _camera_name(env: EcosystemEnv) -> str:
    prefix = "agent_cam_scan_v2" if env._camera_control == "scan_v2" else "agent_cam_scan"
    return f"{prefix}_{env._scan_sector}"


def _patches(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    radius = M84_PATCH_RADIUS
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
    return windows.transpose(0, 1, 3, 4, 2).reshape(-1, (2 * radius + 1) ** 2 * 3)


@dataclass(frozen=True, slots=True)
class TrainingFrame:
    rgb: np.ndarray
    labels: np.ndarray
    calibration_pairs: tuple[tuple[np.ndarray, np.ndarray], ...]


def _segmentation_frame(env: EcosystemEnv, renderer: mujoco.Renderer) -> TrainingFrame:
    """Private offline labeler; it must never be called from a policy method."""

    renderer.update_scene(env.data, camera=_camera_name(env))
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    ids = segmentation[..., 0]
    agent_id = env.model.geom("agent_geom").id
    target_ids = (env.model.geom("food_geom").id, env.model.geom("distractor_geom").id)
    labels = np.full(ids.shape, _BACKGROUND, dtype=np.int8)
    labels[ids == agent_id] = _AGENT
    labels[np.isin(ids, target_ids)] = _TARGET
    agent_pixels = np.argwhere(ids == agent_id)
    if not len(agent_pixels):
        return TrainingFrame(env._rgb_observation(), labels, ())
    agent_centre = np.asarray((float(np.mean(agent_pixels[:, 1])), -float(np.mean(agent_pixels[:, 0]))), dtype=np.float64)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for geom_name, target_xy in (("food_geom", env._food_xy()), ("distractor_geom", env._distractor_xy())):
        pixels = np.argwhere(ids == env.model.geom(geom_name).id)
        if len(pixels):
            centre = np.asarray((float(np.mean(pixels[:, 1])), -float(np.mean(pixels[:, 0]))), dtype=np.float64)
            pairs.append((np.asarray([*(centre - agent_centre), 1.0]), np.asarray(target_xy - env._agent_xy(), dtype=np.float64)))
    return TrainingFrame(env._rgb_observation(), labels, tuple(pairs))


def _scan_visibility(env: EcosystemEnv, renderer: mujoco.Renderer) -> bool:
    """Offline benchmark audit: whether a four-view scan contains the food."""

    food_id = env.model.geom("food_geom").id
    for _ in range(4):
        renderer.update_scene(env.data, camera=_camera_name(env))
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
        if np.any(segmentation[..., 0] == food_id):
            return True
        env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
    return False


def m84_scan_coverage(
    conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    """Audit target visibility before fitting or scoring a camera-only policy.

    MuJoCo segmentation is confined to this offline validator.  It is never
    available to the grounder or the recovery policy.
    """

    coverage: dict[str, dict[str, object]] = {}
    for condition, controls in conditions.items():
        initially_visible = 0
        relocated_visible = 0
        relocation_episodes = 0
        for seed in seeds:
            env = EcosystemEnv(m84_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                env.reset(seed=seed, options=_options(controls))
                initially_visible += int(_scan_visibility(env, renderer))
                if controls.get("disturbance_step") is not None:
                    relocation_episodes += 1
                    # A scan is a public action and triggers the configured
                    # relocation at step one; audit the new target position.
                    env.reset(seed=seed, options=_options(controls))
                    env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
                    relocated_visible += int(_scan_visibility(env, renderer))
            finally:
                renderer.close()
                env.close()
        coverage[condition] = {
            "episodes": len(seeds),
            "initially_visible_episodes": initially_visible,
            "relocation_episodes": relocation_episodes,
            "relocated_visible_episodes": relocated_visible,
            "passes": initially_visible == len(seeds)
            and (not relocation_episodes or relocated_visible == relocation_episodes),
        }
    return coverage


def _require_full_scan_coverage(
    conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...], label: str
) -> dict[str, dict[str, object]]:
    coverage = m84_scan_coverage(conditions, seeds=seeds)
    failed = [condition for condition, result in coverage.items() if not result["passes"]]
    if failed:
        raise RuntimeError(f"{label} has unobservable scan targets: {', '.join(failed)}")
    return coverage


def _collect_static_training_frames() -> list[TrainingFrame]:
    frames: list[TrainingFrame] = []
    for controls in M84_TRAIN_CONDITIONS.values():
        for seed in M84_TRAIN_SEEDS:
            env = EcosystemEnv(m84_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                observation, _ = env.reset(seed=seed, options=_options(controls))
                for _ in range(4):
                    frame = _segmentation_frame(env, renderer)
                    frames.append(TrainingFrame(np.asarray(observation["rgb"], dtype=np.uint8), frame.labels, frame.calibration_pairs))
                    observation, _, _, _, _ = env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
            finally:
                renderer.close()
                env.close()
    return frames


def _pixel_training_data(frames: list[TrainingFrame]) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(84)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for frame in frames:
        patches = _patches(frame.rgb)
        flat_labels = frame.labels.reshape(-1)
        for label, limit in ((_BACKGROUND, 120), (_TARGET, 120), (_AGENT, 120)):
            candidates = np.flatnonzero(flat_labels == label)
            if len(candidates):
                chosen = generator.choice(candidates, size=min(len(candidates), limit), replace=False)
                features.append(patches[chosen])
                labels.append(np.full(len(chosen), label, dtype=np.int64))
    return np.concatenate(features), np.concatenate(labels)


def _fit_pixel_mlp(frames: list[TrainingFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels = _pixel_training_data(frames)
    generator = np.random.default_rng(8_400)
    hidden = generator.normal(0.0, 0.1, size=(features.shape[1], M84_HIDDEN_UNITS))
    hidden_bias = np.zeros(M84_HIDDEN_UNITS, dtype=np.float64)
    output = generator.normal(0.0, 0.1, size=(M84_HIDDEN_UNITS, 3))
    output_bias = np.zeros(3, dtype=np.float64)
    one_hot = np.eye(3, dtype=np.float64)
    for _ in range(M84_TRAIN_STEPS):
        indices = generator.integers(0, len(features), size=M84_BATCH_SIZE)
        batch = features[indices]
        preactivation = batch @ hidden + hidden_bias
        activations = np.maximum(preactivation, 0.0)
        logits = activations @ output + output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        error = (probabilities - one_hot[labels[indices]]) / len(indices)
        output_gradient = activations.T @ error
        output_bias_gradient = np.sum(error, axis=0)
        hidden_error = (error @ output.T) * (preactivation > 0.0)
        hidden_gradient = batch.T @ hidden_error
        hidden_bias_gradient = np.sum(hidden_error, axis=0)
        hidden -= M84_LEARNING_RATE * hidden_gradient
        hidden_bias -= M84_LEARNING_RATE * hidden_bias_gradient
        output -= M84_LEARNING_RATE * output_gradient
        output_bias -= M84_LEARNING_RATE * output_bias_gradient
    return hidden, hidden_bias, output, output_bias


def _fit_pixel_map(frames: list[TrainingFrame]) -> np.ndarray:
    feature_rows = [pair[0] for frame in frames for pair in frame.calibration_pairs]
    target_rows = [pair[1] for frame in frames for pair in frame.calibration_pairs]
    features = np.stack(feature_rows)
    targets = np.stack(target_rows)
    return np.linalg.solve(features.T @ features + 1e-3 * np.eye(features.shape[1]), features.T @ targets)


def _connected_centres(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    seen = np.zeros_like(mask, dtype=bool)
    centres: list[np.ndarray] = []
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
        if len(pixels) >= M84_MIN_COMPONENT_PIXELS:
            ys, xs = zip(*pixels, strict=True)
            centres.append(np.asarray((float(np.mean(xs)), -float(np.mean(ys))), dtype=np.float64))
    return tuple(centres)


class LearnedRgbHeatmapGrounder:
    """Patchwise RGB target/agent heatmaps and learned local-coordinate map."""

    def __init__(self, hidden: np.ndarray, hidden_bias: np.ndarray, output: np.ndarray, output_bias: np.ndarray, pixel_map: np.ndarray) -> None:
        self.hidden = np.asarray(hidden, dtype=np.float64)
        self.hidden_bias = np.asarray(hidden_bias, dtype=np.float64)
        self.output = np.asarray(output, dtype=np.float64)
        self.output_bias = np.asarray(output_bias, dtype=np.float64)
        self.pixel_map = np.asarray(pixel_map, dtype=np.float64)

    def probabilities(self, rgb: np.ndarray) -> np.ndarray:
        features = _patches(rgb)
        activations = np.maximum(features @ self.hidden + self.hidden_bias, 0.0)
        logits = activations @ self.output + self.output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        height, width = np.asarray(rgb).shape[:2]
        return probabilities.reshape(height, width, 3)

    def components(self, rgb: np.ndarray) -> tuple[np.ndarray, ...]:
        probabilities = self.probabilities(rgb)
        agent_centres = _connected_centres(probabilities[..., _AGENT] >= M84_CONFIDENCE_THRESHOLD)
        target_centres = _connected_centres(probabilities[..., _TARGET] >= M84_CONFIDENCE_THRESHOLD)
        if not agent_centres:
            return ()
        agent = max(agent_centres, key=lambda centre: float(np.linalg.norm(centre)))
        targets = [np.asarray([*(target - agent), 1.0]) @ self.pixel_map for target in target_centres]
        return tuple(sorted((np.asarray(target, dtype=np.float32) for target in targets), key=lambda target: float(np.linalg.norm(target))))


@dataclass(slots=True)
class HeatmapRecoveryMemory:
    awaiting_pickup: bool = False
    blocked_attempts: int = 0


class LearnedHeatmapRecoveryPolicy:
    """Public scan/retry shell over learned RGB grounding; no fixed adapter at inference."""

    def __init__(self, grounder: LearnedRgbHeatmapGrounder) -> None:
        self.grounder = grounder

    def reset(self) -> HeatmapRecoveryMemory:
        return HeatmapRecoveryMemory()

    def act(self, observation: dict[str, Any], memory: HeatmapRecoveryMemory) -> dict[str, np.ndarray | int]:
        if bool(observation["holding_food"]):
            return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
        if memory.awaiting_pickup:
            memory.awaiting_pickup = False
            return skill_action(ActionKind.PICK_UP_RELATIVE, np.zeros(2, dtype=np.float32), 0.1)
        if ActionOutcome(list(ActionOutcome)[int(observation["prior_outcome"])]) is ActionOutcome.BLOCKED:
            memory.blocked_attempts += 1
        components = self.grounder.components(np.asarray(observation["rgb"], dtype=np.uint8))
        if not components:
            return skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1)
        target = components[min(memory.blocked_attempts, len(components) - 1)]
        distance = float(np.linalg.norm(target))
        if distance <= m84_config().pickup_radius:
            return skill_action(ActionKind.PICK_UP_RELATIVE, target, 0.1)
        memory.awaiting_pickup = True
        return skill_action(ActionKind.WALK_RELATIVE, target, max(0.1, distance / m84_config().walk_speed_per_second))


def _fit_grounder_after_coverage_check() -> LearnedRgbHeatmapGrounder:
    frames = _collect_static_training_frames()
    hidden, hidden_bias, output, output_bias = _fit_pixel_mlp(frames)
    return LearnedRgbHeatmapGrounder(hidden, hidden_bias, output, output_bias, _fit_pixel_map(frames))


def fit_m84_grounder() -> LearnedRgbHeatmapGrounder:
    _require_full_scan_coverage(M84_TRAIN_CONDITIONS, seeds=M84_TRAIN_SEEDS, label="M8.4 training protocol")
    return _fit_grounder_after_coverage_check()


def m84_policy_fingerprint(grounder: LearnedRgbHeatmapGrounder) -> str:
    """Stable digest of the deterministic learned parameters used by M8.5."""

    digest = hashlib.sha256()
    for values in (grounder.hidden, grounder.hidden_bias, grounder.output, grounder.output_bias, grounder.pixel_map):
        array = np.asarray(values, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def m84_protocol_fingerprint() -> str:
    """Digest every training choice that must remain fixed for M8.5."""

    payload = {
        "protocol_version": M84_PROTOCOL_VERSION,
        "train_seeds": M84_TRAIN_SEEDS,
        "train_conditions": M84_TRAIN_CONDITIONS,
        "patch_radius": M84_PATCH_RADIUS,
        "hidden_units": M84_HIDDEN_UNITS,
        "train_steps": M84_TRAIN_STEPS,
        "batch_size": M84_BATCH_SIZE,
        "learning_rate": M84_LEARNING_RATE,
        "confidence_threshold": M84_CONFIDENCE_THRESHOLD,
        "minimum_component_pixels": M84_MIN_COMPONENT_PIXELS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class M84Episode:
    seed: int
    condition: str
    task_success: bool
    total_reward: float
    steps: int
    scan_actions: int
    blocked_outcomes: int
    disturbance_events: int
    post_disturbance_completion: bool


def _run_episode(policy: LearnedHeatmapRecoveryPolicy, *, seed: int, condition: str, controls: dict[str, Any]) -> M84Episode:
    env = EcosystemEnv(m84_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
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
                return M84Episode(seed, condition, bool(info["task_success"]), total_reward, step, scans, blocked, disturbances, bool(info["post_disturbance_completion"]))
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _run_oracle(*, seed: int, condition: str, controls: dict[str, Any]) -> M84Episode:
    env = EcosystemEnv(m84_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        total_reward = 0.0
        disturbances = 0
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
            disturbances += int(info["disturbance"] is not None)
            if terminated or truncated:
                return M84Episode(seed, condition, bool(info["task_success"]), total_reward, step, 0, 0, disturbances, bool(info["post_disturbance_completion"]))
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _aggregate(episodes: list[M84Episode]) -> dict[str, object]:
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


def _grounding_metrics(
    grounder: LearnedRgbHeatmapGrounder,
    controls: dict[str, Any],
    *,
    seeds: tuple[int, ...] = M84_VALIDATION_SEEDS,
) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    errors: list[float] = []
    for seed in seeds:
        env = EcosystemEnv(m84_config())
        renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
        try:
            observation, _ = env.reset(seed=seed, options=_options(controls))
            for _ in range(4):
                frame = _segmentation_frame(env, renderer)
                predicted = grounder.components(observation["rgb"])
                actual = [target for _, target in frame.calibration_pairs]
                unmatched_actual = list(actual)
                for predicted_target in predicted:
                    if not unmatched_actual:
                        false_positive += 1
                        continue
                    distances = [float(np.linalg.norm(predicted_target - actual_target)) for actual_target in unmatched_actual]
                    nearest = int(np.argmin(distances))
                    if distances[nearest] <= M84_COMPONENT_MATCH_RADIUS:
                        true_positive += 1
                        errors.append(distances[nearest])
                        unmatched_actual.pop(nearest)
                    else:
                        false_positive += 1
                false_negative += len(unmatched_actual)
                observation, _, _, _, _ = env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
        finally:
            renderer.close()
            env.close()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "frames": 4 * len(seeds),
        "component_match_radius": M84_COMPONENT_MATCH_RADIUS,
        "matched_target_components": true_positive,
        "false_positive_components": false_positive,
        "false_negative_components": false_negative,
        "visible_precision": precision,
        "visible_recall": recall,
        "mean_relative_target_error": float(np.mean(errors)) if errors else None,
        "p90_relative_target_error": float(np.percentile(errors, 90)) if errors else None,
    }


def m84_validation(*, seeds: tuple[int, ...] = M84_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M84_VALIDATION_SEEDS:
        raise ValueError("M8.4 validation seeds are frozen; use M84_VALIDATION_SEEDS")
    train_coverage = _require_full_scan_coverage(M84_TRAIN_CONDITIONS, seeds=M84_TRAIN_SEEDS, label="M8.4 training protocol")
    validation_coverage = _require_full_scan_coverage(M84_VALIDATION_CONDITIONS, seeds=seeds, label="M8.4 validation protocol")
    grounder = _fit_grounder_after_coverage_check()
    policy = LearnedHeatmapRecoveryPolicy(grounder)
    results: dict[str, dict[str, object]] = {}
    grounding: dict[str, dict[str, object]] = {}
    for condition, controls in M84_VALIDATION_CONDITIONS.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in seeds]
        results[condition] = {"learned_rgb_heatmap_grounder": _aggregate(learned), "state_oracle_ceiling": _aggregate(oracle)}
        grounding[condition] = _grounding_metrics(grounder, controls)
    all_successes = sum(result["learned_rgb_heatmap_grounder"]["successes"] for result in results.values())
    all_episodes = len(results) * len(seeds)
    condition_success_rates = {
        condition: result["learned_rgb_heatmap_grounder"]["task_success_rate"] for condition, result in results.items()
    }
    passes = all_successes / all_episodes >= 0.75 and all(
        rate >= M84_MIN_CONDITION_SUCCESS_RATE for rate in condition_success_rates.values()
    )
    return {
        "schema_version": "0.84",
        "protocol_version": M84_PROTOCOL_VERSION,
        "train_seeds": list(M84_TRAIN_SEEDS),
        "validation_seeds": list(seeds),
        "train_conditions": M84_TRAIN_CONDITIONS,
        "validation_conditions": M84_VALIDATION_CONDITIONS,
        "scan_coverage": {"training": train_coverage, "validation": validation_coverage},
        "m82_training_episodes": 0,
        "policy_boundary": ["rgb", "holding_food", "prior_outcome", "policy_owned_memory"],
        "results": results,
        "grounding_metrics": grounding,
        "validation_gate": {
            "episodes": all_episodes,
            "successes": all_successes,
            "success_rate": all_successes / all_episodes,
            "predeclared_minimum_success_rate": 0.75,
            "condition_success_rates": condition_success_rates,
            "predeclared_minimum_condition_success_rate": M84_MIN_CONDITION_SUCCESS_RATE,
            "passes": passes,
            "next_step": "define a new sealed scan_v2 audit without reusing M8.2" if passes else "stop: report validation failure without opening M8.2",
        },
        "limits": [
            "The RGB heatmap is learned from offline MuJoCo segmentation labels, but inference uses only RGB and never calls the simulator labeler or M8's fixed colour-component adapter.",
            "The scan, pickup, and blocked-candidate retry shell is authored public control logic. M8.4 is learned target grounding with sequential recovery, not end-to-end learned control or RL.",
            "M8.2 is excluded from M8.4 training and validation and remains an invalid legacy artifact; any later audit must use a newly sealed scan_v2 suite.",
        ],
    }


def write_m84_report(path: str | Path) -> dict[str, object]:
    report = m84_validation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m84_viewer_demo(trace_path: str | Path, *, seed: int = 507) -> ReplayResult:
    grounder = fit_m84_grounder()
    policy = LearnedHeatmapRecoveryPolicy(grounder)
    controls = M84_VALIDATION_CONDITIONS["southeast_red_relocation"]
    with ViewerSession(EcosystemEnv(m84_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m84-learned-heatmap") as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
