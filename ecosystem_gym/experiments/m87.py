"""M8.7 larger-context, class-balanced RGB grounding successor.

M8.6 found blue-food misses rather than a dynamics failure.  M8.7 preserves
the public recovery shell but trains a 9x9 patch classifier with equal target,
agent, and background samples, including additional blue development scenes.
M8.5 remains excluded from every training and validation choice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..actions import ActionKind
from ..env import EcosystemEnv
from .m8 import M8_TASK_ID
from .m84 import (
    LearnedHeatmapRecoveryPolicy,
    TrainingFrame,
    _aggregate,
    _connected_centres,
    _fit_pixel_map,
    _grounding_metrics,
    _run_episode,
    _run_oracle,
    _segmentation_frame,
    m84_config,
    m84_scan_coverage,
)
from ..trajectory import ReplayResult
from ..viewer import ViewerSession


M87_PROTOCOL_VERSION = "m87-blue-balanced-rgb-grounding-v1"
M87_TRAIN_SEEDS = tuple(range(900, 920))
M87_VALIDATION_SEEDS = tuple(range(1_000, 1_020))
M87_PATCH_RADIUS = 4
M87_HIDDEN_UNITS = 64
M87_TRAIN_STEPS = 2_000
M87_BATCH_SIZE = 512
M87_LEARNING_RATE = 0.08
M87_SAMPLES_PER_CLASS = 64
M87_CONFIDENCE_THRESHOLD = 0.50
M87_MIN_COMPONENT_PIXELS = 3
M87_MIN_SUCCESS_RATE = 0.75

M87_TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "reference_red": {"layout_id": "m8_protocol", "food_variant": "red", "camera_control": "scan_v2", "initial_scan_sector": "north"},
    "orange_west": {"layout_id": "m84_train_west", "food_variant": "orange", "camera_control": "scan_v2", "initial_scan_sector": "east"},
    "blue_southeast": {"layout_id": "m84_train_southeast", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "south"},
    "purple_dim_southwest": {"layout_id": "m84_train_southwest", "food_variant": "purple", "lighting_variant": "dim", "camera_control": "scan_v2", "initial_scan_sector": "west"},
    "blue_hard_northeast": {"layout_id": "m87_train_northeast", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "east"},
    "blue_hard_northwest_grippy": {"layout_id": "m87_train_northwest", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "west", "dynamics_variant": "grippy"},
    "blue_blocked_two_peaks": {"layout_id": "m87_train_northeast", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "north", "blocked_distractor": True, "distractor_xy": [-0.06, 0.06]},
}
M87_VALIDATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "northwest_blue_nominal": {"layout_id": "m87_validation_northwest", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "east"},
    "southeast_blue_grippy": {"layout_id": "m87_validation_southeast", "food_variant": "blue", "camera_control": "scan_v2", "initial_scan_sector": "south", "dynamics_variant": "grippy"},
    "northwest_purple_blocked": {"layout_id": "m87_validation_northwest", "food_variant": "purple", "camera_control": "scan_v2", "initial_scan_sector": "west", "blocked_distractor": True, "distractor_xy": [-0.06, 0.08]},
    "southeast_red_relocation": {"layout_id": "m87_validation_southeast", "food_variant": "red", "camera_control": "scan_v2", "initial_scan_sector": "north", "disturbance_step": 1},
}


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def _patches(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    radius = M87_PATCH_RADIUS
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
    return windows.transpose(0, 1, 3, 4, 2).reshape(-1, (2 * radius + 1) ** 2 * 3)


def _require_coverage(conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...], label: str) -> dict[str, dict[str, object]]:
    coverage = m84_scan_coverage(conditions, seeds=seeds)
    failed = [condition for condition, result in coverage.items() if not result["passes"]]
    if failed:
        raise RuntimeError(f"{label} has unobservable scan_v2 targets: {', '.join(failed)}")
    return coverage


def _collect_frames(conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...]) -> list[TrainingFrame]:
    frames: list[TrainingFrame] = []
    for controls in conditions.values():
        for seed in seeds:
            env = EcosystemEnv(m84_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                observation, _ = env.reset(seed=seed, options=_options(controls))
                for _ in range(4):
                    frame = _segmentation_frame(env, renderer)
                    frames.append(frame)
                    observation, _, _, _, _ = env.step(
                        {"kind": int(ActionKind.SCAN), "target": np.zeros(2, dtype=np.float32), "duration": np.float32(0.1)}
                    )
            finally:
                renderer.close()
                env.close()
    return frames


def _balanced_training_data(frames: list[TrainingFrame]) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(8_700)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for frame in frames:
        patches = _patches(frame.rgb)
        labels = frame.labels.reshape(-1)
        for label in (0, 1, 2):
            candidates = np.flatnonzero(labels == label)
            if not len(candidates):
                continue
            chosen = generator.choice(candidates, size=M87_SAMPLES_PER_CLASS, replace=len(candidates) < M87_SAMPLES_PER_CLASS)
            feature_rows.append(patches[chosen])
            label_rows.append(np.full(M87_SAMPLES_PER_CLASS, label, dtype=np.int64))
    return np.concatenate(feature_rows), np.concatenate(label_rows)


def _fit_pixel_mlp(frames: list[TrainingFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels = _balanced_training_data(frames)
    generator = np.random.default_rng(87_000)
    hidden = generator.normal(0.0, 0.08, size=(features.shape[1], M87_HIDDEN_UNITS))
    hidden_bias = np.zeros(M87_HIDDEN_UNITS, dtype=np.float64)
    output = generator.normal(0.0, 0.08, size=(M87_HIDDEN_UNITS, 3))
    output_bias = np.zeros(3, dtype=np.float64)
    one_hot = np.eye(3, dtype=np.float64)
    for _ in range(M87_TRAIN_STEPS):
        indices = generator.integers(0, len(features), size=M87_BATCH_SIZE)
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
        hidden -= M87_LEARNING_RATE * hidden_gradient
        hidden_bias -= M87_LEARNING_RATE * hidden_bias_gradient
        output -= M87_LEARNING_RATE * output_gradient
        output_bias -= M87_LEARNING_RATE * output_bias_gradient
    return hidden, hidden_bias, output, output_bias


@dataclass(frozen=True, slots=True)
class BlueBalancedGrounder:
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    output_bias: np.ndarray
    pixel_map: np.ndarray

    def components(self, rgb: np.ndarray) -> tuple[np.ndarray, ...]:
        features = _patches(rgb)
        activations = np.maximum(features @ self.hidden + self.hidden_bias, 0.0)
        logits = activations @ self.output + self.output_bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        height, width = np.asarray(rgb).shape[:2]
        maps = probabilities.reshape(height, width, 3)
        agents = _connected_centres(maps[..., 2] >= M87_CONFIDENCE_THRESHOLD)
        targets = _connected_centres(maps[..., 1] >= M87_CONFIDENCE_THRESHOLD)
        if not agents:
            return ()
        agent = max(agents, key=lambda centre: float(np.linalg.norm(centre)))
        local_targets = [np.asarray([*(target - agent), 1.0]) @ self.pixel_map for target in targets]
        return tuple(sorted((np.asarray(target, dtype=np.float32) for target in local_targets), key=lambda target: float(np.linalg.norm(target))))


def fit_m87_grounder() -> BlueBalancedGrounder:
    _require_coverage(M87_TRAIN_CONDITIONS, seeds=M87_TRAIN_SEEDS, label="M8.7 training protocol")
    frames = _collect_frames(M87_TRAIN_CONDITIONS, seeds=M87_TRAIN_SEEDS)
    hidden, hidden_bias, output, output_bias = _fit_pixel_mlp(frames)
    return BlueBalancedGrounder(hidden, hidden_bias, output, output_bias, _fit_pixel_map(frames))


def m87_policy_fingerprint(grounder: BlueBalancedGrounder) -> str:
    digest = hashlib.sha256()
    for values in (grounder.hidden, grounder.hidden_bias, grounder.output, grounder.output_bias, grounder.pixel_map):
        array = np.asarray(values, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def m87_protocol_fingerprint() -> str:
    payload = {
        "protocol_version": M87_PROTOCOL_VERSION,
        "train_seeds": M87_TRAIN_SEEDS,
        "train_conditions": M87_TRAIN_CONDITIONS,
        "patch_radius": M87_PATCH_RADIUS,
        "hidden_units": M87_HIDDEN_UNITS,
        "train_steps": M87_TRAIN_STEPS,
        "batch_size": M87_BATCH_SIZE,
        "learning_rate": M87_LEARNING_RATE,
        "samples_per_class": M87_SAMPLES_PER_CLASS,
        "confidence_threshold": M87_CONFIDENCE_THRESHOLD,
        "minimum_component_pixels": M87_MIN_COMPONENT_PIXELS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def m87_validation(*, seeds: tuple[int, ...] = M87_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M87_VALIDATION_SEEDS:
        raise ValueError("M8.7 validation seeds are frozen; use M87_VALIDATION_SEEDS")
    train_coverage = _require_coverage(M87_TRAIN_CONDITIONS, seeds=M87_TRAIN_SEEDS, label="M8.7 training protocol")
    validation_coverage = _require_coverage(M87_VALIDATION_CONDITIONS, seeds=seeds, label="M8.7 validation protocol")
    grounder = fit_m87_grounder()
    policy = LearnedHeatmapRecoveryPolicy(grounder)  # type: ignore[arg-type]
    results: dict[str, dict[str, object]] = {}
    grounding: dict[str, dict[str, object]] = {}
    for condition, controls in M87_VALIDATION_CONDITIONS.items():
        learned = [_run_episode(policy, seed=seed, condition=condition, controls=controls) for seed in seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in seeds]
        results[condition] = {"blue_balanced_rgb_grounder": _aggregate(learned), "state_oracle_ceiling": _aggregate(oracle)}
        grounding[condition] = _grounding_metrics(grounder, controls, seeds=seeds)
    successes = sum(result["blue_balanced_rgb_grounder"]["successes"] for result in results.values())
    episodes = len(results) * len(seeds)
    condition_rates = {condition: result["blue_balanced_rgb_grounder"]["task_success_rate"] for condition, result in results.items()}
    passes = successes / episodes >= M87_MIN_SUCCESS_RATE and all(rate >= M87_MIN_SUCCESS_RATE for rate in condition_rates.values())
    return {
        "schema_version": "0.87",
        "protocol_version": M87_PROTOCOL_VERSION,
        "train_seeds": list(M87_TRAIN_SEEDS),
        "validation_seeds": list(seeds),
        "train_conditions": M87_TRAIN_CONDITIONS,
        "validation_conditions": M87_VALIDATION_CONDITIONS,
        "scan_coverage": {"training": train_coverage, "validation": validation_coverage},
        "m85_training_episodes": 0,
        "policy_boundary": ["rgb", "holding_food", "prior_outcome", "policy_owned_memory"],
        "results": results,
        "grounding_metrics": grounding,
        "validation_gate": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "condition_success_rates": condition_rates,
            "predeclared_minimum_success_rate": M87_MIN_SUCCESS_RATE,
            "passes": passes,
            "next_step": "define a new sealed scan_v2 audit without reusing M8.5" if passes else "stop: report validation failure without opening M8.5",
        },
        "limits": [
            "M8.7 learns RGB target grounding from offline segmentation labels but inference uses only RGB; the scan/pickup/retry shell is authored public control.",
            "Balanced blue examples and larger patches are a targeted successor change motivated by M8.6, not proof of broad perception or physics generalization.",
            "M8.5 remains excluded from M8.7 training and validation; a later audit must use a new sealed suite.",
        ],
    }


def write_m87_report(path: str | Path) -> dict[str, object]:
    report = m87_validation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m87_viewer_demo(trace_path: str | Path, *, seed: int = 1_007) -> ReplayResult:
    grounder = fit_m87_grounder()
    policy = LearnedHeatmapRecoveryPolicy(grounder)  # type: ignore[arg-type]
    with ViewerSession(EcosystemEnv(m84_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m87-blue-balanced") as session:
        observation, _ = session.reset(seed=seed, options=_options(M87_VALIDATION_CONDITIONS["southeast_blue_grippy"]))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(observation, memory)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
