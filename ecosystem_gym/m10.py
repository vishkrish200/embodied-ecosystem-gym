"""M10 persistent-maintenance mechanics and frozen oracle protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .actions import ActionKind, ActionOutcome
from .config import EcosystemConfig
from .env import EcosystemEnv
from .m8 import wilson_interval
from .policies import skill_action
from .trajectory import ReplayResult
from .viewer import ViewerSession


M10_PROTOCOL_VERSION = "m10-persistent-maintenance-v1"
M10_TASK_ID = "persistent_maintenance"
M10_TRAIN_SEEDS = tuple(range(1_700, 1_720))
M10_VALIDATION_SEEDS = tuple(range(1_800, 1_820))
M10_AUDIT_SEEDS = tuple(range(1_900, 1_920))

M10_TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": {
        "layout_id": "m10_train_northeast",
        "food_variant": "orange",
        "toy_variant": "ball",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
    },
    "event_relocation": {
        "layout_id": "m10_train_southwest",
        "food_variant": "purple",
        "toy_variant": "cube",
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "event_relocation_on_first_pickup": True,
    },
}

M10_VALIDATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": {
        "layout_id": "m10_validation_northeast",
        "food_variant": "orange",
        "toy_variant": "ball",
        "camera_control": "scan_v2",
        "initial_scan_sector": "north",
    },
    "renewal_and_morphology": {
        "layout_id": "m10_validation_southwest",
        "food_variant": "purple",
        "food_shape_variant": "capsule",
        "toy_variant": "cube",
        "agent_shape_variant": "capsule",
        "camera_control": "scan_v2",
        "initial_scan_sector": "south",
    },
    "event_relocation": {
        "layout_id": "m10_validation_northwest",
        "food_variant": "blue",
        "food_shape_variant": "box",
        "toy_variant": "capsule",
        "lighting_variant": "dim",
        "camera_control": "scan_v2",
        "initial_scan_sector": "east",
        "event_relocation_on_first_pickup": True,
    },
    "compound": {
        "layout_id": "m10_validation_southeast",
        "food_variant": "red",
        "food_shape_variant": "capsule",
        "toy_variant": "cube",
        "agent_shape_variant": "box",
        "dynamics_variant": "grippy",
        "blocked_distractor": True,
        "distractor_xy": [-0.04, 0.04],
        "camera_control": "scan_v2",
        "initial_scan_sector": "west",
        "event_relocation_on_first_pickup": True,
    },
}

M10_AUDIT_CONDITIONS: dict[str, dict[str, Any]] = {
    "persistent_reference": {**M10_VALIDATION_CONDITIONS["persistent_reference"], "layout_id": "m10_audit_northeast"},
    "renewal_and_morphology": {**M10_VALIDATION_CONDITIONS["renewal_and_morphology"], "layout_id": "m10_audit_southwest"},
    "event_relocation": {**M10_VALIDATION_CONDITIONS["event_relocation"], "layout_id": "m10_audit_northwest"},
    "compound": {**M10_VALIDATION_CONDITIONS["compound"], "layout_id": "m10_audit_southeast"},
}


def m10_config(*, observation_mode: str = "rgb") -> EcosystemConfig:
    return EcosystemConfig(
        observation_mode=observation_mode,  # type: ignore[arg-type]
        max_episode_steps=160,
        satiety_decay_per_second=0.018,
        energy_decay_per_second=0.012,
        boredom_gain_per_second=0.020,
        play_success_boredom_threshold=0.60,
        play_boredom_reduction=0.55,
        eat_satiety_gain=0.55,
        food_respawn_seconds=5.0,
        rest_cycle_energy_threshold=0.45,
        rest_energy_gain=0.65,
    )


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M10_TASK_ID, **controls}


def _camera_name(env: EcosystemEnv) -> str:
    return f"agent_cam_scan_v2_{env._scan_sector}"


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


def m10_scan_coverage(
    conditions: dict[str, dict[str, Any]], *, seeds: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    """Offline segmentation audit; no policy method can access these labels."""

    report: dict[str, dict[str, object]] = {}
    for name, controls in conditions.items():
        initial_food = replenished_food = toy = rest = relocated_food = distractor = 0
        relocation_episodes = distractor_episodes = 0
        for seed in seeds:
            env = EcosystemEnv(m10_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
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
                    raise AssertionError("M10 replenishment did not emit its public event")
                replenished_food += int(_visible_in_public_scan(env, renderer, food_ids))

                if controls.get("event_relocation_on_first_pickup"):
                    relocation_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    env._set_agent_xy(env._food_xy())
                    _, _, _, _, info = env.step(skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1))
                    if info["disturbance"] != "food_relocated" or info["outcome"] != ActionOutcome.BLOCKED.value:
                        raise AssertionError("M10 event relocation must force a stale failed pickup")
                    relocated_food += int(_visible_in_public_scan(env, renderer, food_ids))

                if controls.get("blocked_distractor"):
                    distractor_episodes += 1
                    env.reset(seed=seed, options=_options(controls))
                    distractor += int(_visible_in_public_scan(env, renderer, (env.model.geom("distractor_geom").id,)))
            finally:
                renderer.close()
                env.close()
        count = len(seeds)
        passes = (
            initial_food == count
            and replenished_food == count
            and toy == count
            and rest == count
            and (not relocation_episodes or relocated_food == relocation_episodes)
            and (not distractor_episodes or distractor == distractor_episodes)
        )
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
            "passes": passes,
        }
    return report


def _oracle_action(env: EcosystemEnv, observation: dict[str, Any]) -> dict[str, np.ndarray | int]:
    state = env._require_state()
    agent = np.asarray(observation["agent_xy"], dtype=np.float32)
    if bool(observation["holding_food"]):
        return skill_action(ActionKind.CONSUME, np.zeros(2, dtype=np.float32), 0.1)
    if float(observation["drives"][1]) <= env.config.rest_cycle_energy_threshold:
        target = env._rest_xy()
        distance = float(np.linalg.norm(target - agent))
        if distance <= env.config.rest_interaction_radius:
            return skill_action(ActionKind.REST, np.zeros(2, dtype=np.float32), 1.0)
        return skill_action(ActionKind.WALK_TO, target, max(0.1, distance / env.config.walk_speed_per_second))
    if state.food_available and float(observation["drives"][0]) <= 0.55:
        target = np.asarray(observation["food_xy"], dtype=np.float32)
        distance = float(np.linalg.norm(target - agent))
        if distance <= env.config.pickup_radius:
            return skill_action(ActionKind.PICK_UP, np.zeros(2, dtype=np.float32), 0.1)
        return skill_action(ActionKind.WALK_TO, target, max(0.1, distance / env.config.walk_speed_per_second))
    if float(observation["drives"][2]) >= env.config.play_success_boredom_threshold:
        target = np.asarray(observation["toy_xy"], dtype=np.float32)
        distance = float(np.linalg.norm(target - agent))
        if distance <= env.config.toy_interaction_radius:
            return skill_action(ActionKind.RUN_AROUND, np.zeros(2, dtype=np.float32), 1.0)
        return skill_action(ActionKind.WALK_TO, target, max(0.1, distance / env.config.walk_speed_per_second))
    return skill_action(ActionKind.IDLE, np.zeros(2, dtype=np.float32), 5.0)


@dataclass(frozen=True, slots=True)
class M10Episode:
    seed: int
    condition: str
    success: bool
    survived: bool
    steps: int
    feed_cycles: int
    play_cycles: int
    rest_cycles: int
    safe_drive_fraction: float
    relocation_failures: int
    recovered_relocations: int
    replenishments: int


def _run_oracle(*, seed: int, condition: str, controls: dict[str, Any]) -> M10Episode:
    env = EcosystemEnv(m10_config(observation_mode="state_oracle"))
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        safe_steps = relocations = recoveries = replenishments = 0
        for step in range(1, env.config.max_episode_steps + 1):
            action = _oracle_action(env, observation)
            observation, _, terminated, truncated, info = env.step(action)
            drives = np.asarray(observation["drives"], dtype=np.float32)
            safe_steps += int(float(drives[0]) > 0.15 and float(drives[1]) > 0.15 and float(drives[2]) < 0.90)
            relocations += int(info["disturbance"] == "food_relocated" and info["outcome"] == ActionOutcome.BLOCKED.value)
            recoveries += int(info["post_disturbance_completion"])
            replenishments += int(info["resource_event"] == "food_replenished")
            if terminated or truncated:
                state = env._require_state()
                return M10Episode(
                    seed=seed,
                    condition=condition,
                    success=bool(info["task_success"]),
                    survived=bool(info["survived"]),
                    steps=step,
                    feed_cycles=state.feed_cycles,
                    play_cycles=state.play_cycles,
                    rest_cycles=state.rest_cycles,
                    safe_drive_fraction=safe_steps / step,
                    relocation_failures=relocations,
                    recovered_relocations=recoveries,
                    replenishments=replenishments,
                )
        raise AssertionError("M10 oracle episode did not terminate")
    finally:
        env.close()


def _aggregate(episodes: list[M10Episode]) -> dict[str, object]:
    count = len(episodes)
    successes = sum(item.success for item in episodes)
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "success_wilson_95": list(wilson_interval(successes, count)),
        "survival_rate": sum(item.survived for item in episodes) / count,
        "minimum_feed_cycles": min(item.feed_cycles for item in episodes),
        "minimum_play_cycles": min(item.play_cycles for item in episodes),
        "minimum_rest_cycles": min(item.rest_cycles for item in episodes),
        "mean_safe_drive_fraction": float(np.mean([item.safe_drive_fraction for item in episodes])),
        "forced_relocation_failures": sum(item.relocation_failures for item in episodes),
        "recovered_relocations": sum(item.recovered_relocations for item in episodes),
        "food_replenishments": sum(item.replenishments for item in episodes),
        "mean_steps": float(np.mean([item.steps for item in episodes])),
    }


def m10_protocol_fingerprint() -> str:
    payload = {
        "version": M10_PROTOCOL_VERSION,
        "config": asdict(m10_config()),
        "train_seeds": M10_TRAIN_SEEDS,
        "validation_seeds": M10_VALIDATION_SEEDS,
        "audit_seeds": M10_AUDIT_SEEDS,
        "train_conditions": M10_TRAIN_CONDITIONS,
        "validation_conditions": M10_VALIDATION_CONDITIONS,
        "audit_conditions": M10_AUDIT_CONDITIONS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def m10_validation(*, seeds: tuple[int, ...] = M10_VALIDATION_SEEDS) -> dict[str, object]:
    if seeds != M10_VALIDATION_SEEDS:
        raise ValueError("M10 validation seeds are frozen; use M10_VALIDATION_SEEDS")
    coverage = m10_scan_coverage(M10_VALIDATION_CONDITIONS, seeds=seeds)
    failed_coverage = [name for name, row in coverage.items() if not row["passes"]]
    if failed_coverage:
        raise RuntimeError(f"M10 has unobservable public targets: {', '.join(failed_coverage)}")
    results = {
        name: _aggregate([_run_oracle(seed=seed, condition=name, controls=controls) for seed in seeds])
        for name, controls in M10_VALIDATION_CONDITIONS.items()
    }
    config = m10_config()
    oracle_complete = all(
        row["successes"] == len(seeds)
        and row["minimum_feed_cycles"] >= config.persistent_min_feed_cycles
        and row["minimum_play_cycles"] >= config.persistent_min_play_cycles
        and row["minimum_rest_cycles"] >= config.persistent_min_rest_cycles
        for row in results.values()
    )
    return {
        "schema_version": "0.10",
        "protocol_version": M10_PROTOCOL_VERSION,
        "protocol_fingerprint": m10_protocol_fingerprint(),
        "split": "validation",
        "seeds": list(seeds),
        "conditions": M10_VALIDATION_CONDITIONS,
        "coverage": coverage,
        "privileged_oracle": results,
        "gate": {
            "passes": oracle_complete,
            "oracle_ceiling_complete": oracle_complete,
            "policy_scored": False,
        },
        "limits": [
            "M10 validates persistent mechanics and observability; it does not score an RGB maintenance policy.",
            "Movement and interaction remain typed kinematic skills rather than contact-rich control.",
            "Segmentation is restricted to the offline coverage audit.",
        ],
    }


def write_m10_report(path: str | Path) -> dict[str, object]:
    report = m10_validation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m10_viewer_demo(trace_path: str | Path, *, seed: int = 1_807) -> ReplayResult:
    controls = M10_VALIDATION_CONDITIONS["event_relocation"]
    with ViewerSession(
        EcosystemEnv(m10_config(observation_mode="state_oracle"), render_mode="rgb_array"),
        trace_path=trace_path,
        episode_id="m10-persistent-maintenance",
    ) as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        for _ in range(session.env.config.max_episode_steps):
            action = _oracle_action(session.env, observation)
            observation, _, terminated, truncated, _ = session.step(action)
            if terminated or truncated:
                break
        return session.replay(trace_path)
