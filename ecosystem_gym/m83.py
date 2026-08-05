"""M8.3 one-factor diagnostics for the sealed M8.1/M8.2 failure.

This module does not train a policy.  It runs the frozen M8.1 policies on
new seeds while changing one visual or spatial factor at a time, so a later
learner redesign has an evidence-backed target and M8.2 remains untouched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .actions import ActionKind
from .env import EcosystemEnv
from .m8 import FixedRgbScanRecoveryPolicy, M8_TASK_ID, rgb_food_components
from .m81 import (
    FeedForwardRgbBCPolicy,
    RecurrentRgbBCPolicy,
    _aggregate,
    _run_episode,
    _run_fixed,
    _run_oracle,
    fit_m81_policies,
    m81_config,
    m81_policy_fingerprint,
    m81_protocol_fingerprint,
)
from .m82 import M81_FROZEN_POLICY_FINGERPRINT, M81_FROZEN_PROTOCOL_FINGERPRINT
from .policies import skill_action


M83_PROTOCOL_VERSION = "m83-one-factor-diagnostics-v1-observability-audit"
M83_DIAGNOSTIC_SEEDS = tuple(range(300, 320))

# Every condition changes exactly one factor relative to the M8.1 reference
# environment, except that the landmark row tests its extra visible structure.
M83_CONDITIONS: dict[str, dict[str, Any]] = {
    "reference": {"layout_id": "m8_protocol", "food_variant": "red", "camera_control": "scan", "initial_scan_sector": "north"},
    "initial_sector_east": {"layout_id": "m8_protocol", "food_variant": "red", "camera_control": "scan", "initial_scan_sector": "east"},
    "purple_appearance": {"layout_id": "m8_protocol", "food_variant": "purple", "camera_control": "scan", "initial_scan_sector": "north"},
    "dim_lighting": {"layout_id": "m8_protocol", "food_variant": "red", "lighting_variant": "dim", "camera_control": "scan", "initial_scan_sector": "north"},
    "agent_spawn_northwest": {"layout_id": "m83_agent_northwest", "food_variant": "red", "camera_control": "scan", "initial_scan_sector": "north"},
    "food_quadrant_northwest": {"layout_id": "m83_food_northwest", "food_variant": "red", "camera_control": "scan", "initial_scan_sector": "north"},
    "landmark_structure": {"layout_id": "m8_protocol", "food_variant": "red", "camera_control": "scan", "initial_scan_sector": "north", "geometry_variant": "m82_landmark"},
}


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def m83_observability_audit(*, seeds: tuple[int, ...] = M83_DIAGNOSTIC_SEEDS) -> dict[str, dict[str, object]]:
    """Offline audit of four-view target visibility for the legacy diagnostic."""

    if seeds != M83_DIAGNOSTIC_SEEDS:
        raise ValueError("M8.3 diagnostic seeds are frozen; use M83_DIAGNOSTIC_SEEDS")
    audit: dict[str, dict[str, object]] = {}
    for condition, controls in M83_CONDITIONS.items():
        visible_episodes = visible_frames = 0
        for seed in seeds:
            env = EcosystemEnv(m81_config())
            renderer = mujoco.Renderer(env.model, height=env.config.rgb_height, width=env.config.rgb_width)
            try:
                env.reset(seed=seed, options=_options(controls))
                food_id = env.model.geom("food_geom").id
                visible = False
                for _ in range(4):
                    renderer.update_scene(env.data, camera=f"agent_cam_scan_{env._scan_sector}")
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render()
                    renderer.disable_segmentation_rendering()
                    frame_visible = bool(np.any(segmentation[..., 0] == food_id))
                    visible_frames += int(frame_visible)
                    visible |= frame_visible
                    env.step(skill_action(ActionKind.SCAN, np.zeros(2, dtype=np.float32), 0.1))
                visible_episodes += int(visible)
            finally:
                renderer.close()
                env.close()
        audit[condition] = {
            "episodes": len(seeds),
            "food_visible_in_any_scan_episodes": visible_episodes,
            "food_visible_scan_frames": visible_frames,
            "passes": visible_episodes == len(seeds),
        }
    return audit


def _initial_diagnostic(
    policy: FixedRgbScanRecoveryPolicy | FeedForwardRgbBCPolicy | RecurrentRgbBCPolicy,
    *,
    seed: int,
    controls: dict[str, Any],
) -> tuple[bool, str]:
    """Read only the initial public frame and the policy's first public action."""

    env = EcosystemEnv(m81_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        visible = bool(rgb_food_components(rgb))
        if isinstance(policy, FixedRgbScanRecoveryPolicy):
            action = policy.act(observation, policy.reset())
        elif isinstance(policy, RecurrentRgbBCPolicy):
            action = policy.act(rgb, policy.reset())
        else:
            action = policy.act(rgb)
        return visible, ActionKind(int(action["kind"])).name
    finally:
        env.close()


def _initial_summary(
    policy: FixedRgbScanRecoveryPolicy | FeedForwardRgbBCPolicy | RecurrentRgbBCPolicy,
    controls: dict[str, Any],
) -> dict[str, object]:
    observations = [_initial_diagnostic(policy, seed=seed, controls=controls) for seed in M83_DIAGNOSTIC_SEEDS]
    visible = sum(component_visible for component_visible, _ in observations)
    actions = Counter(action for _, action in observations)
    return {
        "episodes": len(observations),
        "initial_component_visible": visible,
        "initial_component_visible_rate": visible / len(observations),
        "first_action_counts": dict(sorted(actions.items())),
    }


def m83_diagnostics(*, seeds: tuple[int, ...] = M83_DIAGNOSTIC_SEEDS) -> dict[str, object]:
    if seeds != M83_DIAGNOSTIC_SEEDS:
        raise ValueError("M8.3 diagnostic seeds are frozen; use M83_DIAGNOSTIC_SEEDS")
    observability = m83_observability_audit(seeds=seeds)
    feed_forward, recurrent = fit_m81_policies()
    protocol_fingerprint = m81_protocol_fingerprint()
    policy_fingerprint = m81_policy_fingerprint(feed_forward, recurrent)
    if protocol_fingerprint != M81_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.1 protocol fingerprint drifted")
    if policy_fingerprint != M81_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.1 policy fingerprint drifted")

    results: dict[str, dict[str, object]] = {}
    initial_frames: dict[str, dict[str, object]] = {}
    fixed_policy = FixedRgbScanRecoveryPolicy()
    for condition, controls in M83_CONDITIONS.items():
        fixed = [_run_fixed(seed, condition, controls) for seed in seeds]
        feed_forward_episodes = [_run_episode(feed_forward, seed=seed, condition=condition, controls=controls) for seed in seeds]
        recurrent_episodes = [_run_episode(recurrent, seed=seed, condition=condition, controls=controls) for seed in seeds]
        oracle = [_run_oracle(seed, condition, controls) for seed in seeds]
        results[condition] = {
            "m8_fixed_rgb_baseline": _aggregate(fixed),
            "feed_forward_rgb_bc": _aggregate(feed_forward_episodes),
            "recurrent_rgb_bc": _aggregate(recurrent_episodes),
            "state_oracle_ceiling": _aggregate(oracle),
        }
        initial_frames[condition] = {
            "m8_fixed_rgb_baseline": _initial_summary(fixed_policy, controls),
            "feed_forward_rgb_bc": _initial_summary(feed_forward, controls),
            "recurrent_rgb_bc": _initial_summary(recurrent, controls),
        }
    return {
        "schema_version": "0.83",
        "protocol_version": M83_PROTOCOL_VERSION,
        "diagnostic_seeds": list(seeds),
        "conditions": M83_CONDITIONS,
        "m81_freeze": {
            "frozen_protocol_fingerprint": M81_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_protocol_fingerprint": protocol_fingerprint,
            "frozen_policy_fingerprint": M81_FROZEN_POLICY_FINGERPRINT,
            "observed_policy_fingerprint": policy_fingerprint,
            "m83_training_episodes": 0,
        },
        "results": results,
        "observability_audit": {
            "method": "offline MuJoCo segmentation across the same four public scan views; unavailable to every policy",
            "valid": all(result["passes"] for result in observability.values()),
            "by_condition": observability,
        },
        "initial_frame_diagnostics": initial_frames,
        "limits": [
            "M8.3 is a diagnosis of frozen M8.1 behavior, not a new learned-policy result.",
            "Initial component visibility is a measurement from M8's fixed adapter; it does not claim that a future raw-RGB learner needs that adapter.",
            "M8.2 remains sealed and is not present in this diagnostic matrix.",
            "Rows with incomplete scan coverage cannot isolate policy grounding from target invisibility.",
        ],
    }


def write_m83_report(path: str | Path) -> dict[str, object]:
    report = m83_diagnostics()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
