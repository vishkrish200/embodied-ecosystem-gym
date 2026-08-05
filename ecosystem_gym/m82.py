"""M8.2: a sealed post-result external-validity audit for M8.1.

M8.2 never supplies its layouts or outcomes to the M8.1 trainer.  It rebuilds
the deterministic frozen M8.1 policies only to verify their recorded weight
fingerprint, then compares them on new seed, layout, appearance, lighting,
camera-sector, landmark, and disturbance combinations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ActionKind, ActionOutcome
from .env import EcosystemEnv
from .m8 import FixedRgbScanRecoveryPolicy, M8_TASK_ID, wilson_interval
from .m81 import (
    M81_CONDITIONS,
    M81_DIAGNOSTIC_CONDITIONS,
    M81_TEST_SEEDS,
    M81_TRAIN_SEEDS,
    ActionOutcomeMemory,
    FeedForwardRgbBCPolicy,
    RecurrentRgbBCPolicy,
    fit_m81_policies,
    m81_config,
    m81_policy_fingerprint,
    m81_protocol_fingerprint,
)
from .policies import skill_action
from .trajectory import ReplayResult
from .viewer import ViewerSession


M82_PROTOCOL_VERSION = "m82-external-validity-v1"
M82_TEST_SEEDS = tuple(range(200, 220))
M82_PAIRED_WIN_MARGIN = 0.10
M82_BOOTSTRAP_DRAWS = 10_000
M82_BOOTSTRAP_SEED = 20_260_805
# Calculated from M8.1's versioned trainer, train conditions, train seeds,
# ridge value, and the two fitted weight matrices before M8.2 was defined.
M81_FROZEN_POLICY_FINGERPRINT = "bab1d79a31c850391479e37901e99231df0e3cf88eb695dda0747416999f9314"
M81_FROZEN_PROTOCOL_FINGERPRINT = "27795532c4327227d9c5a44bc03be5dc479336af87a6f4c5d68d8f9716f88d54"

# This suite was added after the M8.1 result.  It is disjoint from M8.1's
# layout, seed, and result conditions, and no M8.2 episode is used in fitting.
M82_CONDITIONS: dict[str, dict[str, Any]] = {
    "northwest_blue_west_sector": {
        "layout_id": "m82_northwest",
        "food_variant": "blue",
        "lighting_variant": "dim",
        "camera_control": "scan",
        "initial_scan_sector": "west",
    },
    "southeast_purple_south_sector": {
        "layout_id": "m82_southeast",
        "food_variant": "purple",
        "camera_control": "scan",
        "initial_scan_sector": "south",
        "dynamics_variant": "slippery",
    },
    "northwest_blue_relocation": {
        "layout_id": "m82_northwest",
        "food_variant": "blue",
        "lighting_variant": "dim",
        "camera_control": "scan",
        "initial_scan_sector": "east",
        "disturbance_step": 1,
    },
    "southeast_purple_blocked_landmark": {
        "layout_id": "m82_southeast",
        "food_variant": "purple",
        "camera_control": "scan",
        "initial_scan_sector": "north",
        "blocked_distractor": True,
        "distractor_xy": [-0.12, 0.24],
        "geometry_variant": "m82_landmark",
    },
}


@dataclass(frozen=True, slots=True)
class M82Episode:
    seed: int
    condition: str
    task_success: bool
    total_reward: float
    steps: int
    scan_actions: int
    blocked_outcomes: int
    disturbance_events: int
    post_disturbance_completion: bool


def _options(controls: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": M8_TASK_ID, **controls}


def _episode_result(
    *,
    seed: int,
    condition: str,
    task_success: bool,
    total_reward: float,
    steps: int,
    scan_actions: int,
    blocked_outcomes: int,
    disturbance_events: int,
    post_disturbance_completion: bool,
) -> M82Episode:
    return M82Episode(
        seed,
        condition,
        task_success,
        total_reward,
        steps,
        scan_actions,
        blocked_outcomes,
        disturbance_events,
        post_disturbance_completion,
    )


def _run_fixed(*, seed: int, condition: str, controls: dict[str, Any]) -> M82Episode:
    env = EcosystemEnv(m81_config())
    policy = FixedRgbScanRecoveryPolicy()
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
                return _episode_result(
                    seed=seed,
                    condition=condition,
                    task_success=bool(info["task_success"]),
                    total_reward=total_reward,
                    steps=step,
                    scan_actions=scans,
                    blocked_outcomes=blocked,
                    disturbance_events=disturbances,
                    post_disturbance_completion=bool(info["post_disturbance_completion"]),
                )
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _run_learned(
    policy: FeedForwardRgbBCPolicy | RecurrentRgbBCPolicy, *, seed: int, condition: str, controls: dict[str, Any]
) -> M82Episode:
    env = EcosystemEnv(m81_config())
    try:
        observation, _ = env.reset(seed=seed, options=_options(controls))
        memory: ActionOutcomeMemory | None = policy.reset() if isinstance(policy, RecurrentRgbBCPolicy) else None
        total_reward = 0.0
        scans = blocked = disturbances = 0
        for step in range(1, env.config.max_episode_steps + 1):
            rgb = np.asarray(observation["rgb"], dtype=np.uint8)
            action = policy.act(rgb, memory) if memory is not None else policy.act(rgb)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            scans += int(int(action["kind"]) == int(ActionKind.SCAN))
            blocked += int(info["outcome"] == ActionOutcome.BLOCKED.value)
            disturbances += int(info["disturbance"] is not None)
            if memory is not None:
                memory = policy.remember(memory, action, int(observation["prior_outcome"]))
            if terminated or truncated:
                return _episode_result(
                    seed=seed,
                    condition=condition,
                    task_success=bool(info["task_success"]),
                    total_reward=total_reward,
                    steps=step,
                    scan_actions=scans,
                    blocked_outcomes=blocked,
                    disturbance_events=disturbances,
                    post_disturbance_completion=bool(info["post_disturbance_completion"]),
                )
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _run_oracle(*, seed: int, condition: str, controls: dict[str, Any]) -> M82Episode:
    env = EcosystemEnv(m81_config(observation_mode="state_oracle"))
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
                action = skill_action(
                    ActionKind.WALK_TO,
                    observation["food_xy"],
                    max(0.1, float(np.linalg.norm(delta)) / env.config.walk_speed_per_second),
                )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            disturbances += int(info["disturbance"] is not None)
            if terminated or truncated:
                return _episode_result(
                    seed=seed,
                    condition=condition,
                    task_success=bool(info["task_success"]),
                    total_reward=total_reward,
                    steps=step,
                    scan_actions=0,
                    blocked_outcomes=0,
                    disturbance_events=disturbances,
                    post_disturbance_completion=bool(info["post_disturbance_completion"]),
                )
        raise AssertionError("environment did not terminate")
    finally:
        env.close()


def _aggregate(episodes: list[M82Episode]) -> dict[str, object]:
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


def _paired_bootstrap_interval(recurrent: list[bool], feed_forward: list[bool]) -> tuple[float, float]:
    deltas = np.asarray(recurrent, dtype=np.float64) - np.asarray(feed_forward, dtype=np.float64)
    if not len(deltas):
        raise ValueError("paired comparison needs at least one episode")
    generator = np.random.default_rng(M82_BOOTSTRAP_SEED)
    bootstrap_means = generator.choice(deltas, size=(M82_BOOTSTRAP_DRAWS, len(deltas)), replace=True).mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return float(low), float(high)


def m82_benchmark(*, test_seeds: tuple[int, ...] = M82_TEST_SEEDS) -> dict[str, object]:
    """Run the sealed M8.2 suite; custom seeds are forbidden by the protocol."""

    if test_seeds != M82_TEST_SEEDS:
        raise ValueError("M8.2 test seeds are frozen; use M82_TEST_SEEDS")
    if set(test_seeds) & (set(M81_TRAIN_SEEDS) | set(M81_TEST_SEEDS)):
        raise AssertionError("M8.2 seeds must remain disjoint from all M8.1 seeds")
    feed_forward, recurrent = fit_m81_policies()
    policy_fingerprint = m81_policy_fingerprint(feed_forward, recurrent)
    protocol_fingerprint = m81_protocol_fingerprint()
    if protocol_fingerprint != M81_FROZEN_PROTOCOL_FINGERPRINT:
        raise AssertionError("M8.1 protocol fingerprint drifted; M8.2 must not evaluate an edited protocol")
    if policy_fingerprint != M81_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.1 policy fingerprint drifted; M8.2 must not evaluate changed weights")

    results: dict[str, dict[str, object]] = {}
    paired_ff: list[bool] = []
    paired_rnn: list[bool] = []
    for condition, controls in M82_CONDITIONS.items():
        fixed = [_run_fixed(seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        ff = [_run_learned(feed_forward, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        rnn = [_run_learned(recurrent, seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        oracle = [_run_oracle(seed=seed, condition=condition, controls=controls) for seed in test_seeds]
        results[condition] = {
            "m8_fixed_rgb_baseline": _aggregate(fixed),
            "feed_forward_rgb_bc": _aggregate(ff),
            "recurrent_rgb_bc": _aggregate(rnn),
            "state_oracle_ceiling": _aggregate(oracle),
        }
        paired_ff.extend(episode.task_success for episode in ff)
        paired_rnn.extend(episode.task_success for episode in rnn)

    recurrent_only = sum(r and not f for f, r in zip(paired_ff, paired_rnn, strict=True))
    feed_forward_only = sum(f and not r for f, r in zip(paired_ff, paired_rnn, strict=True))
    paired_episodes = len(paired_ff)
    net_advantage = (recurrent_only - feed_forward_only) / paired_episodes
    ci_low, ci_high = _paired_bootstrap_interval(paired_rnn, paired_ff)
    transfer_win = net_advantage >= M82_PAIRED_WIN_MARGIN and ci_low > 0.0
    return {
        "schema_version": "0.82",
        "protocol_version": M82_PROTOCOL_VERSION,
        "m81_freeze": {
            "train_seeds": list(M81_TRAIN_SEEDS),
            "original_test_seeds": list(M81_TEST_SEEDS),
            "original_conditions": M81_CONDITIONS,
            "original_diagnostic_conditions": M81_DIAGNOSTIC_CONDITIONS,
            "frozen_protocol_fingerprint": M81_FROZEN_PROTOCOL_FINGERPRINT,
            "observed_protocol_fingerprint": protocol_fingerprint,
            "frozen_policy_fingerprint": M81_FROZEN_POLICY_FINGERPRINT,
            "observed_policy_fingerprint": policy_fingerprint,
            "m82_training_episodes": 0,
        },
        "test_seeds": list(test_seeds),
        "conditions": M82_CONDITIONS,
        "policy_boundary": {
            "m8_fixed_rgb_baseline": ["rgb", "drives", "holding_food", "prior_outcome", "policy_owned_memory"],
            "feed_forward_rgb_bc": ["rgb"],
            "recurrent_rgb_bc": ["rgb", "previous_action", "previous_outcome"],
        },
        "results": results,
        "transfer_verdict": {
            "paired_episodes": paired_episodes,
            "predeclared_net_advantage_margin": M82_PAIRED_WIN_MARGIN,
            "bootstrap_draws": M82_BOOTSTRAP_DRAWS,
            "bootstrap_seed": M82_BOOTSTRAP_SEED,
            "recurrent_only_successes": recurrent_only,
            "feed_forward_only_successes": feed_forward_only,
            "paired_net_advantage": net_advantage,
            "paired_net_advantage_bootstrap_95": [ci_low, ci_high],
            "transfer_win": transfer_win,
            "predeclared_stop_rule": "Advance to M9 only when the M8.2 paired advantage is at least 0.10 and its 95% bootstrap lower bound is above zero; otherwise retain this as a negative external-validity result and do not tune or retrain on M8.2.",
            "next_milestone": "M9 learned target grounding" if transfer_win else "stop: document negative external-validity result",
        },
        "limits": [
            "M8.2 reuses the fixed M8 RGB component adapter to ground local targets, so it does not establish end-to-end learned perception or geometry reasoning.",
            "The state-oracle result is a privileged ceiling, not an equal-input baseline.",
            "The blocked distractor remains a kinematic interaction guard rather than contact-physics or obstacle-planning evidence.",
            "M8.2 is post-result held-out only if this fixed suite remains sealed: its seeds, conditions, and outcomes must not be used to retrain or select M8.1 weights.",
        ],
    }


def write_m82_report(path: str | Path) -> dict[str, object]:
    report = m82_benchmark()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_m82_viewer_demo(trace_path: str | Path, *, seed: int = 207) -> ReplayResult:
    """Record/replay one frozen recurrent M8.2 relocation episode."""

    feed_forward, policy = fit_m81_policies()
    if m81_policy_fingerprint(feed_forward, policy) != M81_FROZEN_POLICY_FINGERPRINT:
        raise AssertionError("M8.1 policy fingerprint drifted")
    controls = M82_CONDITIONS["northwest_blue_relocation"]
    with ViewerSession(EcosystemEnv(m81_config(), render_mode="rgb_array"), trace_path=trace_path, episode_id="m82-frozen-rnn") as session:
        observation, _ = session.reset(seed=seed, options=_options(controls))
        memory = policy.reset()
        for _ in range(session.env.config.max_episode_steps):
            action = policy.act(np.asarray(observation["rgb"], dtype=np.uint8), memory)
            observation, _, terminated, truncated, _ = session.step(action)
            memory = policy.remember(memory, action, int(observation["prior_outcome"]))
            if terminated or truncated:
                break
        return session.replay(trace_path)
