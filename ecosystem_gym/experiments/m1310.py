"""M13.10: imitation-initialized RL retention screen.

M13.10 deliberately reuses the complete M13.9 policy boundary, reward,
semi-Markov replay, and evaluation kernel.  The only learned-policy changes are
the frozen imitation initialization and the common exploration schedule.
"""

from __future__ import annotations

import os

# The six spawned jobs each own exactly one numerical-library thread.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..env import EcosystemEnv
from ..tasks import LAYOUTS
from .m10 import m10_scan_coverage
from .m13 import M13Macro, M13Memory, _file_sha256, _options, compile_macro
from .m132 import (
    M132_BATCH_SIZE,
    M132_FEATURE_DIM,
    M132_REPLAY_WARMUP,
    M132_TARGET_UPDATE_EVERY,
    encode_features,
)
from .m139 import (
    BalancedM139Oracle,
    CompactM139QPolicy,
    M139RewardState,
    SeededRandomM139Policy,
    _aggregate_episode_results,
    _duration,
    _DurationReplay,
    _evaluate_policy,
    m139_config,
    m139_learning_reward,
    m139_parameter_fingerprint,
    m139_policy_fingerprint,
    m139_protocol_fingerprint,
    replay_m139_trace,
    run_m139_episode,
)
from .m139_support import assert_compact_report
from .m1310_support import (
    M1310_LEDGER_SCHEMA_VERSION,
    M1310_PARTITION_DEPENDENCIES,
    M1310_SCREEN_REPORT_SCHEMA_VERSION,
    SplitOpenLedger,
    m1310_content_hash,
    m1310_screen_promotes,
)

M1310_PROTOCOL_VERSION = "m13.10-imitation-initialized-rl-r1"
M1310_SCREEN_FIT_SEEDS = tuple(range(5_600, 5_620))
M1310_SCREEN_PROBE_SEEDS = tuple(range(5_620, 5_628))
M1310_CONFIRMATION_FIT_SEEDS = tuple(range(5_700, 5_740))
M1310_CONFIRMATION_EVALUATION_SEEDS = tuple(range(5_800, 5_820))
M1310_AUDIT_SEEDS = tuple(range(5_900, 5_920))
M1310_SCREEN_TRAINING_SEEDS = (20_260_921, 20_260_922)
M1310_CONFIRMATION_TRAINING_SEEDS = tuple(range(20_260_923, 20_260_931))
M1310_SCREEN_EPISODES = 2_000
M1310_CONFIRMATION_EPISODES = 12_000
M1310_BC_EPOCHS = 80
M1310_BC_BATCH_SIZE = 256
M1310_BC_LEARNING_RATE = 1e-3
M1310_WORKERS = 6
M1310_CANONICAL_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "m1310" / "split-open-ledger.json"
)

M1310Arm = Literal[
    "imitation_warmstart_rl",
    "random_init_rl_control",
    "imitation_only_guardrail",
]
M1310_SCREEN_ARMS: tuple[M1310Arm, ...] = (
    "imitation_warmstart_rl",
    "random_init_rl_control",
    "imitation_only_guardrail",
)


def _controls(layout_id: str, **values: Any) -> dict[str, Any]:
    return {"layout_id": layout_id, "camera_control": "scan_v2", **values}


def _conditions(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        "persistent_reference": _controls(
            f"{prefix}_northeast",
            food_variant="orange",
            toy_variant="ball",
            initial_scan_sector="north",
        ),
        "renewal_and_morphology": _controls(
            f"{prefix}_southwest",
            food_variant="purple",
            food_shape_variant="capsule",
            toy_variant="cube",
            agent_shape_variant="capsule",
            initial_scan_sector="south",
        ),
        "event_relocation": _controls(
            f"{prefix}_northwest",
            food_variant="blue",
            food_shape_variant="box",
            toy_variant="capsule",
            lighting_variant="dim",
            initial_scan_sector="east",
            event_relocation_on_first_pickup=True,
        ),
        "compound": _controls(
            f"{prefix}_southeast",
            food_variant="red",
            food_shape_variant="capsule",
            toy_variant="cube",
            agent_shape_variant="box",
            dynamics_variant="grippy",
            blocked_distractor=True,
            distractor_xy=[-0.04, 0.04],
            initial_scan_sector="west",
            event_relocation_on_first_pickup=True,
        ),
    }


M1310_SCREEN_FIT_CONDITIONS = _conditions("m1310_screen")
M1310_SCREEN_PROBE_CONDITIONS = _conditions("m1310_probe")
M1310_CONFIRMATION_FIT_CONDITIONS = _conditions("m1310_confirm_fit")
M1310_CONFIRMATION_EVALUATION_CONDITIONS = _conditions("m1310_confirm_eval")
M1310_AUDIT_CONDITIONS = _conditions("m1310_audit")


def m1310_epsilon(episode: int, *, stage: Literal["screen", "confirmation"] = "screen") -> float:
    """Frozen common exploration schedule for both RL arms."""

    if episode < 0:
        raise ValueError("episode must be non-negative")
    if stage not in {"screen", "confirmation"}:
        raise ValueError(f"unknown M13.10 stage {stage!r}")
    progress = min(episode, M1310_SCREEN_EPISODES - 1) / (M1310_SCREEN_EPISODES - 1)
    return float(0.20 + progress * (0.05 - 0.20))


def _source_hashes() -> dict[str, str]:
    experiment_dir = Path(__file__).resolve().parent
    package_dir = experiment_dir.parent
    repository_dir = package_dir.parent
    paths = {
        "experiments/m1310.py": Path(__file__),
        "experiments/m1310_support.py": experiment_dir / "m1310_support.py",
        "experiments/m139.py": experiment_dir / "m139.py",
        "experiments/m139_support.py": experiment_dir / "m139_support.py",
        "tasks.py": package_dir / "tasks.py",
        "env.py": package_dir / "env.py",
        "docs/M13_10_PROTOCOL.md": repository_dir / "docs" / "M13_10_PROTOCOL.md",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.10 fingerprint sources missing: {missing}")
    return {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(paths.items())}


def m1310_protocol_fingerprint() -> str:
    condition_sets = {
        "screen_fit": M1310_SCREEN_FIT_CONDITIONS,
        "screen_probe": M1310_SCREEN_PROBE_CONDITIONS,
        "confirmation_fit": M1310_CONFIRMATION_FIT_CONDITIONS,
        "confirmation_evaluation": M1310_CONFIRMATION_EVALUATION_CONDITIONS,
        "audit": M1310_AUDIT_CONDITIONS,
    }
    layout_ids = tuple(
        str(row["layout_id"]) for conditions in condition_sets.values() for row in conditions.values()
    )
    payload = {
        "version": M1310_PROTOCOL_VERSION,
        "m139_kernel_fingerprint": m139_protocol_fingerprint(),
        "config": asdict(m139_config()),
        "splits": {
            "screen_fit": M1310_SCREEN_FIT_SEEDS,
            "screen_probe": M1310_SCREEN_PROBE_SEEDS,
            "confirmation_fit": M1310_CONFIRMATION_FIT_SEEDS,
            "confirmation_evaluation": M1310_CONFIRMATION_EVALUATION_SEEDS,
            "audit": M1310_AUDIT_SEEDS,
        },
        "conditions": condition_sets,
        "layouts": {layout_id: asdict(LAYOUTS[layout_id]) for layout_id in layout_ids},
        "ledger": {
            "schema": M1310_LEDGER_SCHEMA_VERSION,
            "dependencies": M1310_PARTITION_DEPENDENCIES,
            "screen_report_schema": M1310_SCREEN_REPORT_SCHEMA_VERSION,
        },
        "teacher": {
            "rows_per_screen": 16_000,
            "features": M132_FEATURE_DIM,
            "classes": len(M13Macro),
            "epochs": M1310_BC_EPOCHS,
            "batch_size": M1310_BC_BATCH_SIZE,
            "learning_rate": M1310_BC_LEARNING_RATE,
            "optimizer": "adam(.9,.999,1e-8)",
            "loss": "class-balanced masked cross-entropy",
        },
        "training": {
            "screen_seeds": M1310_SCREEN_TRAINING_SEEDS,
            "confirmation_seeds": M1310_CONFIRMATION_TRAINING_SEEDS,
            "screen_episodes": M1310_SCREEN_EPISODES,
            "confirmation_episodes": M1310_CONFIRMATION_EPISODES,
            "epsilon": [0.20, 0.05, M1310_SCREEN_EPISODES],
            "rl_rng": "PCG64(seed xor 0x4D31333130)",
            "workers": M1310_WORKERS,
            "worker_threads": 1,
            "replay_warmup": M132_REPLAY_WARMUP,
            "batch_size": M132_BATCH_SIZE,
            "target_update_every": M132_TARGET_UPDATE_EVERY,
            "optimizer_state": "zeroed before RL for both RL arms",
        },
        "arms": M1310_SCREEN_ARMS,
        "sources": _source_hashes(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CompactM1310QPolicy(CompactM139QPolicy):
    """M13.9 learned policy kernel with an M13.10 experiment arm label."""

    def __init__(self, *, m1310_arm: M1310Arm, seed: int) -> None:
        if m1310_arm not in M1310_SCREEN_ARMS:
            raise ValueError(f"unknown M13.10 arm {m1310_arm!r}")
        super().__init__(arm="public_potential_candidate", seed=seed, stage="screen")
        self.m1310_arm = m1310_arm
        self.base_parameter_hash = self.initial_parameter_hash
        self.post_imitation_parameter_hash: str | None = None


def m1310_policy_fingerprint(policy: CompactM1310QPolicy) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "protocol_version": M1310_PROTOCOL_VERSION,
                "arm": policy.m1310_arm,
                "seed": policy.seed,
                "feature_dim": M132_FEATURE_DIM,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for key in sorted(policy.online.params):
        digest.update(key.encode())
        digest.update(np.asarray(policy.online.params[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _memory_from_snapshot(snapshot: dict[str, Any]) -> M13Memory:
    last = snapshot["last_macro"]
    return M13Memory(
        feed_count=int(snapshot["feed_count"]),
        play_count=int(snapshot["play_count"]),
        rest_count=int(snapshot["rest_count"]),
        food_cooldown_bucket=int(snapshot["food_cooldown_bucket"]),
        pending_recovery=bool(snapshot["pending_recovery"]),
        last_macro=None if last == "START" else M13Macro[str(last)],
    )


def _write_dataset_exclusive(
    path: Path,
    *,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, features=features, masks=masks, labels=labels)
        handle.flush()
        os.fsync(handle.fileno())


def load_m1310_teacher_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != {"features", "masks", "labels"}:
            raise ValueError("M13.10 teacher dataset has an invalid field set")
        features = np.asarray(payload["features"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=np.bool_)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    rows = features.shape[0]
    if features.shape != (rows, M132_FEATURE_DIM):
        raise ValueError("M13.10 teacher features have an invalid shape")
    if masks.shape != (rows, len(M13Macro)) or labels.shape != (rows,):
        raise ValueError("M13.10 teacher labels or masks have an invalid shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("M13.10 teacher features must be finite")
    if np.any(labels < 0) or np.any(labels >= len(M13Macro)):
        raise ValueError("M13.10 teacher label is outside the macro surface")
    if not np.all(masks[np.arange(rows), labels]):
        raise ValueError("M13.10 teacher chose an ineligible macro")
    return features, masks, labels


def _collect_teacher_dataset(*, artifact_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_root = artifact_dir / "traces" / "teacher"
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    manifest: list[dict[str, Any]] = []
    episodes = []
    for condition, controls in M1310_SCREEN_FIT_CONDITIONS.items():
        for seed in M1310_SCREEN_FIT_SEEDS:
            policy = BalancedM139Oracle()
            trace = trace_root / condition / f"seed-{seed}.jsonl"
            episode = run_m139_episode(
                policy,
                arm="public_potential_candidate",
                training_seed=0,
                seed=seed,
                condition=condition,
                controls=controls,
                trace_path=trace,
                policy_label="m1310_teacher",
                policy_fingerprint=None,
            )
            replay_m139_trace(trace, BalancedM139Oracle())
            episodes.append(episode)
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()[1:]]
            for row in rows:
                features.append(
                    encode_features(
                        row["policy_observation"],
                        _memory_from_snapshot(row["memory_before"]),
                        include_drives=True,
                        use_memory=True,
                    )
                )
                masks.append(np.asarray(row["mask"], dtype=np.bool_))
                labels.append(int(M13Macro[row["macro"]]))
            manifest.append(
                {
                    "policy": "balanced_public_oracle_teacher",
                    "condition": condition,
                    "env_seed": seed,
                    "path": str(trace.resolve()),
                    "sha256": _file_sha256(trace),
                    "replay_pass": True,
                    "replay_error": None,
                }
            )
    feature_array = np.asarray(features, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=np.bool_)
    label_array = np.asarray(labels, dtype=np.int64)
    expected_rows = (
        len(M1310_SCREEN_FIT_CONDITIONS) * len(M1310_SCREEN_FIT_SEEDS) * m139_config().max_episode_steps
    )
    if feature_array.shape != (expected_rows, M132_FEATURE_DIM):
        raise ValueError(f"M13.10 teacher dataset expected {expected_rows} rows, got {feature_array.shape}")
    dataset_path = artifact_dir / "teacher" / "screen-fit-public.npz"
    _write_dataset_exclusive(dataset_path, features=feature_array, masks=mask_array, labels=label_array)
    loaded = load_m1310_teacher_dataset(dataset_path)
    arrays = (feature_array, mask_array, label_array)
    if not all(np.array_equal(left, right) for left, right in zip(loaded, arrays, strict=True)):
        raise ValueError("M13.10 teacher dataset changed across serialization")
    coverage = m10_scan_coverage(M1310_SCREEN_FIT_CONDITIONS, seeds=M1310_SCREEN_FIT_SEEDS)
    ceiling = {
        condition: _aggregate_episode_results([row for row in episodes if row.condition == condition])
        for condition in M1310_SCREEN_FIT_CONDITIONS
    }
    class_counts = np.bincount(label_array, minlength=len(M13Macro))
    passes = (
        all(bool(row["passes"]) for row in coverage.values())
        and all(
            int(row["episodes"]) == len(M1310_SCREEN_FIT_SEEDS)
            and int(row["full_gate_success"]) == len(M1310_SCREEN_FIT_SEEDS)
            and all(int(value) == 0 for value in row["interventions"].values())
            for row in ceiling.values()
        )
        and bool(np.all(class_counts > 0))
    )
    return {
        "passes": passes,
        "coverage": coverage,
        "ceiling": ceiling,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _file_sha256(dataset_path),
            "rows": int(feature_array.shape[0]),
            "feature_dim": int(feature_array.shape[1]),
            "class_counts": {macro.name: int(class_counts[int(macro)]) for macro in M13Macro},
            "public_fields_only": True,
        },
    }, manifest


def _classification_metrics(
    policy: CompactM1310QPolicy,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    logits = policy.online.predict(features)
    predicted = np.argmax(np.where(masks, logits, -np.inf), axis=1)
    recalls: dict[str, float] = {}
    for macro in M13Macro:
        selected = labels == int(macro)
        recalls[macro.name] = float(np.mean(predicted[selected] == labels[selected]))
    non_wait = labels != int(M13Macro.WAIT)
    result = {
        "accuracy": float(np.mean(predicted == labels)),
        "non_wait_accuracy": float(np.mean(predicted[non_wait] == labels[non_wait])),
        "per_macro_recall": recalls,
    }
    result["passes"] = bool(
        result["accuracy"] >= 0.88 and result["non_wait_accuracy"] >= 0.90 and min(recalls.values()) >= 0.75
    )
    return result


def _supervised_update(
    policy: CompactM1310QPolicy,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    model = policy.online
    pre1 = features @ model.params["w1"] + model.params["b1"]
    hidden1 = np.maximum(pre1, 0.0)
    pre2 = hidden1 @ model.params["w2"] + model.params["b2"]
    hidden2 = np.maximum(pre2, 0.0)
    logits = hidden2 @ model.params["w3"] + model.params["b3"]
    masked = np.where(masks, logits, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exp = np.where(masks, np.exp(masked - maximum), 0.0)
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    row = np.arange(features.shape[0])
    sample_weights = weights[labels]
    loss = -np.mean(sample_weights * np.log(np.maximum(probabilities[row, labels], 1e-12)))
    output_gradient = probabilities
    output_gradient[row, labels] -= 1.0
    output_gradient *= (sample_weights / features.shape[0])[:, None]
    gradients: dict[str, np.ndarray] = {
        "w3": hidden2.T @ output_gradient,
        "b3": output_gradient.sum(axis=0),
    }
    middle = (output_gradient @ model.params["w3"].T) * (pre2 > 0.0)
    gradients["w2"] = hidden1.T @ middle
    gradients["b2"] = middle.sum(axis=0)
    first = (middle @ model.params["w2"].T) * (pre1 > 0.0)
    gradients["w1"] = features.T @ first
    gradients["b1"] = first.sum(axis=0)
    model.step += 1
    for key, gradient in gradients.items():
        model.m[key] = 0.9 * model.m[key] + 0.1 * gradient
        model.v[key] = 0.999 * model.v[key] + 0.001 * gradient * gradient
        corrected_m = model.m[key] / (1.0 - 0.9**model.step)
        corrected_v = model.v[key] / (1.0 - 0.999**model.step)
        model.params[key] -= M1310_BC_LEARNING_RATE * corrected_m / (np.sqrt(corrected_v) + 1e-8)
    return float(loss)


def fit_m1310_imitation(
    policy: CompactM1310QPolicy,
    *,
    dataset_path: str | Path,
    epochs: int = M1310_BC_EPOCHS,
) -> dict[str, Any]:
    if epochs != M1310_BC_EPOCHS:
        raise ValueError(f"M13.10 imitation epochs are frozen at {M1310_BC_EPOCHS}")
    features, masks, labels = load_m1310_teacher_dataset(dataset_path)
    counts = np.bincount(labels, minlength=len(M13Macro)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("M13.10 teacher dataset must contain every macro")
    weights = (labels.size / (len(M13Macro) * counts)).astype(np.float32)
    rng = np.random.default_rng(policy.seed)
    losses: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(labels.size)
        for start in range(0, labels.size, M1310_BC_BATCH_SIZE):
            indexes = order[start : start + M1310_BC_BATCH_SIZE]
            losses.append(
                _supervised_update(
                    policy,
                    features[indexes],
                    masks[indexes],
                    labels[indexes],
                    weights,
                )
            )
    metrics = _classification_metrics(policy, features, masks, labels)
    policy.post_imitation_parameter_hash = m139_parameter_fingerprint(policy)
    return {
        "epochs": epochs,
        "updates": len(losses),
        "mean_cross_entropy": float(np.mean(losses)),
        **metrics,
    }


def _zero_optimizer(policy: CompactM1310QPolicy) -> None:
    policy.online.m = {key: np.zeros_like(value) for key, value in policy.online.params.items()}
    policy.online.v = {key: np.zeros_like(value) for key, value in policy.online.params.items()}
    policy.online.step = 0
    policy.target = policy.online.copy()
    policy.update_count = 0


def train_m1310_rl(
    policy: CompactM1310QPolicy,
    *,
    episodes: int = M1310_SCREEN_EPISODES,
) -> dict[str, float]:
    if policy.m1310_arm not in {"imitation_warmstart_rl", "random_init_rl_control"}:
        raise ValueError("M13.10 imitation-only arm cannot receive RL updates")
    if episodes != M1310_SCREEN_EPISODES:
        raise ValueError(f"M13.10 screen RL budget is frozen at {M1310_SCREEN_EPISODES}")
    _zero_optimizer(policy)
    conditions = tuple(M1310_SCREEN_FIT_CONDITIONS.items())
    rng = np.random.default_rng(policy.seed ^ 0x4D31333130)
    replay = _DurationReplay()
    decisions = 0
    losses: list[float] = []
    env = EcosystemEnv(m139_config())
    try:
        for episode in range(episodes):
            _, controls = conditions[episode % len(conditions)]
            env_seed = M1310_SCREEN_FIT_SEEDS[(episode // len(conditions)) % len(M1310_SCREEN_FIT_SEEDS)]
            observation, _ = env.reset(seed=env_seed, options=_options(controls))
            memory = policy.reset()
            reward_state = M139RewardState(
                required_recovery=bool(controls.get("event_relocation_on_first_pickup"))
            )
            epsilon = m1310_epsilon(episode)
            for _ in range(policy.config.max_episode_steps):
                features = policy.features(observation, memory)
                mask = policy.mask(observation)
                if rng.random() < epsilon:
                    macro = M13Macro(int(rng.choice(np.flatnonzero(mask))))
                else:
                    macro = policy.choose(observation, memory)
                action = compile_macro(macro, observation, policy.config)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                reward, _ = m139_learning_reward(
                    arm="public_potential_candidate",
                    environment_reward=env_reward,
                    state=reward_state,
                    observation_before=observation,
                    observation_after=next_observation,
                    action=action,
                    info=info,
                    terminated=terminated,
                    truncated=truncated,
                    macro=macro,
                )
                policy.observe(
                    memory,
                    observation_before=observation,
                    macro=macro,
                    action=action,
                    observation_after=next_observation,
                )
                replay.add(
                    features,
                    int(macro),
                    reward,
                    policy.features(next_observation, memory),
                    policy.mask(next_observation),
                    terminated or truncated,
                    _duration(action),
                )
                decisions += 1
                if replay.size >= M132_REPLAY_WARMUP and decisions % policy.update_every == 0:
                    losses.append(policy._update(replay, rng))
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        env.close()
    return {
        "episodes": float(episodes),
        "decisions": float(decisions),
        "updates": float(policy.update_count),
        "target_copies": float(policy.update_count // policy.target_update_every),
        "mean_huber_loss": float(np.mean(losses)) if losses else 0.0,
        "epsilon_initial": m1310_epsilon(0),
        "epsilon_final": m1310_epsilon(episodes - 1),
    }


def _train_worker(
    arm: M1310Arm,
    seed: int,
    dataset_path: str,
) -> tuple[M1310Arm, int, CompactM1310QPolicy, dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    policy = CompactM1310QPolicy(m1310_arm=arm, seed=seed)
    imitation: dict[str, Any] = {"applied": False}
    if arm in {"imitation_warmstart_rl", "imitation_only_guardrail"}:
        imitation = {"applied": True, **fit_m1310_imitation(policy, dataset_path=dataset_path)}
    rl: dict[str, Any] = {"applied": False, "episodes": 0.0}
    if arm in {"imitation_warmstart_rl", "random_init_rl_control"}:
        rl = {"applied": True, **train_m1310_rl(policy)}
    training = {
        "imitation": imitation,
        "rl": rl,
        "elapsed_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
        "numerical_threads": 1,
    }
    hashes = {
        "base": policy.base_parameter_hash,
        "post_imitation": policy.post_imitation_parameter_hash,
        "final": m139_parameter_fingerprint(policy),
    }
    return arm, seed, policy, training, hashes


def m1310_screen_jobs() -> tuple[tuple[M1310Arm, int], ...]:
    return tuple((arm, seed) for seed in M1310_SCREEN_TRAINING_SEEDS for arm in M1310_SCREEN_ARMS)


def _train_wave(
    *, dataset_path: Path, workers: int
) -> tuple[
    dict[tuple[M1310Arm, int], CompactM1310QPolicy],
    dict[tuple[M1310Arm, int], dict[str, Any]],
    dict[tuple[M1310Arm, int], dict[str, Any]],
    dict[str, Any],
]:
    if workers != M1310_WORKERS:
        raise ValueError(f"M13.10 worker count is frozen at {M1310_WORKERS}")
    context = multiprocessing.get_context("spawn")
    policies: dict[tuple[M1310Arm, int], CompactM1310QPolicy] = {}
    training: dict[tuple[M1310Arm, int], dict[str, Any]] = {}
    hashes: dict[tuple[M1310Arm, int], dict[str, Any]] = {}
    started = time.perf_counter()
    jobs = m1310_screen_jobs()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(_train_worker, arm, seed, str(dataset_path)): (arm, seed) for arm, seed in jobs
        }
        for future in as_completed(futures):
            arm, seed, policy, result, row_hashes = future.result()
            key = (arm, seed)
            policies[key] = policy
            training[key] = result
            hashes[key] = row_hashes
    if set(policies) != set(jobs):
        raise ValueError("M13.10 six-job wave did not return every policy")
    for seed in M1310_SCREEN_TRAINING_SEEDS:
        rows = [hashes[(arm, seed)] for arm in M1310_SCREEN_ARMS]
        if len({str(row["base"]) for row in rows}) != 1:
            raise ValueError(f"M13.10 seed {seed} arms do not share identical base weights")
        candidate = hashes[("imitation_warmstart_rl", seed)]["post_imitation"]
        guardrail = hashes[("imitation_only_guardrail", seed)]["post_imitation"]
        if candidate is None or candidate != guardrail:
            raise ValueError(f"M13.10 seed {seed} imitation weights are not identical")
    return (
        policies,
        training,
        hashes,
        {
            "start_method": "spawn",
            "workers": workers,
            "worker_threads": 1,
            "jobs": len(jobs),
            "distinct_worker_pids": sorted({int(row["pid"]) for row in training.values()}),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def write_m1310_policy(
    path: str | Path,
    policy: CompactM1310QPolicy,
    *,
    dataset_sha256: str,
) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "m1310-policy-v1",
        "protocol_fingerprint": m1310_protocol_fingerprint(),
        "m139_kernel_fingerprint": m139_protocol_fingerprint(),
        "arm": policy.m1310_arm,
        "seed": policy.seed,
        "dataset_sha256": dataset_sha256,
        "base_parameter_hash": policy.base_parameter_hash,
        "post_imitation_parameter_hash": policy.post_imitation_parameter_hash,
        "network": {key: value.tolist() for key, value in policy.online.params.items()},
        "policy_fingerprint": m1310_policy_fingerprint(policy),
        "m139_policy_fingerprint": m139_policy_fingerprint(policy),
    }
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(output.resolve()),
        "sha256": _file_sha256(output),
        "policy_fingerprint": m1310_policy_fingerprint(policy),
        "m139_policy_fingerprint": m139_policy_fingerprint(policy),
    }


def load_m1310_policy(
    path: str | Path,
    *,
    expected_dataset_sha256: str,
) -> CompactM1310QPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m1310-policy-v1":
        raise ValueError("unsupported M13.10 policy schema")
    if payload.get("protocol_fingerprint") != m1310_protocol_fingerprint():
        raise ValueError("M13.10 policy protocol fingerprint mismatch")
    if payload.get("m139_kernel_fingerprint") != m139_protocol_fingerprint():
        raise ValueError("M13.10 policy M13.9 kernel fingerprint mismatch")
    if payload.get("dataset_sha256") != expected_dataset_sha256:
        raise ValueError("M13.10 policy teacher dataset hash mismatch")
    policy = CompactM1310QPolicy(m1310_arm=payload["arm"], seed=int(payload["seed"]))
    if payload.get("base_parameter_hash") != policy.base_parameter_hash:
        raise ValueError("M13.10 policy base initialization mismatch")
    network = payload.get("network")
    if not isinstance(network, dict) or set(network) != set(policy.online.params):
        raise ValueError("M13.10 policy parameter set is invalid")
    for key, values in network.items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != policy.online.params[key].shape or not np.all(np.isfinite(array)):
            raise ValueError(f"M13.10 policy parameter {key!r} is invalid")
        policy.online.params[key] = array
    post_imitation = payload.get("post_imitation_parameter_hash")
    policy.post_imitation_parameter_hash = None if post_imitation is None else str(post_imitation)
    policy.target = policy.online.copy()
    if payload.get("policy_fingerprint") != m1310_policy_fingerprint(policy):
        raise ValueError("M13.10 policy fingerprint mismatch")
    if payload.get("m139_policy_fingerprint") != m139_policy_fingerprint(policy):
        raise ValueError("M13.10 policy kernel fingerprint mismatch")
    return policy


def _serialize_policies(
    policies: dict[tuple[M1310Arm, int], CompactM1310QPolicy],
    *,
    policy_dir: Path,
    dataset_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[M1310Arm, int], CompactM1310QPolicy]]:
    artifacts: dict[str, Any] = {}
    restored: dict[tuple[M1310Arm, int], CompactM1310QPolicy] = {}
    for seed in M1310_SCREEN_TRAINING_SEEDS:
        artifacts[str(seed)] = {}
        for arm in M1310_SCREEN_ARMS:
            artifact = write_m1310_policy(
                policy_dir / f"seed-{seed}" / f"{arm}.json",
                policies[(arm, seed)],
                dataset_sha256=dataset_sha256,
            )
            loaded = load_m1310_policy(artifact["path"], expected_dataset_sha256=dataset_sha256)
            artifacts[str(seed)][arm] = artifact
            restored[(arm, seed)] = loaded
    if len(restored) != len(m1310_screen_jobs()):
        raise ValueError("M13.10 did not serialize and restore all six policies")
    return artifacts, restored


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"M13.10 report already exists: {path}")
    assert_compact_report(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _report_hash(report: dict[str, Any]) -> str:
    return m1310_content_hash({key: value for key, value in report.items() if key != "content_hash"})


def m1310_screen(
    *,
    artifact_dir: str | Path,
    ledger_path: str | Path = M1310_CANONICAL_LEDGER_PATH,
    workers: int = M1310_WORKERS,
) -> dict[str, Any]:
    """Run the frozen M13.10 fit, six-job wave, and one-shot screen."""

    if workers != M1310_WORKERS:
        raise ValueError(f"M13.10 worker count is frozen at {M1310_WORKERS}")
    protocol_fingerprint = m1310_protocol_fingerprint()
    artifacts = Path(artifact_dir)
    if artifacts.exists():
        raise FileExistsError(f"M13.10 screen artifact directory already exists: {artifacts}")
    ledger = SplitOpenLedger(ledger_path, protocol_fingerprint=protocol_fingerprint)
    ledger.ensure_screen_fit_open()
    artifacts.mkdir(parents=True)
    preflight, manifest = _collect_teacher_dataset(artifact_dir=artifacts)
    if not bool(preflight["passes"]):
        report = {
            "schema_version": "m13.10-screen-preflight-reject-r1",
            "stage": "screen",
            "protocol_version": M1310_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_ledger": ledger.snapshot(),
            "preflight": preflight,
            "trace_manifest": manifest,
            "screen": {"status": "reject", "passes": False, "phase": "preflight"},
            "limits": ["The one-shot probe was not opened.", "Confirmation and audit remain sealed."],
        }
        report["content_hash"] = _report_hash(report)
        return report
    dataset = preflight["dataset"]
    policies, training, hashes, execution = _train_wave(
        dataset_path=Path(str(dataset["path"])), workers=workers
    )
    imitation_gate = all(
        bool(training[(arm, seed)]["imitation"]["passes"])
        for seed in M1310_SCREEN_TRAINING_SEEDS
        for arm in ("imitation_warmstart_rl", "imitation_only_guardrail")
    )
    if not imitation_gate:
        report = {
            "schema_version": "m13.10-screen-fit-gate-reject-r1",
            "stage": "screen",
            "protocol_version": M1310_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_ledger": ledger.snapshot(),
            "preflight": preflight,
            "execution": execution,
            "training": {
                str(seed): {
                    arm: training[(arm, seed)] | {"parameter_hashes": hashes[(arm, seed)]}
                    for arm in M1310_SCREEN_ARMS
                }
                for seed in M1310_SCREEN_TRAINING_SEEDS
            },
            "trace_manifest": manifest,
            "screen": {"status": "reject", "passes": False, "phase": "fit_gate"},
            "limits": ["The one-shot probe was not opened.", "Confirmation and audit remain sealed."],
        }
        report["content_hash"] = _report_hash(report)
        return report
    policy_artifacts, restored = _serialize_policies(
        policies,
        policy_dir=artifacts / "policies",
        dataset_sha256=str(dataset["sha256"]),
    )
    ledger.open_partition("screen_probe")
    probe_coverage = m10_scan_coverage(M1310_SCREEN_PROBE_CONDITIONS, seeds=M1310_SCREEN_PROBE_SEEDS)
    oracle_rows, oracle_aggregates, probe_manifest = _evaluate_policy(
        BalancedM139Oracle(),
        arm="public_potential_candidate",
        training_seed=0,
        conditions=M1310_SCREEN_PROBE_CONDITIONS,
        seeds=M1310_SCREEN_PROBE_SEEDS,
        trace_dir=artifacts / "traces" / "probe_gate",
        label="balanced_public_oracle",
        policy_fingerprint=None,
        replay_mode="deterministic",
    )
    manifest.extend(probe_manifest)
    probe_gate = {
        "passes": all(bool(row["passes"]) for row in probe_coverage.values())
        and all(
            int(row["episodes"]) == len(M1310_SCREEN_PROBE_SEEDS)
            and int(row["full_gate_success"]) == len(M1310_SCREEN_PROBE_SEEDS)
            and all(int(value) == 0 for value in row["interventions"].values())
            for row in oracle_aggregates.values()
        )
        and all(bool(row["replay_pass"]) for row in probe_manifest),
        "coverage": probe_coverage,
        "balanced_public_oracle": {
            "aggregates": oracle_aggregates,
            "compact_episode_summaries": oracle_rows,
        },
    }
    if not bool(probe_gate["passes"]):
        report = {
            "schema_version": "m13.10-screen-probe-mechanics-reject-r1",
            "stage": "screen",
            "protocol_version": M1310_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_ledger": ledger.snapshot(),
            "preflight": preflight,
            "probe_gate": probe_gate,
            "execution": execution,
            "policy_artifacts": policy_artifacts,
            "trace_manifest": manifest,
            "screen": {"status": "reject", "passes": False, "phase": "probe_mechanics"},
            "limits": ["No learned probe score was produced.", "Confirmation and audit remain sealed."],
        }
        report["content_hash"] = _report_hash(report)
        return report
    episode_summaries: dict[str, list[dict[str, Any]]] = {arm: [] for arm in (*M1310_SCREEN_ARMS, "random")}
    aggregates: dict[str, Any] = {}
    for seed in M1310_SCREEN_TRAINING_SEEDS:
        aggregates[str(seed)] = {}
        for arm in M1310_SCREEN_ARMS:
            policy = restored[(arm, seed)]
            rows, arm_aggregates, traces = _evaluate_policy(
                policy,
                arm="public_potential_candidate",
                training_seed=seed,
                conditions=M1310_SCREEN_PROBE_CONDITIONS,
                seeds=M1310_SCREEN_PROBE_SEEDS,
                trace_dir=artifacts / "traces" / "probe",
                label=f"seed-{seed}/{arm}",
                policy_fingerprint=m139_policy_fingerprint(policy),
                replay_mode="learned",
            )
            episode_summaries[arm].extend(rows)
            aggregates[str(seed)][arm] = arm_aggregates
            manifest.extend(traces)
        random_policy = SeededRandomM139Policy(seed)
        rows, random_aggregates, traces = _evaluate_policy(
            random_policy,
            arm="public_potential_candidate",
            training_seed=seed,
            conditions=M1310_SCREEN_PROBE_CONDITIONS,
            seeds=M1310_SCREEN_PROBE_SEEDS,
            trace_dir=artifacts / "traces" / "probe",
            label=f"seed-{seed}/random",
            policy_fingerprint=f"pcg64-{seed}",
            replay_mode="random",
        )
        episode_summaries["random"].extend(rows)
        aggregates[str(seed)]["random"] = random_aggregates
        manifest.extend(traces)
    promotion = m1310_screen_promotes(
        episode_summaries["imitation_warmstart_rl"],
        episode_summaries["random_init_rl_control"],
        episode_summaries["imitation_only_guardrail"],
        episode_summaries["random"],
        training_seeds=M1310_SCREEN_TRAINING_SEEDS,
        conditions=tuple(M1310_SCREEN_PROBE_CONDITIONS),
        env_seeds=M1310_SCREEN_PROBE_SEEDS,
    )
    report = {
        "schema_version": M1310_SCREEN_REPORT_SCHEMA_VERSION,
        "stage": "screen",
        "protocol_version": M1310_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "split_ledger": ledger.snapshot(),
        "preflight": preflight,
        "probe_gate": probe_gate,
        "execution": execution,
        "training": {
            str(seed): {
                arm: training[(arm, seed)] | {"parameter_hashes": hashes[(arm, seed)]}
                for arm in M1310_SCREEN_ARMS
            }
            for seed in M1310_SCREEN_TRAINING_SEEDS
        },
        "policy_artifacts": policy_artifacts,
        "episode_summaries": episode_summaries,
        "aggregates": aggregates,
        "promotion": promotion,
        "trace_manifest": manifest,
        "screen": {
            "status": "promote" if bool(promotion["passes"]) else "reject",
            "passes": bool(promotion["passes"]),
            "phase": "probe",
        },
        "limits": [
            "A screen promotion authorizes confirmation; it is not a final claim.",
            "No confirmation evaluation or audit data is present.",
        ],
    }
    report["content_hash"] = _report_hash(report)
    return report


def write_m1310_screen_report(
    path: str | Path,
    *,
    ledger_path: str | Path = M1310_CANONICAL_LEDGER_PATH,
) -> dict[str, Any]:
    output = Path(path)
    artifacts = output.parent / f"{output.stem}-artifacts"
    if output.exists() or artifacts.exists():
        raise FileExistsError("M13.10 screen report or artifact target already exists")
    report = m1310_screen(artifact_dir=artifacts, ledger_path=ledger_path)
    _write_json_exclusive(output, report)
    return report


__all__ = [
    "M1310_AUDIT_CONDITIONS",
    "M1310_AUDIT_SEEDS",
    "M1310_CANONICAL_LEDGER_PATH",
    "M1310_CONFIRMATION_EVALUATION_CONDITIONS",
    "M1310_CONFIRMATION_EVALUATION_SEEDS",
    "M1310_CONFIRMATION_FIT_CONDITIONS",
    "M1310_CONFIRMATION_FIT_SEEDS",
    "M1310_PROTOCOL_VERSION",
    "M1310_SCREEN_ARMS",
    "M1310_SCREEN_FIT_CONDITIONS",
    "M1310_SCREEN_FIT_SEEDS",
    "M1310_SCREEN_PROBE_CONDITIONS",
    "M1310_SCREEN_PROBE_SEEDS",
    "M1310_SCREEN_TRAINING_SEEDS",
    "CompactM1310QPolicy",
    "fit_m1310_imitation",
    "load_m1310_policy",
    "load_m1310_teacher_dataset",
    "m1310_epsilon",
    "m1310_policy_fingerprint",
    "m1310_protocol_fingerprint",
    "m1310_screen",
    "m1310_screen_jobs",
    "train_m1310_rl",
    "write_m1310_policy",
    "write_m1310_screen_report",
]
