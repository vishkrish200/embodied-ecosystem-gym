"""Pure protocol guards for the frozen M13.9 experiment.

This module deliberately knows nothing about environments, policies, or report
output locations.  It provides the small pieces whose behaviour must remain
stable while the comparatively large experiment runner evolves: the one-way
split ledger, compact gate evidence, paired gates, and screen-report sealing.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

M139_LEDGER_SCHEMA_VERSION = "m13.9-split-ledger-r1"
M139_SCREEN_REPORT_SCHEMA_VERSION = "m13.9-screen-report-r1"
M139_CONDITIONS = (
    "persistent_reference",
    "renewal_and_morphology",
    "event_relocation",
    "compound",
)
M139_RECOVERY_CONDITIONS = frozenset({"event_relocation", "compound"})
M139_PARTITION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "screen_fit": (),
    "screen_probe": ("screen_fit",),
    "confirmation_fit": ("screen_probe",),
    "confirmation_evaluation": ("confirmation_fit",),
    "audit": ("confirmation_evaluation",),
}

_SUMMARY_FIELDS = (
    "training_seed",
    "condition",
    "env_seed",
    "survived",
    "maintenance_complete",
    "full_gate_success",
    "decision_safe_fraction",
    "duration_safe_fraction",
    "recovery_required",
    "recovery_complete",
    "decision_steps",
    "wait_decisions",
    "unsafe_wait_decisions",
    "unsafe_wait_fraction",
    "conformance_violations",
    "replay_violations",
)
_REPORT_ARMS = (
    "public_potential_candidate",
    "static_cost_control",
    "legacy_guardrail",
    "random",
)
_EPSILON = 1e-12


class M139ProtocolError(ValueError):
    """Raised when evidence does not conform to the frozen M13.9 protocol."""


class SplitLedgerError(M139ProtocolError):
    """Base class for one-way split-ledger failures."""


class SplitAlreadyOpenedError(SplitLedgerError):
    """Raised when code attempts to touch an already-open partition again."""


class SplitDependencyError(SplitLedgerError):
    """Raised when a partition is opened before its prerequisites."""


class ScreenReportVerificationError(M139ProtocolError):
    """Raised when a report cannot authorize confirmation."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M139ProtocolError("M13.9 evidence is not canonical-JSON serializable") from exc
    return (encoded + "\n").encode("utf-8")


def m139_content_hash(value: object) -> str:
    """Return the canonical JSON SHA-256 used by M13.9 ledgers/reports."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class SplitOpenLedger:
    """Output-independent, atomic, append-only logical partition ledger.

    ``open_partition`` is intentionally the only mutator.  Call it *before*
    constructing an environment or reading a seed/layout from that partition.
    The marker survives deletion or relocation of experiment output because
    the ledger path is injected independently by the caller.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        protocol_fingerprint: str,
        dependencies: Mapping[str, Sequence[str]] = M139_PARTITION_DEPENDENCIES,
    ) -> None:
        if not isinstance(protocol_fingerprint, str) or not protocol_fingerprint:
            raise SplitLedgerError("protocol_fingerprint must be a non-empty string")
        normalized: dict[str, tuple[str, ...]] = {}
        for partition, prerequisites in dependencies.items():
            if not isinstance(partition, str) or not partition:
                raise SplitLedgerError("partition names must be non-empty strings")
            normalized[partition] = tuple(prerequisites)
        unknown = {
            prerequisite
            for prerequisites in normalized.values()
            for prerequisite in prerequisites
            if prerequisite not in normalized
        }
        if unknown:
            raise SplitLedgerError(f"unknown partition dependencies: {sorted(unknown)!r}")
        self.path = Path(path)
        self.protocol_fingerprint = protocol_fingerprint
        self.dependencies = normalized
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _empty(self) -> dict[str, object]:
        return {
            "schema_version": M139_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_partitions": [],
            "marker_hashes": {},
        }

    def _read_unlocked(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitLedgerError("M13.9 split ledger is unreadable") from exc
        if not isinstance(payload, dict):
            raise SplitLedgerError("M13.9 split ledger root must be an object")
        if payload.get("schema_version") != M139_LEDGER_SCHEMA_VERSION:
            raise SplitLedgerError("M13.9 split ledger schema mismatch")
        if payload.get("protocol_fingerprint") != self.protocol_fingerprint:
            raise SplitLedgerError("M13.9 split ledger protocol fingerprint mismatch")
        opened = payload.get("opened_partitions")
        marker_hashes = payload.get("marker_hashes")
        if not isinstance(opened, list) or any(not isinstance(row, str) for row in opened):
            raise SplitLedgerError("M13.9 split ledger has invalid opened_partitions")
        if len(opened) != len(set(opened)):
            raise SplitLedgerError("M13.9 split ledger contains duplicate partition markers")
        if any(row not in self.dependencies for row in opened):
            raise SplitLedgerError("M13.9 split ledger contains an unknown partition")
        for index, partition in enumerate(opened):
            earlier = set(opened[:index])
            if not set(self.dependencies[partition]).issubset(earlier):
                raise SplitLedgerError("M13.9 split ledger violates dependency ordering")
        if not isinstance(marker_hashes, dict) or set(marker_hashes) != set(opened):
            raise SplitLedgerError("M13.9 split ledger marker hashes are incomplete")
        for index, partition in enumerate(opened):
            expected = m139_content_hash(
                {
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "partition": partition,
                    "ordinal": index,
                }
            )
            if not hmac.compare_digest(str(marker_hashes[partition]), expected):
                raise SplitLedgerError("M13.9 split ledger marker hash mismatch")
        return {
            "schema_version": M139_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_partitions": list(opened),
            "marker_hashes": dict(marker_hashes),
        }

    def snapshot(self) -> dict[str, object]:
        """Read and validate the ledger without opening any partition."""

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def open_partition(self, partition: str) -> dict[str, object]:
        """Atomically mark ``partition`` opened and reject every later reopen."""

        if partition not in self.dependencies:
            raise SplitLedgerError(f"unknown M13.9 partition {partition!r}")
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                payload = self._read_unlocked()
                opened = list(payload["opened_partitions"])
                if partition in opened:
                    raise SplitAlreadyOpenedError(f"M13.9 partition {partition!r} was already opened")
                missing = [
                    dependency for dependency in self.dependencies[partition] if dependency not in opened
                ]
                if missing:
                    raise SplitDependencyError(f"M13.9 partition {partition!r} requires {missing!r} first")
                opened.append(partition)
                marker_hashes = dict(payload["marker_hashes"])
                marker_hashes[partition] = m139_content_hash(
                    {
                        "protocol_fingerprint": self.protocol_fingerprint,
                        "partition": partition,
                        "ordinal": len(opened) - 1,
                    }
                )
                updated = {**payload, "opened_partitions": opened, "marker_hashes": marker_hashes}
                _atomic_write(self.path, updated)
                return updated
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def ensure_screen_fit_open(self) -> dict[str, object]:
        """Mark screen fit once, or reuse only that non-held-out marker."""

        snapshot = self.snapshot()
        opened = list(snapshot["opened_partitions"])
        if not opened:
            return self.open_partition("screen_fit")
        if opened == ["screen_fit"]:
            return snapshot
        raise SplitLedgerError("screen-fit access is unavailable after a held-out M13.9 partition opened")


@dataclass(frozen=True, slots=True)
class CompactEpisodeSummary:
    """The complete episode-level evidence consumed by M13.9 gates."""

    training_seed: int
    condition: str
    env_seed: int
    survived: bool
    maintenance_complete: bool
    full_gate_success: bool
    decision_safe_fraction: float
    duration_safe_fraction: float
    recovery_required: bool
    recovery_complete: bool
    decision_steps: int
    wait_decisions: int
    unsafe_wait_decisions: int
    unsafe_wait_fraction: float
    conformance_violations: int = 0
    replay_violations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.training_seed, Integral) or isinstance(self.training_seed, bool):
            raise M139ProtocolError("training_seed must be an integer")
        if not isinstance(self.env_seed, Integral) or isinstance(self.env_seed, bool):
            raise M139ProtocolError("env_seed must be an integer")
        if not isinstance(self.condition, str) or not self.condition:
            raise M139ProtocolError("condition must be a non-empty string")
        for name in (
            "survived",
            "maintenance_complete",
            "full_gate_success",
            "recovery_required",
            "recovery_complete",
        ):
            if not isinstance(getattr(self, name), bool):
                raise M139ProtocolError(f"{name} must be a bool")
        for name in ("decision_safe_fraction", "duration_safe_fraction", "unsafe_wait_fraction"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise M139ProtocolError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise M139ProtocolError(f"{name} must be finite and in [0, 1]")
        for name in (
            "decision_steps",
            "wait_decisions",
            "unsafe_wait_decisions",
            "conformance_violations",
            "replay_violations",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
                raise M139ProtocolError(f"{name} must be a non-negative integer")
        if self.maintenance_complete and not self.survived:
            raise M139ProtocolError("maintenance_complete cannot be true without survival")
        if self.recovery_complete and not self.recovery_required:
            raise M139ProtocolError("recovery_complete cannot be true when recovery is not required")
        if self.full_gate_success and not self.maintenance_complete:
            raise M139ProtocolError("full_gate_success cannot be true without maintenance")
        if self.unsafe_wait_decisions > self.wait_decisions:
            raise M139ProtocolError("unsafe_wait_decisions cannot exceed wait_decisions")
        expected_unsafe = self.unsafe_wait_decisions / self.wait_decisions if self.wait_decisions else 0.0
        if not math.isclose(self.unsafe_wait_fraction, expected_unsafe, rel_tol=1e-12, abs_tol=1e-12):
            raise M139ProtocolError("unsafe_wait_fraction disagrees with its exact counts")
        if self.wait_decisions > self.decision_steps:
            raise M139ProtocolError("wait_decisions cannot exceed decision_steps")

    def compact_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["training_seed"] = int(self.training_seed)
        result["env_seed"] = int(self.env_seed)
        result["decision_safe_fraction"] = float(self.decision_safe_fraction)
        result["duration_safe_fraction"] = float(self.duration_safe_fraction)
        result["unsafe_wait_fraction"] = float(self.unsafe_wait_fraction)
        result["wait_decisions"] = int(self.wait_decisions)
        result["decision_steps"] = int(self.decision_steps)
        result["unsafe_wait_decisions"] = int(self.unsafe_wait_decisions)
        result["conformance_violations"] = int(self.conformance_violations)
        result["replay_violations"] = int(self.replay_violations)
        return result


def _coerce_summary(value: CompactEpisodeSummary | Mapping[str, object]) -> CompactEpisodeSummary:
    if isinstance(value, CompactEpisodeSummary):
        return value
    if not isinstance(value, Mapping):
        raise M139ProtocolError("episode summaries must be CompactEpisodeSummary or mappings")
    missing = [field for field in _SUMMARY_FIELDS if field not in value]
    if missing:
        raise M139ProtocolError(f"compact episode summary is missing {missing!r}")
    return CompactEpisodeSummary(**{field: value[field] for field in _SUMMARY_FIELDS})  # type: ignore[arg-type]


def compact_episode_summary(
    value: CompactEpisodeSummary | Mapping[str, object],
) -> dict[str, object]:
    """Return only gate evidence; step traces and episode details cannot leak in."""

    return _coerce_summary(value).compact_dict()


def assert_compact_report(value: object) -> None:
    """Reject accidentally embedded transition/episode-detail payloads."""

    forbidden = {"steps", "episodes_detail"}

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            bad = forbidden.intersection(str(key) for key in node)
            if bad:
                raise M139ProtocolError(f"compact M13.9 report contains {sorted(bad)!r}")
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)


def _validated_axes(
    *,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
    required_replicas: int,
    required_env_seeds: int,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    replicas = tuple(training_seeds)
    condition_names = tuple(conditions)
    environments = tuple(env_seeds)
    if len(replicas) != required_replicas or len(set(replicas)) != len(replicas):
        raise M139ProtocolError(f"M13.9 gate requires exactly {required_replicas} unique training seeds")
    if any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in replicas):
        raise M139ProtocolError("training seeds must be integers")
    if len(condition_names) != 4 or set(condition_names) != set(M139_CONDITIONS):
        raise M139ProtocolError("M13.9 gate requires the exact four frozen conditions")
    if any(not isinstance(name, str) or not name for name in condition_names):
        raise M139ProtocolError("condition names must be non-empty strings")
    if len(environments) != required_env_seeds or len(set(environments)) != len(environments):
        raise M139ProtocolError(f"M13.9 gate requires exactly {required_env_seeds} unique environment seeds")
    if any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in environments):
        raise M139ProtocolError("environment seeds must be integers")
    return (
        tuple(int(seed) for seed in replicas),
        condition_names,
        tuple(int(seed) for seed in environments),
    )


def _index_summaries(
    values: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: tuple[int, ...],
    conditions: tuple[str, ...],
    env_seeds: tuple[int, ...],
    arm: str,
) -> dict[int, dict[tuple[str, int], CompactEpisodeSummary]]:
    expected = {(condition, env_seed) for condition in conditions for env_seed in env_seeds}
    indexed = {training_seed: {} for training_seed in training_seeds}
    for value in values:
        summary = _coerce_summary(value)
        if int(summary.training_seed) not in indexed:
            raise M139ProtocolError(
                f"{arm} evidence contains undeclared training seed {summary.training_seed}"
            )
        key = (summary.condition, int(summary.env_seed))
        if key not in expected:
            raise M139ProtocolError(f"{arm} evidence contains undeclared pair {key!r}")
        replica = indexed[int(summary.training_seed)]
        if key in replica:
            raise M139ProtocolError(f"{arm} evidence duplicates {(int(summary.training_seed), *key)!r}")
        replica[key] = summary
    for training_seed, replica in indexed.items():
        missing = expected.difference(replica)
        extra = set(replica).difference(expected)
        if missing or extra:
            raise M139ProtocolError(
                f"{arm} seed {training_seed} has mismatched paired evidence; "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
    return indexed


def _paired_indexes(
    candidate: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    static_control: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    legacy_guardrail: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: tuple[int, ...],
    conditions: tuple[str, ...],
    env_seeds: tuple[int, ...],
) -> tuple[
    dict[int, dict[tuple[str, int], CompactEpisodeSummary]],
    dict[int, dict[tuple[str, int], CompactEpisodeSummary]],
    dict[int, dict[tuple[str, int], CompactEpisodeSummary]],
    dict[int, dict[tuple[str, int], CompactEpisodeSummary]],
]:
    arms = (
        _index_summaries(
            candidate,
            training_seeds=training_seeds,
            conditions=conditions,
            env_seeds=env_seeds,
            arm="candidate",
        ),
        _index_summaries(
            static_control,
            training_seeds=training_seeds,
            conditions=conditions,
            env_seeds=env_seeds,
            arm="static_cost_control",
        ),
        _index_summaries(
            legacy_guardrail,
            training_seeds=training_seeds,
            conditions=conditions,
            env_seeds=env_seeds,
            arm="legacy_guardrail",
        ),
        _index_summaries(
            random,
            training_seeds=training_seeds,
            conditions=conditions,
            env_seeds=env_seeds,
            arm="random",
        ),
    )
    for training_seed in training_seeds:
        for key in arms[0][training_seed]:
            requirements = {arm[training_seed][key].recovery_required for arm in arms}
            if len(requirements) != 1:
                raise M139ProtocolError(
                    f"paired arms disagree on recovery requirement at {(training_seed, *key)!r}"
                )
            expected_recovery = key[0] in M139_RECOVERY_CONDITIONS
            if requirements != {expected_recovery}:
                raise M139ProtocolError(
                    f"evidence has the wrong recovery requirement at {(training_seed, *key)!r}"
                )
    return arms


def _mean(values: Iterable[float]) -> float:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise M139ProtocolError("cannot aggregate empty M13.9 evidence")
    return sum(rows) / len(rows)


def _integrity_pass(rows: Iterable[CompactEpisodeSummary]) -> bool:
    return all(row.conformance_violations == 0 and row.replay_violations == 0 for row in rows)


def _condition_metrics(
    rows: Sequence[CompactEpisodeSummary],
    *,
    survival_required: int,
    maintenance_required: int,
    recovery_required_count: int,
    safe_required: float,
    unsafe_wait_maximum: float,
) -> dict[str, object]:
    count = len(rows)
    recovery_flags = {row.recovery_required for row in rows}
    if len(recovery_flags) != 1:
        raise M139ProtocolError("recovery requirement must be constant within a condition")
    recovery_required = recovery_flags.pop()
    survival_count = sum(row.survived for row in rows)
    maintenance_count = sum(row.maintenance_complete for row in rows)
    recovery_count = sum(row.recovery_complete for row in rows)
    decision_safe = _mean(row.decision_safe_fraction for row in rows)
    duration_safe = _mean(row.duration_safe_fraction for row in rows)
    wait_decisions = sum(row.wait_decisions for row in rows)
    unsafe_wait_decisions = sum(row.unsafe_wait_decisions for row in rows)
    unsafe_wait_fraction = unsafe_wait_decisions / wait_decisions if wait_decisions else 0.0
    integrity = _integrity_pass(rows)
    recovery_pass = not recovery_required or recovery_count >= recovery_required_count
    result: dict[str, object] = {
        "episodes": count,
        "survival_count": survival_count,
        "survival_rate": survival_count / count,
        "maintenance_count": maintenance_count,
        "maintenance_rate": maintenance_count / count,
        "full_objective_count": sum(row.full_gate_success for row in rows),
        "full_objective_rate": sum(row.full_gate_success for row in rows) / count,
        "mean_decision_safe_fraction": decision_safe,
        "mean_duration_safe_fraction": duration_safe,
        "wait_decisions": wait_decisions,
        "unsafe_wait_decisions": unsafe_wait_decisions,
        "unsafe_wait_fraction": unsafe_wait_fraction,
        "recovery_required": recovery_required,
        "recovery_count": recovery_count,
        "recovery_rate": recovery_count / count if recovery_required else None,
        "integrity_pass": integrity,
        "survival_pass": survival_count >= survival_required,
        "maintenance_pass": maintenance_count >= maintenance_required,
        "decision_safe_pass": decision_safe + _EPSILON >= safe_required,
        "duration_safe_pass": duration_safe + _EPSILON >= safe_required,
        "unsafe_wait_pass": unsafe_wait_fraction <= unsafe_wait_maximum + _EPSILON,
        "recovery_pass": recovery_pass,
    }
    result["passes"] = all(
        bool(result[name])
        for name in (
            "integrity_pass",
            "survival_pass",
            "maintenance_pass",
            "decision_safe_pass",
            "duration_safe_pass",
            "unsafe_wait_pass",
            "recovery_pass",
        )
    )
    return result


def _pooled_metrics(rows: Iterable[CompactEpisodeSummary]) -> dict[str, float]:
    episodes = tuple(rows)
    wait_decisions = sum(row.wait_decisions for row in episodes)
    unsafe_wait_decisions = sum(row.unsafe_wait_decisions for row in episodes)
    return {
        "survival_rate": _mean(float(row.survived) for row in episodes),
        "maintenance_rate": _mean(float(row.maintenance_complete) for row in episodes),
        "full_objective_rate": _mean(float(row.full_gate_success) for row in episodes),
        "decision_safe_fraction": _mean(row.decision_safe_fraction for row in episodes),
        "duration_safe_fraction": _mean(row.duration_safe_fraction for row in episodes),
        "wait_occupancy": wait_decisions / sum(row.decision_steps for row in episodes),
        "unsafe_wait_fraction": unsafe_wait_decisions / wait_decisions if wait_decisions else 0.0,
    }


def m139_screen_promotes(
    candidate: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    static_control: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    legacy_guardrail: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
) -> dict[str, object]:
    """Recompute the frozen four-arm, two-replica M13.9 screen gate."""

    replicas, condition_names, environments = _validated_axes(
        training_seeds=training_seeds,
        conditions=conditions,
        env_seeds=env_seeds,
        required_replicas=2,
        required_env_seeds=8,
    )
    candidate_index, static_index, legacy_index, random_index = _paired_indexes(
        candidate,
        static_control,
        legacy_guardrail,
        random,
        training_seeds=replicas,
        conditions=condition_names,
        env_seeds=environments,
    )
    per_replica: dict[str, object] = {}
    strong_effect_pairs: list[int] = []
    for training_seed in replicas:
        candidate_rows = tuple(candidate_index[training_seed].values())
        static_rows = tuple(static_index[training_seed].values())
        legacy_rows = tuple(legacy_index[training_seed].values())
        random_rows = tuple(random_index[training_seed].values())
        condition_gates: dict[str, object] = {}
        for condition in condition_names:
            rows = tuple(candidate_index[training_seed][(condition, env_seed)] for env_seed in environments)
            condition_gates[condition] = _condition_metrics(
                rows,
                survival_required=7,
                maintenance_required=6,
                recovery_required_count=7,
                safe_required=0.80,
                unsafe_wait_maximum=0.15,
            )
        candidate_pooled = _pooled_metrics(candidate_rows)
        static_pooled = _pooled_metrics(static_rows)
        legacy_pooled = _pooled_metrics(legacy_rows)
        random_pooled = _pooled_metrics(random_rows)
        deltas = {
            "random_survival": candidate_pooled["survival_rate"] - random_pooled["survival_rate"],
            "static_decision_safe": candidate_pooled["decision_safe_fraction"]
            - static_pooled["decision_safe_fraction"],
            "static_duration_safe": candidate_pooled["duration_safe_fraction"]
            - static_pooled["duration_safe_fraction"],
            "static_full_objective": candidate_pooled["full_objective_rate"]
            - static_pooled["full_objective_rate"],
            "static_unsafe_wait": static_pooled["unsafe_wait_fraction"]
            - candidate_pooled["unsafe_wait_fraction"],
            "legacy_full_objective": candidate_pooled["full_objective_rate"]
            - legacy_pooled["full_objective_rate"],
            "legacy_decision_safe": candidate_pooled["decision_safe_fraction"]
            - legacy_pooled["decision_safe_fraction"],
            "legacy_duration_safe": candidate_pooled["duration_safe_fraction"]
            - legacy_pooled["duration_safe_fraction"],
        }
        random_pass = deltas["random_survival"] + _EPSILON >= 0.20
        static_pass = (
            deltas["static_decision_safe"] + _EPSILON >= 0.10
            and deltas["static_duration_safe"] + _EPSILON >= 0.10
            and deltas["static_full_objective"] + _EPSILON >= 0.15
            and deltas["static_unsafe_wait"] + _EPSILON >= 0.15
        )
        legacy_pass = (
            deltas["legacy_full_objective"] + _EPSILON >= -0.03
            and deltas["legacy_decision_safe"] + _EPSILON >= -0.03
            and deltas["legacy_duration_safe"] + _EPSILON >= -0.03
        )
        strong_effect = deltas["static_full_objective"] + _EPSILON >= 0.25
        if strong_effect:
            strong_effect_pairs.append(training_seed)
        all_arms_integrity = all(
            _integrity_pass(rows) for rows in (candidate_rows, static_rows, legacy_rows, random_rows)
        )
        replica_pass = (
            all(bool(gate["passes"]) for gate in condition_gates.values())  # type: ignore[index]
            and all_arms_integrity
            and random_pass
            and static_pass
            and legacy_pass
        )
        per_replica[str(training_seed)] = {
            "passes": replica_pass,
            "conditions": condition_gates,
            "candidate_pooled": candidate_pooled,
            "static_cost_control_pooled": static_pooled,
            "legacy_guardrail_pooled": legacy_pooled,
            "random_pooled": random_pooled,
            "deltas": deltas,
            "all_arms_integrity_pass": all_arms_integrity,
            "random_survival_pass": random_pass,
            "static_improvement_pass": static_pass,
            "legacy_noninferiority_pass": legacy_pass,
            "strong_full_objective_effect": strong_effect,
        }
    passes = (
        all(bool(row["passes"]) for row in per_replica.values())  # type: ignore[index]
        and bool(strong_effect_pairs)
    )
    return {
        "passes": passes,
        "per_replica": per_replica,
        "strong_effect_pairs": strong_effect_pairs,
        "strong_effect_pass": bool(strong_effect_pairs),
    }


def _m139_gate_margin(
    learned_index: Mapping[tuple[str, int], CompactEpisodeSummary],
    random_index: Mapping[tuple[str, int], CompactEpisodeSummary],
    *,
    conditions: tuple[str, ...],
    env_seeds: tuple[int, ...],
) -> tuple[float | None, dict[str, float]]:
    margins: dict[str, float] = {}
    for condition in conditions:
        rows = tuple(learned_index[(condition, env_seed)] for env_seed in env_seeds)
        metrics = _condition_metrics(
            rows,
            survival_required=18,
            maintenance_required=18,
            recovery_required_count=18,
            safe_required=0.85,
            unsafe_wait_maximum=0.10,
        )
        margins[f"{condition}.survival"] = float(metrics["survival_rate"]) - 0.90
        margins[f"{condition}.maintenance"] = float(metrics["maintenance_rate"]) - 0.90
        margins[f"{condition}.decision_safe"] = float(metrics["mean_decision_safe_fraction"]) - 0.85
        margins[f"{condition}.duration_safe"] = float(metrics["mean_duration_safe_fraction"]) - 0.85
        margins[f"{condition}.unsafe_wait"] = 0.10 - float(metrics["unsafe_wait_fraction"])
        if bool(metrics["recovery_required"]):
            margins[f"{condition}.recovery"] = float(metrics["recovery_rate"]) - 0.90
    learned_rows = tuple(learned_index.values())
    random_rows = tuple(random_index.values())
    learned_pooled = _pooled_metrics(learned_rows)
    random_pooled = _pooled_metrics(random_rows)
    margins["pooled_random_survival"] = (
        learned_pooled["survival_rate"] - random_pooled["survival_rate"] - 0.20
    )
    if not _integrity_pass(learned_rows) or not _integrity_pass(random_rows):
        return None, margins
    return min(margins.values()), margins


def m139_confirmation_passes(
    candidate: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    static_control: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
) -> dict[str, object]:
    """Recompute the eight-replica hard gate and 7/8 potential-margin test."""

    replicas, condition_names, environments = _validated_axes(
        training_seeds=training_seeds,
        conditions=conditions,
        env_seeds=env_seeds,
        required_replicas=8,
        required_env_seeds=20,
    )
    candidate_index = _index_summaries(
        candidate,
        training_seeds=replicas,
        conditions=condition_names,
        env_seeds=environments,
        arm="candidate",
    )
    static_index = _index_summaries(
        static_control,
        training_seeds=replicas,
        conditions=condition_names,
        env_seeds=environments,
        arm="static_cost_control",
    )
    random_index = _index_summaries(
        random, training_seeds=replicas, conditions=condition_names, env_seeds=environments, arm="random"
    )
    for training_seed in replicas:
        for key in candidate_index[training_seed]:
            requirements = {
                index[training_seed][key].recovery_required
                for index in (candidate_index, static_index, random_index)
            }
            if len(requirements) != 1:
                raise M139ProtocolError(
                    f"paired arms disagree on recovery requirement at {(training_seed, *key)!r}"
                )
            expected_recovery = key[0] in M139_RECOVERY_CONDITIONS
            if requirements != {expected_recovery}:
                raise M139ProtocolError(
                    f"evidence has the wrong recovery requirement at {(training_seed, *key)!r}"
                )
    per_replica: dict[str, object] = {}
    win_seeds: list[int] = []
    for training_seed in replicas:
        candidate_rows = tuple(candidate_index[training_seed].values())
        static_rows = tuple(static_index[training_seed].values())
        random_rows = tuple(random_index[training_seed].values())
        condition_gates: dict[str, object] = {}
        for condition in condition_names:
            rows = tuple(candidate_index[training_seed][(condition, env_seed)] for env_seed in environments)
            condition_gates[condition] = _condition_metrics(
                rows,
                survival_required=18,
                maintenance_required=18,
                recovery_required_count=18,
                safe_required=0.85,
                unsafe_wait_maximum=0.10,
            )
        candidate_pooled = _pooled_metrics(candidate_rows)
        random_pooled = _pooled_metrics(random_rows)
        random_delta = candidate_pooled["survival_rate"] - random_pooled["survival_rate"]
        all_arms_integrity = all(_integrity_pass(rows) for rows in (candidate_rows, static_rows, random_rows))
        hard_pass = (
            all(bool(gate["passes"]) for gate in condition_gates.values())  # type: ignore[index]
            and all_arms_integrity
            and random_delta + _EPSILON >= 0.20
        )
        candidate_margin, candidate_components = _m139_gate_margin(
            candidate_index[training_seed],
            random_index[training_seed],
            conditions=condition_names,
            env_seeds=environments,
        )
        static_margin, static_components = _m139_gate_margin(
            static_index[training_seed],
            random_index[training_seed],
            conditions=condition_names,
            env_seeds=environments,
        )
        win = (
            candidate_margin is not None
            and static_margin is not None
            and candidate_margin > static_margin + _EPSILON
        )
        if win:
            win_seeds.append(training_seed)
        per_replica[str(training_seed)] = {
            "hard_gate_pass": hard_pass,
            "conditions": condition_gates,
            "candidate_pooled": candidate_pooled,
            "random_pooled": random_pooled,
            "random_survival_delta": random_delta,
            "all_arms_integrity_pass": all_arms_integrity,
            "candidate_margin": candidate_margin,
            "static_cost_control_margin": static_margin,
            "candidate_margin_components": candidate_components,
            "static_cost_control_margin_components": static_components,
            "paired_margin_win": win,
        }
    hard_gate_pass = all(bool(row["hard_gate_pass"]) for row in per_replica.values())  # type: ignore[index]
    return {
        "passes": hard_gate_pass and len(win_seeds) >= 7,
        "hard_gate_pass": hard_gate_pass,
        "robust_baseline_pass": hard_gate_pass,
        "paired_margin_wins": len(win_seeds),
        "paired_margin_win_seeds": win_seeds,
        "per_replica": per_replica,
    }


def _compact_report_evidence(
    report: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    evidence = report.get("episode_summaries")
    if not isinstance(evidence, Mapping) or set(evidence) != set(_REPORT_ARMS):
        raise ScreenReportVerificationError(
            "screen report episode_summaries must contain the exact M13.9 four-arm matrix"
        )
    result: dict[str, list[dict[str, object]]] = {}
    for arm in _REPORT_ARMS:
        rows = evidence[arm]
        if not isinstance(rows, (list, tuple)):
            raise ScreenReportVerificationError(f"screen report {arm} evidence must be a list")
        result[arm] = [compact_episode_summary(row) for row in rows]
    return result


def seal_screen_report(
    report: Mapping[str, object],
    *,
    expected_protocol_fingerprint: str,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
) -> dict[str, object]:
    """Canonicalize evidence, recompute promotion, and attach its content hash."""

    if not isinstance(report, Mapping):
        raise ScreenReportVerificationError("screen report must be an object")
    if report.get("stage", "screen") != "screen":
        raise ScreenReportVerificationError("screen report has the wrong stage")
    fingerprint = report.get("protocol_fingerprint", expected_protocol_fingerprint)
    if fingerprint != expected_protocol_fingerprint:
        raise ScreenReportVerificationError("screen report protocol fingerprint mismatch")
    evidence = _compact_report_evidence(report)
    promotion = m139_screen_promotes(
        evidence["public_potential_candidate"],
        evidence["static_cost_control"],
        evidence["legacy_guardrail"],
        evidence["random"],
        training_seeds=training_seeds,
        conditions=conditions,
        env_seeds=env_seeds,
    )
    sealed = {
        key: value
        for key, value in report.items()
        if key not in {"content_hash", "episode_summaries", "promotion", "passes"}
    }
    sealed.update(
        {
            "schema_version": M139_SCREEN_REPORT_SCHEMA_VERSION,
            "stage": "screen",
            "protocol_fingerprint": expected_protocol_fingerprint,
            "episode_summaries": evidence,
            "promotion": promotion,
        }
    )
    assert_compact_report(sealed)
    sealed["content_hash"] = m139_content_hash(sealed)
    return sealed


def verify_screen_report(
    report: Mapping[str, object],
    *,
    expected_protocol_fingerprint: str,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
    require_promotion: bool = True,
) -> dict[str, object]:
    """Verify report integrity and return a freshly recomputed promotion gate.

    In particular, the stored ``promotion.passes`` value is never trusted.
    Confirmation callers should keep ``require_promotion=True`` (the default).
    """

    if not isinstance(report, Mapping):
        raise ScreenReportVerificationError("screen report must be an object")
    assert_compact_report(report)
    if report.get("schema_version") != M139_SCREEN_REPORT_SCHEMA_VERSION:
        raise ScreenReportVerificationError("screen report schema mismatch")
    if report.get("stage") != "screen":
        raise ScreenReportVerificationError("screen report has the wrong stage")
    if report.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ScreenReportVerificationError("screen report protocol fingerprint mismatch")
    content_hash = report.get("content_hash")
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise ScreenReportVerificationError("screen report has no valid content hash")
    unhashed = {key: value for key, value in report.items() if key != "content_hash"}
    expected_hash = m139_content_hash(unhashed)
    if not hmac.compare_digest(content_hash, expected_hash):
        raise ScreenReportVerificationError("screen report content hash mismatch")
    evidence = _compact_report_evidence(report)
    trusted = m139_screen_promotes(
        evidence["public_potential_candidate"],
        evidence["static_cost_control"],
        evidence["legacy_guardrail"],
        evidence["random"],
        training_seeds=training_seeds,
        conditions=conditions,
        env_seeds=env_seeds,
    )
    stored = report.get("promotion")
    if not isinstance(stored, Mapping) or _canonical_json_bytes(stored) != _canonical_json_bytes(trusted):
        raise ScreenReportVerificationError("stored promotion disagrees with episode evidence")
    if require_promotion and not bool(trusted["passes"]):
        raise ScreenReportVerificationError("screen report did not promote M13.9")
    return trusted


# Names used by orchestration code read a little more naturally with this alias.
verify_m139_screen_report = verify_screen_report


__all__ = [
    "M139_CONDITIONS",
    "M139_LEDGER_SCHEMA_VERSION",
    "M139_PARTITION_DEPENDENCIES",
    "M139_RECOVERY_CONDITIONS",
    "M139_SCREEN_REPORT_SCHEMA_VERSION",
    "CompactEpisodeSummary",
    "M139ProtocolError",
    "ScreenReportVerificationError",
    "SplitAlreadyOpenedError",
    "SplitDependencyError",
    "SplitLedgerError",
    "SplitOpenLedger",
    "assert_compact_report",
    "compact_episode_summary",
    "m139_confirmation_passes",
    "m139_content_hash",
    "m139_screen_promotes",
    "seal_screen_report",
    "verify_m139_screen_report",
    "verify_screen_report",
]
