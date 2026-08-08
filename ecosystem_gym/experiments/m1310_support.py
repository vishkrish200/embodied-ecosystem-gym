"""Pure ledger and screen-gate guards for frozen M13.10."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .m139_support import (
    CompactEpisodeSummary,
    _condition_metrics,
    _index_summaries,
    _integrity_pass,
    _pooled_metrics,
)

M1310_LEDGER_SCHEMA_VERSION = "m13.10-split-ledger-r1"
M1310_SCREEN_REPORT_SCHEMA_VERSION = "m13.10-screen-report-r1"
M1310_PARTITION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "screen_fit": (),
    "screen_probe": ("screen_fit",),
    "confirmation_fit": ("screen_probe",),
    "confirmation_evaluation": ("confirmation_fit",),
    "audit": ("confirmation_evaluation",),
}
M1310_RECOVERY_CONDITIONS = frozenset({"event_relocation", "compound"})
_EPSILON = 1e-12


class M1310ProtocolError(ValueError):
    """Raised when evidence violates the frozen M13.10 protocol."""


class SplitLedgerError(M1310ProtocolError):
    """Base split-ledger failure."""


class SplitAlreadyOpenedError(SplitLedgerError):
    """Raised on a repeated one-shot split opening."""


class SplitDependencyError(SplitLedgerError):
    """Raised when a split is opened before its prerequisite."""


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
        raise M1310ProtocolError("M13.10 evidence is not canonical JSON") from exc
    return (encoded + "\n").encode()


def m1310_content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class SplitOpenLedger:
    """Atomic output-independent ordered M13.10 partition ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        protocol_fingerprint: str,
        dependencies: Mapping[str, Sequence[str]] = M1310_PARTITION_DEPENDENCIES,
    ) -> None:
        if not protocol_fingerprint:
            raise SplitLedgerError("protocol_fingerprint must be non-empty")
        self.path = Path(path)
        self.protocol_fingerprint = protocol_fingerprint
        self.dependencies = {name: tuple(rows) for name, rows in dependencies.items()}
        unknown = {row for rows in self.dependencies.values() for row in rows if row not in self.dependencies}
        if unknown:
            raise SplitLedgerError(f"unknown split dependencies: {sorted(unknown)!r}")
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _empty(self) -> dict[str, object]:
        return {
            "schema_version": M1310_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_partitions": [],
            "marker_hashes": {},
        }

    def _read_unlocked(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitLedgerError("M13.10 split ledger is unreadable") from exc
        if not isinstance(payload, dict):
            raise SplitLedgerError("M13.10 split ledger root must be an object")
        if payload.get("schema_version") != M1310_LEDGER_SCHEMA_VERSION:
            raise SplitLedgerError("M13.10 split ledger schema mismatch")
        if payload.get("protocol_fingerprint") != self.protocol_fingerprint:
            raise SplitLedgerError("M13.10 split ledger fingerprint mismatch")
        opened = payload.get("opened_partitions")
        markers = payload.get("marker_hashes")
        if (
            not isinstance(opened, list)
            or not all(isinstance(row, str) for row in opened)
            or len(opened) != len(set(opened))
        ):
            raise SplitLedgerError("M13.10 split ledger has invalid markers")
        if not isinstance(markers, dict) or set(markers) != set(opened):
            raise SplitLedgerError("M13.10 split marker hashes are incomplete")
        for index, partition in enumerate(opened):
            if partition not in self.dependencies:
                raise SplitLedgerError("M13.10 split ledger names an unknown partition")
            if not set(self.dependencies[partition]).issubset(opened[:index]):
                raise SplitLedgerError("M13.10 split dependency order is invalid")
            expected = m1310_content_hash(
                {
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "partition": partition,
                    "ordinal": index,
                }
            )
            if not hmac.compare_digest(str(markers[partition]), expected):
                raise SplitLedgerError("M13.10 split marker hash mismatch")
        return {
            "schema_version": M1310_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_partitions": list(opened),
            "marker_hashes": dict(markers),
        }

    def snapshot(self) -> dict[str, object]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def open_partition(self, partition: str) -> dict[str, object]:
        if partition not in self.dependencies:
            raise SplitLedgerError(f"unknown M13.10 partition {partition!r}")
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                payload = self._read_unlocked()
                opened = list(payload["opened_partitions"])
                if partition in opened:
                    raise SplitAlreadyOpenedError(f"M13.10 partition {partition!r} was already opened")
                missing = [row for row in self.dependencies[partition] if row not in opened]
                if missing:
                    raise SplitDependencyError(f"M13.10 partition {partition!r} requires {missing!r} first")
                opened.append(partition)
                markers = dict(payload["marker_hashes"])
                markers[partition] = m1310_content_hash(
                    {
                        "protocol_fingerprint": self.protocol_fingerprint,
                        "partition": partition,
                        "ordinal": len(opened) - 1,
                    }
                )
                updated = {
                    **payload,
                    "opened_partitions": opened,
                    "marker_hashes": markers,
                }
                _atomic_write(self.path, updated)
                return updated
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def ensure_screen_fit_open(self) -> dict[str, object]:
        """Open screen-fit once, or resume while no held-out split is open."""

        snapshot = self.snapshot()
        opened = list(snapshot["opened_partitions"])
        if not opened:
            return self.open_partition("screen_fit")
        if opened == ["screen_fit"]:
            return snapshot
        raise SplitLedgerError("M13.10 screen-fit access is unavailable after a held-out partition opened")


def _paired_indexes(
    candidate: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random_init: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    imitation_only: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: tuple[int, ...],
    conditions: tuple[str, ...],
    env_seeds: tuple[int, ...],
) -> tuple[dict[int, dict[tuple[str, int], CompactEpisodeSummary]], ...]:
    names = ("candidate", "random_init_rl_control", "imitation_only_guardrail", "random")
    arms = tuple(
        _index_summaries(
            rows,
            training_seeds=training_seeds,
            conditions=conditions,
            env_seeds=env_seeds,
            arm=name,
        )
        for rows, name in zip((candidate, random_init, imitation_only, random), names, strict=True)
    )
    for training_seed in training_seeds:
        for key in arms[0][training_seed]:
            requirements = {arm[training_seed][key].recovery_required for arm in arms}
            expected = key[0] in M1310_RECOVERY_CONDITIONS
            if requirements != {expected}:
                raise M1310ProtocolError(f"paired recovery metadata mismatch at {(training_seed, *key)!r}")
    return arms


def m1310_screen_promotes(
    candidate: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random_init: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    imitation_only: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    random: Iterable[CompactEpisodeSummary | Mapping[str, object]],
    *,
    training_seeds: Sequence[int],
    conditions: Sequence[str],
    env_seeds: Sequence[int],
) -> dict[str, object]:
    replicas = tuple(training_seeds)
    condition_names = tuple(conditions)
    environments = tuple(env_seeds)
    if len(replicas) != 2 or len(set(replicas)) != 2:
        raise M1310ProtocolError("M13.10 screen requires exactly two replicas")
    if len(condition_names) != 4 or len(set(condition_names)) != 4:
        raise M1310ProtocolError("M13.10 screen requires exactly four conditions")
    if len(environments) != 8 or len(set(environments)) != 8:
        raise M1310ProtocolError("M13.10 screen requires exactly eight probe seeds")
    candidate_index, control_index, guardrail_index, random_index = _paired_indexes(
        candidate,
        random_init,
        imitation_only,
        random,
        training_seeds=replicas,
        conditions=condition_names,
        env_seeds=environments,
    )
    results: dict[str, object] = {}
    strong: list[int] = []
    for seed in replicas:
        condition_gates: dict[str, object] = {}
        guardrail_gates: dict[str, object] = {}
        for condition in condition_names:
            candidate_rows = tuple(candidate_index[seed][(condition, env_seed)] for env_seed in environments)
            guardrail_rows = tuple(guardrail_index[seed][(condition, env_seed)] for env_seed in environments)
            condition_gates[condition] = _condition_metrics(
                candidate_rows,
                survival_required=7,
                maintenance_required=6,
                recovery_required_count=7,
                safe_required=0.80,
                unsafe_wait_maximum=0.15,
            )
            guardrail_gates[condition] = _condition_metrics(
                guardrail_rows,
                survival_required=7,
                maintenance_required=6,
                recovery_required_count=7,
                safe_required=0.80,
                unsafe_wait_maximum=0.15,
            )
        rows = tuple(candidate_index[seed].values())
        controls = tuple(control_index[seed].values())
        guardrails = tuple(guardrail_index[seed].values())
        randoms = tuple(random_index[seed].values())
        pooled = _pooled_metrics(rows)
        control = _pooled_metrics(controls)
        guardrail = _pooled_metrics(guardrails)
        random_metrics = _pooled_metrics(randoms)
        deltas = {
            "random_survival": pooled["survival_rate"] - random_metrics["survival_rate"],
            "control_full_objective": pooled["full_objective_rate"] - control["full_objective_rate"],
            "control_decision_safe": pooled["decision_safe_fraction"] - control["decision_safe_fraction"],
            "control_duration_safe": pooled["duration_safe_fraction"] - control["duration_safe_fraction"],
            "control_unsafe_wait": control["unsafe_wait_fraction"] - pooled["unsafe_wait_fraction"],
            "guardrail_full_objective": pooled["full_objective_rate"] - guardrail["full_objective_rate"],
            "guardrail_decision_safe": pooled["decision_safe_fraction"] - guardrail["decision_safe_fraction"],
            "guardrail_duration_safe": pooled["duration_safe_fraction"] - guardrail["duration_safe_fraction"],
            "guardrail_unsafe_wait": guardrail["unsafe_wait_fraction"] - pooled["unsafe_wait_fraction"],
        }
        control_pass = (
            deltas["control_full_objective"] + _EPSILON >= 0.15
            and deltas["control_decision_safe"] + _EPSILON >= 0.05
            and deltas["control_duration_safe"] + _EPSILON >= 0.05
            and deltas["control_unsafe_wait"] + _EPSILON >= 0.10
        )
        retention_pass = (
            deltas["guardrail_full_objective"] + _EPSILON >= -0.05
            and deltas["guardrail_decision_safe"] + _EPSILON >= -0.05
            and deltas["guardrail_duration_safe"] + _EPSILON >= -0.05
            and deltas["guardrail_unsafe_wait"] + _EPSILON >= -0.05
        )
        strong_effect = deltas["control_full_objective"] + _EPSILON >= 0.25
        if strong_effect:
            strong.append(seed)
        integrity = all(_integrity_pass(group) for group in (rows, controls, guardrails, randoms))
        passes = (
            all(bool(row["passes"]) for row in condition_gates.values())  # type: ignore[index]
            and all(bool(row["passes"]) for row in guardrail_gates.values())  # type: ignore[index]
            and integrity
            and deltas["random_survival"] + _EPSILON >= 0.20
            and control_pass
            and retention_pass
        )
        results[str(seed)] = {
            "passes": passes,
            "candidate_conditions": condition_gates,
            "imitation_only_conditions": guardrail_gates,
            "candidate_pooled": pooled,
            "random_init_rl_pooled": control,
            "imitation_only_pooled": guardrail,
            "random_pooled": random_metrics,
            "deltas": deltas,
            "all_arms_integrity_pass": integrity,
            "control_improvement_pass": control_pass,
            "retention_pass": retention_pass,
            "random_survival_pass": deltas["random_survival"] + _EPSILON >= 0.20,
            "strong_full_objective_effect": strong_effect,
        }
    passes = all(bool(row["passes"]) for row in results.values()) and bool(strong)  # type: ignore[index]
    return {
        "passes": passes,
        "per_replica": results,
        "strong_effect_seeds": strong,
        "strong_effect_pass": bool(strong),
    }


__all__ = [
    "M1310_LEDGER_SCHEMA_VERSION",
    "M1310_PARTITION_DEPENDENCIES",
    "M1310ProtocolError",
    "SplitAlreadyOpenedError",
    "SplitDependencyError",
    "SplitLedgerError",
    "SplitOpenLedger",
    "m1310_content_hash",
    "m1310_screen_promotes",
]
