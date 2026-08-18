"""Pure ledger and preflight helpers for the unopened M13.14-r2 protocol."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


M1314R2_LEDGER_SCHEMA_VERSION = "m13.14-r2-stage-ledger-r1"
M1314R2_PREFLIGHT_REPORT_SCHEMA_VERSION = "m13.14-r2-preflight-report-r1"

M1314R2_LEDGER_STAGE_ORDER = (
    "development_preflight",
    "development_fit",
    "development_check",
    "confirmation_preflight",
    "confirmation_fit",
    "confirmation_evaluation",
    "audit_preflight",
    "audit",
)
M1314R2_RESUMABLE_STAGES = frozenset({"development_fit", "confirmation_fit"})
M1314R2_PREFLIGHT_STAGE_TARGETS = {
    "development_preflight": {
        "scan_partitions": ("development_fit", "development_check"),
        "authorizes_command": "m1314r2-development",
    },
    "confirmation_preflight": {
        "scan_partitions": ("confirmation_fit", "confirmation_evaluation"),
        "authorizes_command": "m1314r2-confirmation",
    },
    "audit_preflight": {
        "scan_partitions": ("audit",),
        "authorizes_command": "m1314r2-audit",
    },
}
_COVERAGE_FIELDS = (
    "episodes",
    "initial_food_visible",
    "replenished_food_visible",
    "toy_visible",
    "rest_visible",
    "relocation_episodes",
    "relocated_food_visible",
    "distractor_episodes",
    "distractor_visible",
    "passes",
)


class M1314R2ProtocolError(ValueError):
    """Raised when M13.14-r2 support evidence violates the frozen protocol."""


class M1314R2LedgerError(M1314R2ProtocolError):
    """Base class for one-way M13.14-r2 ledger failures."""


class M1314R2PreflightVerificationError(M1314R2ProtocolError):
    """Raised when a stored preflight report cannot authorize the next stage."""


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
        raise M1314R2ProtocolError("M13.14-r2 evidence is not canonical-JSON serializable") from exc
    return (encoded + "\n").encode("utf-8")


def m1314r2_content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


class M1314R2StageLedger:
    """Append-only stage ledger for future M13.14-r2 preflight and run stages."""

    def __init__(self, path: str | Path, *, protocol_fingerprint: str) -> None:
        if not isinstance(protocol_fingerprint, str) or not protocol_fingerprint:
            raise M1314R2LedgerError("protocol_fingerprint must be a non-empty string")
        self.path = Path(path)
        self.protocol_fingerprint = protocol_fingerprint
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _empty(self) -> dict[str, object]:
        return {
            "schema_version": M1314R2_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_stages": [],
            "marker_hashes": {},
        }

    def _read_unlocked(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise M1314R2LedgerError("M13.14-r2 ledger is unreadable") from exc
        if not isinstance(payload, dict):
            raise M1314R2LedgerError("M13.14-r2 ledger root must be an object")
        if payload.get("schema_version") != M1314R2_LEDGER_SCHEMA_VERSION:
            raise M1314R2LedgerError("M13.14-r2 ledger schema mismatch")
        if payload.get("protocol_fingerprint") != self.protocol_fingerprint:
            raise M1314R2LedgerError("M13.14-r2 ledger protocol fingerprint mismatch")
        opened = payload.get("opened_stages")
        markers = payload.get("marker_hashes")
        if not isinstance(opened, list) or any(not isinstance(item, str) for item in opened):
            raise M1314R2LedgerError("M13.14-r2 ledger opened_stages is malformed")
        if not isinstance(markers, dict) or any(not isinstance(key, str) for key in markers):
            raise M1314R2LedgerError("M13.14-r2 ledger marker_hashes is malformed")
        if opened != list(M1314R2_LEDGER_STAGE_ORDER[: len(opened)]):
            raise M1314R2LedgerError("M13.14-r2 ledger violates append-only stage ordering")
        if len(opened) != len(set(opened)):
            raise M1314R2LedgerError("M13.14-r2 ledger contains duplicate stage markers")
        if set(markers) != set(opened):
            raise M1314R2LedgerError("M13.14-r2 ledger markers do not match opened stages")
        for ordinal, stage in enumerate(opened):
            expected = m1314r2_content_hash(
                {
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "stage": stage,
                    "ordinal": ordinal,
                }
            )
            if markers.get(stage) != expected:
                raise M1314R2LedgerError("M13.14-r2 ledger marker hash mismatch")
        return {
            "schema_version": M1314R2_LEDGER_SCHEMA_VERSION,
            "protocol_fingerprint": self.protocol_fingerprint,
            "opened_stages": list(opened),
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

    def open(self, stage: str, *, resumable_fit: bool = False) -> dict[str, object]:
        if stage not in M1314R2_LEDGER_STAGE_ORDER:
            raise M1314R2LedgerError(f"unknown M13.14-r2 stage {stage!r}")
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                payload = self._read_unlocked()
                opened = list(payload["opened_stages"])
                markers = dict(payload["marker_hashes"])
                expected = (
                    M1314R2_LEDGER_STAGE_ORDER[len(opened)]
                    if len(opened) < len(M1314R2_LEDGER_STAGE_ORDER)
                    else None
                )
                if stage in opened:
                    if resumable_fit and stage in M1314R2_RESUMABLE_STAGES and opened[-1] == stage:
                        return payload
                    raise M1314R2LedgerError(f"M13.14-r2 stage {stage!r} was already opened")
                if stage != expected:
                    raise M1314R2LedgerError(
                        f"M13.14-r2 next stage must be {expected!r}, not {stage!r}"
                    )
                opened.append(stage)
                markers[stage] = m1314r2_content_hash(
                    {
                        "protocol_fingerprint": self.protocol_fingerprint,
                        "stage": stage,
                        "ordinal": len(opened) - 1,
                    }
                )
                updated = {
                    **payload,
                    "opened_stages": opened,
                    "marker_hashes": markers,
                }
                _atomic_write(self.path, updated)
                return updated
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M1314R2PreflightVerificationError(f"{label} must be an object")
    return value


def _normalize_coverage_row(
    value: object,
    *,
    condition: str,
    expected_episodes: int,
) -> dict[str, object]:
    row = _require_mapping(value, label=f"coverage row for {condition}")
    normalized: dict[str, object] = {}
    for key in _COVERAGE_FIELDS:
        if key not in row:
            raise M1314R2PreflightVerificationError(
                f"coverage row for {condition} is missing {key!r}"
            )
    for key in _COVERAGE_FIELDS:
        field = row[key]
        if key == "passes":
            normalized[key] = bool(field)
            continue
        if not isinstance(field, int) or isinstance(field, bool):
            raise M1314R2PreflightVerificationError(
                f"coverage row for {condition} field {key!r} must be an integer"
            )
        normalized[key] = int(field)
    if normalized["episodes"] != expected_episodes:
        raise M1314R2PreflightVerificationError(
            f"coverage row for {condition} expected {expected_episodes} episodes"
        )
    if normalized["relocated_food_visible"] > normalized["relocation_episodes"]:
        raise M1314R2PreflightVerificationError(
            f"coverage row for {condition} has impossible relocated-food counts"
        )
    if normalized["distractor_visible"] > normalized["distractor_episodes"]:
        raise M1314R2PreflightVerificationError(
            f"coverage row for {condition} has impossible distractor counts"
        )
    return normalized


def _normalize_partition_coverage(
    value: object,
    *,
    partition: str,
    expected_conditions: Sequence[str],
    expected_episodes: int,
) -> dict[str, dict[str, object]]:
    coverage = _require_mapping(value, label=f"{partition} coverage")
    expected = tuple(str(name) for name in expected_conditions)
    if set(coverage) != set(expected):
        raise M1314R2PreflightVerificationError(
            f"{partition} coverage conditions must exactly match {list(expected)!r}"
        )
    return {
        condition: _normalize_coverage_row(
            coverage[condition],
            condition=condition,
            expected_episodes=expected_episodes,
        )
        for condition in expected
    }


def _normalize_stage_coverage(
    value: object,
    *,
    expected_partitions: Sequence[str],
    expected_conditions: Sequence[str],
    expected_episode_counts: Mapping[str, int],
) -> dict[str, dict[str, dict[str, object]]]:
    coverage = _require_mapping(value, label="preflight coverage")
    partitions = tuple(str(name) for name in expected_partitions)
    if set(coverage) != set(partitions):
        raise M1314R2PreflightVerificationError(
            f"preflight coverage partitions must exactly match {list(partitions)!r}"
        )
    return {
        partition: _normalize_partition_coverage(
            coverage[partition],
            partition=partition,
            expected_conditions=expected_conditions,
            expected_episodes=int(expected_episode_counts[partition]),
        )
        for partition in partitions
    }


def seal_m1314r2_preflight_report(
    report: Mapping[str, object],
    *,
    expected_protocol_fingerprint: str,
    expected_manifest_fingerprint: str,
    expected_stage: str,
    expected_conditions: Sequence[str],
    expected_episode_counts: Mapping[str, int],
) -> dict[str, object]:
    if expected_stage not in M1314R2_PREFLIGHT_STAGE_TARGETS:
        raise M1314R2PreflightVerificationError(f"unknown M13.14-r2 preflight stage {expected_stage!r}")
    if not isinstance(report, Mapping):
        raise M1314R2PreflightVerificationError("preflight report must be an object")
    target = M1314R2_PREFLIGHT_STAGE_TARGETS[expected_stage]
    fingerprint = report.get("protocol_fingerprint", expected_protocol_fingerprint)
    if fingerprint != expected_protocol_fingerprint:
        raise M1314R2PreflightVerificationError("preflight report protocol fingerprint mismatch")
    manifest_fingerprint = report.get("manifest_fingerprint", expected_manifest_fingerprint)
    if manifest_fingerprint != expected_manifest_fingerprint:
        raise M1314R2PreflightVerificationError("preflight report manifest fingerprint mismatch")
    stage = report.get("stage", expected_stage)
    if stage != expected_stage:
        raise M1314R2PreflightVerificationError("preflight report has the wrong stage")
    coverage = _normalize_stage_coverage(
        report.get("coverage"),
        expected_partitions=target["scan_partitions"],
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_episode_counts,
    )
    failing_rows = [
        {"partition": partition, "condition": condition}
        for partition, partition_rows in coverage.items()
        for condition, row in partition_rows.items()
        if not bool(row["passes"])
    ]
    sealed = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "schema_version",
            "stage",
            "protocol_fingerprint",
            "manifest_fingerprint",
            "scanned_partitions",
            "authorizes_command",
            "coverage",
            "coverage_passes",
            "failing_rows",
            "content_hash",
        }
    }
    sealed.update(
        {
            "schema_version": M1314R2_PREFLIGHT_REPORT_SCHEMA_VERSION,
            "stage": expected_stage,
            "protocol_fingerprint": expected_protocol_fingerprint,
            "manifest_fingerprint": expected_manifest_fingerprint,
            "scanned_partitions": list(target["scan_partitions"]),
            "authorizes_command": str(target["authorizes_command"]),
            "coverage": coverage,
            "coverage_passes": not failing_rows,
            "failing_rows": failing_rows,
        }
    )
    sealed["content_hash"] = m1314r2_content_hash(
        {key: value for key, value in sealed.items() if key != "content_hash"}
    )
    return sealed


def verify_m1314r2_preflight_report(
    report: Mapping[str, object],
    *,
    expected_protocol_fingerprint: str,
    expected_manifest_fingerprint: str,
    expected_stage: str,
    expected_conditions: Sequence[str],
    expected_episode_counts: Mapping[str, int],
    require_pass: bool = True,
) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise M1314R2PreflightVerificationError("preflight report must be an object")
    if report.get("schema_version") != M1314R2_PREFLIGHT_REPORT_SCHEMA_VERSION:
        raise M1314R2PreflightVerificationError("preflight report schema mismatch")
    if report.get("stage") != expected_stage:
        raise M1314R2PreflightVerificationError("preflight report has the wrong stage")
    if report.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise M1314R2PreflightVerificationError("preflight report protocol fingerprint mismatch")
    if report.get("manifest_fingerprint") != expected_manifest_fingerprint:
        raise M1314R2PreflightVerificationError("preflight report manifest fingerprint mismatch")
    content_hash = report.get("content_hash")
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise M1314R2PreflightVerificationError("preflight report has no valid content hash")
    unhashed = {key: value for key, value in report.items() if key != "content_hash"}
    expected_hash = m1314r2_content_hash(unhashed)
    if not hmac.compare_digest(content_hash, expected_hash):
        raise M1314R2PreflightVerificationError("preflight report content hash mismatch")
    trusted = seal_m1314r2_preflight_report(
        report,
        expected_protocol_fingerprint=expected_protocol_fingerprint,
        expected_manifest_fingerprint=expected_manifest_fingerprint,
        expected_stage=expected_stage,
        expected_conditions=expected_conditions,
        expected_episode_counts=expected_episode_counts,
    )
    if _canonical_json_bytes(report) != _canonical_json_bytes(trusted):
        raise M1314R2PreflightVerificationError(
            "stored preflight report disagrees with canonical coverage evidence"
        )
    if require_pass and not bool(trusted["coverage_passes"]):
        raise M1314R2PreflightVerificationError("preflight report did not pass coverage")
    return trusted


verify_preflight_report = verify_m1314r2_preflight_report


__all__ = [
    "M1314R2_LEDGER_SCHEMA_VERSION",
    "M1314R2_LEDGER_STAGE_ORDER",
    "M1314R2_PREFLIGHT_REPORT_SCHEMA_VERSION",
    "M1314R2_PREFLIGHT_STAGE_TARGETS",
    "M1314R2_RESUMABLE_STAGES",
    "M1314R2LedgerError",
    "M1314R2PreflightVerificationError",
    "M1314R2ProtocolError",
    "M1314R2StageLedger",
    "m1314r2_content_hash",
    "seal_m1314r2_preflight_report",
    "verify_m1314r2_preflight_report",
    "verify_preflight_report",
]
