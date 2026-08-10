"""Append-only split ledger for the unopened M13.13 policy-family protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


M1313_LEDGER_SCHEMA_VERSION = "m13.13-policy-family-ledger-r1"


def content_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class M1313SplitLedger:
    partitions = (
        "development_fit",
        "development_check",
        "confirmation_fit",
        "confirmation_evaluation",
        "audit",
    )
    resumable = {"development_fit", "confirmation_fit"}

    def __init__(self, path: str | Path, *, protocol_fingerprint: str) -> None:
        self.path = Path(path)
        self.protocol_fingerprint = str(protocol_fingerprint)

    def snapshot(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "schema_version": M1313_LEDGER_SCHEMA_VERSION,
                "protocol_fingerprint": self.protocol_fingerprint,
                "opened_partitions": [],
                "marker_hashes": {},
            }
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != M1313_LEDGER_SCHEMA_VERSION
            or value.get("protocol_fingerprint") != self.protocol_fingerprint
        ):
            raise ValueError("M13.13 ledger schema or protocol mismatch")
        opened = value.get("opened_partitions")
        markers = value.get("marker_hashes")
        if not isinstance(opened, list) or not isinstance(markers, dict):
            raise ValueError("M13.13 ledger is malformed")
        if opened != list(self.partitions[: len(opened)]) or set(markers) != set(opened):
            raise ValueError("M13.13 ledger partition order is malformed")
        for ordinal, partition in enumerate(opened):
            expected = content_hash(
                {
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "partition": partition,
                    "ordinal": ordinal,
                }
            )
            if markers[partition] != expected:
                raise ValueError("M13.13 ledger marker mismatch")
        return value

    def open(self, partition: str, *, resumable_fit: bool = False) -> dict[str, object]:
        if partition not in self.partitions:
            raise ValueError(f"unknown M13.13 partition {partition!r}")
        value = self.snapshot()
        opened = list(value["opened_partitions"])
        expected = self.partitions[len(opened)] if len(opened) < len(self.partitions) else None
        if partition in opened:
            if resumable_fit and partition in self.resumable and opened[-1] == partition:
                return value
            raise ValueError(f"M13.13 partition {partition!r} is already opened")
        if partition != expected:
            raise ValueError(f"M13.13 next partition must be {expected!r}, not {partition!r}")
        opened.append(partition)
        markers = dict(value["marker_hashes"])
        markers[partition] = content_hash(
            {
                "protocol_fingerprint": self.protocol_fingerprint,
                "partition": partition,
                "ordinal": len(opened) - 1,
            }
        )
        result = {**value, "opened_partitions": opened, "marker_hashes": markers}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result
