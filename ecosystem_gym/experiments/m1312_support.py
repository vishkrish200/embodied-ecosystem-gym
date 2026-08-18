"""One-way ledger primitives for the additive M13.12 experiment."""

from __future__ import annotations

import json
from pathlib import Path

from .m1311_support import content_hash

M1312_LEDGER_SCHEMA_VERSION = "m13.12-development-ledger-r1"


class M1312DevelopmentLedger:
    partitions = ("development_fit", "development_check")

    def __init__(self, path: str | Path, *, protocol_fingerprint: str) -> None:
        self.path = Path(path)
        self.protocol_fingerprint = protocol_fingerprint

    def snapshot(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "schema_version": M1312_LEDGER_SCHEMA_VERSION,
                "protocol_fingerprint": self.protocol_fingerprint,
                "opened_partitions": [],
                "marker_hashes": {},
            }
        value = json.loads(self.path.read_text())
        if value.get("schema_version") != M1312_LEDGER_SCHEMA_VERSION or value.get("protocol_fingerprint") != self.protocol_fingerprint:
            raise ValueError("M13.12 ledger schema or protocol mismatch")
        opened = value.get("opened_partitions", [])
        markers = value.get("marker_hashes", {})
        if opened != list(dict.fromkeys(opened)) or any(item not in self.partitions for item in opened) or set(markers) != set(opened):
            raise ValueError("M13.12 ledger is malformed")
        for ordinal, partition in enumerate(opened):
            if partition == "development_check" and opened[:ordinal] != ["development_fit"]:
                raise ValueError("development check requires fit")
            expected = content_hash({"protocol_fingerprint": self.protocol_fingerprint, "partition": partition, "ordinal": ordinal})
            if markers[partition] != expected:
                raise ValueError("M13.12 ledger marker mismatch")
        return value

    def open(self, partition: str, *, resumable_fit: bool = False) -> dict[str, object]:
        if partition not in self.partitions:
            raise ValueError("unknown M13.12 partition")
        value = self.snapshot()
        opened = list(value["opened_partitions"])
        if partition in opened:
            if partition == "development_fit" and resumable_fit and opened == ["development_fit"]:
                return value
            raise ValueError(f"M13.12 partition {partition!r} is already opened")
        if partition == "development_check" and opened != ["development_fit"]:
            raise ValueError("development check requires completed development fit")
        opened.append(partition)
        markers = dict(value["marker_hashes"])
        markers[partition] = content_hash({"protocol_fingerprint": self.protocol_fingerprint, "partition": partition, "ordinal": len(opened) - 1})
        value = {**value, "opened_partitions": opened, "marker_hashes": markers}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return value
