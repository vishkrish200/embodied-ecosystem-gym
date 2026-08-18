"""One-way development ledger for the non-promotional M13.11 bakeoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

M1311_LEDGER_SCHEMA_VERSION = "m13.11-development-ledger-r1"


def content_hash(value: object) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()


class DevelopmentLedger:
    """Append-only two-partition ledger; smoke callers use a separate path."""
    partitions = ("development_fit", "development_check")
    def __init__(self, path: str|Path, *, protocol_fingerprint: str) -> None:
        self.path, self.protocol_fingerprint = Path(path), protocol_fingerprint
    def snapshot(self) -> dict[str, object]:
        if not self.path.exists(): return {"schema_version":M1311_LEDGER_SCHEMA_VERSION,"protocol_fingerprint":self.protocol_fingerprint,"opened_partitions":[],"marker_hashes":{}}
        value=json.loads(self.path.read_text())
        if value.get("schema_version") != M1311_LEDGER_SCHEMA_VERSION or value.get("protocol_fingerprint") != self.protocol_fingerprint: raise ValueError("M13.11 ledger schema or protocol mismatch")
        opened=value.get("opened_partitions",[]); markers=value.get("marker_hashes",{})
        if opened != list(dict.fromkeys(opened)) or any(row not in self.partitions for row in opened) or set(markers) != set(opened): raise ValueError("M13.11 ledger is malformed")
        for ordinal,row in enumerate(opened):
            if row == "development_check" and opened[:ordinal] != ["development_fit"]: raise ValueError("development check requires fit")
            if markers[row] != content_hash({"protocol_fingerprint":self.protocol_fingerprint,"partition":row,"ordinal":ordinal}): raise ValueError("M13.11 ledger marker mismatch")
        return value
    def open(self, partition: str, *, resumable_fit: bool = False) -> dict[str, object]:
        if partition not in self.partitions: raise ValueError("unknown M13.11 partition")
        value=self.snapshot(); opened=list(value["opened_partitions"])
        if partition in opened:
            if partition == "development_fit" and resumable_fit and opened == ["development_fit"]: return value
            raise ValueError(f"M13.11 partition {partition!r} is already opened")
        if partition == "development_check" and opened != ["development_fit"]: raise ValueError("development check requires completed development fit")
        opened.append(partition); markers=dict(value["marker_hashes"]); markers[partition]=content_hash({"protocol_fingerprint":self.protocol_fingerprint,"partition":partition,"ordinal":len(opened)-1})
        value={**value,"opened_partitions":opened,"marker_hashes":markers}; self.path.parent.mkdir(parents=True,exist_ok=True); self.path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n"); return value
