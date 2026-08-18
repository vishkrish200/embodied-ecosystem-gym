"""Tamper-evident artifacts for the additive M13.14 modular-Q family."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .modular_q import ModularQConfig, ModularQPolicy, QNetworkKind, QNetworkState
from .policy_state import public_dynamics_signature


MODULAR_Q_ARTIFACT_SCHEMA = "m1314-modular-q-artifact-v1"


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "ecosystem_gym/config.py",
        root / "ecosystem_gym/maintenance/contract.py",
        root / "ecosystem_gym/maintenance/policy_state.py",
        root / "ecosystem_gym/maintenance/reward.py",
        root / "ecosystem_gym/maintenance/shielded.py",
        root / "ecosystem_gym/maintenance/modular_q.py",
        root / "ecosystem_gym/maintenance/modular_q_artifacts.py",
        root / "ecosystem_gym/maintenance/teacher_data.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M13.14 artifact fingerprint sources are missing: {missing}")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _reject_m1313_path(path: str | Path) -> Path:
    resolved = Path(path)
    if any("m1313" in part.lower() for part in resolved.parts):
        raise ValueError("M13.14 artifact paths must reject frozen M13.13 paths")
    return resolved


def _tensor_payload(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _tensor_from_payload(payload: dict[str, Any]) -> np.ndarray:
    if not isinstance(payload, dict) or payload.get("dtype") != "float32":
        raise ValueError("artifact tensor payload is malformed")
    shape = payload.get("shape")
    encoded = payload.get("data_b64")
    if not isinstance(shape, list) or not all(isinstance(item, int) and item >= 0 for item in shape) or not isinstance(encoded, str):
        raise ValueError("artifact tensor shape or data is malformed")
    raw = base64.b64decode(encoded.encode("ascii"))
    array = np.frombuffer(raw, dtype=np.float32)
    expected = int(np.prod(shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError("artifact tensor byte count does not match its shape")
    return array.reshape(tuple(shape)).copy()


def _network_payload(state: QNetworkState) -> dict[str, Any]:
    return {
        "kind": state.kind.value,
        "parameter_fingerprint": state.parameter_fingerprint(),
        "parameters": {name: _tensor_payload(values) for name, values in state.parameter_items().items()},
    }


def _network_from_payload(payload: dict[str, Any]) -> QNetworkState:
    if not isinstance(payload, dict):
        raise ValueError("artifact network payload must be an object")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("artifact network parameters are malformed")
    state = QNetworkState(
        kind=QNetworkKind(str(payload.get("kind"))),
        w1=_tensor_from_payload(parameters["w1"]),
        b1=_tensor_from_payload(parameters["b1"]),
        w2=_tensor_from_payload(parameters["w2"]),
        b2=_tensor_from_payload(parameters["b2"]),
        head_w=_tensor_from_payload(parameters["head_w"]),
        head_b=_tensor_from_payload(parameters["head_b"]),
    )
    if payload.get("parameter_fingerprint") != state.parameter_fingerprint():
        raise ValueError("artifact network fingerprint mismatch")
    return state


def modular_q_policy_fingerprint(policy: ModularQPolicy) -> str:
    return policy.parameter_fingerprint()


def modular_q_weight_fingerprint(policy: ModularQPolicy) -> str:
    return policy.weight_fingerprint()


def modular_q_artifact_payload(policy: ModularQPolicy) -> dict[str, Any]:
    core = {
        "schema_version": MODULAR_Q_ARTIFACT_SCHEMA,
        "family": policy.family,
        "arm": policy.arm.value,
        "seed": policy.seed,
        "protocol_fingerprint": policy.protocol_fingerprint,
        "public_dynamics": public_dynamics_signature(policy.config),
        "learner_config": asdict(policy.learner),
        "source_hashes": _source_hashes(),
        "update_count": int(policy.update_count),
        "initial_parameter_fingerprint": str(policy.initial_parameter_fingerprint),
        "initial_weight_fingerprint": str(policy.initial_weight_fingerprint),
        "parameter_bytes_sha256": hashlib.sha256(policy.parameter_bytes()).hexdigest(),
        "weight_fingerprint": modular_q_weight_fingerprint(policy),
        "weight_bytes_sha256": hashlib.sha256(policy.weight_bytes()).hexdigest(),
        "online": _network_payload(policy.online),
        "target": _network_payload(policy.target),
    }
    payload = dict(core)
    payload["policy_fingerprint"] = modular_q_policy_fingerprint(policy)
    payload["content_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def save_modular_q_policy(path: str | Path, policy: ModularQPolicy) -> dict[str, Any]:
    output = _reject_m1313_path(path)
    if output.exists():
        raise FileExistsError(f"M13.14 modular-Q artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = modular_q_artifact_payload(policy)
    with output.open("xb") as handle:
        handle.write(_canonical(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy_fingerprint": payload["policy_fingerprint"],
        "weight_fingerprint": payload["weight_fingerprint"],
        "parameter_bytes_sha256": payload["parameter_bytes_sha256"],
        "weight_bytes_sha256": payload["weight_bytes_sha256"],
    }


def load_modular_q_policy(
    path: str | Path,
    *,
    config: Any,
    expected_protocol_fingerprint: str | None = None,
    require_current_source_hashes: bool = False,
) -> ModularQPolicy:
    artifact_path = _reject_m1313_path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("M13.14 modular-Q artifact is not readable JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != MODULAR_Q_ARTIFACT_SCHEMA:
        raise ValueError("unsupported M13.14 modular-Q artifact schema")

    supplied_content = payload.get("content_sha256")
    without_content = dict(payload)
    without_content.pop("content_sha256", None)
    if supplied_content != hashlib.sha256(_canonical(without_content)).hexdigest():
        raise ValueError("M13.14 modular-Q artifact content hash mismatch")

    supplied_policy = without_content.get("policy_fingerprint")
    core = dict(without_content)
    core.pop("policy_fingerprint", None)
    if not isinstance(supplied_policy, str) or len(supplied_policy) != 64:
        raise ValueError("M13.14 modular-Q artifact fingerprint mismatch")
    if expected_protocol_fingerprint is not None and core.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ValueError("M13.14 modular-Q artifact protocol mismatch")
    if core.get("public_dynamics") != public_dynamics_signature(config):
        raise ValueError("M13.14 modular-Q artifact public dynamics mismatch")
    if require_current_source_hashes and core.get("source_hashes") != _source_hashes():
        raise ValueError("M13.14 modular-Q artifact source hash mismatch")

    learner_config = core.get("learner_config")
    if not isinstance(learner_config, dict):
        raise ValueError("M13.14 modular-Q artifact learner configuration is malformed")
    policy = ModularQPolicy(
        arm=str(core["arm"]),
        seed=int(core["seed"]),
        config=config,
        learner=ModularQConfig(**learner_config),
        protocol_fingerprint=str(core["protocol_fingerprint"]),
    )
    policy.online = _network_from_payload(core["online"])
    policy.target = _network_from_payload(core["target"])
    if learner_config != asdict(policy.learner):
        raise ValueError("M13.14 modular-Q artifact learner configuration mismatch")
    policy.update_count = int(core["update_count"])
    policy.initial_parameter_fingerprint = str(core["initial_parameter_fingerprint"])
    policy.initial_weight_fingerprint = str(core["initial_weight_fingerprint"])
    if core.get("parameter_bytes_sha256") != hashlib.sha256(policy.parameter_bytes()).hexdigest():
        raise ValueError("M13.14 modular-Q artifact parameter bytes mismatch")
    if core.get("weight_bytes_sha256") != hashlib.sha256(policy.weight_bytes()).hexdigest():
        raise ValueError("M13.14 modular-Q artifact weight bytes mismatch")
    if core.get("weight_fingerprint") != modular_q_weight_fingerprint(policy):
        raise ValueError("M13.14 modular-Q artifact weight fingerprint mismatch")
    if modular_q_policy_fingerprint(policy) != supplied_policy:
        raise ValueError("reconstructed M13.14 modular-Q fingerprint mismatch")
    return policy
