"""Tamper-evident serialization for additive maintenance policy families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model_based import ModelBasedSchedulerPolicy, model_config_from_dict
from .policy_state import policy_parameter_dict, public_dynamics_signature
from .shielded import PublicMLPActor, ShieldedLearnedPolicy, shield_config_from_dict
from .urgency import UrgencySchedulerPolicy, urgency_config_from_dict


POLICY_ARTIFACT_SCHEMA = "maintenance-policy-artifact-v1"
Policy = UrgencySchedulerPolicy | ModelBasedSchedulerPolicy | ShieldedLearnedPolicy


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _core_payload(policy: Policy) -> dict[str, Any]:
    if isinstance(policy, UrgencySchedulerPolicy):
        parameters = policy_parameter_dict(policy.scheduler)
        actor = None
    elif isinstance(policy, ModelBasedSchedulerPolicy):
        parameters = policy_parameter_dict(policy.planner)
        actor = None
    elif isinstance(policy, ShieldedLearnedPolicy):
        parameters = policy_parameter_dict(policy.shield)
        actor = {
            "schema_version": "maintenance-public-actor-v1",
            "fingerprint": policy.actor.fingerprint(),
            "network": policy.actor.payload(),
        }
    else:  # pragma: no cover - the type alias is closed, runtime callers are not
        raise TypeError(f"unsupported maintenance policy type {type(policy)!r}")
    return {
        "schema_version": POLICY_ARTIFACT_SCHEMA,
        "family": policy.family,
        "protocol_fingerprint": policy.protocol_fingerprint,
        "public_dynamics": public_dynamics_signature(policy.config),
        "parameters": parameters,
        "actor": actor,
    }


def policy_fingerprint(policy: Policy) -> str:
    return hashlib.sha256(_canonical(_core_payload(policy))).hexdigest()


def artifact_payload(policy: Policy) -> dict[str, Any]:
    payload = _core_payload(policy)
    payload["policy_fingerprint"] = hashlib.sha256(_canonical(payload)).hexdigest()
    payload["content_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def save_policy(path: str | Path, policy: Policy) -> dict[str, str]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"maintenance policy artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact_payload(policy)
    output.write_bytes(_canonical(payload) + b"\n")
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy_fingerprint": str(payload["policy_fingerprint"]),
    }


def load_policy(
    path: str | Path,
    *,
    config: Any,
    expected_protocol_fingerprint: str | None = None,
) -> Policy:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("maintenance policy artifact is not readable JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != POLICY_ARTIFACT_SCHEMA:
        raise ValueError("unsupported maintenance policy artifact schema")

    supplied_content_hash = payload.get("content_sha256")
    without_content = dict(payload)
    without_content.pop("content_sha256", None)
    if supplied_content_hash != hashlib.sha256(_canonical(without_content)).hexdigest():
        raise ValueError("maintenance policy artifact content hash mismatch")

    supplied_policy_hash = without_content.get("policy_fingerprint")
    core = dict(without_content)
    core.pop("policy_fingerprint", None)
    if supplied_policy_hash != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("maintenance policy artifact fingerprint mismatch")
    protocol = core.get("protocol_fingerprint")
    if expected_protocol_fingerprint is not None and protocol != expected_protocol_fingerprint:
        raise ValueError("maintenance policy artifact protocol mismatch")
    if core.get("public_dynamics") != public_dynamics_signature(config):
        raise ValueError("maintenance policy artifact public dynamics mismatch")

    family = core.get("family")
    parameters = core.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("maintenance policy artifact parameters are malformed")
    if family == UrgencySchedulerPolicy.family:
        if core.get("actor") is not None:
            raise ValueError("deterministic urgency artifact unexpectedly contains an actor")
        policy: Policy = UrgencySchedulerPolicy(
            config=config,
            scheduler=urgency_config_from_dict(parameters),
            protocol_fingerprint=str(protocol),
        )
    elif family == ModelBasedSchedulerPolicy.family:
        if core.get("actor") is not None:
            raise ValueError("model-based artifact unexpectedly contains an actor")
        policy = ModelBasedSchedulerPolicy(
            config=config,
            planner=model_config_from_dict(parameters),
            protocol_fingerprint=str(protocol),
        )
    elif family == ShieldedLearnedPolicy.family:
        actor_payload = core.get("actor")
        if not isinstance(actor_payload, dict) or actor_payload.get("schema_version") != "maintenance-public-actor-v1":
            raise ValueError("shielded policy artifact actor is malformed")
        actor = PublicMLPActor.from_payload(actor_payload.get("network"))
        if actor_payload.get("fingerprint") != actor.fingerprint():
            raise ValueError("shielded policy actor fingerprint mismatch")
        policy = ShieldedLearnedPolicy(
            actor=actor,
            config=config,
            shield=shield_config_from_dict(parameters),
            protocol_fingerprint=str(protocol),
        )
    else:
        raise ValueError(f"unknown maintenance policy family {family!r}")
    if policy_fingerprint(policy) != supplied_policy_hash:
        raise ValueError("reconstructed maintenance policy fingerprint mismatch")
    return policy
