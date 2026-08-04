"""Versioned JSONL trajectory writing and deterministic replay validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import EcosystemEnv
from .config import EcosystemConfig


SCHEMA_VERSION = "0.2"
LEGACY_SCHEMA_VERSION = "0.1"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


class TrajectoryWriter:
    """Append policy-visible records, optionally headed by exact reset metadata.

    Existing callers that provide only ``path``, ``episode_id``, and ``seed``
    retain compact step-only traces. New live/viewer callers can supply reset
    metadata so a non-default episode can be replayed exactly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        seed: int,
        reset_options: dict[str, Any] | None = None,
        observation_mode: str = "state_oracle",
        config: EcosystemConfig | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.episode_id = episode_id
        self.seed = seed
        self.observation_mode = observation_mode
        if reset_options is not None:
            header = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "episode_metadata",
                "episode_id": self.episode_id,
                "seed": self.seed,
                "reset_options": _json_value(reset_options),
                "observation_mode": self.observation_mode,
                "config": asdict(config) if config is not None else asdict(EcosystemConfig(observation_mode=observation_mode)),
            }
            self._file.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
            self._file.flush()

    def record(
        self,
        *,
        step: int,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step": step,
            "seed": self.seed,
            "observation_mode": self.observation_mode,
            "observation": _json_value(observation),
            "action": _json_value(action),
            "outcome": info["outcome"],
            "reward": reward,
            "task_success": info["task_success"],
            "terminated": terminated,
            "truncated": truncated,
            "environment_version": info["environment_version"],
        }
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ReplayResult:
    steps: int
    task_success: bool


def replay_and_validate(path: str | Path) -> ReplayResult:
    """Replay a JSONL trace and reject any changed observable transition."""

    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("trajectory has no records")
    schema_version = records[0].get("schema_version")
    if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION} or any(
        record.get("schema_version") != schema_version for record in records
    ):
        raise ValueError("trajectory schema version is unsupported")
    header = records[0] if records[0].get("record_type") == "episode_metadata" else None
    steps = records[1:] if header is not None else records
    if not steps:
        raise ValueError("trajectory has no step records")
    seed = (header or steps[0])["seed"]
    if any(record.get("seed") != seed for record in steps):
        raise ValueError("trajectory contains multiple seeds")
    reset_options = dict(header.get("reset_options", {})) if header is not None else None
    config_values = dict(header.get("config", {})) if header is not None else {}
    observation_mode = str(config_values.get("observation_mode", (header or steps[0]).get("observation_mode", "state_oracle")))
    config_values["observation_mode"] = observation_mode
    env = EcosystemEnv(EcosystemConfig(**config_values))
    env.reset(seed=seed, options=reset_options)
    try:
        for expected in steps:
            action = expected["action"]
            gym_action = {
                "kind": int(action["kind"]),
                "target": np.asarray(action["target"], dtype=np.float32),
                "duration": np.asarray(action["duration"], dtype=np.float32),
            }
            observation, reward, terminated, truncated, info = env.step(gym_action)
            if info["outcome"] != expected["outcome"]:
                raise ValueError(f"outcome mismatch at step {expected['step']}")
            if not np.isclose(reward, expected["reward"]):
                raise ValueError(f"reward mismatch at step {expected['step']}")
            if bool(terminated) != expected["terminated"] or bool(truncated) != expected["truncated"]:
                raise ValueError(f"terminal state mismatch at step {expected['step']}")
            actual_observation = _json_value(observation)
            expected_observation = expected["observation"]
            if schema_version == LEGACY_SCHEMA_VERSION:
                observation_matches = all(actual_observation.get(key) == value for key, value in expected_observation.items())
            else:
                observation_matches = actual_observation == expected_observation
            if not observation_matches:
                raise ValueError(f"observation mismatch at step {expected['step']}")
        return ReplayResult(steps=len(steps), task_success=bool(steps[-1]["task_success"]))
    finally:
        env.close()
