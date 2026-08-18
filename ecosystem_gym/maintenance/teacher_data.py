"""Immutable public-only teacher dataset helpers for M13.14 successors."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import MaintenanceMacro, MaintenanceMemory, eligible_macro_mask, encode_features


PUBLIC_OBSERVATION_KEYS = frozenset(
    {"agent_xy", "food_xy", "toy_xy", "rest_xy", "drives", "holding_food", "prior_outcome"}
)
TEACHER_DATASET_FIELDS = ("features", "masks", "labels")


def validate_public_observation(observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        raise ValueError("public maintenance observation must be a dictionary")
    keys = set(observation)
    missing = PUBLIC_OBSERVATION_KEYS.difference(keys)
    extra = keys.difference(PUBLIC_OBSERVATION_KEYS)
    if missing:
        raise ValueError(f"public maintenance observation is missing {sorted(missing)!r}")
    if extra:
        raise ValueError(f"public maintenance observation contains private keys {sorted(extra)!r}")


def teacher_example(
    observation: dict[str, Any],
    memory: MaintenanceMemory,
    *,
    label: MaintenanceMacro | int,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    validate_public_observation(observation)
    features = encode_features(observation, memory)
    mask = eligible_macro_mask(observation, config)
    macro = int(label)
    if macro < 0 or macro >= len(MaintenanceMacro):
        raise ValueError("teacher label is outside the maintenance macro surface")
    if not bool(mask[macro]):
        raise ValueError("teacher chose an ineligible maintenance macro")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(mask, dtype=np.bool_),
        macro,
    )


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_arrays(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    mask_values = np.asarray(masks, dtype=np.bool_)
    label_values = np.asarray(labels, dtype=np.int64)
    rows = int(feature_values.shape[0])
    if feature_values.shape != (rows, 30):
        raise ValueError("teacher features must have shape (rows, 30)")
    if mask_values.shape != (rows, len(MaintenanceMacro)):
        raise ValueError("teacher masks must have shape (rows, 8)")
    if label_values.shape != (rows,):
        raise ValueError("teacher labels must have shape (rows,)")
    if rows == 0:
        raise ValueError("teacher dataset must contain at least one row")
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("teacher features must be finite")
    if np.any(label_values < 0) or np.any(label_values >= len(MaintenanceMacro)):
        raise ValueError("teacher label is outside the maintenance macro surface")
    if not np.all(mask_values[np.arange(rows), label_values]):
        raise ValueError("teacher chose an ineligible maintenance macro")
    class_counts = np.bincount(label_values, minlength=len(MaintenanceMacro))
    if np.any(class_counts == 0):
        missing = [MaintenanceMacro(index).name for index, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"teacher dataset must cover every macro; missing {missing!r}")
    return (
        _readonly_copy(feature_values, dtype=np.float32),
        _readonly_copy(mask_values, dtype=np.bool_),
        _readonly_copy(label_values, dtype=np.int64),
    )


def teacher_dataset_fingerprint(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"m1314-teacher-dataset-v1")
    for name, values in (
        ("features", np.asarray(features, dtype=np.float32)),
        ("masks", np.asarray(masks, dtype=np.bool_)),
        ("labels", np.asarray(labels, dtype=np.int64)),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(values.shape)).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherDataset:
    """Validated immutable public teacher rows."""

    features: np.ndarray
    masks: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        features, masks, labels = _validate_arrays(self.features, self.masks, self.labels)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "labels", labels)

    @property
    def rows(self) -> int:
        return int(self.labels.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(MaintenanceMacro))
        return {MaintenanceMacro(index).name: int(count) for index, count in enumerate(counts)}

    def fingerprint(self) -> str:
        return teacher_dataset_fingerprint(self.features, self.masks, self.labels)


def save_teacher_dataset(path: str | Path, dataset: TeacherDataset) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"teacher dataset already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **{name: getattr(dataset, name) for name in TEACHER_DATASET_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fingerprint": dataset.fingerprint(),
        "rows": dataset.rows,
        "feature_dim": dataset.feature_dim,
        "class_counts": dataset.class_counts,
        "public_fields_only": True,
    }


def load_teacher_dataset(path: str | Path) -> TeacherDataset:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if tuple(sorted(payload.files)) != tuple(sorted(TEACHER_DATASET_FIELDS)):
                raise ValueError("teacher dataset has an invalid field set")
            dataset = TeacherDataset(
                features=payload["features"],
                masks=payload["masks"],
                labels=payload["labels"],
            )
    except OSError as exc:
        raise ValueError("teacher dataset is not readable") from exc
    return dataset
