from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EcosystemConfig


@dataclass(frozen=True, slots=True)
class Drives:
    """All drives are normalized to [0, 1]; satiety=1 means full."""

    satiety: float = 1.0
    energy: float = 1.0
    boredom: float = 0.0

    def evolve(self, config: EcosystemConfig, elapsed_seconds: float) -> "Drives":
        return Drives(
            satiety=float(np.clip(self.satiety - config.satiety_decay_per_second * elapsed_seconds, 0.0, 1.0)),
            energy=float(np.clip(self.energy - config.energy_decay_per_second * elapsed_seconds, 0.0, 1.0)),
            boredom=float(np.clip(self.boredom + config.boredom_gain_per_second * elapsed_seconds, 0.0, 1.0)),
        )

    def as_array(self) -> np.ndarray:
        return np.asarray([self.satiety, self.energy, self.boredom], dtype=np.float32)
