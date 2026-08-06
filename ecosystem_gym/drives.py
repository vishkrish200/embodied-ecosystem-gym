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

    def relieve_boredom(self, amount: float) -> "Drives":
        """Apply a bounded positive play effect without changing survival drives."""
        return Drives(
            satiety=self.satiety,
            energy=self.energy,
            boredom=float(np.clip(self.boredom - amount, 0.0, 1.0)),
        )

    def restore_energy(self, amount: float) -> "Drives":
        """Apply a bounded rest effect without changing satiety or boredom."""
        return Drives(
            satiety=self.satiety,
            energy=float(np.clip(self.energy + amount, 0.0, 1.0)),
            boredom=self.boredom,
        )

    def as_array(self) -> np.ndarray:
        return np.asarray([self.satiety, self.energy, self.boredom], dtype=np.float32)
