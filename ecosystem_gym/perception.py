"""Inspectable RGB-only food perception used by M3 hybrid and visual baselines."""

from __future__ import annotations

import numpy as np


def food_detection_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return [visible, u, v] from red-pixel segmentation, with no world-state input."""

    image = np.asarray(rgb, dtype=np.uint8)
    red = (image[..., 0] > 150) & (image[..., 1] < 125) & (image[..., 2] < 125)
    ys, xs = np.nonzero(red)
    if len(xs) < 3:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    height, width = image.shape[:2]
    u = (float(np.mean(xs)) / max(width - 1, 1)) * 2.0 - 1.0
    v = (float(np.mean(ys)) / max(height - 1, 1)) * 2.0 - 1.0
    return np.asarray([1.0, u, v], dtype=np.float32)


def detection_to_relative_xy(detection: np.ndarray, *, camera_half_extent: float = 1.15) -> np.ndarray:
    """Project the top-down agent-camera detection into a local XY offset."""

    visible, u, v = np.asarray(detection, dtype=np.float32)
    if visible <= 0.0:
        return np.zeros(2, dtype=np.float32)
    return np.asarray([u * camera_half_extent, -v * camera_half_extent], dtype=np.float32)
