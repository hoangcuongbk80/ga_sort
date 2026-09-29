from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import PhotometryConfig


def build_saturation_candidate_mask(
    frame_bgr: NDArray[np.uint8], config: PhotometryConfig
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Return the cleaned candidate mask and the normalized HSV value channel."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for photometric processing") from exc

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3)")
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    raw = (value > config.value_threshold) & (
        saturation < config.saturation_threshold
    )

    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=np.bool_)
    if count > 1:
        keep[1:] = statistics[1:, cv2.CC_STAT_AREA] >= config.minimum_component_area
    return keep[labels], value

