from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..geometry import clip_xyxy, pairwise_iou


def _raster_bounds(
    xyxy: NDArray[np.float64], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    clipped = clip_xyxy(xyxy, image_width, image_height)
    x1 = int(np.floor(clipped[0]))
    y1 = int(np.floor(clipped[1]))
    x2 = int(np.ceil(clipped[2]))
    y2 = int(np.ceil(clipped[3]))
    return x1, y1, x2, y2


def extract_degradation_descriptor(
    candidate_mask: NDArray[np.bool_],
    value_channel: NDArray[np.float32],
    predicted_xyxy: NDArray[np.float64],
    value_threshold: float,
    boundary_band_fraction: float = 0.10,
    radial_bins: int = 4,
    epsilon: float = 1.0e-8,
) -> NDArray[np.float64]:
    """Extract [area, strength, 13 geometry values] from a predicted region.

    Boundary and radial occupancies are fractions of the candidate pixels, so
    they encode spatial distribution independently of the candidate-area ratio.
    """
    if candidate_mask.shape != value_channel.shape:
        raise ValueError("candidate_mask and value_channel must have equal shapes")
    if radial_bins != 4:
        raise ValueError("the manuscript configuration requires four radial bins")

    image_height, image_width = candidate_mask.shape
    x1, y1, x2, y2 = _raster_bounds(predicted_xyxy, image_width, image_height)
    output = np.zeros(15, dtype=np.float64)
    if x2 <= x1 or y2 <= y1:
        return output

    local_mask = candidate_mask[y1:y2, x1:x2]
    local_value = value_channel[y1:y2, x1:x2]
    region_area = float(local_mask.size)
    candidate_count = int(local_mask.sum())
    output[0] = candidate_count / region_area
    excess = np.maximum(local_value - value_threshold, 0.0)
    output[1] = float((local_mask * excess).sum()) / (
        region_area * (1.0 - value_threshold + epsilon)
    )
    if candidate_count == 0:
        return output

    rows, columns = np.nonzero(local_mask)
    pixel_x = x1 + columns.astype(np.float64) + 0.5
    pixel_y = y1 + rows.astype(np.float64) + 0.5
    box = np.asarray(predicted_xyxy, dtype=np.float64)
    width = max(box[2] - box[0], epsilon)
    height = max(box[3] - box[1], epsilon)
    center_x = (box[0] + box[2]) * 0.5
    center_y = (box[1] + box[3]) * 0.5

    centroid_x = float(pixel_x.mean())
    centroid_y = float(pixel_y.mean())
    output[2] = (centroid_x - center_x) / width
    output[3] = (centroid_y - center_y) / height

    centered_x = pixel_x - centroid_x
    centered_y = pixel_y - centroid_y
    output[4] = float(np.mean(centered_x**2)) / (width**2)
    output[5] = float(np.mean(centered_y**2)) / (height**2)
    output[6] = float(np.mean(centered_x * centered_y)) / (width * height)

    distance_to_sides = np.stack(
        (
            (pixel_x - box[0]) / width,
            (box[2] - pixel_x) / width,
            (pixel_y - box[1]) / height,
            (box[3] - pixel_y) / height,
        ),
        axis=1,
    )
    nearest_side = np.argmin(distance_to_sides, axis=1)
    nearest_distance = np.min(distance_to_sides, axis=1)
    in_boundary_band = nearest_distance <= boundary_band_fraction
    for side in range(4):
        output[7 + side] = np.mean(in_boundary_band & (nearest_side == side))

    half_diagonal = 0.5 * np.sqrt(width**2 + height**2)
    radius = np.sqrt((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2)
    radius = np.clip(radius / max(half_diagonal, epsilon), 0.0, 1.0)
    radial_index = np.minimum((radius * radial_bins).astype(int), radial_bins - 1)
    for index in range(radial_bins):
        output[11 + index] = np.mean(radial_index == index)
    return output


def extract_baseline_descriptor(
    predicted_xyxy: NDArray[np.float64],
    all_predicted_xyxy: NDArray[np.float64],
    track_index: int,
    image_width: int,
    image_height: int,
    recent_detection_history: Sequence[bool],
) -> NDArray[np.float64]:
    """Extract the 11-dimensional class-independent baseline descriptor."""
    box = np.asarray(predicted_xyxy, dtype=np.float64)
    width = max(box[2] - box[0], 1.0e-8)
    height = max(box[3] - box[1], 1.0e-8)
    center_x = (box[0] + box[2]) * 0.5
    center_y = (box[1] + box[3]) * 0.5

    max_overlap = 0.0
    if len(all_predicted_xyxy) > 1:
        overlaps = pairwise_iou(box[None, :], all_predicted_xyxy)[0]
        overlaps[track_index] = 0.0
        max_overlap = float(overlaps.max())
    recent_rate = (
        float(np.mean(np.asarray(recent_detection_history, dtype=np.float64)))
        if recent_detection_history
        else 1.0
    )
    return np.asarray(
        [
            center_x / image_width,
            center_y / image_height,
            np.log(width / image_width),
            np.log(height / image_height),
            np.log(width / height),
            box[0] / image_width,
            box[1] / image_height,
            (image_width - box[2]) / image_width,
            (image_height - box[3]) / image_height,
            max_overlap,
            recent_rate,
        ],
        dtype=np.float64,
    )
