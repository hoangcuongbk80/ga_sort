from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def xyxy_to_measurement(xyxy: FloatArray) -> FloatArray:
    box = np.asarray(xyxy, dtype=np.float64)
    width = box[..., 2] - box[..., 0]
    height = box[..., 3] - box[..., 1]
    if np.any(width <= 0.0) or np.any(height <= 0.0):
        raise ValueError("box width and height must be positive")
    center_x = (box[..., 0] + box[..., 2]) * 0.5
    center_y = (box[..., 1] + box[..., 3]) * 0.5
    return np.stack((center_x, center_y, np.log(width), np.log(height)), axis=-1)


def measurement_to_xyxy(measurement: FloatArray) -> FloatArray:
    z = np.asarray(measurement, dtype=np.float64)
    width = np.exp(z[..., 2])
    height = np.exp(z[..., 3])
    return np.stack(
        (
            z[..., 0] - width * 0.5,
            z[..., 1] - height * 0.5,
            z[..., 0] + width * 0.5,
            z[..., 1] + height * 0.5,
        ),
        axis=-1,
    )


def clip_xyxy(xyxy: FloatArray, image_width: int, image_height: int) -> FloatArray:
    box = np.asarray(xyxy, dtype=np.float64).copy()
    box[..., (0, 2)] = np.clip(box[..., (0, 2)], 0.0, float(image_width))
    box[..., (1, 3)] = np.clip(box[..., (1, 3)], 0.0, float(image_height))
    return box


def pairwise_iou(first: FloatArray, second: FloatArray) -> FloatArray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
