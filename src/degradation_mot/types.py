from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Detection:
    xyxy: FloatArray
    confidence: float
    category: int

    @classmethod
    def from_xyxy(
        cls,
        xyxy: tuple[float, float, float, float] | FloatArray,
        confidence: float,
        category: int,
    ) -> "Detection":
        box = np.asarray(xyxy, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError("xyxy must contain four values")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detection width and height must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        return cls(box, float(confidence), int(category))


@dataclass(frozen=True)
class TrackOutput:
    track_id: int
    xyxy: FloatArray
    category: int
    existence_probability: float

