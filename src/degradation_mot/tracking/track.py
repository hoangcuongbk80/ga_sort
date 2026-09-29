from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Track:
    track_id: int
    category: int
    mean: FloatArray
    covariance: FloatArray
    existence: float
    history_window: int
    detection_history: deque[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.detection_history = deque([True], maxlen=self.history_window)

    def record_detection(self, observed: bool) -> None:
        self.detection_history.append(observed)

