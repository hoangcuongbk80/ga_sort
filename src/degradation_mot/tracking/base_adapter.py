from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeVar

import numpy as np

from ..config import PhotometryConfig
from ..models.confidence import beta_log_tail
from ..models.observation import ObservationModel
from ..photometry.descriptors import extract_baseline_descriptor, extract_degradation_descriptor
from ..photometry.mask import build_saturation_candidate_mask
from ..types import Detection, FloatArray
from .association import beta_log_density

Output = TypeVar("Output", covariant=True)


@dataclass(frozen=True)
class BaseObservation:
    detection_probability: float
    bias: FloatArray
    covariance: FloatArray
    log_confidence_density: Callable[[float], float]


@dataclass(frozen=True)
class BaseTrackPrediction:
    xyxy: FloatArray
    recent_detections: Sequence[bool]
    residual_scale: FloatArray
    native_observation: BaseObservation


ObservationOverride = Callable[[Sequence[BaseTrackPrediction]], list[BaseObservation]]


class BaseHost(Protocol[Output]):
    def step(self, frame: np.ndarray, detections: list[Detection], timestamp: float) -> Output: ...

    def step_with_observations(
        self, frame: np.ndarray, detections: list[Detection], timestamp: float,
        override: ObservationOverride,
    ) -> Output:
        """Invoke override after native prediction; use its quantities in the native update."""
        ...


@dataclass(frozen=True)
class ObservationSwitches:
    detection_probability: bool = False
    bias: bool = False
    covariance: bool = False
    confidence: bool = False


class BaseObservationAdapter:
    """Replace observation quantities inside a configured BASE host step."""

    def __init__(
        self, model: ObservationModel | None, switches: ObservationSwitches,
        photometry: PhotometryConfig = PhotometryConfig(),
    ) -> None:
        self.model = model
        self.switches = switches
        self.photometry = photometry
        if switches != ObservationSwitches():
            if model is None:
                raise ValueError("enabled BASE observation switches require a fitted model")
            threshold = getattr(model, "confidence_threshold", None)
            if threshold is not None and not np.isclose(threshold, 0.10, rtol=0.0, atol=1.0e-8):
                raise ValueError("BASE variants use retained scores at threshold 0.10")

    def step(
        self, host: BaseHost[Output], frame: np.ndarray, detections: list[Detection], timestamp: float,
    ) -> Output:
        if self.switches == ObservationSwitches():
            return host.step(frame, detections, timestamp)
        if any(not 0.10 <= detection.confidence < 1.0 for detection in detections):
            raise ValueError("pass the common ordered detections already filtered at 0.10")
        return host.step_with_observations(
            frame, detections, timestamp, lambda predictions: self._replace(frame, predictions)
        )

    def _replace(
        self, frame: np.ndarray, predictions: Sequence[BaseTrackPrediction],
    ) -> list[BaseObservation]:
        if not predictions:
            return []
        assert self.model is not None
        height, width = frame.shape[:2]
        mask, value = build_saturation_candidate_mask(frame, self.photometry)
        boxes = np.asarray([track.xyxy for track in predictions])
        observations = []
        for index, track in enumerate(predictions):
            degradation = extract_degradation_descriptor(
                mask, value, track.xyxy, self.photometry.value_threshold,
                self.photometry.boundary_band_fraction, self.photometry.radial_bins,
            )
            baseline = extract_baseline_descriptor(
                track.xyxy, boxes, index, width, height, track.recent_detections,
            )
            predicted = self.model.predict(degradation, baseline, 1.0, 1.0)
            native = track.native_observation
            scale = track.residual_scale
            observations.append(BaseObservation(
                detection_probability=(predicted.detection_probability if self.switches.detection_probability
                                       else native.detection_probability),
                bias=predicted.bias * scale if self.switches.bias else native.bias,
                covariance=(predicted.covariance * np.outer(scale, scale) if self.switches.covariance
                            else native.covariance),
                log_confidence_density=(_confidence_mark(predicted.confidence_alpha, predicted.confidence_beta)
                                        if self.switches.confidence else native.log_confidence_density),
            ))
        return observations


def _confidence_mark(alpha: float, beta: float) -> Callable[[float], float]:
    tail = beta_log_tail(0.10, alpha, beta)

    def log_density(score: float) -> float:
        if not 0.10 <= score < 1.0:
            return -np.inf
        return beta_log_density(float(np.clip(score, 1.0e-4, 1.0 - 1.0e-4)), alpha, beta) - tail

    return log_density
