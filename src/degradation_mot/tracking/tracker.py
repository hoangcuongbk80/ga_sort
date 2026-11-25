from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
from numpy.typing import NDArray

from ..config import TrackerConfig
from ..geometry import clip_xyxy, measurement_to_xyxy, xyxy_to_measurement
from ..models.density import (
    DiagonalGaussianMixtureDensity, MarkedDensity, SpatialClutterDensity,
    UniformMarkedDensity, marked_log_intensity,
)
from ..models.observation import ConstantObservationModel, ObservationModel, ObservationParameters
from ..photometry.descriptors import (
    extract_baseline_descriptor,
    extract_degradation_descriptor,
)
from ..photometry.mask import build_saturation_candidate_mask
from ..types import Detection, TrackOutput
from .association import associate, birth_existence, missed_existence
from .kalman import (
    birth_covariance,
    correct,
    innovation_statistics,
    predict,
    transition_and_process_covariance,
)
from .track import Track


@dataclass(frozen=True)
class FrameDiagnostics:
    degradation_descriptors: tuple[NDArray[np.float64], ...]
    baseline_descriptors: tuple[NDArray[np.float64], ...]
    observation_parameters: tuple[ObservationParameters, ...]
    matches: tuple[tuple[int, int], ...]
    missed_track_indices: tuple[int, ...]
    unused_detection_indices: tuple[int, ...]


class DegradationConditionedTracker:
    @classmethod
    def internal_fixed_observation_reference(
        cls, config: TrackerConfig, spatial_clutter: DiagonalGaussianMixtureDensity,
    ) -> DegradationConditionedTracker:
        if config.tracking.confidence_threshold != 0.10:
            raise ValueError("the internal reference uses threshold 0.10")
        model = ConstantObservationModel(confidence_alpha=1.0, confidence_beta=1.0)
        return cls(config, model, SpatialClutterDensity(spatial_clutter), fixed_reference=True)

    def __init__(
        self,
        config: TrackerConfig,
        observation_model: ObservationModel,
        clutter_density: MarkedDensity | None = None,
        birth_density: MarkedDensity | None = None,
        *, fixed_reference: bool = False,
    ) -> None:
        self.config = config
        self.observation_model = observation_model
        degradation_mean = getattr(observation_model, "degradation_mean", None)
        if degradation_mean is not None and len(degradation_mean) != 11 + config.photometry.radial_bins:
            raise ValueError("checkpoint descriptor dimension does not match radial_bins")
        self.fixed_reference = fixed_reference
        threshold = config.tracking.confidence_threshold
        self.clutter_density = clutter_density or UniformMarkedDensity(1.0, confidence_threshold=threshold)
        self.birth_density = birth_density or UniformMarkedDensity(0.25, confidence_threshold=threshold)
        for name, model in (("observation", observation_model), ("clutter", self.clutter_density),
                            ("birth", self.birth_density)):
            fitted_threshold = getattr(model, "confidence_threshold", None)
            if fitted_threshold is not None and not np.isclose(
                fitted_threshold, threshold, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(f"{name} model was calibrated at threshold {fitted_threshold}, expected {threshold}")
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self.previous_timestamp: float | None = None
        self.last_diagnostics: FrameDiagnostics | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.previous_timestamp = None
        self.last_diagnostics = None

    def update(
        self,
        frame_bgr: NDArray[np.uint8],
        detections: list[Detection],
        timestamp: float,
    ) -> list[TrackOutput]:
        image_height, image_width = frame_bgr.shape[:2]
        delta_time = self._delta_time(timestamp)
        if delta_time > self.config.motion.maximum_timestamp_gap:
            self.tracks.clear()
            self.last_diagnostics = None
            delta_time = self.config.tracking.reference_frame_interval

        transition, process = transition_and_process_covariance(
            delta_time, image_width, image_height, self.config.motion
        )
        survival = self.config.tracking.survival_probability ** (
            delta_time / self.config.tracking.reference_frame_interval
        )
        for track in self.tracks:
            track.mean, track.covariance = predict(
                track.mean, track.covariance, transition, process
            )
            if not self.fixed_reference:
                track.existence *= survival

        mask, value = build_saturation_candidate_mask(
            frame_bgr, self.config.photometry
        )
        predicted_boxes = np.asarray(
            [measurement_to_xyxy(track.mean[:4]) for track in self.tracks],
            dtype=np.float64,
        ).reshape((-1, 4))
        degradation_descriptors: list[NDArray[np.float64]] = []
        baseline_descriptors: list[NDArray[np.float64]] = []
        parameters: list[ObservationParameters] = []
        for index, track in enumerate(self.tracks):
            box = predicted_boxes[index]
            degradation = extract_degradation_descriptor(
                mask,
                value,
                box,
                self.config.photometry.value_threshold,
                self.config.photometry.boundary_band_fraction,
                self.config.photometry.radial_bins,
            )
            baseline = extract_baseline_descriptor(
                box,
                predicted_boxes,
                index,
                image_width,
                image_height,
                list(track.detection_history),
            )
            width = exp(track.mean[2])
            height = exp(track.mean[3])
            degradation_descriptors.append(degradation)
            baseline_descriptors.append(baseline)
            parameters.append(
                self.observation_model.predict(degradation, baseline, width, height)
            )

        measurements = (
            np.asarray([xyxy_to_measurement(detection.xyxy) for detection in detections])
            if detections
            else np.empty((0, 4), dtype=np.float64)
        )
        gate_threshold = self._gate_threshold()
        assignment = associate(
            [track.mean for track in self.tracks],
            [track.covariance for track in self.tracks],
            [track.existence for track in self.tracks],
            [track.category for track in self.tracks],
            parameters,
            detections,
            measurements,
            self.clutter_density,
            (image_width, image_height),
            gate_threshold,
            self.config.tracking.gate_probability,
            self.config.tracking.probability_clip,
            self.config.tracking.confidence_clip,
            self.config.tracking.density_floor,
            self.config.tracking.logarithm_floor,
            self.config.tracking.confidence_threshold,
        )

        for track_index, detection_index in assignment.matches:
            track = self.tracks[track_index]
            parameter = parameters[track_index]
            innovation, innovation_covariance = innovation_statistics(
                track.mean,
                track.covariance,
                parameter.bias,
                parameter.covariance,
                measurements[detection_index],
            )
            track.mean, track.covariance = correct(
                track.mean,
                track.covariance,
                innovation,
                innovation_covariance,
                parameter.covariance,
            )
            track.existence = 1.0
            track.record_detection(True)

        for track_index in assignment.missed_tracks:
            track = self.tracks[track_index]
            if not self.fixed_reference:
                track.existence = missed_existence(
                    track.existence, parameters[track_index].detection_probability,
                    self.config.tracking.gate_probability,
                )
            track.record_detection(False)

        if self.fixed_reference:
            self.tracks = [track for track in self.tracks if track.consecutive_misses
                           <= (30 if track.confirmed else 0)]
        else:
            self.tracks = [track for track in self.tracks
                           if track.existence >= self.config.tracking.delete_threshold]
        for detection_index in assignment.unused_detections:
            self._initialize_track(
                detections[detection_index],
                measurements[detection_index],
                image_width,
                image_height,
            )

        self.last_diagnostics = FrameDiagnostics(
            tuple(degradation_descriptors),
            tuple(baseline_descriptors),
            tuple(parameters),
            assignment.matches,
            assignment.missed_tracks,
            assignment.unused_detections,
        )
        self.previous_timestamp = timestamp
        return self._reported_tracks(image_width, image_height)

    def _initialize_track(
        self,
        detection: Detection,
        measurement: NDArray[np.float64],
        image_width: int,
        image_height: int,
    ) -> None:
        if self.fixed_reference:
            if detection.confidence < 0.50:
                return
            existence = 1.0
        else:
            image_size = (image_width, image_height)
            log_birth = marked_log_intensity(
                self.birth_density, measurement, detection.confidence, image_size,
                self.config.tracking.confidence_clip,
            )
            log_clutter = marked_log_intensity(
                self.clutter_density, measurement, detection.confidence, image_size,
                self.config.tracking.confidence_clip,
            )
            existence = birth_existence(log_birth, log_clutter)
        mean = np.concatenate((measurement, np.zeros(4, dtype=np.float64)))
        self.tracks.append(
            Track(
                track_id=self.next_track_id,
                category=detection.category,
                mean=mean,
                covariance=birth_covariance(
                    image_width, image_height, self.config.motion
                ),
                existence=existence,
                history_window=self.config.tracking.recent_detection_window,
            )
        )
        self.next_track_id += 1

    def _reported_tracks(self, image_width: int, image_height: int) -> list[TrackOutput]:
        outputs: list[TrackOutput] = []
        for track in self.tracks:
            report = (track.confirmed and track.consecutive_misses == 0 if self.fixed_reference
                      else track.existence >= self.config.tracking.report_threshold)
            if not report:
                continue
            box = measurement_to_xyxy(track.mean[:4])
            outputs.append(
                TrackOutput(
                    track_id=track.track_id,
                    xyxy=clip_xyxy(box, image_width, image_height),
                    category=track.category,
                    existence_probability=track.existence,
                )
            )
        return outputs

    def _delta_time(self, timestamp: float) -> float:
        if self.previous_timestamp is None:
            return self.config.tracking.reference_frame_interval
        delta_time = timestamp - self.previous_timestamp
        if delta_time <= 0.0:
            raise ValueError("timestamps must increase strictly")
        return delta_time

    def _gate_threshold(self) -> float:
        try:
            from scipy.stats import chi2
        except ImportError as exc:
            raise ImportError("SciPy is required to compute the chi-square gate") from exc
        return float(chi2.ppf(self.config.tracking.gate_probability, df=4))
