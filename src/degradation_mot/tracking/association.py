from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log, pi

import numpy as np
from numpy.typing import NDArray

from ..models.density import MarkedDensity
from ..models.observation import ObservationParameters
from ..types import Detection
from .kalman import innovation_statistics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]
    missed_tracks: tuple[int, ...]
    unused_detections: tuple[int, ...]


def beta_log_density(confidence: float, alpha: float, beta: float) -> float:
    value = float(np.clip(confidence, 1.0e-12, 1.0 - 1.0e-12))
    normalization = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    return normalization + (alpha - 1.0) * log(value) + (beta - 1.0) * log(1.0 - value)


def gaussian_log_density(residual: FloatArray, covariance: FloatArray) -> float:
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("innovation covariance is not positive definite")
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    dimension = len(residual)
    return -0.5 * (dimension * log(2.0 * pi) + log_determinant + mahalanobis)


def associate(
    track_means: list[FloatArray],
    track_covariances: list[FloatArray],
    track_existences: list[float],
    track_categories: list[int],
    observation_parameters: list[ObservationParameters],
    detections: list[Detection],
    detection_measurements: FloatArray,
    clutter_density: MarkedDensity,
    image_size: tuple[int, int],
    gate_threshold: float,
    gate_probability: float,
    probability_clip: tuple[float, float],
    confidence_clip: tuple[float, float],
    density_floor: float,
    logarithm_floor: float,
) -> AssociationResult:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("SciPy is required for Hungarian assignment") from exc

    track_count = len(track_means)
    detection_count = len(detections)
    if track_count == 0:
        return AssociationResult((), (), tuple(range(detection_count)))

    cost = np.full((track_count, detection_count + track_count), np.inf)
    for track_index in range(track_count):
        parameters = observation_parameters[track_index]
        probability = float(np.clip(parameters.detection_probability, *probability_clip))
        miss_probability = max(
            1.0 - track_existences[track_index] * probability * gate_probability,
            logarithm_floor,
        )
        cost[track_index, detection_count + track_index] = -log(miss_probability)

        for detection_index, detection in enumerate(detections):
            if detection.category != track_categories[track_index]:
                continue
            measurement = detection_measurements[detection_index]
            innovation, innovation_covariance = innovation_statistics(
                track_means[track_index],
                track_covariances[track_index],
                parameters.bias,
                parameters.covariance,
                measurement,
            )
            mahalanobis = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
            if mahalanobis > gate_threshold:
                continue

            clipped_confidence = float(np.clip(detection.confidence, *confidence_clip))
            log_target = gaussian_log_density(innovation, innovation_covariance)
            log_target += beta_log_density(
                clipped_confidence,
                parameters.confidence_alpha,
                parameters.confidence_beta,
            )
            log_clutter = clutter_density.log_intensity(
                measurement, clipped_confidence, image_size
            )
            log_clutter = max(log_clutter, log(density_floor))
            log_ratio = (
                log(max(track_existences[track_index], logarithm_floor))
                + log(probability)
                + log_target
                - log_clutter
            )
            cost[track_index, detection_index] = -log_ratio

    row_indices, column_indices = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    missed_tracks: list[int] = []
    used_detections: set[int] = set()
    assigned_columns = dict(zip(row_indices.tolist(), column_indices.tolist()))
    for track_index in range(track_count):
        column = assigned_columns[track_index]
        if column < detection_count:
            matches.append((track_index, column))
            used_detections.add(column)
        else:
            missed_tracks.append(track_index)
    unused = tuple(index for index in range(detection_count) if index not in used_detections)
    return AssociationResult(tuple(matches), tuple(missed_tracks), unused)


def birth_existence(log_birth_intensity: float, log_clutter_intensity: float) -> float:
    difference = log_clutter_intensity - log_birth_intensity
    if difference >= 0:
        scaled = exp(-difference)
        return scaled / (1.0 + scaled)
    scaled = exp(difference)
    return 1.0 / (1.0 + scaled)


def missed_existence(
    predicted_existence: float, detection_probability: float, gate_probability: float
) -> float:
    missed_likelihood = 1.0 - detection_probability * gate_probability
    numerator = predicted_existence * missed_likelihood
    denominator = 1.0 - predicted_existence * detection_probability * gate_probability
    return float(np.clip(numerator / max(denominator, 1.0e-15), 0.0, 1.0))

