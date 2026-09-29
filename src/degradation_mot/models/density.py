from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class MarkedDensity(Protocol):
    cardinality_rate: float

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float: ...


@dataclass
class UniformMarkedDensity:
    """Finite fallback density over image position, log scale, and confidence."""

    cardinality_rate: float = 1.0
    log_scale_range: float = 20.0

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float:
        del measurement, confidence
        image_width, image_height = image_size
        support = image_width * image_height * self.log_scale_range**2
        return log(max(self.cardinality_rate, 1.0e-300)) - log(max(support, 1.0))


@dataclass
class DiagonalGaussianMixtureDensity:
    """Marked intensity in normalized box and logit-confidence coordinates."""

    weights: FloatArray
    means: FloatArray
    variances: FloatArray
    cardinality_rate: float

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.variances = np.asarray(self.variances, dtype=np.float64)
        if self.means.ndim != 2 or self.means.shape[1] != 5:
            raise ValueError("means must have shape (components, 5)")
        if self.variances.shape != self.means.shape:
            raise ValueError("variances and means must have equal shapes")
        if self.weights.shape != (len(self.means),):
            raise ValueError("weights must contain one entry per component")
        if np.any(self.variances <= 0.0) or np.any(self.weights <= 0.0):
            raise ValueError("weights and variances must be positive")
        self.weights = self.weights / self.weights.sum()

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float:
        image_width, image_height = image_size
        clipped_confidence = float(np.clip(confidence, 1.0e-12, 1.0 - 1.0e-12))
        feature = np.asarray(
            [
                measurement[0] / image_width,
                measurement[1] / image_height,
                measurement[2] - log(image_width),
                measurement[3] - log(image_height),
                log(clipped_confidence / (1.0 - clipped_confidence)),
            ],
            dtype=np.float64,
        )
        difference = feature[None, :] - self.means
        component_log_density = (
            np.log(self.weights)
            - 0.5
            * (
                np.sum(difference**2 / self.variances, axis=1)
                + np.sum(np.log(self.variances), axis=1)
                + 5.0 * log(2.0 * pi)
            )
        )
        maximum = float(component_log_density.max())
        log_mixture = maximum + log(float(np.exp(component_log_density - maximum).sum()))
        log_jacobian = -log(image_width) - log(image_height) - log(
            clipped_confidence * (1.0 - clipped_confidence)
        )
        return log(max(self.cardinality_rate, 1.0e-300)) + log_mixture + log_jacobian

    @classmethod
    def fit(
        cls,
        measurements: FloatArray,
        confidences: FloatArray,
        image_sizes: FloatArray,
        cardinality_rate: float,
        components: int = 8,
        random_state: int = 0,
    ) -> "DiagonalGaussianMixtureDensity":
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            raise ImportError("scikit-learn is required to fit nuisance densities") from exc
        measurements = np.asarray(measurements, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)
        image_sizes = np.asarray(image_sizes, dtype=np.float64)
        if measurements.ndim != 2 or measurements.shape[1] != 4:
            raise ValueError("measurements must have shape (samples, 4)")
        if image_sizes.shape != (len(measurements), 2):
            raise ValueError("image_sizes must have shape (samples, 2)")
        width = image_sizes[:, 0]
        height = image_sizes[:, 1]
        confidence = np.clip(confidences, 1.0e-4, 1.0 - 1.0e-4)
        features = np.column_stack(
            (
                measurements[:, 0] / width,
                measurements[:, 1] / height,
                measurements[:, 2] - np.log(width),
                measurements[:, 3] - np.log(height),
                np.log(confidence / (1.0 - confidence)),
            )
        )
        mixture = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            random_state=random_state,
            reg_covar=1.0e-6,
        ).fit(features)
        return cls(
            weights=mixture.weights_,
            means=mixture.means_,
            variances=mixture.covariances_,
            cardinality_rate=cardinality_rate,
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                weights=self.weights,
                means=self.means,
                variances=self.variances,
                cardinality_rate=np.asarray(self.cardinality_rate),
            )

    @classmethod
    def load(cls, path: str | Path) -> "DiagonalGaussianMixtureDensity":
        with np.load(Path(path)) as archive:
            return cls(
                weights=archive["weights"],
                means=archive["means"],
                variances=archive["variances"],
                cardinality_rate=float(archive["cardinality_rate"]),
            )
