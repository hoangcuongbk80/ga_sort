from __future__ import annotations

from dataclasses import dataclass, field
from math import log, pi
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import log_ndtr, logit, logsumexp

FloatArray = NDArray[np.float64]


class MarkedDensity(Protocol):
    cardinality_rate: float

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float: ...


def marked_log_intensity(
    density: MarkedDensity, measurement: FloatArray, confidence: float,
    image_size: tuple[int, int], confidence_clip: tuple[float, float],
) -> float:
    if isinstance(density, DiagonalGaussianMixtureDensity):
        return density.log_intensity(
            measurement, confidence, image_size, confidence_clip=confidence_clip
        )
    return density.log_intensity(measurement, confidence, image_size)


@dataclass
class UniformMarkedDensity:
    """Finite fallback density over image position, log scale, and confidence."""

    cardinality_rate: float = 1.0
    log_scale_range: float = 20.0
    confidence_threshold: float = 0.0

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float:
        del measurement
        if not self.confidence_threshold <= confidence < 1.0:
            return -np.inf
        image_width, image_height = image_size
        support = (image_width * image_height * self.log_scale_range**2
                   * (1.0 - self.confidence_threshold))
        return log(max(self.cardinality_rate, 1.0e-300)) - log(max(support, 1.0))


@dataclass
class DiagonalGaussianMixtureDensity:
    """Marked intensity in normalized box and logit-confidence coordinates."""

    weights: FloatArray
    means: FloatArray
    variances: FloatArray
    cardinality_rate: float
    confidence_threshold: float = 0.0
    _log_tail: float = field(init=False, repr=False)

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
        if not 0.0 <= self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1)")
        self._log_tail = self.log_confidence_tail(self.confidence_threshold)

    def log_confidence_tail(self, threshold: float) -> float:
        standardized = (self.means[:, 4] - logit(threshold)) / np.sqrt(self.variances[:, 4])
        return float(logsumexp(np.log(self.weights) + log_ndtr(standardized)))

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int],
        *, confidence_clip: tuple[float, float] = (1.0e-4, 1.0 - 1.0e-4),
    ) -> float:
        if not self.confidence_threshold <= confidence < 1.0:
            return -np.inf
        image_width, image_height = image_size
        clipped_confidence = float(np.clip(confidence, *confidence_clip))
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
        log_mixture = float(logsumexp(component_log_density))
        log_jacobian = -log(image_width) - log(image_height) - log(
            clipped_confidence * (1.0 - clipped_confidence)
        )
        return log(max(self.cardinality_rate, 1.0e-300)) + log_mixture + log_jacobian - self._log_tail

    @classmethod
    def fit(
        cls,
        measurements: FloatArray,
        confidences: FloatArray,
        image_sizes: FloatArray,
        cardinality_rate: float,
        components: int = 8,
        random_state: int = 0,
        confidence_threshold: float = 0.0,
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
        if not 0.0 <= confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1)")
        if confidences.shape != (len(measurements),) or not np.all(
            (confidences >= confidence_threshold) & (confidences < 1.0)
        ):
            raise ValueError("fit requires scores within the retained confidence range")
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
        weights, means, variances = mixture.weights_, mixture.means_, mixture.covariances_
        if confidence_threshold > 0.0:
            weights, means, variances = _fit_truncated_mixture(
                features, weights, means, variances, confidence_threshold
            )
        return cls(
            weights=weights,
            means=means,
            variances=variances,
            cardinality_rate=cardinality_rate,
            confidence_threshold=confidence_threshold,
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                weights=self.weights,
                means=self.means,
                variances=self.variances,
                cardinality_rate=np.asarray(self.cardinality_rate),
                confidence_threshold=np.asarray(self.confidence_threshold),
            )

    @classmethod
    def load(cls, path: str | Path) -> "DiagonalGaussianMixtureDensity":
        with np.load(Path(path)) as archive:
            return cls(
                weights=archive["weights"],
                means=archive["means"],
                variances=archive["variances"],
                cardinality_rate=float(archive["cardinality_rate"]),
                confidence_threshold=float(archive["confidence_threshold"])
                if "confidence_threshold" in archive else 0.0,
            )


def _fit_truncated_mixture(
    features: FloatArray, weights: FloatArray, means: FloatArray,
    variances: FloatArray, threshold: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    components = len(weights)
    cutoff = logit(threshold)
    sample_count = len(features)

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        logits = parameters[:components]
        means, log_variances = parameters[components:].reshape(2, components, 5)
        variances = np.exp(log_variances)
        delta = features[:, None, :] - means
        squared_error = delta**2 / variances
        # The common softmax normalizer and the coordinate Jacobian cancel
        # or are constant in the conditional likelihood optimization.
        joint = logits - 0.5 * (squared_error + log_variances + log(2.0 * pi)).sum(axis=2)
        log_joint = logsumexp(joint, axis=1)
        responsibility = np.exp(joint - log_joint[:, None]) / sample_count

        standard_deviation = np.sqrt(variances[:, 4])
        standardized = (means[:, 4] - cutoff) / standard_deviation
        log_tail = log_ndtr(standardized)
        log_mass = logsumexp(logits + log_tail)
        tail_responsibility = np.exp(logits + log_tail - log_mass)
        tail_gradient = np.exp(logits - log_mass - 0.5 * (standardized**2 + log(2.0 * pi)))

        mass_gradient = tail_responsibility - responsibility.sum(axis=0)
        mean_gradient = -(responsibility[..., None] * delta / variances).sum(axis=0)
        variance_gradient = -0.5 * (responsibility[..., None] * (squared_error - 1.0)).sum(axis=0)
        mean_gradient[:, 4] += tail_gradient / standard_deviation
        variance_gradient[:, 4] -= 0.5 * tail_gradient * standardized
        gradient = np.concatenate((mass_gradient, mean_gradient.ravel(), variance_gradient.ravel()))
        return float(log_mass - log_joint.mean()), gradient

    initial = np.concatenate((np.log(weights), means.ravel(), np.log(variances).ravel()))
    bounds = [(None, None)] * (6 * components) + [(log(1.0e-6), None)] * (5 * components)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1.0e-10})
    if not np.isfinite(result.fun) or result.fun > objective(initial)[0] + 1.0e-8:
        raise RuntimeError(f"conditional mixture fit failed: {result.message}")
    logits = result.x[:components]
    fitted_means, log_variances = result.x[components:].reshape(2, components, 5)
    return np.exp(logits - logsumexp(logits)), fitted_means, np.exp(log_variances)


@dataclass(frozen=True)
class SpatialClutterDensity:
    """Box-only mixture with a uniform retained confidence mark."""

    mixture: DiagonalGaussianMixtureDensity
    confidence_threshold: float = 0.10
    cardinality_rate: float = 1.20

    def log_intensity(
        self, measurement: FloatArray, confidence: float, image_size: tuple[int, int]
    ) -> float:
        if not self.confidence_threshold <= confidence < 1.0:
            return -np.inf
        width, height = image_size
        feature = measurement / [width, height, 1.0, 1.0]
        feature[2:] -= np.log(image_size)
        variance = self.mixture.variances[:, :4]
        squared_error = (feature - self.mixture.means[:, :4]) ** 2 / variance
        component_log_density = np.log(self.mixture.weights) - 0.5 * (
            squared_error + np.log(variance) + log(2.0 * pi)
        ).sum(axis=1)
        return float(logsumexp(component_log_density)) + log(
            self.cardinality_rate / (width * height * (1.0 - self.confidence_threshold))
        )
