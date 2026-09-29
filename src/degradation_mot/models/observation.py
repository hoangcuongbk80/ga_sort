from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_numpy(value: object) -> FloatArray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class ObservationParameters:
    detection_probability: float
    bias: FloatArray
    covariance: FloatArray
    confidence_alpha: float
    confidence_beta: float


class ObservationModel(Protocol):
    def predict(
        self,
        degradation: FloatArray,
        baseline: FloatArray,
        predicted_width: float,
        predicted_height: float,
    ) -> ObservationParameters: ...


def restricted_block_covariance(
    raw: FloatArray,
    predicted_width: float,
    predicted_height: float,
    sigma_min: FloatArray,
) -> FloatArray:
    """Construct Equation (10) from four scale values and one correlation value."""
    raw = np.asarray(raw, dtype=np.float64)
    sigma_min = np.asarray(sigma_min, dtype=np.float64)
    if raw.shape != (5,) or sigma_min.shape != (4,):
        raise ValueError("raw and sigma_min must have shapes (5,) and (4,)")
    sigma = sigma_min + np.logaddexp(0.0, raw[:4])
    correlation = float(np.tanh(raw[4]))
    correlation_matrix = np.eye(4, dtype=np.float64)
    correlation_matrix[2, 3] = correlation
    correlation_matrix[3, 2] = correlation
    normalized = np.diag(sigma) @ correlation_matrix @ np.diag(sigma)
    scale = np.diag([predicted_width, predicted_height, 1.0, 1.0])
    return scale @ normalized @ scale.T


class ConstantObservationModel:
    """Non-learned fallback for integration and synthetic tests."""

    def __init__(
        self,
        detection_probability: float = 0.90,
        normalized_standard_deviations: tuple[float, float, float, float] = (
            0.05,
            0.05,
            0.10,
            0.10,
        ),
        confidence_alpha: float = 4.0,
        confidence_beta: float = 2.0,
    ) -> None:
        self.probability = detection_probability
        self.standard_deviations = np.asarray(
            normalized_standard_deviations, dtype=np.float64
        )
        self.alpha = confidence_alpha
        self.beta = confidence_beta

    def predict(
        self,
        degradation: FloatArray,
        baseline: FloatArray,
        predicted_width: float,
        predicted_height: float,
    ) -> ObservationParameters:
        del degradation, baseline
        scale = np.asarray(
            [predicted_width, predicted_height, 1.0, 1.0], dtype=np.float64
        )
        covariance = np.diag((self.standard_deviations * scale) ** 2)
        return ObservationParameters(
            detection_probability=self.probability,
            bias=np.zeros(4, dtype=np.float64),
            covariance=covariance,
            confidence_alpha=self.alpha,
            confidence_beta=self.beta,
        )


class TorchObservationModel:
    """Inference wrapper for trained calibration networks and standardizers."""

    def __init__(
        self,
        networks: object,
        degradation_mean: FloatArray,
        degradation_std: FloatArray,
        baseline_mean: FloatArray,
        baseline_std: FloatArray,
        sigma_min: tuple[float, float, float, float] = (0.01, 0.01, 0.02, 0.02),
        device: str = "cpu",
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for TorchObservationModel") from exc
        self.torch = torch
        self.networks = networks.to(device).eval()
        self.device = device
        self.degradation_mean = _as_numpy(degradation_mean)
        self.degradation_std = _as_numpy(degradation_std)
        self.baseline_mean = _as_numpy(baseline_mean)
        self.baseline_std = _as_numpy(baseline_std)
        self.sigma_min = np.asarray(sigma_min, dtype=np.float64)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "TorchObservationModel":
        import torch

        from .networks import CalibrationNetworks

        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        networks = CalibrationNetworks(**checkpoint.get("architecture", {}))
        networks.load_state_dict(checkpoint["model"])
        return cls(
            networks=networks,
            degradation_mean=checkpoint["degradation_mean"],
            degradation_std=checkpoint["degradation_std"],
            baseline_mean=checkpoint["baseline_mean"],
            baseline_std=checkpoint["baseline_std"],
            sigma_min=tuple(checkpoint.get("sigma_min", (0.01, 0.01, 0.02, 0.02))),
            device=device,
        )

    def predict(
        self,
        degradation: FloatArray,
        baseline: FloatArray,
        predicted_width: float,
        predicted_height: float,
    ) -> ObservationParameters:
        degradation_scaled = (degradation - self.degradation_mean) / np.maximum(
            self.degradation_std, 1.0e-8
        )
        baseline_scaled = (baseline - self.baseline_mean) / np.maximum(
            self.baseline_std, 1.0e-8
        )
        with self.torch.no_grad():
            degradation_tensor = self.torch.as_tensor(
                degradation_scaled[None], dtype=self.torch.float32, device=self.device
            )
            baseline_tensor = self.torch.as_tensor(
                baseline_scaled[None], dtype=self.torch.float32, device=self.device
            )
            prediction = self.networks(degradation_tensor, baseline_tensor)
        probability = float(prediction["detection_probability"][0].cpu())
        normalized_bias = prediction["normalized_bias"][0].cpu().numpy().astype(np.float64)
        covariance_raw = prediction["covariance_raw"][0].cpu().numpy().astype(np.float64)
        scale = np.asarray(
            [predicted_width, predicted_height, 1.0, 1.0], dtype=np.float64
        )
        return ObservationParameters(
            detection_probability=probability,
            bias=normalized_bias * scale,
            covariance=restricted_block_covariance(
                covariance_raw,
                predicted_width,
                predicted_height,
                self.sigma_min,
            ),
            confidence_alpha=float(prediction["confidence_alpha"][0].cpu()),
            confidence_beta=float(prediction["confidence_beta"][0].cpu()),
        )

