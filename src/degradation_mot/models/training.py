from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationLoss:
    total: Tensor
    detection: Tensor
    localization: Tensor
    confidence: Tensor


def normalized_covariance(
    covariance_raw: Tensor,
    sigma_min: tuple[float, float, float, float],
) -> Tensor:
    lower_bound = covariance_raw.new_tensor(sigma_min)
    standard_deviation = lower_bound + F.softplus(covariance_raw[..., :4])
    correlation = torch.tanh(covariance_raw[..., 4])
    covariance = torch.diag_embed(standard_deviation.square())
    cross = correlation * standard_deviation[..., 2] * standard_deviation[..., 3]
    covariance[..., 2, 3] = cross
    covariance[..., 3, 2] = cross
    return covariance


def calibration_loss(
    prediction: dict[str, Tensor],
    detection_labels: Tensor,
    normalized_residuals: Tensor,
    confidences: Tensor,
    localization_eligible: Tensor,
    sigma_min: tuple[float, float, float, float] = (0.01, 0.01, 0.02, 0.02),
) -> CalibrationLoss:
    probability = prediction["detection_probability"]
    detection = F.binary_cross_entropy(probability, detection_labels.float())

    eligible = localization_eligible.bool()
    if eligible.any():
        residual = normalized_residuals[eligible] - prediction["normalized_bias"][eligible]
        covariance = normalized_covariance(prediction["covariance_raw"][eligible], sigma_min)
        cholesky = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), cholesky).squeeze(-1)
        mahalanobis = torch.sum(residual * solved, dim=-1)
        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        localization = 0.5 * (mahalanobis + log_determinant).mean()

        alpha = prediction["confidence_alpha"][eligible]
        beta = prediction["confidence_beta"][eligible]
        confidence = confidences[eligible].clamp(1.0e-4, 1.0 - 1.0e-4)
        confidence_nll = -(
            (alpha - 1.0) * torch.log(confidence)
            + (beta - 1.0) * torch.log1p(-confidence)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + torch.lgamma(alpha + beta)
        ).mean()
    else:
        zero = probability.sum() * 0.0
        localization = zero
        confidence_nll = zero
    total = detection + localization + confidence_nll
    return CalibrationLoss(total, detection, localization, confidence_nll)
