from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class DetectionProbabilityHead(nn.Module):
    """Equation (8), including monotonic area and strength contributions."""

    def __init__(
        self, baseline_dim: int = 11, geometry_dim: int = 13, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.baseline = MLP(baseline_dim, 1, hidden_dim)
        self.geometry = MLP(geometry_dim, 1, hidden_dim)
        self.raw_area_weight = nn.Parameter(torch.zeros(()))
        self.raw_strength_weight = nn.Parameter(torch.zeros(()))

    def forward(self, baseline: Tensor, degradation: Tensor) -> Tensor:
        area = degradation[..., 0]
        strength = degradation[..., 1]
        geometry = degradation[..., 2:]
        logit = (
            self.baseline(baseline).squeeze(-1)
            + self.geometry(geometry).squeeze(-1)
            - F.softplus(self.raw_area_weight) * area
            - F.softplus(self.raw_strength_weight) * strength
        )
        return torch.sigmoid(logit)


class CalibrationNetworks(nn.Module):
    def __init__(
        self,
        degradation_dim: int = 15,
        baseline_dim: int = 11,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.detection_probability = DetectionProbabilityHead(
            baseline_dim=baseline_dim,
            geometry_dim=degradation_dim - 2,
            hidden_dim=hidden_dim,
        )
        self.bias = MLP(degradation_dim, 4, hidden_dim)
        self.covariance = MLP(degradation_dim, 5, hidden_dim)
        self.confidence = MLP(degradation_dim + baseline_dim, 2, hidden_dim)

    def forward(self, degradation: Tensor, baseline: Tensor) -> dict[str, Tensor]:
        confidence_raw = self.confidence(torch.cat((baseline, degradation), dim=-1))
        return {
            "detection_probability": self.detection_probability(baseline, degradation),
            "normalized_bias": self.bias(degradation),
            "covariance_raw": self.covariance(degradation),
            "confidence_alpha": 1.0 + F.softplus(confidence_raw[..., 0]),
            "confidence_beta": 1.0 + F.softplus(confidence_raw[..., 1]),
        }

