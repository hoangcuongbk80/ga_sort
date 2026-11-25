from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        self.network = nn.Sequential(*layers, nn.Linear(input_dim, output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class DetectionProbabilityHead(nn.Module):
    """Equation (8), including monotonic area and strength contributions."""

    def __init__(
        self, baseline_dim: int = 11, geometry_dim: int = 13, hidden_dim: int = 32,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.baseline = MLP(baseline_dim, 1, hidden_dim, hidden_layers)
        self.geometry = MLP(geometry_dim, 1, hidden_dim, hidden_layers)
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
        hidden_layers: int = 2,
        visibility_only: bool = False,
    ) -> None:
        super().__init__()
        self.detection_probability = DetectionProbabilityHead(
            baseline_dim=baseline_dim,
            geometry_dim=degradation_dim - 2,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.visibility_only = visibility_only
        self.bias = MLP(degradation_dim, 4, hidden_dim, hidden_layers)
        self.covariance = MLP(degradation_dim, 5, hidden_dim, hidden_layers)
        self.confidence = MLP(degradation_dim + baseline_dim, 2, hidden_dim, hidden_layers)

    def forward(self, degradation: Tensor, baseline: Tensor) -> dict[str, Tensor]:
        confidence_raw = self.confidence(torch.cat((baseline, degradation), dim=-1))
        probability = (
            self.detection_probability.baseline(baseline).squeeze(-1).sigmoid()
            if self.visibility_only else self.detection_probability(baseline, degradation)
        )
        return {
            "detection_probability": probability,
            "normalized_bias": self.bias(degradation),
            "covariance_raw": self.covariance(degradation),
            "confidence_alpha": 1.0 + F.softplus(confidence_raw[..., 0]),
            "confidence_beta": 1.0 + F.softplus(confidence_raw[..., 1]),
        }

