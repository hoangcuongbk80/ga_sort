from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhotometryConfig:
    value_threshold: float = 0.90
    saturation_threshold: float = 0.25
    minimum_component_area: int = 16
    boundary_band_fraction: float = 0.10
    radial_bins: int = 4


@dataclass(frozen=True)
class MotionConfig:
    position_acceleration_fraction: float = 0.02
    log_scale_acceleration: float = 0.10
    birth_position_std_fraction: float = 0.02
    birth_log_scale_std: float = 0.05
    birth_velocity_std_fraction: float = 0.05
    birth_log_scale_velocity_std: float = 0.20
    maximum_timestamp_gap: float = 2.0


@dataclass(frozen=True)
class TrackingConfig:
    gate_probability: float = 0.995
    survival_probability: float = 0.995
    reference_frame_interval: float = 0.04
    report_threshold: float = 0.85
    delete_threshold: float = 0.05
    probability_clip: tuple[float, float] = (0.01, 0.99)
    confidence_clip: tuple[float, float] = (1.0e-4, 1.0 - 1.0e-4)
    density_floor: float = 1.0e-12
    logarithm_floor: float = 1.0e-12
    recent_detection_window: int = 10


@dataclass(frozen=True)
class TrackerConfig:
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackerConfig":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load configuration files") from exc

        with Path(path).open("r", encoding="utf-8") as stream:
            raw: dict[str, Any] = yaml.safe_load(stream) or {}
        return cls(
            photometry=PhotometryConfig(**raw.get("photometry", {})),
            motion=MotionConfig(**raw.get("motion", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
        )

