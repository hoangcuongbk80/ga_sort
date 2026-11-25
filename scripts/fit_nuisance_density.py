from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from degradation_mot.models.density import DiagonalGaussianMixtureDensity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--spatial-only", action="store_true", help="Fit pooled box clutter for the internal reference")
    args = parser.parse_args()

    with np.load(args.samples) as archive:
        required = {"measurement", "confidence", "image_size", "frame_count"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        frame_count = int(archive["frame_count"])
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        archive_threshold = float(archive["confidence_threshold"]) if "confidence_threshold" in archive else 0.10
        if not np.isclose(archive_threshold, args.confidence_threshold, atol=1.0e-8, rtol=0.0):
            raise ValueError("nuisance labels must be prepared at the requested confidence threshold")
        confidence = archive["confidence"]
        if not np.all((confidence >= args.confidence_threshold) & (confidence < 1.0)):
            raise ValueError("nuisance labels contain detections outside retained score support")
        density = DiagonalGaussianMixtureDensity.fit(
            archive["measurement"],
            np.full_like(confidence, 0.5) if args.spatial_only else confidence,
            archive["image_size"],
            cardinality_rate=len(archive["measurement"]) / frame_count,
            components=args.components,
            random_state=args.seed,
            confidence_threshold=0.0 if args.spatial_only else args.confidence_threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    density.save(args.output)


if __name__ == "__main__":
    main()
