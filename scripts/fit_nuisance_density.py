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
    args = parser.parse_args()

    with np.load(args.samples) as archive:
        required = {"measurement", "confidence", "image_size", "frame_count"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        frame_count = int(archive["frame_count"])
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        density = DiagonalGaussianMixtureDensity.fit(
            archive["measurement"],
            archive["confidence"],
            archive["image_size"],
            cardinality_rate=len(archive["measurement"]) / frame_count,
            components=args.components,
            random_state=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    density.save(args.output)


if __name__ == "__main__":
    main()
