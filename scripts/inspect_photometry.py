from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from degradation_mot.config import PhotometryConfig
from degradation_mot.photometry import (
    build_saturation_candidate_mask,
    extract_degradation_descriptor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--box", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(args.image)
    config = PhotometryConfig()
    mask, value = build_saturation_candidate_mask(frame, config)
    box = np.asarray(args.box, dtype=np.float64)
    descriptor = extract_degradation_descriptor(
        mask,
        value,
        box,
        config.value_threshold,
        config.boundary_band_fraction,
        config.radial_bins,
    )
    overlay = frame.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise OSError(f"could not write {args.output}")
    print(np.array2string(descriptor, precision=6, separator=", "))


if __name__ == "__main__":
    main()
