from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from degradation_mot.config import TrackerConfig
from degradation_mot.models.density import DiagonalGaussianMixtureDensity
from degradation_mot.models.observation import (
    ConstantObservationModel,
    TorchObservationModel,
)
from degradation_mot.tracking.tracker import DegradationConditionedTracker
from degradation_mot.types import Detection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the tracker on frames with cached detector outputs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clutter-density", type=Path)
    parser.add_argument("--birth-density", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = TrackerConfig.from_yaml(args.config)
    observation_model = (
        TorchObservationModel.from_checkpoint(args.checkpoint, args.device)
        if args.checkpoint
        else ConstantObservationModel()
    )
    clutter_density = (
        DiagonalGaussianMixtureDensity.load(args.clutter_density)
        if args.clutter_density
        else None
    )
    birth_density = (
        DiagonalGaussianMixtureDensity.load(args.birth_density)
        if args.birth_density
        else None
    )
    tracker = DegradationConditionedTracker(
        config, observation_model, clutter_density, birth_density
    )
    rows: list[str] = []
    with args.manifest.open("r", encoding="utf-8") as stream:
        for frame_index, line in enumerate(stream, start=1):
            record = json.loads(line)
            image_path = Path(record["image"])
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(image_path)
            detections = [
                Detection.from_xyxy(
                    tuple(item["xyxy"]), item["confidence"], item["category"]
                )
                for item in record.get("detections", [])
            ]
            outputs = tracker.update(frame, detections, float(record["timestamp"]))
            for output in outputs:
                x1, y1, x2, y2 = output.xyxy
                rows.append(
                    f"{frame_index},{output.track_id},{x1:.6f},{y1:.6f},"
                    f"{x2 - x1:.6f},{y2 - y1:.6f},"
                    f"{output.existence_probability:.6f},{output.category},-1,-1"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
