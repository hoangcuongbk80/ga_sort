from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from degradation_mot.bdd_evaluation import TrackingCounts, VideoResult, bdd_validation_summary, score_video_classes
from degradation_mot.data.correspondence import Annotation
from degradation_mot.evaluation import EvaluationFrame, GlareEvent, score_events
from degradation_mot.types import TrackOutput


def read_sequence(sequence: dict, input_dir: Path) -> tuple[list[EvaluationFrame], list[list[TrackOutput]]]:
    frames, outputs = [], []
    for frame in sequence["frames"]:
        annotations = [Annotation(np.asarray(row["xyxy"], dtype=float), row["category"], row["identity"])
                       for row in frame.get("spatial_annotations", [])]
        scores = {int(key): value for key, value in frame.get("verified_detector_scores", {}).items()}
        frames.append(EvaluationFrame(frame["timestamp"], annotations, scores))
        outputs.append([
            TrackOutput(row["track_id"], np.asarray(row["xyxy"], dtype=float), row["category"], 1.0)
            for row in frame.get("outputs", [])
        ])
    if "tracks" in sequence:
        if any(outputs):
            raise ValueError("specify either a tracks file or embedded outputs")
        with (input_dir / sequence["tracks"]).open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                index, track_id = map(int, row[:2])
                x, y, width, height, score = map(float, row[2:7])
                if not 1 <= index <= len(frames):
                    raise ValueError(f"tracker frame {index} falls outside the complete sequence")
                outputs[index - 1].append(TrackOutput(
                    track_id, np.array([x, y, x + width, y + height]), int(row[7]), score,
                ))
    return frames, outputs


def event_summary(sequences: list[dict], input_dir: Path) -> dict:
    outcomes = []
    for sequence in sequences:
        frames, outputs = read_sequence(sequence, input_dir)
        events = [GlareEvent(**row) for row in sequence["events"]]
        outcomes.extend({"sequence_id": sequence["sequence_id"], **asdict(decision)}
                        for decision in score_events(frames, events, outputs))
    eligible = sum(result["success"] is not None for result in outcomes)
    successes = sum(result["success"] is True for result in outcomes)
    groups = {}
    for label in sorted({label for result in outcomes for label in result["event"]["attributes"]}):
        members = [result for result in outcomes if label in result["event"]["attributes"]]
        kept = sum(result["success"] is not None for result in members)
        passed = sum(result["success"] is True for result in members)
        groups[label] = {"all": len(members), "eligible": kept, "successes": passed,
                         "EISR": 100.0 * passed / kept if kept else None}
    return {"successes": successes, "eligible": eligible,
            "EISR": 100.0 * successes / eligible if eligible else None,
            "exclusions": {reason: sum(row["exclusion"] == reason for row in outcomes) for reason in ("P", "A", "U")},
            "by_attribute": groups, "events": outcomes}


def video_result(video: dict, input_dir: Path) -> VideoResult:
    counts = ({name: TrackingCounts(**row) for name, row in video["classes"].items()}
              if "classes" in video else score_video_classes(*read_sequence(video, input_dir)))
    return VideoResult(
        video["video_id"], video["illumination"], video["saturation_selected"], video["matched_control"], counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score revised EISR or pool complete BDD100K videos")
    parser.add_argument("mode", choices=["eisr", "bdd-tracking"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if args.mode == "eisr":
        result = event_summary(records, args.input.parent)
    else:
        videos = [video_result(video, args.input.parent) for video in records]
        result = bdd_validation_summary(videos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
