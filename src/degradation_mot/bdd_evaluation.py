from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .evaluation import EvaluationFrame
from .geometry import pairwise_iou
from .types import TrackOutput

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")
COCO_VEHICLE_IDS = {"car": 2, "truck": 7, "bus": 5, "motorcycle": 3}


@dataclass(frozen=True)
class TrackingCounts:
    """Sufficient statistics from one complete video and one vehicle category."""

    ground_truth: int
    predictions: int
    hota_tp: Sequence[int]
    hota_association_sum: Sequence[float]
    idtp: int
    false_negatives: int
    false_positives: int
    identity_switches: int


@dataclass(frozen=True)
class VideoResult:
    video_id: str
    illumination: str
    saturation_selected: bool
    matched_control: bool
    classes: Mapping[str, TrackingCounts]


def score_video_classes(
    frames: Sequence[EvaluationFrame], outputs: Sequence[Sequence[TrackOutput]],
) -> dict[str, TrackingCounts]:
    """Run TrackEval on a complete video after the annotation ignore rules."""
    try:
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as exc:
        raise ImportError("TrackEval is required to score video boxes; install ga_sort with the eval extra") from exc
    if not frames or len(frames) != len(outputs):
        raise ValueError("expected the complete, aligned video annotations and tracker outputs")
    config = {"THRESHOLD": 0.50, "PRINT_CONFIG": False}
    hota_metric, identity_metric, clear_metric = HOTA(), Identity(config), CLEAR(config)

    def contiguous_ids(ids: list[list[int]]) -> tuple[list[np.ndarray], int]:
        unique = {identity for frame in ids for identity in frame}
        mapping = {identity: index for index, identity in enumerate(sorted(unique))}
        if any(len(frame) != len(set(frame)) for frame in ids):
            raise ValueError("each identity may occur at most once per frame and category")
        return [np.asarray([mapping[identity] for identity in frame], dtype=int) for frame in ids], len(unique)

    counts = {}
    for name, category in COCO_VEHICLE_IDS.items():
        ground_truth = [[row for row in frame.spatial_annotations if row.category == category]
                        for frame in frames]
        predictions = [[row for row in frame if row.category == category] for frame in outputs]
        gt_ids, num_gt_ids = contiguous_ids([[row.identity for row in frame] for frame in ground_truth])
        pred_ids, num_pred_ids = contiguous_ids([[row.track_id for row in frame] for frame in predictions])
        similarity = [pairwise_iou(np.asarray([row.xyxy for row in gt]),
                                   np.asarray([row.xyxy for row in pred]))
                      for gt, pred in zip(ground_truth, predictions)]
        sequence = {
            "gt_ids": gt_ids, "tracker_ids": pred_ids,
            "num_gt_ids": num_gt_ids, "num_tracker_ids": num_pred_ids,
            "num_gt_dets": sum(map(len, gt_ids)), "num_tracker_dets": sum(map(len, pred_ids)),
            "num_timesteps": len(frames), "similarity_scores": similarity,
        }
        hota = hota_metric.eval_sequence(sequence)
        identity = identity_metric.eval_sequence(sequence)
        clear = clear_metric.eval_sequence(sequence)
        counts[name] = TrackingCounts(
            sequence["num_gt_dets"], sequence["num_tracker_dets"], hota["HOTA_TP"].tolist(),
            (hota["HOTA_TP"] * hota["AssA"]).tolist(), int(identity["IDTP"]),
            int(clear["CLR_FN"]), int(clear["CLR_FP"]), int(clear["IDSW"]),
        )
    return counts


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


def pooled_tracking_metrics(counts: Iterable[TrackingCounts]) -> dict[str, float | int]:
    rows = list(counts)
    tp = np.asarray([row.hota_tp for row in rows])
    ass_sum = np.asarray([row.hota_association_sum for row in rows], dtype=float)
    if not rows or tp.shape != (len(rows), 19) or ass_sum.shape != tp.shape:
        raise ValueError("expected one count and association sum at each of the 19 IoU thresholds")
    g = sum(row.ground_truth for row in rows)
    p = sum(row.predictions for row in rows)
    tp = tp.sum(axis=0)
    ass_sum = ass_sum.sum(axis=0)
    det = _divide(tp, g + p - tp)
    ass = _divide(ass_sum, tp)
    idsw = sum(row.identity_switches for row in rows)
    errors = idsw + sum(row.false_negatives + row.false_positives for row in rows)
    return {
        "HOTA": 100.0 * float(np.sqrt(det * ass).mean()),
        "DetA": 100.0 * float(det.mean()),
        "AssA": 100.0 * float(ass.mean()),
        "IDF1": 200.0 * sum(row.idtp for row in rows) / (g + p) if g + p else 0.0,
        "MOTA": 100.0 * (1.0 - errors / g) if g else 0.0,
        "IDSW": idsw,
    }


def bdd_validation_summary(videos: Sequence[VideoResult]) -> dict:
    """Pool over all 200 complete videos, with the 38/162/40 secondary views."""
    if len(videos) != 200 or len({video.video_id for video in videos}) != 200:
        raise ValueError("the primary BDD100K validation population requires 200 unique videos")
    selected = [video for video in videos if video.saturation_selected]
    complement = [video for video in videos if not video.saturation_selected]
    controls = [video for video in complement if video.matched_control]
    if len(selected) != 38 or len(controls) != 40 or any(video.matched_control for video in selected):
        raise ValueError("expected 38 saturation-selected videos and 40 controls in their complement")
    if any(set(video.classes) != set(VEHICLE_CLASSES) for video in videos):
        raise ValueError("provide all four vehicle classes per video, including zero-count classes")
    illumination = {label: [video for video in videos if video.illumination == label]
                    for label in ("daytime", "night", "dawn/dusk")}
    if [len(group) for group in illumination.values()] != [120, 60, 20]:
        raise ValueError("expected 120 daytime, 60 night and 20 dawn/dusk videos")

    def summarize(group: Sequence[VideoResult]) -> dict:
        if not group:
            return {"videos": 0}
        by_category = {category: pooled_tracking_metrics(video.classes[category] for video in group)
                       for category in VEHICLE_CLASSES}
        return {
            "videos": len(group),
            **pooled_tracking_metrics(row for video in group for row in video.classes.values()),
            "mHOTA4": sum(row["HOTA"] for row in by_category.values()) / len(VEHICLE_CLASSES),
            "by_category": by_category,
        }

    groups = {"all": videos, "selected": selected, "complement": complement, "controls": controls}
    return {
        **{name: summarize(group) for name, group in groups.items()},
        "by_illumination": {label: summarize(group) for label, group in illumination.items()},
        "illumination_by_selection": {
            label: {name: summarize([video for video in source if video.illumination == label])
                    for name, source in (("selected", selected), ("complement", complement))}
            for label in illumination
        },
    }


def paired_hota_interval(
    proposed: Sequence[Sequence[VideoResult]], comparator: Sequence[Sequence[VideoResult]],
    category: str | None = None, replicates: int = 10_000, seed: int = 0,
) -> tuple[float, float]:
    """Pool complete-video statistics for each paired bootstrap and seed."""
    if not proposed or not comparator or not proposed[0]:
        raise ValueError("provide at least one run per method on a common video population")
    ids = [video.video_id for video in proposed[0]]
    if len(set(ids)) != len(ids) or any([video.video_id for video in run] != ids
                                       for run in (*proposed, *comparator)):
        raise ValueError("all runs must have the same ordered unique video IDs")
    categories = VEHICLE_CLASSES if category is None else (category,)

    def video_counts(video: VideoResult) -> np.ndarray:
        rows = [video.classes[c] for c in categories]
        return np.vstack((
            np.sum([row.hota_tp for row in rows], axis=0),
            np.sum([row.hota_association_sum for row in rows], axis=0),
            np.full(19, sum(row.ground_truth + row.predictions for row in rows)),
        ))

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=replicates)

    def mean_hota(runs: Sequence[Sequence[VideoResult]]) -> np.ndarray:
        scores = []
        for run in runs:
            sufficient = np.asarray([video_counts(video) for video in run])
            tp, association, totals = (draws @ sufficient.reshape(len(ids), -1)).reshape(-1, 3, 19).transpose(1, 0, 2)
            scores.append(100.0 * np.sqrt(_divide(tp, totals - tp) * _divide(association, tp)).mean(axis=1))
        return np.mean(scores, axis=0)

    lower, upper = np.percentile(mean_hota(proposed) - mean_hota(comparator), [2.5, 97.5])
    return float(lower), float(upper)
