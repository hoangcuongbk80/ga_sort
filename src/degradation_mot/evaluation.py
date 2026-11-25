from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data.correspondence import Annotation
from .geometry import pairwise_iou
from .types import TrackOutput


@dataclass(frozen=True)
class EvaluationFrame:
    timestamp: float
    spatial_annotations: Sequence[Annotation]
    verified_detector_scores: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GlareEvent:
    event_id: str
    target_id: int
    onset: int
    offset: int
    identity_resolved: bool
    exit_frame: int | None = None
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventDecision:
    event: GlareEvent
    exclusion: str | None
    success: bool | None


def event_eligibility(frames: Sequence[EvaluationFrame], event: GlareEvent) -> str | None:
    """Return the first failed rule, using the fixed 0.10 detector reference."""
    pre_event, end = _event_window(frames, event)
    if sum(0.10 <= frames[i].verified_detector_scores.get(event.target_id, 0.0) < 1.0
           for i in pre_event) < 3:
        return "P"
    if not any(any(box.identity == event.target_id for box in frames[i].spatial_annotations)
               for i in range(event.offset + 1, end)):
        return "A"
    return None if event.identity_resolved else "U"


def _event_window(frames: Sequence[EvaluationFrame], event: GlareEvent) -> tuple[range, int]:
    end = len(frames) if event.exit_frame is None else event.exit_frame
    if not 0 <= event.onset <= event.offset < end <= len(frames):
        raise ValueError(f"invalid boundaries for event {event.event_id}")
    start = event.onset
    onset_time = frames[start].timestamp
    while start > 0 and onset_time - frames[start - 1].timestamp <= 1.0 + 1.0e-9:
        start -= 1
    return range(start, event.onset), end


def match_outputs(
    annotations: Sequence[Annotation], outputs: Sequence[TrackOutput],
) -> dict[int, int]:
    if not annotations or not outputs:
        return {}
    iou = pairwise_iou(np.asarray([box.xyxy for box in annotations]),
                       np.asarray([track.xyxy for track in outputs]))
    same_class = np.equal.outer([box.category for box in annotations],
                                [track.category for track in outputs])
    accepted = same_class & (iou >= 0.50)
    scores = np.where(accepted, iou + len(annotations), 0.0)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return {annotations[row].identity: outputs[col].track_id
            for row, col in zip(rows, cols) if accepted[row, col]}


def score_events(
    frames: Sequence[EvaluationFrame], events: Sequence[GlareEvent],
    outputs: Sequence[Sequence[TrackOutput]],
) -> list[EventDecision]:
    if len(outputs) != len(frames):
        raise ValueError("outputs must cover the complete sequence in frame order")
    if any(left.timestamp >= right.timestamp for left, right in zip(frames, frames[1:])):
        raise ValueError("frame timestamps must be strictly increasing")
    matches = [match_outputs(frame.spatial_annotations, tracks)
               for frame, tracks in zip(frames, outputs)]
    decisions = []
    for event in events:
        exclusion = event_eligibility(frames, event)
        success = None if exclusion else _survives(frames, matches, event)
        decisions.append(EventDecision(event, exclusion, success))
    return decisions


def _survives(
    frames: Sequence[EvaluationFrame], matches: Sequence[dict[int, int]], event: GlareEvent,
) -> bool:
    pre_event, end = _event_window(frames, event)
    target = event.target_id
    # In a count tie, the most recently matched ID represents the pre-event state.
    history = [matches[i][target] for i in reversed(pre_event) if target in matches[i]]
    if not history:
        return False
    track_id, count = Counter(history).most_common(1)[0]
    if count < 3:
        return False
    if any(other != target and predicted_id == track_id
           for i in range(event.onset, event.offset + 1)
           for other, predicted_id in matches[i].items()):
        return False
    first_recovered = next((matches[i][target] for i in range(event.offset + 1, end)
                            if target in matches[i]), None)
    return first_recovered == track_id


def bdd_event_strata(
    occluded: Sequence[bool], verified_detector_scores: Sequence[float],
) -> tuple[str, str]:
    """Classify the original inclusive 11-instant BDD100K event interval."""
    if len(occluded) != 11 or len(verified_detector_scores) != 11:
        raise ValueError("BDD100K strata require the original 11 annotated instants")
    count = sum(occluded)
    occlusion = "low" if count <= 2 else "moderate" if count <= 6 else "high"
    current = longest = 0
    for confidence in verified_detector_scores:
        current = 0 if 0.10 <= confidence < 1.0 else current + 1
        longest = max(longest, current)
    gap = "none" if longest == 0 else "short" if longest <= 3 else "long"
    return occlusion, gap


def bdd_photometric_strata(
    coverage: Sequence[float], area_weighted_strength: Sequence[float],
) -> tuple[str, str]:
    """Use reference-region statistics over the original 11 annotated instants."""
    area = np.asarray(coverage, dtype=float)
    strength = np.asarray(area_weighted_strength, dtype=float)
    if area.shape != (11,) or strength.shape != (11,) or not np.all(
        (0.0 <= strength) & (strength <= area) & (area <= 1.0)
    ):
        raise ValueError("expected 11 paired area and strength values in [0, 1]")
    median_area = np.median(area)
    median_brightness = np.median(np.divide(strength, area, out=np.zeros_like(area), where=area > 0))
    size = "low" if median_area < 0.08 else "moderate" if median_area < 0.20 else "high"
    brightness = "low" if median_brightness < 0.60 else "moderate" if median_brightness < 0.85 else "high"
    return size, brightness


def paired_event_interval(
    proposed: np.ndarray, comparator: np.ndarray, replicates: int = 10_000, seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap [cluster, seed, (success, eligible)] counts with shared draws."""
    if (proposed.ndim != 3 or comparator.ndim != 3 or proposed.shape[2] != 2
            or comparator.shape[2] != 2 or len(proposed) != len(comparator) or len(proposed) == 0):
        raise ValueError("expected aligned cluster counts with one column pair per seed")
    eligible = proposed[:, :1, 1]
    if not (np.all(eligible == proposed[:, :, 1]) and np.all(eligible == comparator[:, :, 1])):
        raise ValueError("all methods and seeds must share the same eligible events")
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(proposed), size=(replicates, len(proposed)))
    proposed_counts = proposed[samples].sum(axis=1)
    comparator_counts = comparator[samples].sum(axis=1)
    valid = proposed_counts[:, 0, 1] > 0
    if not valid.any():
        return float("nan"), float("nan")
    def mean_rate(counts: np.ndarray) -> np.ndarray:
        counts = counts[valid]
        return (counts[..., 0] / counts[..., 1]).mean(axis=1)

    lower, upper = np.percentile(100.0 * (mean_rate(proposed_counts) - mean_rate(comparator_counts)), [2.5, 97.5])
    return float(lower), float(upper)
