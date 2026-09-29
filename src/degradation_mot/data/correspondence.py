from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..geometry import pairwise_iou
from ..types import Detection


@dataclass(frozen=True)
class Annotation:
    xyxy: NDArray[np.float64]
    category: int
    identity: int


@dataclass(frozen=True)
class CorrespondenceResult:
    matches: tuple[tuple[int, int], ...]
    missed_annotations: tuple[int, ...]
    clutter_detections: tuple[int, ...]
    ambiguous_matches: tuple[tuple[int, int], ...]


def match_detections_to_annotations(
    annotations: list[Annotation],
    detections: list[Detection],
    alternative_cost_margin: float = 0.15,
) -> CorrespondenceResult:
    """Apply the permissive calibration gate and matching from the supplement.

    Ambiguous matches are returned separately and must be reviewed. They should
    not be used as target, miss, or clutter labels until adjudicated.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("SciPy is required for calibration matching") from exc

    annotation_count = len(annotations)
    detection_count = len(detections)
    if annotation_count == 0:
        return CorrespondenceResult((), (), tuple(range(detection_count)), ())
    if detection_count == 0:
        return CorrespondenceResult((), tuple(range(annotation_count)), (), ())

    annotation_boxes = np.asarray([item.xyxy for item in annotations], dtype=np.float64)
    detection_boxes = np.asarray([item.xyxy for item in detections], dtype=np.float64)
    overlaps = pairwise_iou(annotation_boxes, detection_boxes)
    annotation_width = annotation_boxes[:, 2] - annotation_boxes[:, 0]
    annotation_height = annotation_boxes[:, 3] - annotation_boxes[:, 1]
    detection_width = detection_boxes[:, 2] - detection_boxes[:, 0]
    detection_height = detection_boxes[:, 3] - detection_boxes[:, 1]
    annotation_area = annotation_width * annotation_height
    detection_area = detection_width * detection_height
    annotation_center = 0.5 * (annotation_boxes[:, :2] + annotation_boxes[:, 2:])
    detection_center = 0.5 * (detection_boxes[:, :2] + detection_boxes[:, 2:])
    annotation_diagonal = np.sqrt(annotation_width**2 + annotation_height**2)
    center_distance = np.linalg.norm(
        annotation_center[:, None, :] - detection_center[None, :, :], axis=2
    ) / np.maximum(annotation_diagonal[:, None], 1.0e-8)
    scale_distance = np.abs(
        np.log(detection_width[None, :] / annotation_width[:, None])
    ) + np.abs(np.log(detection_height[None, :] / annotation_height[:, None]))
    area_ratio = detection_area[None, :] / annotation_area[:, None]
    class_compatible = np.asarray(
        [
            [annotation.category == detection.category for detection in detections]
            for annotation in annotations
        ],
        dtype=np.bool_,
    )
    admissible = (
        class_compatible
        & (area_ratio >= 0.1)
        & (area_ratio <= 10.0)
        & ((overlaps >= 0.05) | (center_distance <= 0.75))
    )
    costs = (
        1.0
        - overlaps
        + 0.25 * np.minimum(center_distance, 2.0)
        + 0.10 * np.minimum(scale_distance, 2.0)
    )
    assignment_cost = np.where(admissible, costs, 1.0e9)
    row_indices, column_indices = linear_sum_assignment(assignment_cost)

    provisional_matches: list[tuple[int, int]] = []
    ambiguous: list[tuple[int, int]] = []
    multi_overlap = (overlaps >= 0.10).sum(axis=0) >= 2
    for annotation_index, detection_index in zip(row_indices, column_indices):
        if not admissible[annotation_index, detection_index]:
            continue
        selected_cost = costs[annotation_index, detection_index]
        annotation_alternatives = costs[annotation_index, admissible[annotation_index]]
        detection_alternatives = costs[admissible[:, detection_index], detection_index]
        near_tie_annotation = np.sum(
            np.abs(annotation_alternatives - selected_cost) < alternative_cost_margin
        ) > 1
        near_tie_detection = np.sum(
            np.abs(detection_alternatives - selected_cost) < alternative_cost_margin
        ) > 1
        pair = (int(annotation_index), int(detection_index))
        if near_tie_annotation or near_tie_detection or multi_overlap[detection_index]:
            ambiguous.append(pair)
            continue
        provisional_matches.append(pair)

    ambiguous_annotations = {pair[0] for pair in ambiguous}
    ambiguous_detections = {pair[1] for pair in ambiguous}
    changed = True
    while changed:
        changed = False
        connected_detections = set(
            np.flatnonzero(admissible[list(ambiguous_annotations)].any(axis=0)).tolist()
        ) if ambiguous_annotations else set()
        connected_annotations = set(
            np.flatnonzero(admissible[:, list(ambiguous_detections)].any(axis=1)).tolist()
        ) if ambiguous_detections else set()
        if not connected_detections.issubset(ambiguous_detections):
            ambiguous_detections.update(connected_detections)
            changed = True
        if not connected_annotations.issubset(ambiguous_annotations):
            ambiguous_annotations.update(connected_annotations)
            changed = True

    matches = [
        pair
        for pair in provisional_matches
        if pair[0] not in ambiguous_annotations and pair[1] not in ambiguous_detections
    ]
    used_annotations = {pair[0] for pair in matches}
    used_detections = {pair[1] for pair in matches}
    missed = tuple(
        index
        for index in range(annotation_count)
        if index not in used_annotations and index not in ambiguous_annotations
    )
    clutter = tuple(
        index
        for index in range(detection_count)
        if index not in used_detections and index not in ambiguous_detections
    )
    return CorrespondenceResult(tuple(matches), missed, clutter, tuple(ambiguous))

