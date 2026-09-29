import numpy as np

from degradation_mot.data.correspondence import Annotation, match_detections_to_annotations
from degradation_mot.types import Detection


def test_permissive_correspondence_matching() -> None:
    annotations = [Annotation(np.asarray([10.0, 10.0, 30.0, 50.0]), 0, 7)]
    detections = [Detection.from_xyxy((11.0, 9.0, 31.0, 49.0), 0.8, 0)]
    result = match_detections_to_annotations(annotations, detections)
    assert result.matches == ((0, 0),)
    assert result.missed_annotations == ()
    assert result.clutter_detections == ()


def test_class_mismatch_yields_miss_and_clutter() -> None:
    annotations = [Annotation(np.asarray([10.0, 10.0, 30.0, 50.0]), 0, 7)]
    detections = [Detection.from_xyxy((10.0, 10.0, 30.0, 50.0), 0.8, 1)]
    result = match_detections_to_annotations(annotations, detections)
    assert result.matches == ()
    assert result.missed_annotations == (0,)
    assert result.clutter_detections == (0,)


def test_ambiguous_component_is_not_labeled_as_clutter() -> None:
    annotations = [
        Annotation(np.asarray([10.0, 10.0, 30.0, 50.0]), 0, 7),
        Annotation(np.asarray([12.0, 10.0, 32.0, 50.0]), 0, 8),
    ]
    detections = [
        Detection.from_xyxy((11.0, 10.0, 31.0, 50.0), 0.8, 0),
        Detection.from_xyxy((13.0, 10.0, 33.0, 50.0), 0.7, 0),
    ]
    result = match_detections_to_annotations(annotations, detections)
    assert result.ambiguous_matches
    assert result.matches == ()
    assert result.missed_annotations == ()
    assert result.clutter_detections == ()
