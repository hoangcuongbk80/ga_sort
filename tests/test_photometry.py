import numpy as np

from degradation_mot.config import PhotometryConfig, TrackerConfig
from degradation_mot.models.observation import ConstantObservationModel
from degradation_mot.photometry.mask import build_saturation_candidate_mask
from degradation_mot.photometry.descriptors import (
    extract_baseline_descriptor,
    extract_degradation_descriptor,
)
from degradation_mot.tracking.tracker import DegradationConditionedTracker
from degradation_mot.types import Detection


def test_saturation_mask_removes_small_components() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[5:9, 5:9] = 255
    frame[15, 15] = 255
    mask, value = build_saturation_candidate_mask(frame, PhotometryConfig())
    assert mask[5:9, 5:9].all()
    assert not mask[15, 15]
    assert value[5, 5] == 1.0


def test_degradation_descriptor_shape_and_mass() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    value = np.zeros((20, 20), dtype=np.float32)
    value[mask] = 1.0
    descriptor = extract_degradation_descriptor(
        mask, value, np.asarray([0.0, 0.0, 20.0, 20.0]), 0.90
    )
    assert descriptor.shape == (15,)
    assert descriptor[0] == 16 / 400
    assert descriptor[1] > 0.0
    np.testing.assert_allclose(descriptor[2:4], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(descriptor[11:15].sum(), 1.0)


def test_empty_candidate_region_returns_zero() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    value = np.zeros((10, 10), dtype=np.float32)
    descriptor = extract_degradation_descriptor(
        mask, value, np.asarray([1.0, 1.0, 9.0, 9.0]), 0.90
    )
    np.testing.assert_array_equal(descriptor, np.zeros(15))


def test_baseline_descriptor_has_eleven_values() -> None:
    boxes = np.asarray([[10.0, 10.0, 30.0, 50.0], [20.0, 20.0, 40.0, 60.0]])
    descriptor = extract_baseline_descriptor(
        boxes[0], boxes, 0, 100, 80, [True, False, True]
    )
    assert descriptor.shape == (11,)
    np.testing.assert_allclose(descriptor[-1], 2.0 / 3.0)
    assert descriptor[-2] > 0.0


def test_tracker_confirms_birth_after_second_detection() -> None:
    tracker = DegradationConditionedTracker(
        TrackerConfig(), ConstantObservationModel()
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection.from_xyxy((40.0, 30.0, 60.0, 70.0), 0.8, 0)
    assert tracker.update(frame, [detection], 0.0) == []
    outputs = tracker.update(frame, [detection], 0.04)
    assert len(outputs) == 1
    assert outputs[0].track_id == 1
    np.testing.assert_allclose(outputs[0].xyxy, detection.xyxy, atol=1.0e-6)

