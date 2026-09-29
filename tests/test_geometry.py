import numpy as np

from degradation_mot.geometry import measurement_to_xyxy, pairwise_iou, xyxy_to_measurement


def test_box_measurement_round_trip() -> None:
    box = np.asarray([10.0, 20.0, 50.0, 100.0])
    recovered = measurement_to_xyxy(xyxy_to_measurement(box))
    np.testing.assert_allclose(recovered, box)


def test_pairwise_iou() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    overlap = pairwise_iou(boxes, boxes)
    np.testing.assert_allclose(np.diag(overlap), 1.0)
    assert overlap[0, 1] == 0.0

