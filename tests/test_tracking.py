import numpy as np

from degradation_mot.config import MotionConfig
from degradation_mot.models.density import UniformMarkedDensity
from degradation_mot.models.observation import ObservationParameters
from degradation_mot.tracking.association import associate, missed_existence
from degradation_mot.tracking.kalman import (
    correct,
    innovation_statistics,
    predict,
    transition_and_process_covariance,
)
from degradation_mot.types import Detection


def test_lower_detection_probability_preserves_more_existence_after_miss() -> None:
    high_probability = missed_existence(0.95, 0.95, 0.995)
    low_probability = missed_existence(0.95, 0.20, 0.995)
    assert low_probability > high_probability


def test_bias_corrected_kalman_update() -> None:
    mean = np.asarray([50.0, 50.0, np.log(20.0), np.log(40.0), 0.0, 0.0, 0.0, 0.0])
    covariance = np.eye(8)
    measurement_covariance = np.eye(4)
    measurement = mean[:4] + np.asarray([5.0, 0.0, 0.0, 0.0])
    innovation, innovation_covariance = innovation_statistics(
        mean,
        covariance,
        np.asarray([5.0, 0.0, 0.0, 0.0]),
        measurement_covariance,
        measurement,
    )
    corrected_mean, corrected_covariance = correct(
        mean, covariance, innovation, innovation_covariance, measurement_covariance
    )
    np.testing.assert_allclose(corrected_mean, mean)
    assert np.all(np.linalg.eigvalsh(corrected_covariance) >= -1.0e-12)


def test_process_covariance_prediction_is_symmetric() -> None:
    transition, process = transition_and_process_covariance(0.04, 1920, 1080, MotionConfig())
    mean, covariance = predict(np.zeros(8), np.eye(8), transition, process)
    np.testing.assert_allclose(covariance, covariance.T)
    assert mean.shape == (8,)


def test_assignment_prefers_compatible_nearby_detection() -> None:
    mean = np.asarray([50.0, 50.0, np.log(20.0), np.log(40.0), 0.0, 0.0, 0.0, 0.0])
    covariance = np.eye(8)
    parameters = ObservationParameters(
        detection_probability=0.9,
        bias=np.zeros(4),
        covariance=np.diag([4.0, 4.0, 0.01, 0.01]),
        confidence_alpha=4.0,
        confidence_beta=2.0,
    )
    detection = Detection.from_xyxy((40.0, 30.0, 60.0, 70.0), 0.8, 0)
    measurement = np.asarray([[50.0, 50.0, np.log(20.0), np.log(40.0)]])
    result = associate(
        [mean],
        [covariance],
        [0.95],
        [0],
        [parameters],
        [detection],
        measurement,
        UniformMarkedDensity(cardinality_rate=1.0),
        (100, 100),
        14.86,
        0.995,
        (0.01, 0.99),
        (1.0e-4, 1.0 - 1.0e-4),
        1.0e-12,
        1.0e-12,
    )
    assert result.matches == ((0, 0),)
    assert result.unused_detections == ()

