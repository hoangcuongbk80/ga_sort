from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import MotionConfig

FloatArray = NDArray[np.float64]
MEASUREMENT_MATRIX = np.concatenate(
    (np.eye(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64)), axis=1
)


def transition_and_process_covariance(
    delta_time: float,
    image_width: int,
    image_height: int,
    config: MotionConfig,
) -> tuple[FloatArray, FloatArray]:
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    identity = np.eye(4, dtype=np.float64)
    transition = np.block(
        [[identity, delta_time * identity], [np.zeros((4, 4)), identity]]
    )
    acceleration = np.diag(
        [
            (config.position_acceleration_fraction * image_width) ** 2,
            (config.position_acceleration_fraction * image_height) ** 2,
            config.log_scale_acceleration**2,
            config.log_scale_acceleration**2,
        ]
    )
    process = np.block(
        [
            [delta_time**4 / 4.0 * acceleration, delta_time**3 / 2.0 * acceleration],
            [delta_time**3 / 2.0 * acceleration, delta_time**2 * acceleration],
        ]
    )
    return transition, process


def predict(mean: FloatArray, covariance: FloatArray, transition: FloatArray, process: FloatArray) -> tuple[FloatArray, FloatArray]:
    predicted_mean = transition @ mean
    predicted_covariance = transition @ covariance @ transition.T + process
    return predicted_mean, _symmetrize(predicted_covariance)


def innovation_statistics(
    mean: FloatArray,
    covariance: FloatArray,
    bias: FloatArray,
    measurement_covariance: FloatArray,
    measurement: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    innovation = measurement - MEASUREMENT_MATRIX @ mean - bias
    innovation_covariance = (
        MEASUREMENT_MATRIX @ covariance @ MEASUREMENT_MATRIX.T
        + measurement_covariance
    )
    return innovation, _symmetrize(innovation_covariance)


def correct(
    mean: FloatArray,
    covariance: FloatArray,
    innovation: FloatArray,
    innovation_covariance: FloatArray,
    measurement_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    cross_covariance = covariance @ MEASUREMENT_MATRIX.T
    kalman_gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    corrected_mean = mean + kalman_gain @ innovation
    residual_transform = np.eye(8) - kalman_gain @ MEASUREMENT_MATRIX
    corrected_covariance = (
        residual_transform @ covariance @ residual_transform.T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    return corrected_mean, _symmetrize(corrected_covariance)


def birth_covariance(
    image_width: int, image_height: int, config: MotionConfig
) -> FloatArray:
    measurement_variance = np.asarray(
        [
            (config.birth_position_std_fraction * image_width) ** 2,
            (config.birth_position_std_fraction * image_height) ** 2,
            config.birth_log_scale_std**2,
            config.birth_log_scale_std**2,
        ]
    )
    velocity_variance = np.asarray(
        [
            (config.birth_velocity_std_fraction * image_width) ** 2,
            (config.birth_velocity_std_fraction * image_height) ** 2,
            config.birth_log_scale_velocity_std**2,
            config.birth_log_scale_velocity_std**2,
        ]
    )
    return np.diag(np.concatenate((measurement_variance, velocity_variance)))


def _symmetrize(matrix: FloatArray) -> FloatArray:
    return 0.5 * (matrix + matrix.T)

