import numpy as np

from degradation_mot.models.observation import restricted_block_covariance


def test_restricted_covariance_is_positive_definite() -> None:
    covariance = restricted_block_covariance(
        np.asarray([-2.0, -1.0, -0.5, 0.5, 1.3]),
        predicted_width=40.0,
        predicted_height=80.0,
        sigma_min=np.asarray([0.01, 0.01, 0.02, 0.02]),
    )
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert covariance[0, 1] == 0.0
    assert covariance[2, 3] != 0.0

