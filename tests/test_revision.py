from math import exp

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logit

from degradation_mot.bdd_evaluation import TrackingCounts, VideoResult, VEHICLE_CLASSES, paired_hota_interval
from degradation_mot.config import TrackerConfig
from degradation_mot.data.correspondence import Annotation
from degradation_mot.evaluation import EvaluationFrame, GlareEvent, bdd_event_strata, score_events
from degradation_mot.models.confidence import beta_log_tail
from degradation_mot.models.density import DiagonalGaussianMixtureDensity, _fit_truncated_mixture
from degradation_mot.models.observation import ConstantObservationModel
from degradation_mot.tracking.association import beta_log_density
from degradation_mot.tracking.base_adapter import (
    BaseObservation, BaseObservationAdapter, BaseTrackPrediction, ObservationSwitches,
)
from degradation_mot.tracking.tracker import DegradationConditionedTracker
from degradation_mot.types import Detection, TrackOutput


@pytest.mark.parametrize("threshold", [0.05, 0.10, 0.20])
def test_retained_target_and_nuisance_likelihoods(threshold):
    target, _ = quad(lambda c: exp(beta_log_density(c, 4.0, 6.0) - beta_log_tail(threshold, 4.0, 6.0)), threshold, 1)
    assert target == pytest.approx(1.0, abs=1.0e-12)

    rate = 1.18
    density = DiagonalGaussianMixtureDensity(
        np.array([1.0]), np.zeros((1, 5)), np.ones((1, 5)), rate, threshold,
    )
    spatial_density = (2 * np.pi) ** -2
    nuisance, _ = quad(lambda c: exp(density.log_intensity(np.zeros(4), c, (1, 1),
                        confidence_clip=(1.0e-12, 1.0 - 1.0e-12))) / spatial_density, threshold, 1)
    assert nuisance == pytest.approx(rate, abs=1.0e-9)
    assert density.log_intensity(np.zeros(4), threshold / 2, (1, 1)) == -np.inf


def test_conditional_mixture_estimates_original_score_distribution():
    rng = np.random.default_rng(6)
    threshold = 0.20
    full_samples = rng.normal(size=(30_000, 5))
    samples = full_samples[full_samples[:, 4] >= logit(threshold)]
    mean = samples.mean(axis=0, keepdims=True)
    variance = samples.var(axis=0, keepdims=True)
    _, fitted_mean, fitted_variance = _fit_truncated_mixture(samples, np.ones(1), mean, variance, threshold)
    assert abs(fitted_mean[0, 4]) < abs(mean[0, 4])
    assert fitted_variance[0, 4] == pytest.approx(1.0, abs=0.05)


def test_cutoff_never_births_low_confidence_and_reference_uses_three_hits():
    config = TrackerConfig()
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracker = DegradationConditionedTracker(config, ConstantObservationModel())
    low = Detection.from_xyxy((20, 20, 30, 40), 0.09, 0)
    tracker.update(frame, [low], 0.0)
    assert tracker.tracks == []
    at_threshold = Detection.from_xyxy((20, 20, 30, 40), 0.10, 0)
    tracker.update(frame, [low, at_threshold], 0.04)
    assert tracker.last_diagnostics.unused_detection_indices == (1,)
    assert len(tracker.tracks) == 1

    mixture = DiagonalGaussianMixtureDensity(np.ones(1), np.zeros((1, 5)), np.ones((1, 5)), 1.2)
    reference = tracker.internal_fixed_observation_reference(config, mixture)
    good = Detection.from_xyxy((20, 20, 30, 40), 0.80, 0)
    assert reference.update(frame, [good], 0.0) == []
    assert reference.update(frame, [good], 0.04) == []
    assert len(reference.update(frame, [good], 0.08)) == 1
    for frame_index in range(3, 33):
        assert reference.update(frame, [], frame_index * 0.04) == []
    assert len(reference.tracks) == 1
    reference.update(frame, [], 33 * 0.04)
    assert reference.tracks == []


def test_eisr_first_recovery_and_competitor_use_original_bounds():
    box = np.array([1.0, 1.0, 4.0, 4.0])
    competitor = np.array([5.0, 5.0, 8.0, 8.0])
    frames = [EvaluationFrame(i / 5, [Annotation(box, 2, 1), Annotation(competitor, 2, 2)], {1: 0.10})
              for i in range(9)]
    event = GlareEvent("glare", 1, 4, 5, True)
    outputs = [[TrackOutput(20, box, 2, 1.0)] for _ in frames]
    assert score_events(frames, [event], outputs)[0].success is True
    outputs[6] = [TrackOutput(21, box, 2, 1.0)]
    assert score_events(frames, [event], outputs)[0].success is False
    outputs[6] = [TrackOutput(20, box, 2, 1.0)]
    outputs[5] = [TrackOutput(20, competitor, 2, 1.0)]
    assert score_events(frames, [event], outputs)[0].success is False
    assert bdd_event_strata([True] * 7 + [False] * 4, [0.0] * 3 + [0.10] * 8) == ("high", "short")


def test_base_with_disabled_switches_delegates_without_model_or_frame_access():
    class Host:
        def step(self, *args):
            return args

    arguments = (None, [], 1.0)
    adapter = BaseObservationAdapter(None, ObservationSwitches())
    observed = adapter.step(Host(), *arguments)
    assert all(value is expected for value, expected in zip(observed, arguments))


def test_base_replacement_preserves_native_quantities_for_disabled_switches():
    native = BaseObservation(0.3, np.ones(4), np.eye(4), lambda score: 0.1)
    prediction = BaseTrackPrediction(np.array([1, 1, 10, 10]), [True], np.array([9, 9, 1, 1]), native)

    class Host:
        def step_with_observations(self, frame, detections, timestamp, override):
            return override([prediction])[0]

    model = ConstantObservationModel(detection_probability=0.7)
    adapter = BaseObservationAdapter(model, ObservationSwitches(detection_probability=True, confidence=True))
    observed = adapter.step(Host(), np.zeros((16, 16, 3), np.uint8), [], 0.0)
    assert observed.detection_probability == 0.7
    assert observed.bias is native.bias and observed.covariance is native.covariance
    assert observed.log_confidence_density(0.09) == -np.inf
    assert observed.log_confidence_density(0.8) == pytest.approx(
        beta_log_density(0.8, 4.0, 2.0) - beta_log_tail(0.10, 4.0, 2.0)
    )


def test_video_bootstrap_pools_complete_video_counts():
    perfect = TrackingCounts(10, 10, [10] * 19, [10.0] * 19, 10, 0, 0, 0)
    failed = TrackingCounts(10, 10, [0] * 19, [0.0] * 19, 0, 10, 10, 0)
    good = VideoResult("a", "daytime", False, False, dict.fromkeys(VEHICLE_CLASSES, perfect))
    bad = VideoResult("a", "daytime", False, False, dict.fromkeys(VEHICLE_CLASSES, failed))
    assert paired_hota_interval([[good]], [[bad]], replicates=32) == (100.0, 100.0)


def test_truncated_training_loss_has_finite_shape_gradients():
    torch = pytest.importorskip("torch")
    from degradation_mot.models.training import calibration_loss

    alpha = torch.tensor([4.0], dtype=torch.double, requires_grad=True)
    beta = torch.tensor([6.0], dtype=torch.double, requires_grad=True)
    prediction = {"confidence_alpha": alpha, "confidence_beta": beta,
                  "detection_probability": torch.tensor([0.85]),
                  "normalized_bias": torch.zeros((1, 4)), "covariance_raw": torch.zeros((1, 5))}
    loss = calibration_loss(prediction, torch.ones(1), torch.zeros((1, 4)),
                            torch.tensor([0.45], dtype=torch.double), torch.ones(1, dtype=torch.bool),
                            confidence_threshold=0.20)
    assert loss.confidence.item() == pytest.approx(-beta_log_density(0.45, 4.0, 6.0) + beta_log_tail(0.20, 4.0, 6.0))
    loss.total.backward()
    assert torch.isfinite(alpha.grad).all() and torch.isfinite(beta.grad).all()
