# Degradation-conditioned probabilistic tracker

This repository provides a Python implementation of degradation-conditioned
multi-object tracking under photometric saturation. The tracker estimates the
probability of detection, systematic bounding-box bias, and heteroscedastic
measurement covariance from each predicted target region. These quantities are
used consistently in likelihood-based data association, Kalman correction, and
probabilistic track management.

The implementation uses the measurement representation

```text
z = [center_x, center_y, log(width), log(height)]
```

and an eight-dimensional constant-velocity state. Image boxes exposed by the
public API use `xyxy` coordinates with an exclusive lower-right boundary.

## Implemented components

- HSV saturation-candidate mask and the 15-dimensional degradation descriptor.
- The 11-dimensional baseline detectability descriptor.
- Detection-probability, bias, restricted block-covariance, and confidence heads.
- Timestamp-aware Kalman prediction and bias-aware Joseph-form correction.
- Statistical gating, target-to-clutter likelihood-ratio costs, track-specific
  miss alternatives, and Hungarian assignment.
- Bernoulli existence updates and likelihood-based tentative-track birth.
- Diagonal Gaussian-mixture marked densities for clutter and birth.
- Unit tests for descriptor dimensions, covariance construction, Kalman
  behavior, assignment, and existence updates.

## Installation

Python 3.10 is recommended for the provided dependency versions.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[train,test]"
pytest
```

## Minimal use

```python
from degradation_mot.config import TrackerConfig
from degradation_mot.models.observation import ConstantObservationModel
from degradation_mot.tracking.tracker import DegradationConditionedTracker
from degradation_mot.types import Detection

tracker = DegradationConditionedTracker(
    config=TrackerConfig(),
    observation_model=ConstantObservationModel(),
)

# frame is a BGR uint8 image. Each detection is Detection.from_xyxy(...).
reported_tracks = tracker.update(frame, detections, timestamp_seconds)
```

The constant observation model is intended only for integration testing. A
trained `TorchObservationModel` should be used for reported experiments.

## Cached-detection manifest

`scripts/run_tracker.py` accepts one JSON object per frame:

```json
{"image":"data/sequence/images/000001.jpg","timestamp":0.0,"detections":[{"xyxy":[120,80,190,260],"confidence":0.83,"category":0}]}
```

Run the constant-model integration version with:

```bash
python scripts/run_tracker.py --manifest sequence.jsonl --config configs/headlampmot.yaml --output tracks.txt
```

Add `--checkpoint`, `--clutter-density`, and `--birth-density` for the complete
learned model. The output follows the ten-column MOT text convention, with track
existence probability in the score column.

## Calibration archive format

`scripts/train_observation_model.py` expects training and validation `.npz`
archives containing:

- `degradation`: `N x 15` causal track-side descriptors;
- `baseline`: `N x 11` baseline descriptors;
- `detected`: binary pre-gating observation labels;
- `normalized_residual`: `N x 4` values ordered as normalized center residuals
  and log-width/log-height residuals;
- `confidence`: detector confidence for verified target-generated observations;
- `localization_eligible`: binary mask selecting verified observations with
  reliable spatial annotations.

Clutter and birth archives for `scripts/fit_nuisance_density.py` contain
`measurement`, `confidence`, `image_size`, and `frame_count`.

## Reproducibility rules

1. Cache detector outputs once and reuse them for all trackers.
2. Generate calibration descriptors from causal predicted track regions. Do
   not use the current-frame annotation to position an input crop.
3. Use annotations only to construct labels and localization residuals.
4. Compute feature-standardization statistics from the training split only.
5. Keep ambiguous correspondence decisions in a persistent review table.
6. Do not use BDD100K to fit the primary observation model.

