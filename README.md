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
- Threshold-truncated beta target scores and marked Gaussian-mixture densities
  for clutter and birth, fitted on the retained support.
- Internal fixed-observation reference, BASE observation adapter, and evaluation
  of the fixed EISR population and the complete BDD100K validation split.

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
from degradation_mot.models.density import DiagonalGaussianMixtureDensity
from degradation_mot.models.observation import TorchObservationModel
from degradation_mot.tracking.tracker import DegradationConditionedTracker
from degradation_mot.types import Detection

tracker = DegradationConditionedTracker(
    config=TrackerConfig.from_yaml("configs/headlampmot.yaml"),
    observation_model=TorchObservationModel.from_checkpoint("observation.pt"),
    clutter_density=DiagonalGaussianMixtureDensity.load("clutter.npz"),
    birth_density=DiagonalGaussianMixtureDensity.load("birth.npz"),
)

# frame is a BGR uint8 image. Each detection is Detection.from_xyxy(...).
reported_tracks = tracker.update(frame, detections, timestamp_seconds)
```

## Cached-detection manifest

`scripts/run_tracker.py` accepts one JSON object per frame:

```json
{"image":"data/sequence/images/000001.jpg","timestamp":0.0,"detections":[{"xyxy":[120,80,190,260],"confidence":0.83,"category":0}]}
```

Run the learned observation model with:

```bash
python scripts/run_tracker.py --manifest sequence.jsonl --config configs/headlampmot.yaml --checkpoint observation.pt --clutter-density clutter.npz --birth-density birth.npz --output tracks.txt
```

The output follows the ten-column MOT text convention, with track
existence probability in the score column.

`tracking.confidence_threshold` defaults to `0.10`. Use the same frozen post-NMS
cache at all thresholds, with scores retained from `0.05`, and set the YAML
threshold and `--confidence-threshold` during fitting to the same value. Each
threshold needs its own verified correspondences, observation checkpoint,
clutter fit, and birth fit. The EISR reference stays at `0.10`. Checkpoints
must record their fitting confidence threshold.

For the standalone fixed-observation reference, fit pooled training clutter
using `scripts/fit_nuisance_density.py --spatial-only` and run
`scripts/run_tracker.py --internal-reference --clutter-density spatial_clutter.npz`
along with the same manifest, config, and output arguments. This reference uses
three consecutive hits, reports matched confirmed tracks, and deletes a
confirmed track after its 31st consecutive miss.

## Calibration archive format

`scripts/train_observation_model.py` expects training and validation `.npz`
archives containing:

- `degradation`: `N x 15` causal track-side descriptors at the default four
  radial bins (the `--radial-bins` option sets the alternate dimension);
- `baseline`: `N x 11` baseline descriptors;
- `detected`: binary pre-gating observation labels;
- `normalized_residual`: `N x 4` values ordered as normalized center residuals
  and log-width/log-height residuals;
- `confidence`: detector confidence for verified target-generated observations;
- `localization_eligible`: binary mask selecting verified observations with
  reliable spatial annotations.

Clutter and birth archives for `scripts/fit_nuisance_density.py` contain
`measurement`, `confidence`, `image_size`, and `frame_count`. Each archive's
`frame_count` includes eligible training frames with zero samples. At nondefault
thresholds, both training and nuisance archives must record the scalar
`confidence_threshold`. Model files record the threshold used during fitting.

## Controlled BASE variants

`tracking.base_adapter.BaseObservationAdapter` supplies independently switched
retained detection probability, bias, covariance, and target confidence to a
BASE host. A host implementing `BaseHost` invokes the override once after native
prediction and uses the returned quantities in the native scoring, correction,
and track-validity paths. It supplies its native confidence mark on `[0.10, 1)`
and its residual normalization scales in native measurement coordinates.
Train the BASE heads on residuals formed in those same coordinates. For the
visibility-only control, train with `--visibility-only` and enable only the
detection-probability switch. With every switch disabled the adapter calls the
original host step directly. Supply the configured BASE host and use the
vehicle-ReID tracker outputs for appearance-baseline evaluation.

## Evaluation

```bash
python scripts/evaluate_revision.py eisr --input events.json --output eisr.json
python scripts/evaluate_revision.py bdd-tracking --input videos.json --output bdd.json
```

For EISR, the input JSON is a list of sequence objects with `sequence_id`,
`frames`, and `events`. A frame has `timestamp`, `spatial_annotations` (rows with
`identity`, `category`, and `xyxy`), and `verified_detector_scores` keyed by
reference identity. Include reliable spatial boxes only, with no interpolation
through presence-only intervals. Give each sequence a `tracks` path relative to
the input JSON, pointing to a MOT output from `run_tracker.py`, or put `outputs`
on each frame as rows with `track_id`, `category`, and `xyxy`. Event rows contain
`event_id`, `target_id`, `onset`, `offset`, and `identity_resolved`, with optional
`exit_frame` and `attributes` (for example `independent`, `severe`, `competitive`,
or `detector_miss`). Event indices are zero-based, onset and offset are inclusive,
and exit is exclusive. For ties in pre-event ID counts, use the most recently
matched ID. The event ledger and reference scores must be shared across runs.

For BDD100K supply each of the 200 videos with `video_id`, `illumination`
(`daytime`, `night`, or `dawn/dusk`), `saturation_selected`, `matched_control`, and
complete `frames` and tracks in the format above. Apply the spatial ignore rules
before passing box records, and use COCO categories. Install the `eval` extra
for box-level TrackEval scoring. An alternative input is a per-video `classes`
mapping with sufficient statistics in the `TrackingCounts` format. Reports pool
counts and association sums over the complete videos and categories, and include
the separate saturation-selected, complement, control, illumination, and category
results. `paired_event_interval` and `paired_hota_interval` resample common
clusters or complete videos across runs and use within-replicate seed averages.

## Reproducibility rules

1. Cache detector outputs once and reuse them for all trackers.
2. Generate calibration descriptors from causal predicted track regions. Do
   not use the current-frame annotation to position an input crop.
3. Use annotations only to construct labels and localization residuals.
4. Compute feature-standardization statistics from the training split only.
5. Keep ambiguous correspondence decisions in a persistent review table.
6. Do not use BDD100K to fit the primary observation model.

