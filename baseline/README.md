# GlareTrack

**Glare-robust, motion-only, appearance-free multi-object tracking for mining glare.**

In underground mining footage a worker's head-lamp can *burn out their whole body in the
image* - total photometric feature loss, not occlusion. Every conventional tracker drops the
track the instant the detections vanish. GlareTrack holds a correctly-identified box through the
burn until the person reappears, **without any appearance/Re-ID model and without any learned
motion model** - just an OC-SORT Kalman backbone plus a saturation-conditioned controller that
acts on a single learned observability signal.

It is the official tracker for the **GlareMOT** benchmark and the reference implementation for the
paper *"Absence of Evidence is not Evidence of Absence: Observability-Aware Multi-Object Tracking
under Sensor-Induced Measurement Dropout."*

---

## Headline results (held-out test; identical YOLO11m detector at conf 0.25 for every method)

| Benchmark | Metric | GlareTrack | Best baseline |
|---|---|---:|---:|
| **Real test** (20 seq) | HOTA | **69.17** | TrackTrack 67.36 |
| Real test | IDF1 / IDSW | **85.19 / 14** | 81.14 / 29 |
| **Synth full-scene** (15 burn twins) | HOTA | **61.22** | HybridSORT 55.13 |
| **Recovery through burn** (total/heavy) | box recovery @IoU0.5 | **72.5% / 70.1%** | **0.0–1.4%** (all baselines) |

GlareTrack leads ordinary tracking (HOTA, IDF1, fewest identity switches - with **no appearance
model**) *and* is the only method that survives glare burn, where every baseline - including a
learned-motion tracker (DiffMOT) and an amodal/occlusion specialist (DIP) - collapses to ~0%.
The per-table logs that back the paper are under [`results/paper/`](results/paper/).

---

## Method

The tracker (`glaremot/tracking/tracker.py`) is OC-SORT (Kalman constant-velocity prediction +
velocity-direction association + observation-centric rematching) plus a **Saturation-Conditioned
Existence Controller (SCEC)**. It runs as the default config with **no env flags**.

**Perception (computed before the per-frame loop, on the *frozen* YOLO11m detector):**

- An **analytic block** (`glaremot/perception/analytic.py`) gives three physics-based,
  training-free signals per box: the photometric observability `m_glare` (torso-weighted surviving
  fraction), the saturated-core fraction `phi` (gates lamp blobs out of birth/association), and the
  lamp beacon `mu` (saturated-core centroid - a weak centre-only pseudo-measurement).
- A **lightweight learned head** `h_psi` (`glaremot/perception/h_psi.py`) refines `m_glare` with
  image context. It is the **only** learned signal the controller consumes - a single scalar,
  trained with a single smooth-L1 regression on frozen detector features. No Re-ID, no learned
  motion forecaster.

**SCEC controller (the glare-specific machinery):**

1. **Track-aware (overlap-band) birth.** A new track is suppressed only as a *duplicate* (high IoU
   with an existing track), never merely for being centre-close to a distinct neighbour - so two
   miners walking together are both tracked. Appearance-free, track-aware-NMS in spirit.
2. **Lamp beacon + earned Kalman coast.** While a confirmed track is unobserved, the analytic lamp
   beacon anchors a Kalman coast whose budget grows with the track's reliable evidence, holding the
   box through the burn. A re-entry gate stops a track reviving onto a lamp blob.
3. **Observability-weighted fusion.** A matched detection updates the Kalman state with
   heteroscedastic noise `R = R0 / max(m_glare, m_min)`, so a low-trust detection nudges rather than
   drags the state.
4. **Identity guards, exit gate, confirmation, emission cap.** Contested boxes re-awarded by last
   observation; order-inverting swaps undone; a track leaving through a frame edge dies fast; a box
   is emitted only while the unobserved gap stays within the single emission cap `G`.

A fixed-camera scene-veto (online EMA background residual) is also baked on as a deployment
birth-guard against detector hallucinations on static person-shaped structures.

> The deployed operating point is `G=12`. The recovery benchmark raises only this single cap to
> `G=80` to emit the earned box deeper into the burn; every gate and the coast are unchanged.

---

## Install

```bash
# 1) PyTorch matching your CUDA - https://pytorch.org/get-started/
pip install -r requirements.txt
# 2) MOT metrics for scripts/09_eval_mot.py (HOTA/IDF1/MOTA):
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

The two trained weights are **shipped in-repo and already public**: `weights/detector/best.pt`
(YOLO11m@1536) and `weights/hpsi/h_psi_head.pt` (glare observability head) - both are ready to use
as-is for inference/reproduction, no retraining needed. Only the auto-downloaded base
`yolo11m.pt` (`weights/pretrained/`) is git-ignored.

The detector *weight* is public, but the detector-training *corpus* it was fine-tuned on
(`data.detector` in `configs/paths.yaml`) is a separate dataset not bundled in this release; it
will be published separately. `scripts/01_prepare_dataset.py` / `02_train_detector.py` are
provided purely for reference, or for retraining on your own data pointed at that path - they are
not required to reproduce any result in the paper, since the trained checkpoint is already
included.

## Data

Datasets live **outside** the repo; set their locations in
[`configs/paths.yaml`](configs/paths.yaml) (edit the `data.*` paths, or point `GLAREMOT_CONFIG` at
your own copy). Only two roots are needed to reproduce the paper:

- `GlareMOT/<seq>/{images, mot/gt/gt.txt}` - real held-out MOT footage.
- `GlareMOT-Synth/<seq>/{images, meta/glare_schedule.json}` - synthetic glare-burn twins (the
  per-frame burn schedule is the recovery-benchmark ground truth).

Split: 80 / 17 / 20 train/val/**test** (`splits/`). Hyper-parameters are selected on val; every
reported number is on the held-out test.

## Reproduce the paper numbers

From the repo root, with `configs/paths.yaml` pointing at your data:

```bash
# Real test (deployed G=12, no flags needed)  -> HOTA 69.17
python scripts/09_eval_mot.py --split test --tracker_name glaretrack

# Synth full-scene MOT (every person scored)   -> HOTA 61.22
python scripts/09_eval_mot.py --split test --tracker_name glaretrack_synth \
    --data_root /path/to/GlareMOT-Synth --results_subdir eval_synth

# Recovery through burn at the benchmark cap G=80  -> total 72.5%, heavy 70.1%, all-burn 71.2%
python scripts/10_eval_recovery.py --split test --max_gap 80 --tracker_name glaretrack_rcv

# Visualize one sequence (annotated mp4; red ○ = lamp-core beacon, as in the paper teaser)
python scripts/08_run_tracker.py --seq 136_sequence_2q
```

Or run the bundled suites (set `PYTHON` / `SYNTH` to override the interpreter and synth root):

```bash
PYTHON=python bash reproduce/run_realtest_ablation_g12.sh     # ablation table (real test)
PYTHON=python bash reproduce/run_synth_and_recovery_g80.sh    # synth MOT + recovery @ G=80
PYTHON=python bash reproduce/run_recovery_g12_deployed.sh     # deployed-point recovery
```

See [`reproduce/REPRODUCE.md`](reproduce/REPRODUCE.md) for the full mapping of commands to paper
tables. (Set `YOLO_CONFIG_DIR` to a writable folder if ultralytics complains about its config path.)

## Repository layout

```
glaremot/
  tracking/tracker.py     # GlareTrack: OC-SORT backbone + SCEC controller (the method)
  perception/             # analytic block + h_psi observability head + photometric ops + synth glare
  eval/                   # eval_mot.py (HOTA/IDF1/MOTA), eval_recovery.py (glare recovery)
  data/                   # splits, glare schedule -> m_glare, h_psi dataset/labels
  external/ocsort/        # vendored OC-SORT Kalman filter + association
scripts/                  # 01..07 prepare/train, 08 run+visualize, 09/10 eval
reproduce/                # one-command scripts that regenerate the paper tables
configs/paths.yaml        # dataset / weight / output paths
splits/                   # the 80/17/20 split
weights/                  # detector + h_psi weights (shipped)
results/                  # the paper tables + per-method benchmark reports
```

## Citation

```bibtex
@inproceedings{glaretrack2026,
  title     = {Absence of Evidence is not Evidence of Absence: Observability-Aware
               Multi-Object Tracking under Sensor-Induced Measurement Dropout},
  author    = {TODO},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```
