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

