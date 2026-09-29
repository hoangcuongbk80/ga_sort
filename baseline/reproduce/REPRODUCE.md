# Reproducing the paper numbers

GlareTrack has a **single emission cap `G`** implemented as two coupled environment
variables, `GMOT_OUTPUT_MAX_GAP` (pure-Kalman coast) and `GMOT_OUTPUT_MAX_GAP_BEACON`
(beacon-anchored coast). **Always set both to the same value `G`.** The official
deployed default is `G=12` (both vars default to 12 in `glaremot/tracking/tracker.py`),
so a plain run reproduces the real-test headline out of the box.

| Operating point | `G` | Used for |
|---|---|---|
| Deployed (default) | 12 | Real-test MOT (Table 1), synthetic MOT (Table 2), deployed recovery |
| Benchmark | 80 | Recovery benchmark (Table 3, `G=80` row) |

Detector: YOLO11m @ 1536 px, conf 0.25 (paths in `configs/paths.yaml`). The two trained
weights are **shipped in-repo** under `weights/` (`detector/best.pt`, `hpsi/h_psi_head.pt`).

## Real-test MOT - Table 1 headline (HOTA 69.17)

```bash
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 \
  python scripts/09_eval_mot.py --split test
```
Expected: `COMBINED 69.17 / DetA 65.40 / AssA 74.10 / IDF1 85.19 / MOTA 79.61 /
IDSW 14 / FP 1710 / FN 2540` - saved log: `results/paper/realtest_glaretrack_g12_HOTA6917.txt`.

Ablation rows (Table - controller ablation), all at coupled `G=12`:
`GMOT_OVERLAP_BIRTH=0` (− overlap birth), `GMOT_OUTPUT_MAX_GAP=0
GMOT_OUTPUT_MAX_GAP_BEACON=0` (− persistence), learned-head→analytic HSV monkeypatch
(− learned head). See `reproduce/run_realtest_ablation_g12.sh`.

## Synthetic full-scene MOT - Table 2 (HOTA 61.22)

```bash
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 \
  python scripts/09_eval_mot.py --split test \
  --data_root <GlareMOT-Synth root> --results_subdir eval_synth
```
Saved log: `results/paper/synthmot_glaretrack_g12.txt`.

## Recovery benchmark - Table 3 (both operating points)

`scripts/10_eval_recovery.py` couples both emission caps to `--max_gap` automatically,
so you only pass `--max_gap`.

```bash
# Benchmark point G=80  -> total-burn recovery 72.5%
python scripts/10_eval_recovery.py --split test --max_gap 80

# Deployed point G=12   -> total-burn recovery 30.1% @ 94% held-id precision
python scripts/10_eval_recovery.py --split test --max_gap 12
```
Saved logs: `results/paper/recovery_glaretrack_g80_benchmark.txt`,
`results/paper/recovery_glaretrack_g12_deployed.txt`. The test split has 15 burnt
twins / 30 burn episodes / 21 burnt targets / 850 burn frames / 9570 clear-reference
frames.

The `reproduce/run_*.sh` scripts are the runners that regenerate the committed logs in
`results/paper/`. They run from the repo root and read dataset locations from
`configs/paths.yaml`; override the interpreter and synth root with `PYTHON=` and `SYNTH=`,
e.g. `PYTHON=python SYNTH=/path/to/GlareMOT-Synth bash reproduce/run_synth_and_recovery_g80.sh`.
