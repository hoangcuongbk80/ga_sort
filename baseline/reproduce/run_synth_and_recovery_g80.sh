#!/usr/bin/env bash
# Synthetic glare benchmark at G=80 (coast+beacon coupled), test split = 15 q-twins.
#   - full-scene MOT, every person scored (HOTA 61.22)
#   - recovery on burnt targets (all-burn 71.2%, total 72.5%)
#   - the -learned-head (HSV) recovery ablation row
# SYNTH must point at the synthetic glare-twin root (see configs/paths.yaml: data.glare).
# Usage:  PYTHON=python SYNTH=/path/to/GlareMOT-Synth bash reproduce/run_synth_and_recovery_g80.sh
set -u
PY="${PYTHON:-python}"
SYNTH="${SYNTH:-$("$PY" -c 'from glaremot.config import load_config as L;print(L(None)["data"]["glare"])' 2>/dev/null)}"
cd "$(dirname "$0")/.."                 # repo root

echo "##### [1/3] GlareTrack full-scene MOT on 15 synth twins @ G=80"
GMOT_OUTPUT_MAX_GAP=80 GMOT_OUTPUT_MAX_GAP_BEACON=80 "$PY" scripts/09_eval_mot.py \
  --split test --data_root "$SYNTH" --tracker_name glaretrack_synthmot_g80 \
  --results_subdir eval_synth 2>&1 | tail -3

echo "##### [2/3] GlareTrack recovery on burnt targets @ G=80, --split test"
"$PY" scripts/10_eval_recovery.py --split test --max_gap 80 \
  --tracker_name glaretrack_rcv_g80 2>&1 | tail -3

echo "##### [3/3] -learned head (HSV) recovery ablation @ G=80, --split test"
"$PY" reproduce/abl_hsvonly_recovery.py 2>&1 | tail -3

echo "===================== SUMMARY ====================="
echo "--- full-scene synth MOT COMBINED ---"
grep -E "COMBINED" "results/eval_synth/benchmark_glaretrack_synthmot_g80_test.txt" 2>/dev/null
echo "--- GlareTrack recovery report ---"
sed -n '2,16p' results/eval_recovery/recovery_glaretrack_rcv_g80_test.txt 2>/dev/null
echo "--- HSV recovery ablation report ---"
sed -n '2,16p' results/eval_recovery/recovery_abl_hsvonly_rcv_test.txt 2>/dev/null
echo "DONE_SYNTH_G80"
