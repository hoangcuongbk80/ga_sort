#!/usr/bin/env bash
# Deployed operating point (G=12) measurements:
#   1. synth full-scene MOT @ G=12  (HOTA 61.22)
#   2. recovery @ G=12 --split test (deployed Rcv@.5 / held precision)
#   3. -overlap-birth recovery @ G=80 --split test (recovery ablation column)
# SYNTH must point at the synthetic glare-twin root (see configs/paths.yaml: data.glare).
# Usage:  PYTHON=python SYNTH=/path/to/GlareMOT-Synth bash reproduce/run_recovery_g12_deployed.sh
set -u
PY="${PYTHON:-python}"
SYNTH="${SYNTH:-$("$PY" -c 'from glaremot.config import load_config as L;print(L(None)["data"]["glare"])' 2>/dev/null)}"
cd "$(dirname "$0")/.."                 # repo root

echo "##### [1/3] synth full-scene MOT @ G=12 -> glaretrack_synthmot_g12"
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 "$PY" scripts/09_eval_mot.py \
  --split test --data_root "$SYNTH" --tracker_name glaretrack_synthmot_g12 \
  --results_subdir eval_synth 2>&1 | tail -3

echo "##### [2/3] recovery @ G=12 --split test -> glaretrack_rcv_g12"
"$PY" scripts/10_eval_recovery.py --split test --max_gap 12 \
  --tracker_name glaretrack_rcv_g12 2>&1 | tail -3

echo "##### [3/3] -overlap birth recovery @ G=80 --split test -> glaretrack_nooverlap_rcv_g80"
GMOT_OVERLAP_BIRTH=0 "$PY" scripts/10_eval_recovery.py --split test --max_gap 80 \
  --tracker_name glaretrack_nooverlap_rcv_g80 2>&1 | tail -3

echo "===================== SUMMARY ====================="
echo "--- synth MOT G=12 (deployed MOTA/FP) ---"
grep -E "COMBINED" "results/eval_synth/benchmark_glaretrack_synthmot_g12_test.txt" 2>/dev/null
echo "--- recovery G=12 (deployed Rcv/held-prec) ---"
sed -n '2,16p' results/eval_recovery/recovery_glaretrack_rcv_g12_test.txt 2>/dev/null
echo "--- -overlap recovery G=80 ---"
sed -n '2,16p' results/eval_recovery/recovery_glaretrack_nooverlap_rcv_g80_test.txt 2>/dev/null
echo "DONE_REMAINING_G12"
