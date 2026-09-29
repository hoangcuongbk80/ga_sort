#!/usr/bin/env bash
# Real-test ablation suite at deploy G=12. "G" = BOTH coast (GMOT_OUTPUT_MAX_GAP) AND
# beacon (GMOT_OUTPUT_MAX_GAP_BEACON) coupled. Rows for the ablation table:
#   Full / -overlap birth / -persistence / -learned head (HSV).
# Usage:  PYTHON=python bash reproduce/run_realtest_ablation_g12.sh
set -u
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."                 # repo root
EV() { "$PY" scripts/09_eval_mot.py --split test --tracker_name "$1" --results_subdir eval_abl; }

echo "##### [1/4] Full (G=12 = coast12+beacon12) -> abl_full_g12  (expect HOTA 69.17)"
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 EV abl_full_g12 2>&1 | tail -2

echo "##### [2/4] -overlap birth -> abl_nooverlap_g12"
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 GMOT_OVERLAP_BIRTH=0 EV abl_nooverlap_g12 2>&1 | tail -2

echo "##### [3/4] -persistence (coast+beacon=0) -> abl_nopersist_g12"
GMOT_OUTPUT_MAX_GAP=0 GMOT_OUTPUT_MAX_GAP_BEACON=0 EV abl_nopersist_g12 2>&1 | tail -2

echo "##### [4/4] -learned head / HSV-only -> abl_hsvonly (swaps head m_glare for analytic, G=12 coupled)"
GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 "$PY" reproduce/abl_hsvonly.py 2>&1 | tail -2

echo "===================== SUMMARY (COMBINED: HOTA DetA AssA IDF1 MOTA IDSW Rcll Prcn FP FN) ====================="
for t in abl_full_g12 abl_nooverlap_g12 abl_nopersist_g12 abl_hsvonly; do
  f="results/eval_abl/benchmark_${t}_test.txt"
  if [ -f "$f" ]; then printf "%-22s " "$t"; grep -E "COMBINED" "$f" | tail -1; else printf "%-22s MISSING\n" "$t"; fi
done
echo "DONE_ABL_SUITE"
