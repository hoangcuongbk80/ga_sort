"""Step 10 - Glare Recovery Benchmark (amodal box + ID retention through burn).

Scores trackers on the SYNTHETIC-GLARE twin (GlareMOT-Synth), isolating the
glare-recovery question that real-TEST can't (real-TEST is detector-bound). See
glaremot/eval/eval_recovery.py.

Default split is 'heldout' = val+test q-twins, the held-out clean pool.

Examples:
  # OFFICIAL GlareTrack (emission cap lifted so the held box is scored through the burn):
  python scripts/10_eval_recovery.py --split test --tracker_name glaretrack_rcv

  # re-score an existing tracker txt only (e.g. a baseline dropped into the workspace):
  python scripts/10_eval_recovery.py --tracker_name deepocsort_rcv --skip_track
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.eval.eval_recovery import main

if __name__ == "__main__":
    main()
