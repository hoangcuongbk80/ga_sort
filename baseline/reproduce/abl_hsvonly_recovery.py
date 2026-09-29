r"""Recovery ablation for the -learned-head (HSV-only) row at the benchmark cap (max_gap=80).
Swaps the head m_glare for the analytic m_glare_analytic, then runs the recovery eval.
Run via reproduce/run_synth_and_recovery_g80.sh, or directly from the repo root:

    python reproduce/abl_hsvonly_recovery.py
"""
import os, sys

# allow running from anywhere: put the repo root (parent of reproduce/) on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.tracking import tracker as T
from glaremot.perception.analytic import m_glare_analytic

_orig_call = T.Perception.__call__

def _patched_call(self, frame):
    dets = _orig_call(self, frame)
    for d in dets:
        x1, y1, x2, y2 = d["xyxy_s"]
        d["m_glare"] = float(m_glare_analytic(frame, (int(x1), int(y1), int(x2), int(y2))))
    return dets

T.Perception.__call__ = _patched_call
print("[abl_hsvonly_recovery] head m_glare -> analytic HSV; max_gap=80", flush=True)

from glaremot.eval import eval_recovery as ev
sys.argv = ["eval_recovery", "--split", "test", "--max_gap", "80", "--tracker_name", "abl_hsvonly_rcv"]
ev.main()
