r"""Ablation: replace the learned head m_glare with the analytic HSV m_glare_analytic
(the target the head regresses to), then run the real-test MOT eval. This is the
"-learned head (HSV)" / "no h_psi" row. Run via reproduce/run_realtest_ablation_g12.sh
(it sets the G=12 coast/beacon caps), or directly from the repo root:

    GMOT_OUTPUT_MAX_GAP=12 GMOT_OUTPUT_MAX_GAP_BEACON=12 python reproduce/abl_hsvonly.py
"""
import os, sys

# allow running from anywhere: put the repo root (parent of reproduce/) on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# deploy G=12 = BOTH coast + beacon caps coupled (unless the caller already set them)
os.environ.setdefault("GMOT_OUTPUT_MAX_GAP", "12")
os.environ.setdefault("GMOT_OUTPUT_MAX_GAP_BEACON", "12")

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
print("[abl_hsvonly] head m_glare -> analytic HSV m_glare_analytic; "
      f"BEACON_GAP={os.environ['GMOT_OUTPUT_MAX_GAP_BEACON']}", flush=True)

from glaremot.eval import eval_mot as ev
sys.argv = ["eval_mot", "--split", "test", "--tracker_name", "abl_hsvonly",
            "--results_subdir", "eval_abl"]
ev.main()
