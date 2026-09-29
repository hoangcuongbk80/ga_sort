"""Step 15 -- REAL-val FP composition diagnostic (headroom for matching the
photometric-gate FP cut, but via the existence posterior).

Question: the baseline's ~2406 real-val FP -- how many are departed-ghost coasts
(track's owned GT target has LEFT the frame, box keeps being emitted) vs
present-miss (target still there, just not covered)? Departed-ghosts are exactly
what an output-gate (suppress emission when existence has collapsed) can cut
WITHOUT killing the track / flickering identity.

Reuses fp_breakdown/owned_target from scripts/13 on REAL GT (no glare events).
Txt-only, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.data.splits import make_splits
from glaremot.eval.eval_recovery import load_mot

# reuse the proven breakdown
import importlib.util
spec = importlib.util.spec_from_file_location(
    "s13", os.path.join(os.path.dirname(__file__), "13_fp_selectivity.py"))
s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)

REAL_ROOT = "D:/datasets/GlareMOT"
EVAL_ROOT = "results/eval/trackers"

TRACKERS = [
    ("gt_official_val", "official(base)"),
    ("gt_union_val", "union"),
    ("gt_v2_val", "gate(v2)"),
]


def main():
    sp = make_splits()
    seqs = [s for s in sp["val"] if os.path.isdir(os.path.join(REAL_ROOT, s))]
    gt_cache = {s: load_mot(os.path.join(REAL_ROOT, s, "mot", "gt", "gt.txt")) for s in seqs}

    print(f"\nREAL-val FP composition over {len(seqs)} seqs (IoU>={s13.IOU_THR}). Owner=last-matched GT id.\n")
    hdr = f"{'tracker':<16}{'FP':>7}{'departed-ghost':>16}{'present-miss':>14}{'unowned':>9}"
    print(hdr); print("-" * len(hdr))
    for tdir, lab in TRACKERS:
        d = os.path.join(EVAL_ROOT, tdir, "data")
        if not os.path.isdir(d):
            print(f"{lab:<16}  (no outputs at {d})"); continue
        T = D = P = U = 0
        for s in seqs:
            trk = load_mot(os.path.join(d, f"{s}.txt"))
            t, dep, p, u = s13.fp_breakdown(gt_cache[s], trk, mode="last")
            T += t; D += dep; P += p; U += u
        print(f"{lab:<16}{T:>7}{D:>8} ({100*D/T if T else 0:>4.0f}%){P:>7} "
              f"({100*P/T if T else 0:>4.0f}%){U:>6} ({100*U/T if T else 0:>4.0f}%)")
        # per-seq departed-ghost concentration (top 3)
        rows = []
        for s in seqs:
            trk = load_mot(os.path.join(d, f"{s}.txt"))
            t, dep, p, u = s13.fp_breakdown(gt_cache[s], trk, mode="last")
            rows.append((dep, t, s))
        rows.sort(reverse=True)
        top = "  ".join(f"{s.split('_')[0]}:{dep}g/{t}fp" for dep, t, s in rows[:4])
        print(f"   top departed-ghost seqs: {top}")


if __name__ == "__main__":
    main()
