"""Step 13 -- Coast-FP SELECTIVITY analysis (the 'missed-because-gone' half).

The per-burn-duration analysis (step 12) showed union == dumb-coast on RECOVERY
(both hold burnt-present workers equally). What separates them is FALSE POSITIVES.
This script shows WHERE the dumb-coast FP come from and that union suppresses
exactly the ghost kind: tracks whose target has DEPARTED the scene.

For each tracker (deployed full-scene synth configs), every emitted box that
matches NO GT at IoU>=0.5 is a FP. We split FP by the state of the track's OWNED
GT target (the GT id it matched most across the sequence):
  * 'departed-ghost' : owned target ABSENT from the frame  -> coasting is wrong
                       (the person is gone; this is a pure ghost box).
  * 'present-miss'   : owned target PRESENT in the frame    -> a localization/assoc
                       miss while the target is still there (defensible to keep).
  * 'unowned'        : track never matched any GT (spurious birth).

Mechanism claim: union and dumb-coast tie on recovery, but dumb-coast's EXTRA FP
are overwhelmingly departed-ghosts -- exactly what an observability-conditioned
existence posterior retires (high observability + persistent miss => target gone).
Txt-only (GT + tracker MOT), NO GPU, NO new data.
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.data.splits import make_splits
from glaremot.eval.eval_recovery import _iou, load_events, load_mot

GLARE_ROOT = "D:/datasets/GlareMOT-Synth"
EVAL_ROOT = "results/eval_synth/trackers"
IOU_THR = 0.5

TRACKERS = [
    ("gt_official_synth_val", "official(g12)"),
    ("gt_union_r09_synth_val", "union(r0.9)"),
    ("gt_longcoast30_synth_val", "dumb30"),
    ("gt_longcoast90_synth_val", "dumb90"),
]


def owned_target(gt, trk):
    """tracker id -> GT id it matches most often across the sequence (greedy/frame)."""
    pair_cnt = defaultdict(Counter)
    for f, trows in trk.items():
        grows = gt.get(f, [])
        if not grows:
            continue
        for (ttid, tx, ty, tw, th) in trows:
            best_g, best_i = None, IOU_THR
            for (gtid, gx, gy, gw, gh) in grows:
                i = _iou((tx, ty, tw, th), (gx, gy, gw, gh))
                if i >= best_i:
                    best_g, best_i = gtid, i
            if best_g is not None:
                pair_cnt[ttid][best_g] += 1
    return {ttid: c.most_common(1)[0][0] for ttid, c in pair_cnt.items()}


def fp_breakdown(gt, trk, mode="global"):
    """mode='global' -> owner = most-matched GT id over whole seq.
       mode='last'   -> owner = the GT id this track was LAST matched to before
                        the FP frame (Codex robustness check: more local owner)."""
    owned = owned_target(gt, trk)
    gt_ids_at = {f: {r[0] for r in rows} for f, rows in gt.items()}
    last_owner = {}  # ttid -> last GT id matched, updated as frames advance
    dep = pres = unowned = total_fp = 0
    for f in sorted(trk):
        trows = trk[f]
        grows = gt.get(f, [])
        for (ttid, tx, ty, tw, th) in trows:
            best_g, best_i = None, IOU_THR
            for (gtid, gx, gy, gw, gh) in grows:
                i = _iou((tx, ty, tw, th), (gx, gy, gw, gh))
                if i >= best_i:
                    best_g, best_i = gtid, i
            if best_g is not None:           # matched -> update last owner, not a FP
                last_owner[ttid] = best_g
                continue
            total_fp += 1
            own = last_owner.get(ttid) if mode == "last" else owned.get(ttid)
            if own is None:
                unowned += 1
            elif own in gt_ids_at.get(f, set()):
                pres += 1
            else:
                dep += 1
    return total_fp, dep, pres, unowned


def main():
    sp = make_splits()
    seqs = [s for s in sp["val"]
            if os.path.isdir(os.path.join(GLARE_ROOT, s, "images"))
            and load_events(s, glare_root=GLARE_ROOT)]
    gt_cache = {s: load_mot(os.path.join(GLARE_ROOT, s, "mot", "gt", "gt.txt")) for s in seqs}

    print(f"\nCoast-FP selectivity over {len(seqs)} synth val seqs (IoU>={IOU_THR}).")
    print("FP split by state of the track's OWNED GT target.")
    for mode, title in (("global", "OWNER = most-matched GT id over sequence"),
                        ("last", "OWNER = last GT id matched before the FP (robustness check)")):
        print(f"\n--- {title} ---")
        hdr = f"{'tracker':<16}{'FP':>7}{'departed-ghost':>16}{'present-miss':>14}{'unowned':>9}"
        print(hdr); print("-" * len(hdr))
        res = {}
        for tdir, lab in TRACKERS:
            T = D = P = U = 0
            for s in seqs:
                trk = load_mot(os.path.join(EVAL_ROOT, tdir, "data", f"{s}.txt"))
                t, d, p, u = fp_breakdown(gt_cache[s], trk, mode=mode)
                T += t; D += d; P += p; U += u
            res[tdir] = (T, D, P, U)
            print(f"{lab:<16}{T:>7}{D:>8} ({100*D/T if T else 0:>4.0f}%){P:>7} "
                  f"({100*P/T if T else 0:>4.0f}%){U:>6} ({100*U/T if T else 0:>4.0f}%)")
        print("-" * len(hdr))
        uo = res["gt_union_r09_synth_val"]; d30 = res["gt_longcoast30_synth_val"]
        d90 = res["gt_longcoast90_synth_val"]
        print(f"EXTRA FP of dumb-coast over union, by type:")
        for lab, dd in (("dumb30", d30), ("dumb90", d90)):
            extra = dd[0] - uo[0]
            gh = dd[1] - uo[1]
            print(f"  {lab:<8} extra FP = {extra:>5}  "
                  f"(departed-ghost {gh:+5} = {100*gh/extra if extra else 0:>3.0f}% of extra, "
                  f"present-miss {dd[2]-uo[2]:+5}, unowned {dd[3]-uo[3]:+5})")


if __name__ == "__main__":
    main()
