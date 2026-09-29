r"""Precision-penalized recovery scorer (reviewer rebuttal, request C).

The official Glare Recovery Benchmark (glaremot/eval/eval_recovery.py) is PURE
RECALL: it counts how many amodal-GT burn boxes a tracker covers, and never
penalizes a coasted box that drifts off the target or is emitted while the
target is effectively gone. So a longer blind coast can only ever help the
score -- which is exactly why the deployed cap (G=12) and the reported cap
(G=90) diverge, and why a naive 90-frame coast looks good on recall alone.

This scorer adds, per burnt target, a PRECISION term on the entering identity:

  held-id precision = (burn frames where the entering id's box covers the GT)
                      / (burn frames where the entering id is emitted at all).

A track that holds its id but lets the box drift off the burnt person (a ghost
hold) is now penalized. Recovery-F1 = harmonic mean of Rcv-Recall and held-id
precision. This is the metric that separates "emit a coasted box" from
"keep the box ON the person through the burn" -- i.e. observability-awareness.

Reuses eval_recovery's loaders/geometry verbatim; only the scoring is new.
Reads MOT txt from results/eval_recovery/trackers/<name>/data/<seq>.txt.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from glaremot.config import data_path, load_config, results_dir
from glaremot.data.glare_m import glare_m_map
from glaremot.data.splits import make_splits
from glaremot.eval.eval_recovery import (BURN_THR, CLEAR_THR, BUCKETS, burn_index,
                                         load_events, load_mot, match_frame)


def score_seq(seq, gt_mot, trk_mot, glare_root, iou_thr=0.5):
    bi = burn_index(seq, glare_root)
    if not bi:
        return None
    burn_tids = {tid for (tid, _f) in bi}
    frames = sorted(gt_mot)
    match = {f: match_frame(gt_mot[f], trk_mot.get(f, []), iou_thr) for f in frames}
    trk_ids = {f: {r[0] for r in trk_mot.get(f, [])} for f in frames}

    # entering id = tracker id matched to the target most often in its CLEAR frames
    entering = {}
    for tid in burn_tids:
        from collections import Counter
        cnt = Counter()
        for f in frames:
            if bi.get((tid, f), 1.0) >= CLEAR_THR and tid in match[f]:
                cnt[match[f][tid][0]] += 1
        entering[tid] = cnt.most_common(1)[0][0] if cnt else None

    # buckets: recall (cover) + precision (eid emitted vs eid-on-target)
    keys = ["all", "heavy", "total"]
    acc = {k: dict(gt=0, cover=0, idcover=0, eid_emit=0, eid_on=0) for k in keys}

    def bucket_keys(m):
        ks = ["all"]
        for name, lo, hi in BUCKETS:
            if name in ("heavy", "total") and lo <= m < hi:
                ks.append(name)
        return ks

    for f in frames:
        for row in gt_mot[f]:
            tid = row[0]
            if tid not in burn_tids:
                continue
            m = bi.get((tid, f), 1.0)
            if m >= BURN_THR:
                continue
            eid = entering[tid]
            pair = match[f].get(tid)
            covered = pair is not None
            id_on = covered and eid is not None and pair[0] == eid
            eid_emitted = eid is not None and eid in trk_ids[f]
            for k in bucket_keys(m):
                a = acc[k]
                a["gt"] += 1
                a["cover"] += int(covered)
                a["idcover"] += int(id_on)
                a["eid_emit"] += int(eid_emitted)
                a["eid_on"] += int(id_on)   # eid emitted AND on the GT
    return acc


def add(agg, acc):
    for k, d in acc.items():
        for f in d:
            agg[k][f] += d[f]


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def f1(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+",
                    default=["glt_g90", "glt_g12", "oc_naivecoast", "oc_plain"])
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(None)
    glare_root = data_path(cfg, "glare")
    sp = make_splits()
    seqs = [s for s in sp["test"]
            if os.path.isdir(os.path.join(glare_root, s, "images"))
            and glare_m_map(s, glare_root=glare_root)]
    troot = os.path.join(results_dir(cfg, "eval_recovery"), "trackers")

    print(f"\n{'tracker':<16}{'Rcv@.5':>8}{'heavy':>7}{'total':>7}{'ID-Rcv':>8}"
          f"{'HeldPrec':>10}{'Rcv-F1':>8}{'ghost%':>8}")
    print("-" * 72)
    results = {}
    for name in args.trackers:
        data_dir = os.path.join(troot, name, "data")
        if not os.path.isdir(data_dir):
            print(f"{name:<16}  (no data dir, skipped)")
            continue
        agg = {k: dict(gt=0, cover=0, idcover=0, eid_emit=0, eid_on=0)
               for k in ["all", "heavy", "total"]}
        for seq in seqs:
            gt = load_mot(os.path.join(glare_root, seq, "mot", "gt", "gt.txt"))
            trk = load_mot(os.path.join(data_dir, f"{seq}.txt"))
            acc = score_seq(seq, gt, trk, glare_root, args.iou)
            if acc:
                add(agg, acc)
        a = agg["all"]
        rcv = pct(a["cover"], a["gt"])
        idrcv = pct(a["idcover"], a["gt"])
        prec = pct(a["eid_on"], a["eid_emit"])
        rf1 = f1(rcv, prec)
        ghost = pct(a["eid_emit"] - a["eid_on"], a["eid_emit"])
        results[name] = dict(rcv=rcv, heavy=pct(agg["heavy"]["cover"], agg["heavy"]["gt"]),
                             total=pct(agg["total"]["cover"], agg["total"]["gt"]),
                             idrcv=idrcv, prec=prec, f1=rf1, ghost=ghost)
        print(f"{name:<16}{rcv:>7.1f}%{results[name]['heavy']:>6.1f}%"
              f"{results[name]['total']:>6.1f}%{idrcv:>7.1f}%{prec:>9.1f}%"
              f"{rf1:>7.1f}{ghost:>7.1f}%")
    print("-" * 72)
    print(f"HeldPrec = among burn frames the entering id is emitted, fraction it covers "
          f"the GT at IoU>{args.iou}.")
    print("ghost%% = 100 - HeldPrec = id held but box OFF the person.".replace("%%", "%"))


if __name__ == "__main__":
    main()
