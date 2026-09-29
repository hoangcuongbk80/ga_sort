"""Step 12 -- Per-burn-DURATION mechanism analysis (Codex lever #1).

Tests the causal claim behind the observability-existence win: the advantage of
union (observability-extended coast) over OFFICIAL earned-coast should GROW as
the burn lasts longer than the official emission horizon (~12 frames). A dumb
long-coast also recovers long burns, but the full-scene Pareto already showed it
pays for that with a FP explosion; here we only score the burnt-target recovery
side, per event, bucketed by burn length.

Reuses the EXISTING full-scene synth tracker outputs (deployed configs that
produced the Pareto in results/eval_synth) -- NO re-tracking, NO new data. Reuses
the proven GT<->track matching from glaremot.eval.eval_recovery.

Frame convention: glare_m_map is 0-indexed; gt/mot txt 1-indexed (see eval_recovery).
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.data.splits import make_splits
from glaremot.eval.eval_recovery import (
    BURN_THR, CLEAR_THR, burn_index, load_events, load_mot, match_frame,
)

GLARE_ROOT = "D:/datasets/GlareMOT-Synth"
EVAL_ROOT = "results/eval_synth/trackers"
IOU_THR = 0.5

# tracker dir name -> label (deployed full-scene configs)
TRACKERS = [
    ("gt_official_synth_val", "official(g12)"),
    ("gt_union_r09_synth_val", "union(r0.9)"),
    ("gt_longcoast30_synth_val", "dumb30"),
    ("gt_longcoast90_synth_val", "dumb90"),
]

# burn-frame-count buckets (the true sub-threshold unobservable span per event)
DUR_BUCKETS = [("<=15", 0, 15), ("16-25", 16, 25), ("26-35", 26, 35),
               ("36-45", 36, 45), (">45", 46, 10 ** 9)]


def entering_ids(seq, gt, trk, burn_tids, bi):
    """Per burnt target, the tracker id matched most in its CLEAR frames."""
    frames = sorted(gt)
    match = {f: match_frame(gt[f], trk.get(f, []), IOU_THR) for f in frames}
    ent = {}
    for tid in burn_tids:
        cnt = Counter()
        for f in frames:
            if bi.get((tid, f), 1.0) >= CLEAR_THR and tid in match[f]:
                cnt[match[f][tid][0]] += 1
        ent[tid] = cnt.most_common(1)[0][0] if cnt else None
    return match, ent


def main():
    sp = make_splits()
    seqs = [s for s in sp["val"]
            if os.path.isdir(os.path.join(GLARE_ROOT, s, "images"))
            and load_events(s, glare_root=GLARE_ROOT)]

    gt_cache = {s: load_mot(os.path.join(GLARE_ROOT, s, "mot", "gt", "gt.txt")) for s in seqs}
    bi_cache = {s: burn_index(s, GLARE_ROOT) for s in seqs}

    # per (tracker) -> list of per-event records
    events = []  # (seq, tid, dur, n_burn) collected once
    # first collect events + their burn-frame sets (tracker-independent)
    burnframes = {}  # (seq,tid,start,dur) -> [frames]
    for s in seqs:
        gt, bi = gt_cache[s], bi_cache[s]
        gt_at = {f: {r[0] for r in gt[f]} for f in gt}
        for tid, s0, dur in load_events(s, glare_root=GLARE_ROOT):
            ep = [F for F in range(s0 + 1, s0 + dur + 1)
                  if F in gt_at and tid in gt_at[F] and bi.get((tid, F), 1.0) < BURN_THR]
            if ep:
                burnframes[(s, tid, s0, dur)] = ep
                events.append((s, tid, s0, dur, len(ep)))

    # per-tracker coverage on each event's burn frames
    cov = defaultdict(dict)   # tname -> {(s,tid,s0,dur): (rcv_frac, id_frac)}
    for tdir, _lab in TRACKERS:
        for s in seqs:
            gt, bi = gt_cache[s], bi_cache[s]
            trk = load_mot(os.path.join(EVAL_ROOT, tdir, "data", f"{s}.txt"))
            burn_tids = {tid for (tid, _f) in bi}
            match, ent = entering_ids(s, gt, trk, burn_tids, bi)
            for (ss, tid, s0, dur), ep in burnframes.items():
                if ss != s:
                    continue
                n = len(ep)
                rcv = sum(1 for F in ep if tid in match[F])
                idr = sum(1 for F in ep if tid in match[F]
                          and ent[tid] is not None and match[F][tid][0] == ent[tid])
                cov[tdir][(s, tid, s0, dur)] = (rcv / n, idr / n)

    # ---- per-event table ----
    print(f"\n{len(events)} burn episodes over {len(seqs)} synth val seqs "
          f"(burn frame = m<{BURN_THR}). Rcv@.5 = frac of burn frames covered.\n")
    hdr = f"{'seq':<24}{'tid':>4}{'dur':>5}{'nburn':>6}"
    for _t, lab in TRACKERS:
        hdr += f"{lab:>15}"
    print(hdr); print("-" * len(hdr))
    for (s, tid, s0, dur, n) in sorted(events, key=lambda e: e[4]):
        row = f"{s:<24}{tid:>4}{dur:>5}{n:>6}"
        for tdir, _lab in TRACKERS:
            rcv, _idr = cov[tdir][(s, tid, s0, dur)]
            row += f"{rcv * 100:>14.1f}%"
        print(row)

    # ---- bucketed-by-burn-length aggregate (the causal figure) ----
    print(f"\n=== RECOVERY vs BURN LENGTH (mean Rcv@.5 over episodes in bucket) ===")
    hdr2 = f"{'burnlen':<9}{'nEp':>5}"
    for _t, lab in TRACKERS:
        hdr2 += f"{lab:>15}"
    hdr2 += f"{'union-official':>16}"
    print(hdr2); print("-" * len(hdr2))
    for bname, lo, hi in DUR_BUCKETS:
        sel = [e for e in events if lo <= e[4] <= hi]
        if not sel:
            continue
        line = f"{bname:<9}{len(sel):>5}"
        means = {}
        for tdir, _lab in TRACKERS:
            m = np.mean([cov[tdir][(s, tid, s0, dur)][0] for (s, tid, s0, dur, n) in sel])
            means[tdir] = m
            line += f"{m * 100:>14.1f}%"
        delta = (means["gt_union_r09_synth_val"] - means["gt_official_synth_val"]) * 100
        line += f"{delta:>15.1f}%"
        print(line)

    # ID-recovery version (same buckets)
    print(f"\n=== ID-RECOVERY vs BURN LENGTH (mean ID-Rcv@.5; held box w/ entering id) ===")
    print(hdr2); print("-" * len(hdr2))
    for bname, lo, hi in DUR_BUCKETS:
        sel = [e for e in events if lo <= e[4] <= hi]
        if not sel:
            continue
        line = f"{bname:<9}{len(sel):>5}"
        means = {}
        for tdir, _lab in TRACKERS:
            m = np.mean([cov[tdir][(s, tid, s0, dur)][1] for (s, tid, s0, dur, n) in sel])
            means[tdir] = m
            line += f"{m * 100:>14.1f}%"
        delta = (means["gt_union_r09_synth_val"] - means["gt_official_synth_val"]) * 100
        line += f"{delta:>15.1f}%"
        print(line)


if __name__ == "__main__":
    main()
