"""Glare Recovery Benchmark - amodal box + ID retention THROUGH controlled burn.

HANDOFF_PAPER.md §6c.  The real-TEST split is a wash on HOTA because it is
*detector-bound* (held-out pose/occlusion domain shift): both trackers lose the
same recall the detector loses, so neither can pull ahead.  This benchmark
removes that confound by scoring on the **synthetic-glare twin** of clean
sequences, where:

  * the detector works well (same distribution as train, no pose shift),
  * the burn is *controlled* - ``meta/glare_schedule.json`` records exactly which
    target is burnt, when, and how hard (``glare_m`` raised-cosine envelope), and
  * the GT box is *amodal* - synth only paints pixels, so ``mot/gt/gt.txt`` keeps
    the full body box right through the burn.

So we can ask the one question real-TEST can't isolate: **while a person is burnt
white (detector blind), who keeps drawing the box, and who keeps the ID?**  Other
trackers coast a Kalman box (not zero, but drifts off the manifold and is capped
by a fixed max-age); GlareTrack holds an amodal box on an earned Kalman coast
anchored by a weak lamp-core beacon and re-associates the *same* ID through the
burn where Re-ID is meaningless (the person is white).

Metrics (HANDOFF §6c), all transparent reductions of standard CLEAR/Identity
quantities restricted to the *burnt* targets - not a bespoke self-scoring metric:

  * **Recovery-Recall (RcvR) @ IoU 0.5 / 0.75 / 0.9** - fraction of amodal-GT boxes
    inside the burn the tracker still covers, at rising localization strictness
    (the mAP IoU-sweep, as recall), plus **mean IoU** of the held boxes.  Geometry
    only -> fair across trackers regardless of their confidence scores (a proper
    COCO-mAP would be unfair here: the baseline emits a constant score, GlareMOT a
    varying one, so confidence-ranked AP is apples-to-oranges).
  * **ID-Recovery-Recall (ID-RcvR)** - same, but the covering box must carry the
    *entering* tracker id (the id the target had just before the burn).  This is
    the recall side of IDF1 for the burnt target: "held the box AND the right ID."
  * **ID continuity through burn** (per burnt episode = one glare event):
      - **IDSW** - id switches: the matched tracker id on the target changes.
      - **Frag** - fragmentation: covered->uncovered transitions (track breaks).
      - **ID-off** - frames where the target's ENTERING tracker id is gone from the
        output entirely (the track was killed / not emitted = "ID turned off").
      - **Survival%** - episodes whose entering id is still matched to the target
        at the last burn frame (rode the whole burn without losing the track).
  * **clear-ref recall** - recall on the SAME burnt targets in their CLEAR frames
    (m >= CLEAR_THR).  The control: it shows the targets are detectable when not
    burnt, so a low RcvR is the *burn's* fault, not an always-hard target.

Stratified by burn severity ``m_glare`` (observability; LOW = burnt):
  partial 0.3-0.7 / heavy 0.1-0.3 / total <0.1.

HYGIENE: the head h_psi trains on TRAIN glare twins, so the benchmark must run on
sequences held out of head training.  Default split is ``val`` (no training
gradient; same easy distribution -> detector strong).
``test`` twins are fully held out but carry the pose/occlusion shift; ``--split``
lets you choose.

Frame convention: ``glare_m_map`` is 0-indexed (image index); gt.txt / tracker
MOT txt are 1-indexed.  So MOT frame F maps to glare_m_map key (tid, F-1).
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from ..config import data_path, load_config, results_dir, weight_path
from ..data.glare_m import glare_m_map
from ..data.splits import make_splits

# m_glare strata (observability in [0,1]; LOW = strongly burnt).  HANDOFF §6c.
BUCKETS = [
    ("partial", 0.3, 0.7),
    ("heavy", 0.1, 0.3),
    ("total", 0.0, 0.1),
]
BURN_THR = 0.7    # m < BURN_THR  -> counted as "in burn" (union of the 3 buckets)
CLEAR_THR = 0.9   # m >= CLEAR_THR -> clear-reference frame for the same target
# amodal box localization quality: recovery recall at increasing IoU strictness
# (the mAP IoU-sweep, as recall) + mean IoU of held boxes.  Geometry only, so it
# is fair across trackers regardless of their confidence scores.
IOU_LEVELS = (0.5, 0.75, 0.9)


# --------------------------------------------------------------------------- #
# IO + geometry
# --------------------------------------------------------------------------- #
def load_mot(path):
    """MOT txt -> {frame(1-idx): [(tid, x, y, w, h), ...]}.  Pixel xywh, top-left."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 6:
                continue
            fr = int(float(p[0]))
            tid = int(float(p[1]))
            x, y, w, h = (float(v) for v in p[2:6])
            out.setdefault(fr, []).append((tid, x, y, w, h))
    return out


def _iou(a, b):
    """IoU of two (x, y, w, h) top-left pixel boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def match_frame(gt_boxes, trk_boxes, iou_thr=0.5):
    """One-frame optimal GT<->tracker assignment (Hungarian, fall back to greedy).

    ``gt_boxes`` / ``trk_boxes``: lists of (tid, x, y, w, h).
    Returns {gt_tid: (trk_tid, iou)} for pairs with IoU > iou_thr (so one tracker
    box is never credited to two GTs); the IoU lets us also score @0.75/@0.9."""
    if not gt_boxes or not trk_boxes:
        return {}
    iou = np.zeros((len(gt_boxes), len(trk_boxes)), np.float32)
    for i, g in enumerate(gt_boxes):
        for j, t in enumerate(trk_boxes):
            iou[i, j] = _iou(g[1:], t[1:])
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-iou)
        pairs = list(zip(ri.tolist(), ci.tolist()))
    except Exception:  # pragma: no cover - scipy is present in this env
        pairs, used_g, used_t = [], set(), set()
        order = np.dstack(
            np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for i, j in order:
            i, j = int(i), int(j)
            if i in used_g or j in used_t or iou[i, j] <= iou_thr:
                continue
            pairs.append((i, j))
            used_g.add(i)
            used_t.add(j)
    return {gt_boxes[i][0]: (trk_boxes[j][0], float(iou[i, j]))
            for i, j in pairs if iou[i, j] > iou_thr}


# --------------------------------------------------------------------------- #
# burn index + per-sequence recovery scoring
# --------------------------------------------------------------------------- #
def burn_index(seq, glare_root=None):
    """{(tid, mot_frame_1idx): m_glare} from the glare schedule (or {} if none)."""
    mp = glare_m_map(seq, glare_root=glare_root) if glare_root else glare_m_map(seq)
    return {(tid, f0 + 1): m for (tid, f0), m in mp.items()}


def load_events(seq, glare_root=None):
    """[(target_id, start_frame0, duration), ...] from the glare schedule."""
    root = glare_root or data_path(load_config(), "glare")
    sched = os.path.join(root, seq, "meta", "glare_schedule.json")
    if not os.path.exists(sched):
        return []
    with open(sched, encoding="utf-8") as f:
        evs = json.load(f).get("events", [])
    return [(int(e["target_id"]), int(e["start"]), int(e["duration"])) for e in evs]


def n_events(seq, glare_root):
    return len(load_events(seq, glare_root))


# continuity counters (summed across burnt-target episodes):
#   idsw   - ID switches: matched tracker id on the target changes during burn
#   frag   - fragmentation: covered -> uncovered transitions (track interruptions)
#   id_off - frames where the target's ENTERING tracker id is gone from output
#            entirely (the track was killed / not emitted = "ID turned off")
#   n_ep   - burnt episodes scored (denominator for survival)
#   n_surv - episodes whose entering id is still matched to the target at burn end
_CONT = ("idsw", "frag", "id_off", "n_ep", "n_surv")


def _blank_bucket():
    # rcv = recovered @ IoU 0.5; rcv75/rcv90 = @0.75/@0.9; iou = sum IoU of held
    # boxes (mean = iou/rcv); idrcv = held with the entering id.
    return dict(n=0, rcv=0, rcv75=0, rcv90=0, iou=0.0, idrcv=0)


def _blank_stats():
    s = {b[0]: _blank_bucket() for b in BUCKETS}
    s["all"] = _blank_bucket()
    s["clear"] = _blank_bucket()
    for k in _CONT:
        s[k] = 0
    s["targets"] = 0
    return s


def recovery_seq(seq, gt_mot, trk_mot, iou_thr=0.5, glare_root=None):
    """Score one sequence.  Returns the per-bucket stats dict, or None if the
    sequence has no glare events (not part of the recovery benchmark)."""
    bi = burn_index(seq, glare_root)
    if not bi:
        return None
    burn_tids = {tid for (tid, _f) in bi}
    frames = sorted(gt_mot)

    # one optimal match per frame (reused for both recall and id checks)
    match = {f: match_frame(gt_mot[f], trk_mot.get(f, []), iou_thr) for f in frames}
    # tracker ids emitted at each frame (to detect a track being killed = "ID off")
    trk_ids = {f: {r[0] for r in trk_mot.get(f, [])} for f in frames}

    # entering id per burnt target = the tracker id it is matched to most often in
    # its CLEAR frames (m >= CLEAR_THR, default m = 1.0 outside any event window).
    entering = {}
    for tid in burn_tids:
        cnt = Counter()
        for f in frames:
            if bi.get((tid, f), 1.0) >= CLEAR_THR and tid in match[f]:
                cnt[match[f][tid][0]] += 1
        entering[tid] = cnt.most_common(1)[0][0] if cnt else None

    def _accum(d, matched, iou, id_ok):
        d["n"] += 1
        d["rcv"] += int(matched)
        d["rcv75"] += int(matched and iou > 0.75)
        d["rcv90"] += int(matched and iou > 0.90)
        d["iou"] += iou if matched else 0.0
        d["idrcv"] += int(id_ok)

    st = _blank_stats()
    st["targets"] = len(burn_tids)
    for f in frames:
        for row in gt_mot[f]:
            tid = row[0]
            if tid not in burn_tids:
                continue
            m = bi.get((tid, f), 1.0)
            pair = match[f].get(tid)
            matched = pair is not None
            iou = pair[1] if matched else 0.0
            id_ok = matched and entering[tid] is not None and pair[0] == entering[tid]
            if m < BURN_THR:
                for k in ["all"] + [b[0] for b in BUCKETS if b[1] <= m < b[2]]:
                    _accum(st[k], matched, iou, id_ok)
            elif m >= CLEAR_THR:
                _accum(st["clear"], matched, iou, id_ok)

    # continuity through burn, scored PER EPISODE (one glare event) so a clear gap
    # between two events on the same target is NOT counted as an interruption.
    gt_at = {f: {r[0] for r in gt_mot[f]} for f in frames}
    for tid, s0, dur in load_events(seq, glare_root):
        eid = entering.get(tid)
        ep = [F for F in range(s0 + 1, s0 + dur + 1)            # 1-indexed frames
              if F in gt_at and tid in gt_at[F] and bi.get((tid, F), 1.0) < BURN_THR]
        if not ep:
            continue
        st["n_ep"] += 1
        prev_cov, prev_id = True, eid          # confirmed track entering the burn
        for F in ep:
            pair = match[F].get(tid)
            cov = pair is not None
            cur = pair[0] if cov else None
            if prev_cov and not cov:                       # coverage broke
                st["frag"] += 1
            if cov and prev_id is not None and cur != prev_id:   # id changed
                st["idsw"] += 1
            if eid is not None and eid not in trk_ids[F]:  # entering track gone
                st["id_off"] += 1
            if cov:
                prev_id = cur
            prev_cov = cov
        last = match[ep[-1]].get(tid)
        if eid is not None and last is not None and last[0] == eid:   # survived to end
            st["n_surv"] += 1
    return st


# --------------------------------------------------------------------------- #
# aggregation + reporting
# --------------------------------------------------------------------------- #
def _add(agg, st):
    for k in [b[0] for b in BUCKETS] + ["all", "clear"]:
        for f in ("n", "rcv", "rcv75", "rcv90", "iou", "idrcv"):
            agg[k][f] += st[k][f]
    for k in _CONT:
        agg[k] += st[k]
    agg["targets"] += st["targets"]


def _rate(d, key="rcv"):
    return 100.0 * d[key] / d["n"] if d["n"] else float("nan")


def _miou(d):
    """Mean IoU of the HELD boxes (localization quality, mAP-style)."""
    return d["iou"] / d["rcv"] if d["rcv"] else float("nan")


def format_report(title, agg, n_seq, n_evt, meta=""):
    """Render the combined recovery table.  ASCII only (stdout is cp1252)."""
    rows = [f"\n===== GLARE RECOVERY -- {title} =====",
            f"seqs w/ glare: {n_seq} | events: {n_evt} | "
            f"burnt targets: {agg['targets']} | burn instances: {agg['all']['n']} "
            f"| clear-ref: {agg['clear']['n']}"]
    if meta:
        rows.append(meta)
    # RcvR at IoU 0.5/0.75/0.9 (amodal localization, mAP-style sweep) + mean IoU.
    hdr = (f"{'bucket':<11}{'m_glare':<10}{'n':>6}{'Rcv@.5':>8}{'@.75':>7}"
           f"{'@.9':>7}{'mIoU':>7}{'ID-Rcv':>8}")
    rows += [hdr, "-" * len(hdr)]

    def _line(name, rng, d):
        return (f"{name:<11}{rng:<10}{d['n']:>6}{_rate(d):>7.1f}%"
                f"{_rate(d, 'rcv75'):>6.1f}%{_rate(d, 'rcv90'):>6.1f}%"
                f"{_miou(d):>7.3f}{_rate(d, 'idrcv'):>7.1f}%")

    rows.append(_line("clear-ref", ">=0.90", agg["clear"]))
    rng = {"partial": "0.3-0.7", "heavy": "0.1-0.3", "total": "<0.10"}
    for name, _lo, _hi in BUCKETS:
        rows.append(_line(name, rng[name], agg[name]))
    rows.append("-" * len(hdr))
    d = agg["all"]
    rows.append(_line("all-burn", "<0.70", d))
    # ID continuity through burn (the headline ID story: hold the SAME id, don't
    # drop the track, don't fragment).
    surv = 100.0 * agg["n_surv"] / agg["n_ep"] if agg["n_ep"] else float("nan")
    off = 100.0 * agg["id_off"] / d["n"] if d["n"] else float("nan")
    rows += ["-" * len(hdr),
             f"ID continuity over {agg['n_ep']} burnt episodes ({d['n']} burn frames):",
             f"  ID switches (IDSW)      : {agg['idsw']}",
             f"  fragmentation (Frag)    : {agg['frag']}   "
             f"(track interruptions during burn)",
             f"  ID turned off (track gone): {agg['id_off']} frames ({off:.1f}% of burn)",
             f"  ID survived to burn end : {agg['n_surv']}/{agg['n_ep']} episodes "
             f"({surv:.1f}%)"]
    out = "\n".join(rows)
    print(out)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="heldout",
                    choices=["val", "test", "train", "heldout", "all"],
                    help="which sequences' glare twins to score. default 'heldout'"
                         " = val+test (the held-out clean pool).")
    ap.add_argument("--tracker_name", default="glaretrack_recovery")
    ap.add_argument("--max_gap", type=int, default=90,
                    help="emission cap for recovery (benchmark point 80; deploy 12). "
                         "Couples GMOT_OUTPUT_MAX_GAP and GMOT_OUTPUT_MAX_GAP_BEACON to "
                         "this value so the held box is emitted through the burn.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--glare_root", default=None,
                    help="override the glare dataset root (e.g. a denser eval-only "
                         "twin built by synth_glare --split heldout --max_events N). "
                         "Defaults to data.glare from the config.")
    ap.add_argument("--skip_track", action="store_true",
                    help="reuse existing MOT txt, only score")
    ap.add_argument("--resume", action="store_true",
                    help="skip sequences whose MOT txt already exists")
    ap.add_argument("--per_seq", action="store_true",
                    help="also print a per-sequence table")
    args = ap.parse_args()

    # tracker.py reads these at IMPORT time, so set them before importing it.
    # Couple BOTH the pure-Kalman coast cap and the beacon coast cap: at a low
    # deploy cap (e.g. 12) the beacon cap must move with it, else beacon-anchored
    # holds keep emitting to the old default and inflate recovery.
    os.environ["GMOT_OUTPUT_MAX_GAP"] = str(args.max_gap)
    os.environ["GMOT_OUTPUT_MAX_GAP_BEACON"] = str(args.max_gap)

    cfg = load_config(args.config)
    glare_root = args.glare_root or data_path(cfg, "glare")
    sp = make_splits()
    if args.split == "all":
        seqs = sp["train"] + sp["val"] + sp["test"]
    elif args.split == "heldout":
        seqs = sp["val"] + sp["test"]
    else:
        seqs = sp[args.split]
    # keep only glare twins that exist AND actually contain a burn schedule
    seqs = [s for s in seqs
            if os.path.isdir(os.path.join(glare_root, s, "images"))
            and glare_m_map(s, glare_root=glare_root)]
    print(f"{args.split} split: {len(seqs)} glare sequences with events | "
          f"max_gap={args.max_gap}", flush=True)
    if not seqs:
        sys.exit("no glare sequences with events in this split")

    eval_root = results_dir(cfg, "eval_recovery")
    data_dir = os.path.join(eval_root, "trackers", args.tracker_name, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not args.skip_track:
        import torch
        from .eval_mot import track_sequence  # triggers tracker import (env set above)
        from ..tracking.tracker import Perception
        device = "cuda" if torch.cuda.is_available() else "cpu"
        percep = Perception(weight_path(cfg, "detector"), weight_path(cfg, "hpsi"),
                            device, args.conf)
        for i, seq in enumerate(seqs, 1):
            out_txt = os.path.join(data_dir, f"{seq}.txt")
            if args.resume and os.path.exists(out_txt):
                print(f"[{i}/{len(seqs)}] {seq}: resume, skip", flush=True)
                continue
            n, fps = track_sequence(os.path.join(glare_root, seq), percep, out_txt)
            print(f"[{i}/{len(seqs)}] {seq}: {n} frames @ {fps:.1f} FPS", flush=True)

    print("\nscoring recovery ...", flush=True)
    agg = _blank_stats()
    per_seq, n_evt = [], 0
    for seq in seqs:
        gt = load_mot(os.path.join(glare_root, seq, "mot", "gt", "gt.txt"))
        trk = load_mot(os.path.join(data_dir, f"{seq}.txt"))
        st = recovery_seq(seq, gt, trk, args.iou, glare_root=glare_root)
        if st is None:
            continue
        _add(agg, st)
        n_evt += n_events(seq, glare_root)
        if args.per_seq:
            per_seq.append((seq, st))

    meta = f"tracker={args.tracker_name} | IoU>{args.iou} | max_gap={args.max_gap}"
    report = [format_report(args.tracker_name, agg, len(seqs), n_evt, meta)]

    if args.per_seq:
        lines = ["\n----- per sequence (all-burn Rcv@.5 mIoU ID-Rcv / clear "
                 "| IDSW Frag ID-off Surv) -----"]
        for seq, st in per_seq:
            surv = f"{st['n_surv']}/{st['n_ep']}"
            lines.append(f"{seq:<34} burn={st['all']['n']:>5} "
                         f"Rcv={_rate(st['all']):>5.1f}% "
                         f"mIoU={_miou(st['all']):>4.2f} "
                         f"ID={_rate(st['all'], 'idrcv'):>5.1f}% "
                         f"clear={_rate(st['clear']):>5.1f}% | "
                         f"IDSW={st['idsw']:>3} Frag={st['frag']:>3} "
                         f"off={st['id_off']:>4} surv={surv}")
        block = "\n".join(lines)
        print(block)
        report.append(block)

    out_txt = os.path.join(eval_root, f"recovery_{args.tracker_name}_{args.split}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nrecovery report saved -> {out_txt}")


if __name__ == "__main__":
    main()
