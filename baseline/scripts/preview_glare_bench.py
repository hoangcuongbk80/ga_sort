"""Visual-QA previews for the densified glare-bench twins.

For each held-out q-seq in the bench glare set, write a short mp4 around every
burn event so you can eyeball that the synthetic glare lands on the RIGHT person
at the RIGHT time and severity.  Each frame is overlaid with:
  * the burnt target's amodal GT box (thick green) - confirms the box still
    bounds the now-burnt person (the recovery benchmark scores exactly this box),
  * all other GT boxes (thin grey),
  * a banner: sequence, event index, target id, frame, and m_glare (observability;
    LOW = strongly burnt).

Streams frame-by-frame (no big in-memory buffer), downscaled to 720p.
Output: results/glare_bench_preview/<seq>.mp4

Run:
  python scripts/preview_glare_bench.py \
      --glare_root D:/datasets/GlareMOT-Synth --split heldout
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from glaremot.config import data_path, load_config, results_dir
from glaremot.data.glare_m import glare_m_map
from glaremot.data.splits import make_splits

MARGIN = 15          # frames of context before/after each event
TARGET_LONG = 720    # downscale longest side to this


def read_gt(gt_path):
    """{frame(1-idx): [(tid, x, y, w, h), ...]}  pixel xywh top-left."""
    per_frame = {}
    if not os.path.exists(gt_path):
        return per_frame
    with open(gt_path, encoding="utf-8") as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 6:
                continue
            fr = int(float(p[0]))
            tid = int(float(p[1]))
            x, y, w, h = (float(v) for v in p[2:6])
            per_frame.setdefault(fr, []).append((tid, x, y, w, h))
    return per_frame


def _draw(fr, seq, F, per_frame, mmap, banner):
    """Overlay GT boxes (green = currently burning at this frame) + banner."""
    fi0 = F - 1
    for gid, x, y, w, h in per_frame.get(F, []):
        m = mmap.get((gid, fi0), 1.0)
        burning = m < 0.95
        col = (0, 255, 0) if burning else (200, 200, 200)
        cv2.rectangle(fr, (int(x), int(y)), (int(x + w), int(y + h)), col,
                      3 if burning else 2)
        # EVERY GT person carries an id (the grey/non-burning ones too)
        label = f"id{gid} m={m:.2f}" if burning else f"id{gid}"
        cv2.putText(fr, label, (int(x), max(12, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(fr, seq, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(fr, banner, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), 2, cv2.LINE_AA)


def make_preview(seq, glare_root, out_path, full=False):
    """full=False: concatenate a clip per burn event (short, jumps between events).
    full=True: the whole sequence continuously (verifies no cuts)."""
    seq_dir = os.path.join(glare_root, seq)
    sched_path = os.path.join(seq_dir, "meta", "glare_schedule.json")
    if not os.path.exists(sched_path):
        return None
    events = json.load(open(sched_path, encoding="utf-8")).get("events", [])
    imgs = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    if not events or not imgs:
        return None
    per_frame = read_gt(os.path.join(seq_dir, "mot", "gt", "gt.txt"))
    mmap = glare_m_map(seq, glare_root=glare_root)   # {(tid, frame0): m}

    h0, w0 = cv2.imread(imgs[0]).shape[:2]
    scale = TARGET_LONG / float(max(h0, w0))
    ow, oh = int(round(w0 * scale)), int(round(h0 * scale))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 22 if full else 15,
                         (ow, oh))
    n_written = 0
    if full:
        ranges = [(0, len(imgs) - 1, 0)]
    else:
        ranges = [(max(0, int(ev["start"]) - MARGIN),
                   min(len(imgs) - 1, int(ev["start"]) + int(ev["duration"]) + MARGIN), ei)
                  for ei, ev in enumerate(events, 1)]
    for lo, hi, ei in ranges:
        for fi0 in range(lo, hi + 1):
            fr = cv2.imread(imgs[fi0])
            if fr is None:
                continue
            F = fi0 + 1
            banner = (f"F={F}/{len(imgs)}  (continuous)" if full
                      else f"event {ei}/{len(events)}  F={F}")
            _draw(fr, seq, F, per_frame, mmap, banner)
            vw.write(cv2.resize(fr, (ow, oh)))
            n_written += 1
    vw.release()
    return n_written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--glare_root", default=None,
                    help="bench glare root (defaults to data.glare from the config)")
    ap.add_argument("--split", default="heldout",
                    choices=["val", "test", "train", "heldout", "all"])
    ap.add_argument("--seq", default=None, help="only this sequence")
    ap.add_argument("--full", action="store_true",
                    help="render the whole sequence continuously (no event cuts)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    glare_root = args.glare_root or data_path(cfg, "glare")
    sp = make_splits()
    if args.seq:
        seqs = [args.seq]
    elif args.split == "all":
        seqs = sp["train"] + sp["val"] + sp["test"]
    elif args.split == "heldout":
        seqs = sp["val"] + sp["test"]
    else:
        seqs = sp[args.split]
    seqs = [s for s in seqs if os.path.isdir(os.path.join(glare_root, s, "images"))]

    out_dir = results_dir(cfg, "glare_bench_preview")
    print(f"{len(seqs)} seqs -> {out_dir}", flush=True)
    for i, seq in enumerate(seqs, 1):
        suffix = "_full" if args.full else ""
        out_path = os.path.join(out_dir, f"{seq}{suffix}.mp4")
        n = make_preview(seq, glare_root, out_path, full=args.full)
        if n:
            print(f"[{i}/{len(seqs)}] {seq}: {n} frames -> {os.path.basename(out_path)}",
                  flush=True)
        else:
            print(f"[{i}/{len(seqs)}] {seq}: no events/images, skip", flush=True)


if __name__ == "__main__":
    main()
