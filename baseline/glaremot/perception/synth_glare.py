"""Improved synthetic headlamp-glare generator (spec v4 stage1 §9.1) for the 22fps MOT
person sequences `D:\\datasets\\GlareMOT` (q-suffixed seqs).

Goal: turn a CLEAN tracking sequence (frames + MOT gt boxes/IDs that stay valid as amodal
GT) into a paired "burnt" sequence where a person's headlamp glare rises smoothly, holds
(fully burning the person -> total feature loss), then fades - so the forecaster can be
trained/eval'd to hold the box through the glare gap.

Improvements over the reference v5 engine:
  - Smooth raised-cosine temporal envelope (C1-continuous): gradual brighten -> hold ->
    gradual fade, event duration <= 3s, no abrupt 1-2 frame flashes.
  - Physically-based render in LINEAR light (core + bloom halo + starburst), peak bloom
    sized to cover the WHOLE target person.
  - 1 output version per sequence (no leakage), with 1-3 non-overlapping events on random
    target tracks for temporal coverage.
  - NO mask / reliability outputs (m_glare is analytic from the burnt image at train time).
    Labels + MOT gt are copied unchanged; a small glare_schedule.json logs the events for
    evaluation bucketing.

Run (from the repo root):
  python -m glaremot.perception.synth_glare --seq 125_sequence_12q --preview   # one seq + preview
  python -m glaremot.perception.synth_glare                                              # all q-seqs
"""
import os
import re
import json
import glob
import math
import zlib
import random
import shutil
import argparse

import cv2
import numpy as np

from ..config import data_path, load_config

_cfg = load_config()
SRC_ROOT = data_path(_cfg, "tracking")
OUT_ROOT = data_path(_cfg, "glare")
FPS = 22.0
MAX_EVENT_SEC = 3.0
MIN_EVENT_SEC = 0.8


# ----------------------------- linear light -------------------------------
def srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x):
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


# ----------------------------- envelope -----------------------------------
def env_value(k, attack, hold, decay, peak):
    """Raised-cosine attack/hold/decay value at frame offset k (C1-smooth)."""
    if k < attack:
        u = (k + 1) / max(1, attack)
        return peak * 0.5 * (1 - math.cos(math.pi * u))
    if k < attack + hold:
        return peak
    u = (k - attack - hold) / max(1, decay)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, u)))


# ----------------------------- gt parsing ---------------------------------
def read_gt(gt_path):
    """MOT gt.txt -> (per_frame {frame0:[{id,box_xywh}]}, tracks {id:[frame0,...]}).
    frame in gt is 1-indexed; we store 0-indexed to match frame_%06d.jpg."""
    per_frame, tracks = {}, {}
    if not os.path.exists(gt_path):
        return per_frame, tracks
    for ln in open(gt_path, encoding="utf-8"):
        p = ln.strip().split(",")
        if len(p) < 6:
            continue
        f = int(float(p[0])) - 1
        tid = int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        per_frame.setdefault(f, []).append({"id": tid, "box": (x, y, w, h)})
        tracks.setdefault(tid, []).append(f)
    return per_frame, tracks


def make_events(n_frames, tracks, rng, min_ev=1, max_ev=3):
    """Pick min_ev..max_ev non-overlapping glare events, each on a track present
    for the window."""
    fps = FPS
    margin = max(8, int(0.06 * n_frames))
    n_events = rng.randint(min_ev, max_ev)
    # candidate tracks: contiguous-ish presence long enough for a min event
    cand = []
    for tid, frs in tracks.items():
        frs = sorted(frs)
        if len(frs) >= int(MIN_EVENT_SEC * fps):
            cand.append((tid, frs[0], frs[-1]))
    if not cand:
        return []
    events, busy = [], []
    for _ in range(n_events * 3):
        if len([e for e in events]) >= n_events:
            break
        tid, t0, t1 = rng.choice(cand)
        t0 = max(t0, margin); t1 = min(t1, n_frames - margin)
        D = rng.randint(int(MIN_EVENT_SEC * fps), int(MAX_EVENT_SEC * fps))
        if t1 - t0 < D:
            continue
        start = rng.randint(t0, t1 - D)
        if any(not (start + D <= b0 or start >= b1) for b0, b1 in busy):   # overlap
            continue
        attack = int(rng.uniform(0.20, 0.35) * D)     # slower rise
        hold = int(rng.uniform(0.45, 0.65) * D)        # long hold (real burns sit at peak ~67%)
        decay = max(0, D - attack - hold)              # short, fast fade
        peak = rng.uniform(0.95, 1.25)                 # >0.85 fires the whole-frame veil at hold
        events.append({"target_id": tid, "start": start, "duration": D,
                       "attack": attack, "hold": hold, "decay": decay, "peak": peak,
                       "peak_frame": start + attack + hold // 2})
        busy.append((start, start + D))
    return sorted(events, key=lambda e: e["start"])


# ----------------------------- render -------------------------------------
def _brightest_head_anchor(frame, box):
    """Anchor the lamp on the person's HEAD. Search the brightest pixel only INSIDE the
    head region of the person box (never the background, so it can't latch onto far
    tunnel lights); fall back to the head centre if no real lamp is present."""
    x, y, w, h = box
    H, W = frame.shape[:2]
    head_c = (x + w / 2, y + 0.12 * h)
    x1 = int(max(0, x)); x2 = int(min(W, x + w))
    y1 = int(max(0, y)); y2 = int(min(H, y + 0.30 * h))         # top 30% INSIDE the box
    if x2 <= x1 or y2 <= y1:
        return head_c
    crop = frame[y1:y2, x1:x2].max(2)
    if int(crop.max()) > 200:                                   # a real bright lamp on the head
        cy, cx = np.unravel_index(int(np.argmax(crop)), crop.shape)
        return (x1 + cx, y1 + cy)
    return head_c


# real-calibrated LED tint (BGR), bluer than the old warm defaults
CORE_COLOR = np.array([1.00, 0.97, 0.78], np.float32)      # white-hot core, slight blue (real R/B ~0.64-0.89)
HALO_COLOR = np.array([1.00, 0.78, 0.50], np.float32)      # inner burn
EDGE_BLUE = np.array([1.00, 0.45, 0.16], np.float32)       # deep-blue bloom (real bloom R/B ~0.14-0.37)


def apply_glare(frame_bgr, anchor, intensity, box, rng):
    """Composite a real-calibrated headlamp glare radiating from the LAMP `anchor`
    (helmet), strength `intensity`, in linear light:
      - a small white-hot saturated core,
      - a broad near-white burn sized to engulf the whole body at peak,
      - a wide blue-white bloom (ROI = 3.5*s_bloom, so the soft halo is never
        truncated into a square),
      - a whole-frame veiling lift for strong burns (I > 0.85),
      - a faint outer ghost and a few short diffraction spikes.
    All components radiate from the lamp (not the body centre), matching real
    mine-lamp glare (contained, blue-tinted, head-anchored)."""
    if intensity <= 1e-3:
        return frame_bgr
    I = float(intensity)
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    bh = max(w, h)
    ax, ay = anchor
    box_diag = 0.5 * math.hypot(w, h)                      # centre -> corner
    s_core = max(2.0, 0.11 * bh)
    s_burn = box_diag + (0.6 + 0.7 * I) * bh               # bigger white burn: engulf whole body
    s_bloom = s_burn + (0.6 + 0.8 * I) * bh                # wider halo (real r_bloom ~5.5x box-diag)
    R = int(3.5 * s_bloom) + 8                             # no square truncation of the soft halo
    frame_bgr = frame_bgr.copy()

    # whole-frame veiling glare for strong burns, centred at the lamp (204-style lift)
    if I > 0.85:
        gyf, gxf = np.mgrid[0:H, 0:W].astype(np.float32)
        sg_ = 0.6 * math.hypot(W, H) / 2.0
        veil = np.exp(-((gxf - ax) ** 2 + (gyf - ay) ** 2) / (2 * sg_ * sg_)) \
               * (0.16 * (I - 0.85) / 0.15)
        full = srgb_to_linear(frame_bgr.astype(np.float32) / 255.0) + veil[..., None] * HALO_COLOR
        frame_bgr = (linear_to_srgb(full) * 255.0).astype(np.uint8)

    x1 = max(0, int(ax - R)); x2 = min(W, int(ax + R))
    y1 = max(0, int(ay - R)); y2 = min(H, int(ay + R))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    gx, gy = np.meshgrid(np.arange(x1, x2, dtype=np.float32),
                         np.arange(y1, y2, dtype=np.float32))
    r2 = (gx - ax) ** 2 + (gy - ay) ** 2                   # ALL components radiate from the lamp
    core = np.exp(-r2 / (2 * s_core * s_core)) * (2.6 * I)
    burn = np.exp(-r2 / (2 * s_burn * s_burn)) * (1.7 * I)
    bloom = np.exp(-r2 / (2 * s_bloom * s_bloom)) * (0.85 * I)
    add = (core[..., None] * CORE_COLOR + burn[..., None] * HALO_COLOR
           + bloom[..., None] * EDGE_BLUE)

    # faint ghost at the outer edge (secondary internal reflection)
    if I > 0.45:
        ga = rng.uniform(0, 2 * math.pi); gd = rng.uniform(1.0, 1.5) * s_bloom
        gx0, gy0 = ax + gd * math.cos(ga), ay + gd * math.sin(ga)
        gr2 = (gx - gx0) ** 2 + (gy - gy0) ** 2
        add = add + (np.exp(-gr2 / (2 * (0.40 * s_bloom) ** 2)) * (0.16 * I))[..., None] * EDGE_BLUE

    # a few short diffraction spikes from the lamp core
    dlx, dly = gx - ax, gy - ay
    for _ in range(rng.randint(2, 4)):
        ang = rng.uniform(0, math.pi)
        u = dlx * math.cos(ang) + dly * math.sin(ang)
        v = -dlx * math.sin(ang) + dly * math.cos(ang)
        s_len = rng.uniform(0.8, 1.6) * bh
        spike = np.exp(-(u * u) / (2 * s_len * s_len) - (v * v) / (2 * 1.8 ** 2)) * (0.18 * I)
        add = add + spike[..., None] * CORE_COLOR

    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32) / 255.0
    out = (linear_to_srgb(srgb_to_linear(roi) + add) * 255.0).astype(np.uint8)
    frame_bgr[y1:y2, x1:x2] = out
    return frame_bgr


# ----------------------------- per sequence -------------------------------
def seq_number(folder):
    m = re.search(r"sequence_(\d+)", folder)
    return int(m.group(1)) if m else None


def is_q_sequence(folder):
    return bool(re.search(r"sequence_\d+q", folder))


def process_sequence(seq_dir, out_dir, preview=False, min_ev=1, max_ev=3):
    name = os.path.basename(seq_dir)
    rng = random.Random(zlib.crc32(name.encode()))          # stable, reproducible per-seq seed
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    n_frames = len(img_paths)
    if n_frames == 0:
        return None
    per_frame, tracks = read_gt(os.path.join(seq_dir, "mot", "gt", "gt.txt"))
    events = make_events(n_frames, tracks, rng, min_ev=min_ev, max_ev=max_ev)

    # frame -> (intensity, target_id) from the events
    sched = {}
    for ev in events:
        for k in range(ev["duration"]):
            fi = ev["start"] + k
            val = env_value(k, ev["attack"], ev["hold"], ev["decay"], ev["peak"])
            if val > sched.get(fi, (0.0, None))[0]:
                sched[fi] = (val, ev["target_id"])

    out_img = os.path.join(out_dir, "images")
    os.makedirs(out_img, exist_ok=True)
    preview_frames = []
    for fi, ip in enumerate(img_paths):
        dst = os.path.join(out_img, os.path.basename(ip))
        if fi not in sched:
            shutil.copy2(ip, dst)                            # clean frame: fast copy
            continue
        intensity, tid = sched[fi]
        frame = cv2.imread(ip)
        box = next((d["box"] for d in per_frame.get(fi, []) if d["id"] == tid), None)
        if box is not None:
            anchor = _brightest_head_anchor(frame, box)
            frame = apply_glare(frame, anchor, intensity, box, rng)
        cv2.imwrite(dst, frame)
        if preview:
            preview_frames.append(frame)

    # copy labels + mot (GT unchanged), write schedule
    for sub in ("labels", "labels_all", "mot", "meta", "classes.txt"):
        s = os.path.join(seq_dir, sub)
        d = os.path.join(out_dir, sub)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    with open(os.path.join(out_dir, "meta", "glare_schedule.json"), "w") as f:
        json.dump({"sequence": name, "n_frames": n_frames, "fps": FPS, "events": events}, f, indent=2)

    if preview and preview_frames:
        pv = os.path.join(out_dir, "meta", "glare_preview.mp4")
        h, w = preview_frames[0].shape[:2]
        vw = cv2.VideoWriter(pv, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for fr in preview_frames:
            vw.write(fr)
        vw.release()
    return {"name": name, "n_frames": n_frames, "events": len(events),
            "glare_frames": len(sched)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_ROOT)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--seq", default=None, help="process only this sequence folder")
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test", "heldout", "all"],
                    help="restrict to q-seqs in this split (heldout = val+test). "
                         "Use with a dedicated --out to build a denser eval-only "
                         "twin without touching the canonical training glare set.")
    ap.add_argument("--min_events", type=int, default=1)
    ap.add_argument("--max_events", type=int, default=3,
                    help="more events -> denser burn coverage for the recovery "
                         "benchmark (default 1-3 = the canonical training set)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="write a preview mp4 of glare frames")
    ap.add_argument("--resume", action="store_true",
                    help="skip sequences already fully rendered in --out")
    args = ap.parse_args()

    if args.seq:
        seqs = [args.seq]
    else:
        seqs = sorted(s for s in os.listdir(args.src) if is_q_sequence(s))
        if args.split:
            from ..data.splits import make_splits
            sp = make_splits()
            keep = (sp["val"] + sp["test"] if args.split == "heldout"
                    else sp["train"] + sp["val"] + sp["test"] if args.split == "all"
                    else sp[args.split])
            seqs = [s for s in seqs if s in set(keep)]
        if args.limit:
            seqs = seqs[:args.limit]
    print(f"{len(seqs)} q-sequence(s) to process | events {args.min_events}-{args.max_events} "
          f"| out={args.out}")
    for i, name in enumerate(seqs):
        src_dir = os.path.join(args.src, name)
        out_dir = os.path.join(args.out, name)
        if args.resume:
            n_src = len(glob.glob(os.path.join(src_dir, "images", "*.jpg")))
            n_out = len(glob.glob(os.path.join(out_dir, "images", "*.jpg")))
            if n_src > 0 and n_out == n_src:
                print(f"[{i+1}/{len(seqs)}] {name}: resume, skip ({n_out} frames)", flush=True)
                continue
        r = process_sequence(src_dir, out_dir,
                             preview=args.preview, min_ev=args.min_events,
                             max_ev=args.max_events)
        if r:
            print(f"[{i+1}/{len(seqs)}] {r['name']}: frames={r['n_frames']} "
                  f"events={r['events']} glare_frames={r['glare_frames']}", flush=True)
    print("DONE ->", args.out)


if __name__ == "__main__":
    main()
