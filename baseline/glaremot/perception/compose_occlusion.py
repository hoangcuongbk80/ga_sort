r"""Person-person OCCLUSION compositor (spec v4 stage1 §9.2, build item #3) for the
22fps MOT person sequences ``D:\datasets\GlareMOT``.

WHY. Glare erases a person's OWN pixels; occlusion erases them because a NEARER
person stands in front. The forecaster must hold a track's amodal box through BOTH
gaps. §9.1 (`synth_glare.py`) makes glare pairs; this module makes the occlusion
pairs: clean sequences -> sequences where a person is partly hidden by a nearer
person, with the amodal box kept and a geometric ``vis_geom_gt`` recorded.

HOW (spec §9.2 "ghép 2-3 track, composite theo foot-y"). Real mine sequences rarely
have two tracks that already cross, so we SYNTHESIZE a crossing from tracks of the
SAME sequence (same camera => consistent background, minimal paste seam):
  - pick a TARGET track (the one we want occluded) and a DONOR/occluder track;
  - RE-TIME the donor (frame offset) and TRANSLATE it so that, over a short window,
    it sits over the target with a LARGER foot-y => it is NEARER => the occluder;
  - composite the donor crop ON TOP (overlap pixels come from the nearer person,
    exactly §9.2); feather the crop border to soften the rectangular seam.

GROUND TRUTH (spec §9.2 + §13).
  - ``amodal_box`` for everyone is the ORIGINAL label, unchanged (full extent).
  - ``vis_geom_gt`` = fraction NOT hidden by a nearer person, from the SAME
    unit-tested occlusion-graph used at runtime (`tracking/occlusion_graph.py`) ->
    train==runtime by construction. It also catches NATURAL overlaps already present
    in the clean sequence.
  - the donor becomes a real visible person, so it is ADDED to the GT (a new id,
    ``vis_geom`` ~ 1 because it is in front).
  - there is no glare here, so ``m_glare`` = 1 and ``m = m_glare * vis_geom = vis_geom``.
    (Mix with §9.1 later for occ+glare; the formats line up.)

OUTPUT mirrors `synth_glare.py`: a parallel dataset with composited ``images/``,
an EXTENDED MOT ``gt.txt`` (§13 columns), regenerated amodal YOLO ``labels/``, and
``meta/occlusion_schedule.json`` logging the events for eval bucketing.

OBB note: boxes in this tracking set are axis-aligned, so foot-y ordering and the
xyxy occlusion-graph are exact. The OBB swap (spec §10.3) only needs to replace the
xyxy `compute_vis_geom` call with the polygon path in `perception/obb_geometry.py`;
the rest (events, paste, GT columns) is unchanged.

Run (from the repo root):
  python -m glaremot.perception.compose_occlusion --seq 320_sequence_116q --preview
  python -m glaremot.perception.compose_occlusion            # all multi-track q-seqs
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

from ..tracking.occlusion_graph import compute_vis_geom, covered_fraction
from ..data.visibility_labels import combine_m, classify_provenance, visibility_band

from ..config import data_path, load_config

_cfg = load_config()
SRC_ROOT = data_path(_cfg, "tracking")
OUT_ROOT = data_path(_cfg, "occlusion")
FPS = 22.0
MIN_OCC_SEC = 0.8                 # shortest sustained occlusion event
MAX_OCC_SEC = 1.8                 # keep short so foot-y ordering stays stable over the window
SYN_ID_BASE = 9000               # synthetic-occluder ids start here (avoid clashing real ids)
TAU_OV = 0.2                     # occlusion-graph IoU gate (same default as runtime §7)
DMIN = 8.0                       # occlusion-graph depth-tie guard (px)
MIN_CENTER_COVER = 0.12         # require the donor to cover >=12% of the target at event centre
FEATHER_FRAC = 0.12             # crop-border alpha falloff (fraction of crop size)


# ----------------------------- gt parsing ---------------------------------
def read_gt(gt_path):
    """MOT gt.txt -> (per_frame {f0:[{id,box_xywh}]}, tracks {id:[f0,...]}, box_of {(id,f0):box}).

    Frames in gt are 1-indexed; we store 0-indexed to match ``frame_%06d.jpg``. Only
    the first 6 MOT columns are read (frame,id,x,y,w,h); extra columns are ignored, so
    this also round-trips our own extended output.
    """
    per_frame, tracks, box_of = {}, {}, {}
    for ln in open(gt_path, encoding="utf-8"):
        p = ln.strip().split(",")
        if len(p) < 6:
            continue
        f = int(float(p[0])) - 1
        tid = int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        per_frame.setdefault(f, []).append({"id": tid, "box": (x, y, w, h)})
        tracks.setdefault(tid, []).append(f)
        box_of[(tid, f)] = (x, y, w, h)
    return per_frame, tracks, box_of


# ----------------------------- box helpers --------------------------------
def xywh_to_xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def foot_y(b):
    """Foot point = bottom edge. Larger = lower in the image = NEARER (occluder)."""
    return b[1] + b[3]


def translate(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2], b[3])


def longest_run(frames_sorted):
    """Longest contiguous (step-1) run in a sorted, possibly-gappy frame list.
    Returns (start, end_inclusive)."""
    best_s = best_e = frames_sorted[0]
    s = prev = frames_sorted[0]
    for f in frames_sorted[1:]:
        if f == prev + 1:
            prev = f
        else:
            if prev - s > best_e - best_s:
                best_s, best_e = s, prev
            s = prev = f
    if prev - s > best_e - best_s:
        best_s, best_e = s, prev
    return best_s, best_e


# ----------------------------- event selection ----------------------------
def choose_events(tracks, box_of, n_frames, W, H, rng):
    """Pick 1-2 non-overlapping occlusion events. Each event re-times+translates a
    donor track to pass NEARER and IN FRONT of a target track for ``L`` frames.

    Returns a list of event dicts with keys: ``target_id, occluder_id, syn_id,
    start`` (canvas frame), ``L, off`` (donor_frame = canvas_frame + off), ``dx, dy``
    (translation), ``center_cover``.
    """
    Lmin = max(2, int(MIN_OCC_SEC * FPS))
    Lmax = max(Lmin, int(MAX_OCC_SEC * FPS))
    runs = {}
    for tid, frs in tracks.items():
        s, e = longest_run(sorted(set(frs)))
        if e - s + 1 >= Lmin:
            runs[tid] = (s, e)
    if len(runs) < 2:
        return []

    ids = list(runs)
    n_events = rng.randint(1, min(2, len(ids) - 1))
    events, busy = [], []
    for _ in range(n_events * 8):
        if len(events) >= n_events:
            break
        tid_t, tid_o = rng.sample(ids, 2)
        (ts0, te0), (os0, oe0) = runs[tid_t], runs[tid_o]
        L = rng.randint(Lmin, min(Lmax, te0 - ts0 + 1, oe0 - os0 + 1))
        ts = rng.randint(ts0, te0 - L + 1)
        ds = rng.randint(os0, oe0 - L + 1)
        off = ds - ts
        if any(not (ts + L <= b0 or ts >= b1) for b0, b1 in busy):     # canvas-window overlap
            continue
        fc = ts + L // 2
        gc = fc + off
        Bp, Bd = box_of.get((tid_t, fc)), box_of.get((tid_o, gc))
        if Bp is None or Bd is None:
            continue
        pcx = Bp[0] + Bp[2] / 2.0
        delta = rng.uniform(0.06, 0.18)                                # donor foot just BELOW target -> nearer
        dx = pcx - (Bd[0] + Bd[2] / 2.0)
        dy = (foot_y(Bp) + delta * Bp[3]) - foot_y(Bd)
        Bd_c = translate(Bd, dx, dy)
        cover = covered_fraction(xywh_to_xyxy(Bp), xywh_to_xyxy(Bd_c))  # share of target hidden at centre
        if cover < MIN_CENTER_COVER:
            continue
        events.append({"target_id": tid_t, "occluder_id": tid_o,
                       "syn_id": SYN_ID_BASE + len(events), "start": ts, "L": L,
                       "off": off, "dx": float(dx), "dy": float(dy),
                       "center_cover": float(cover)})
        busy.append((ts, ts + L))
    return sorted(events, key=lambda e: e["start"])


# ----------------------------- compositing --------------------------------
def feather_mask(h, w, frac=FEATHER_FRAC):
    """2-D alpha in [0,1]: 1 in the interior, raised-cosine falloff to 0 over a
    border of ``frac`` of each side. Softens the rectangular crop seam."""
    def ramp(n):
        a = np.ones(n, np.float32)
        f = int(round(frac * n))
        if f > 0:
            t = (np.arange(f, dtype=np.float32) + 1) / (f + 1)
            edge = 0.5 * (1 - np.cos(np.pi * t))
            a[:f] = edge
            a[-f:] = edge[::-1]
        return a
    return np.outer(ramp(h), ramp(w))


def paste_feathered(canvas, src, src_box_xyxy, dst_topleft, frac=FEATHER_FRAC):
    """Alpha-blend ``src``'s crop ``src_box_xyxy`` onto ``canvas`` at ``dst_topleft``
    (pixel x,y). Clamps both crop and destination to their image bounds. In-place."""
    H, W = canvas.shape[:2]
    sH, sW = src.shape[:2]
    sx1, sy1, sx2, sy2 = (int(round(v)) for v in src_box_xyxy)
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(sW, sx2), min(sH, sy2)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    dx1, dy1 = int(round(dst_topleft[0])), int(round(dst_topleft[1]))
    # shift the source crop to absorb out-of-canvas destination
    if dx1 < 0:
        sx1 -= dx1; dx1 = 0
    if dy1 < 0:
        sy1 -= dy1; dy1 = 0
    cw, ch = sx2 - sx1, sy2 - sy1
    cw = min(cw, W - dx1)
    ch = min(ch, H - dy1)
    if cw <= 0 or ch <= 0:
        return
    crop = src[sy1:sy1 + ch, sx1:sx1 + cw].astype(np.float32)
    a = feather_mask(ch, cw, frac)[..., None]
    dst = canvas[dy1:dy1 + ch, dx1:dx1 + cw].astype(np.float32)
    canvas[dy1:dy1 + ch, dx1:dx1 + cw] = (a * crop + (1 - a) * dst).astype(np.uint8)


def _composite_frame(base, f, per_frame, donors, box_of, img_paths):
    """Z-order ALL people by foot-y and paint far->near so the nearest always wins
    the overlap pixels (spec §9.2). Real people are repainted from the ORIGINAL frame
    (seamless, exact background) so a real that is NEARER than a donor correctly
    occludes it; donors are alpha-feathered crops from their own source frame.
    """
    orig = base                                              # untouched original pixels
    canvas = base.copy()
    layers = []                                             # (foot_y, src_img, src_box_xyxy, dst_xy, frac)
    for d in per_frame.get(f, []):                          # real persons (from the original frame)
        b = d["box"]
        layers.append((foot_y(b), orig, xywh_to_xyxy(b), (b[0], b[1]), 0.0))
    for dn in donors:                                       # synthetic occluders (re-timed + translated)
        bd = box_of.get((dn["occluder_id"], f + dn["off"]))
        if bd is None:
            continue
        placed = translate(bd, dn["dx"], dn["dy"])
        src = cv2.imread(img_paths[f + dn["off"]])
        layers.append((foot_y(placed), src, xywh_to_xyxy(bd), (placed[0], placed[1]), FEATHER_FRAC))
    for _fy, src, sbox, dxy, frac in sorted(layers, key=lambda L: L[0]):
        paste_feathered(canvas, src, sbox, dxy, frac)
    return canvas


# ----------------------------- per-frame GT -------------------------------
def frame_entries(f, per_frame, frame_donors, box_of):
    """All amodal boxes present at canvas frame ``f``: real persons + translated
    donor(s). Returns a list of ``(id, box_xywh, is_real)``."""
    entries = [(d["id"], d["box"], True) for d in per_frame.get(f, [])]
    for dn in frame_donors.get(f, []):
        bd = box_of.get((dn["occluder_id"], f + dn["off"]))
        if bd is not None:
            entries.append((dn["syn_id"], translate(bd, dn["dx"], dn["dy"]), False))
    return entries


def compute_frame_vis(entries):
    """vis_geom for every entry via the runtime occlusion-graph (xyxy + foot-y depth)."""
    if not entries:
        return []
    boxes = [xywh_to_xyxy(b) for _, b, _ in entries]
    keys = [foot_y(b) for _, b, _ in entries]
    return compute_vis_geom(boxes, keys, tau_ov=TAU_OV, dmin=DMIN).tolist()


def extended_gt_rows(n_frames, per_frame, frame_donors, box_of):
    """Build the §13 extended MOT rows for every frame. Columns:
    frame,id,bb_left,bb_top,bb_w,bb_h,flag,class,visibility(=m),m_glare,vis_geom,foot_y,prov,beacon_present
    """
    rows = []
    for f in range(n_frames):
        entries = frame_entries(f, per_frame, frame_donors, box_of)
        vis = compute_frame_vis(entries)
        for (tid, box, _is_real), vg in zip(entries, vis):
            m_glare = 1.0                                  # no glare in the occlusion-only set
            m = combine_m(m_glare, vg)
            prov = classify_provenance(m_glare, vg)
            band, _name, _mode = visibility_band(m)
            x, y, w, h = box
            rows.append(
                f"{f + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,"
                f"{m:.4f},{m_glare:.4f},{vg:.4f},{foot_y(box):.2f},{prov},0,{band}"
            )
    return rows


def yolo_lines_for_frame(entries, W, H):
    """Amodal YOLO label lines (class cx cy w h, normalized) for one frame."""
    out = []
    for _tid, (x, y, w, h), _real in entries:
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        out.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    return out


# ----------------------------- per sequence -------------------------------
def is_q_sequence(folder):
    return bool(re.search(r"sequence_\d+q", folder))


def _copy_aux(seq_dir, out_dir):
    """Copy the non-image, non-gt aux folders/files unchanged (HDD-friendly)."""
    for sub in ("labels", "labels_all", "meta", "classes.txt"):
        s, d = os.path.join(seq_dir, sub), os.path.join(out_dir, sub)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif os.path.isfile(s):
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(s, d)


def process_sequence(seq_dir, out_dir, preview=False, write_images=True):
    """Compose one sequence. ``write_images=False`` is the HDD-light mode: it skips
    decoding/copying frames and only emits the extended GT + schedule (the trajectory
    pairs the forecaster §10 actually consumes); appearance stages aren't retrained."""
    name = os.path.basename(seq_dir)
    rng = random.Random(zlib.crc32(name.encode()))            # stable per-seq seed (matches synth_glare style)
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    n_frames = len(img_paths)
    gt_path = os.path.join(seq_dir, "mot", "gt", "gt.txt")
    if n_frames == 0 or not os.path.exists(gt_path):
        return None
    per_frame, tracks, box_of = read_gt(gt_path)
    H, W = cv2.imread(img_paths[0]).shape[:2]                  # one decode for the frame size
    events = choose_events(tracks, box_of, n_frames, W, H, rng)

    # canvas frame -> list of active donors
    frame_donors = {}
    for ev in events:
        for f in range(ev["start"], ev["start"] + ev["L"]):
            frame_donors.setdefault(f, []).append(ev)

    os.makedirs(out_dir, exist_ok=True)

    preview_frames = []
    if write_images:
        out_img = os.path.join(out_dir, "images")
        os.makedirs(out_img, exist_ok=True)
        _copy_aux(seq_dir, out_dir)
        # ---- render images (only touched frames are decoded; the rest are copied) ----
        for f, ip in enumerate(img_paths):
            dst = os.path.join(out_img, os.path.basename(ip))
            donors = frame_donors.get(f)
            if not donors:
                shutil.copy2(ip, dst)
                continue
            base = cv2.imread(ip)
            out = _composite_frame(base, f, per_frame, donors, box_of, img_paths)
            cv2.imwrite(dst, out)
            if preview:
                preview_frames.append(out)
        # ---- add the donor to YOLO labels (only the touched frames are rewritten) ----
        lbl_dir = os.path.join(out_dir, "labels")
        for f in frame_donors:
            entries = frame_entries(f, per_frame, frame_donors, box_of)
            stem = os.path.splitext(os.path.basename(img_paths[f]))[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fp:
                fp.write("\n".join(yolo_lines_for_frame(entries, W, H)) + "\n")

    # ---- extended MOT gt (§13) - always written (trajectory ground truth) ----
    gt_dir = os.path.join(out_dir, "mot", "gt")
    os.makedirs(gt_dir, exist_ok=True)
    rows = extended_gt_rows(n_frames, per_frame, frame_donors, box_of)
    with open(os.path.join(gt_dir, "gt.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(rows) + ("\n" if rows else ""))
    src_labels_txt = os.path.join(seq_dir, "mot", "gt", "labels.txt")
    if os.path.exists(src_labels_txt):
        shutil.copy2(src_labels_txt, os.path.join(gt_dir, "labels.txt"))

    # ---- schedule ----
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    with open(os.path.join(out_dir, "meta", "occlusion_schedule.json"), "w") as fp:
        json.dump({"sequence": name, "n_frames": n_frames, "fps": FPS,
                   "W": W, "H": H, "events": events}, fp, indent=2)

    if preview and preview_frames:
        pv = os.path.join(out_dir, "meta", "occlusion_preview.mp4")
        h, w = preview_frames[0].shape[:2]
        vw = cv2.VideoWriter(pv, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for fr in preview_frames:
            vw.write(fr)
        vw.release()

    occ_frames = sum(1 for r in rows if float(r.split(",")[10]) < 0.999)
    return {"name": name, "n_frames": n_frames, "events": len(events),
            "donor_frames": len(frame_donors), "occluded_rows": occ_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_ROOT)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--seq", default=None, help="process only this sequence folder")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="write a preview mp4 of occlusion frames")
    ap.add_argument("--no-images", dest="no_images", action="store_true",
                    help="HDD-light: emit only the extended GT + schedule (no frame I/O)")
    args = ap.parse_args()

    if args.seq:
        seqs = [args.seq]
    else:
        seqs = sorted(s for s in os.listdir(args.src) if is_q_sequence(s))
        if args.limit:
            seqs = seqs[:args.limit]
    print(f"{len(seqs)} q-sequence(s) to consider")
    made = 0
    for i, name in enumerate(seqs):
        r = process_sequence(os.path.join(args.src, name), os.path.join(args.out, name),
                             preview=args.preview, write_images=not args.no_images)
        if r is None:
            print(f"[{i+1}/{len(seqs)}] {name}: skipped (no frames/gt)", flush=True)
            continue
        if r["events"] == 0:
            print(f"[{i+1}/{len(seqs)}] {r['name']}: no occlusion event (single/short tracks)", flush=True)
            continue
        made += 1
        print(f"[{i+1}/{len(seqs)}] {r['name']}: events={r['events']} "
              f"donor_frames={r['donor_frames']} occluded_rows={r['occluded_rows']}", flush=True)
    print(f"DONE -> {args.out}  ({made} sequence(s) with occlusion)")


if __name__ == "__main__":
    main()
