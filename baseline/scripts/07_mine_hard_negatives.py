"""Step 07 - Mine lamp-blob hard negatives for h_psi, then retrain (step 06 --hardneg).

WHY. The detector fires on bright lamp blobs; without counter-examples the
h_psi head scores them present ~0.7 and the tracker would give birth to "lamp
tracks". This step runs the detector over the synthetic-glare TRAIN/VAL
sequences, keeps every detection that is mostly saturated core
(``core_frac > 0.4``), and encodes those boxes as ``present = 0`` negatives.

GT-OVERLAP FILTER (critical). A bloom ON a person is a person-under-glare - ``present`` must stay high there, only ``m_glare`` drops. Any lamp-like det
overlapping a GT person box (IoU >= 0.2) is therefore AMBIGUOUS and dropped;
labelling it 0 would teach the head that "a person under glare is not a
person" and collapse recall.

The 12 real-glare TEST sequences are never touched: the new detector has not
seen them, and mining them would leak test imagery into h_psi training.

Run:  python scripts/07_mine_hard_negatives.py            # mine caches
      python scripts/06_train_hpsi.py --hardneg            # then retrain
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ultralytics"))

import cv2
import numpy as np
import torch

from glaremot.config import cache_dir, data_path, load_config, results_dir, weight_path
from glaremot.data.splits import make_splits
from glaremot.perception.h_psi import HPsiModel
from glaremot.tracking.tracker import Perception, detector_debug

CORE_FRAC_MIN = 0.4   # det that is mostly saturated core -> lamp-like
GT_IOU_MAX = 0.2      # overlap with a GT person -> ambiguous, DROP


def _frame_gt_boxes(seq_dir, frame_idx):
    """Normalized xyxy GT boxes of one frame from per-frame 5-col labels
    (labels/frame_%06d.txt; frame_idx is 1-based -> file index 0-based)."""
    p = os.path.join(seq_dir, "labels", f"frame_{frame_idx - 1:06d}.txt")
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding="utf-8"):
        v = ln.split()
        if len(v) >= 5:
            cx, cy, w, h = (float(x) for x in v[1:5])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def _iou_norm(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / max(ua, 1e-9)


def load_csv_rows(path, gt_seq_dir):
    """{frame: [(cx,cy,bw,bh)]} of lamp-like dets, GT-overlap filtered."""
    by_frame = {}
    n_amb = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["core_frac"]) <= CORE_FRAC_MIN:
                continue
            fi = int(row["frame"])
            cx, cy, bw, bh = (float(row[k]) for k in ("cx", "cy", "bw", "bh"))
            det = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if any(_iou_norm(det, g) >= GT_IOU_MAX
                   for g in _frame_gt_boxes(gt_seq_dir, fi)):
                n_amb += 1
                continue
            by_frame.setdefault(fi, []).append((cx, cy, bw, bh))
    if n_amb:
        print(f"    ({os.path.basename(path)}: dropped {n_amb} on-person ambiguous dets)")
    return by_frame


@torch.no_grad()
def encode_seq(model, seq_dir, by_frame, S, device, batch=4):
    """Pooled backbone features for all lamp dets of one sequence."""
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    pooled_out = []
    frames = sorted(by_frame)
    for s in range(0, len(frames), batch):
        chunk = frames[s:s + batch]
        imgs, rois = [], []
        for fi in chunk:
            bgr = cv2.imread(img_paths[fi - 1])
            if bgr is None:
                continue
            if bgr.shape[0] != S or bgr.shape[1] != S:
                bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            for cx, cy, bw, bh in by_frame[fi]:
                rois.append([len(imgs) - 1, (cx - bw / 2) * S, (cy - bh / 2) * S,
                             (cx + bw / 2) * S, (cy + bh / 2) * S])
        if not rois:
            continue
        pooled = model.pool(torch.stack(imgs, 0).to(device),
                            torch.tensor(rois, dtype=torch.float32, device=device))
        pooled_out.append(pooled.cpu())
    return torch.cat(pooled_out, 0) if pooled_out else None


def neg_targets(n):
    return {
        "m_glare_analytic": torch.zeros(n),
        "m_valid": torch.zeros(n, dtype=torch.bool),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frame_stride", type=int, default=5,
                    help="detector-debug stride over the tracking sequences")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mine_root = data_path(cfg, "tracking")
    csv_dir = results_dir(cfg, "negmine_tracking")
    out_dir = cache_dir(cfg, "hpsi")
    sp = make_splits()

    # h_psi head weights may not exist yet on the first pass - Perception needs
    # SOME head checkpoint; use the freshly trained step-06 head.
    percep = Perception(weight_path(cfg, "detector"), weight_path(cfg, "hpsi"),
                        device, conf=0.25)
    model = HPsiModel(__import__("ultralytics").YOLO(weight_path(cfg, "detector")).model,
                      tap_index=percep.tap).to(device).eval()

    for split in ("train", "val"):
        seqs = [s for s in sp[split]
                if os.path.isdir(os.path.join(mine_root, s, "images"))]
        print(f"[{split}] {len(seqs)} tracking sequences")
        pooled_all = []
        for i, seq in enumerate(seqs, 1):
            seq_dir = os.path.join(mine_root, seq)
            csv_path = os.path.join(csv_dir, f"{seq}.csv")
            if not os.path.exists(csv_path):
                print(f"  [{i}/{len(seqs)}] {seq}: detector-debug ...", flush=True)
                detector_debug(seq_dir, percep, csv_path, frame_stride=args.frame_stride)
            by_frame = load_csv_rows(csv_path, seq_dir)
            if not by_frame:
                continue
            pooled = encode_seq(model, seq_dir, by_frame, percep.S, device)
            if pooled is not None:
                pooled_all.append(pooled)
                print(f"  [{i}/{len(seqs)}] {seq}: +{pooled.shape[0]} lamp negatives", flush=True)
        if not pooled_all:
            print(f"[{split}] no lamp negatives found")
            continue
        pooled = torch.cat(pooled_all, 0)
        cache = {"pooled": pooled,
                 "meta": {"in_channels": model.backbone.out_channels,
                          "roi_size": model.head.roi_size, "stride": model.stride,
                          "tap": percep.tap, "imgsz": percep.S},
                 "core_frac_min": CORE_FRAC_MIN}
        cache.update(neg_targets(pooled.shape[0]))
        path = os.path.join(out_dir, f"cache_hardneg_{split}.pt")
        torch.save(cache, path)
        print(f"[{split}] saved {pooled.shape[0]} lamp negatives -> {path}")
    print("STEP 07 DONE - now retrain:  python scripts/06_train_hpsi.py --hardneg")


if __name__ == "__main__":
    main()
