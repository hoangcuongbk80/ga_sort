"""Dataset for h_psi training - analytic m_glare target + free negative mining.

Frames come from the 22 fps TRACKING dataset (per-frame 5-col YOLO labels),
sampled with a stride (consecutive 22 fps frames are near-duplicates), and for
the q-sequences a configurable fraction of samples is swapped for the
SYNTHETIC-GLARE twin of the same frame (same labels, burnt pixels) - so the
``m_glare`` regression sees many genuinely burnt bodies.

Per frame, ROI-level items WITHOUT any manual annotation:

  PERSON ROIs (one per labelled person box):
    m_glare = m_glare_analytic (from the image) -> regression target, m_valid = 1

  MINED ENVIRONMENT-LIGHT NEGATIVES:
    every compact saturated core whose centroid is NOT in any person's head
    band (floor reflection / ghost / streak / wall) -> m_valid = 0.

The image is resized to SxS (frames are square) so normalized labels map to
pixels by *S. __getitem__ runs the cv2 analytic on CPU (parallelised by
DataLoader workers); the GPU only runs the frozen backbone in scripts/05.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..perception.analytic import head_roi_box, m_glare_analytic
from ..photometric import saturated_core_mask


def _read_label_boxes(path):
    """5-col YOLO label file -> list of normalized (cx, cy, w, h)."""
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 5:
            out.append([float(v) for v in p[1:5]])
    return out


def build_image_list(tracking_root, glare_root, seqs, frame_stride=5,
                     glare_mix=0.5, seed=0):
    """Sampled frame paths for the given sequences. For q-sequences with a
    glare twin, ``glare_mix`` of the sampled frames use the twin path."""
    rng = np.random.default_rng(seed)
    items = []
    for seq in seqs:
        img_dir = os.path.join(tracking_root, seq, "images")
        if not os.path.isdir(img_dir):
            continue
        frames = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        frames = frames[::frame_stride]
        glare_dir = os.path.join(glare_root, seq, "images")
        has_glare = os.path.isdir(glare_dir)
        for f in frames:
            if has_glare and rng.random() < glare_mix:
                items.append(os.path.join(glare_dir, f))
            else:
                items.append(os.path.join(img_dir, f))
    return items


def _img_to_label(p):
    return os.path.splitext(p.replace(os.sep + "images" + os.sep,
                                      os.sep + "labels" + os.sep))[0] + ".txt"


class HPsiDataset(Dataset):
    def __init__(self, image_list, imgsz=1536, max_neg=8, min_neg_area=12):
        self.images = list(image_list)
        self.S = imgsz
        self.max_neg = max_neg
        self.min_neg_area = min_neg_area

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        S = self.S
        img_path = self.images[i]
        bgr = cv2.imread(img_path)
        if bgr is None:
            return self._empty()
        if bgr.shape[0] != S or bgr.shape[1] != S:
            bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
        core = saturated_core_mask(bgr)

        boxes_norm = _read_label_boxes(_img_to_label(img_path))
        rois, mglare, mvalid = [], [], []
        head_rois = []                                   # for negative mining

        for cx, cy, w, h in boxes_norm:
            x1, y1 = (cx - w / 2) * S, (cy - h / 2) * S
            x2, y2 = (cx + w / 2) * S, (cy + h / 2) * S
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            box = (x1, y1, x2, y2)
            mg = m_glare_analytic(bgr, box)
            hr = head_roi_box(box)
            head_rois.append(hr)

            rois.append(list(box))
            mglare.append(float(mg)); mvalid.append(True)

        # ---- mine environment-light negatives: cores not in any head band ----
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(
            (core > 0).astype(np.uint8), 8)
        negs = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < self.min_neg_area:
                continue
            cxk, cyk = cents[k]
            in_head = any(hr[0] <= cxk <= hr[2] and hr[1] <= cyk <= hr[3]
                          for hr in head_rois)
            if in_head:
                continue
            x = stats[k, cv2.CC_STAT_LEFT]; y = stats[k, cv2.CC_STAT_TOP]
            ww = stats[k, cv2.CC_STAT_WIDTH]; hh = stats[k, cv2.CC_STAT_HEIGHT]
            pad = 0.5 * max(ww, hh)
            negs.append((area, [x - pad, y - pad, x + ww + pad, y + hh + pad]))
        negs.sort(key=lambda t: -t[0])
        for _, box in negs[:self.max_neg]:
            rois.append(box)
            mglare.append(0.0); mvalid.append(False)

        if not rois:
            return self._empty()
        return {
            "img": self._img_tensor(bgr),
            "rois": torch.tensor(rois, dtype=torch.float32),
            "m_glare_analytic": torch.tensor(mglare, dtype=torch.float32),
            "m_valid": torch.tensor(mvalid, dtype=torch.bool),
        }

    def _img_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _empty(self):
        bgr = np.zeros((self.S, self.S, 3), np.uint8)
        return {
            "img": self._img_tensor(bgr),
            "rois": torch.zeros((0, 4), dtype=torch.float32),
            "m_glare_analytic": torch.zeros(0),
            "m_valid": torch.zeros(0, dtype=torch.bool),
        }


def collate(batch):
    """Stack images; concat ROIs with a batch index column; concat targets."""
    imgs = torch.stack([b["img"] for b in batch], 0)
    keys = ["m_glare_analytic", "m_valid"]
    rois = []
    cat = {k: [] for k in keys}
    for bi, b in enumerate(batch):
        n = b["rois"].shape[0]
        if n:
            bidx = torch.full((n, 1), float(bi))
            rois.append(torch.cat([bidx, b["rois"]], 1))
            for k in keys:
                cat[k].append(b[k])
    rois = torch.cat(rois, 0) if rois else torch.zeros((0, 5))
    out = {"img": imgs, "rois": rois}
    for k in keys:
        out[k] = torch.cat(cat[k], 0) if cat[k] else torch.zeros(
            0, dtype=torch.bool if k == "m_valid" else torch.float32)
    return out
