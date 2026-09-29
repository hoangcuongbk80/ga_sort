"""Step 05 - Precompute frozen-backbone ROI features + analytic targets for h_psi.

Because the detector backbone is FROZEN, its ROI features never change across
epochs, so we run it ONCE here and cache pooled features + analytic targets.
Training (step 06) then reads straight from this cache - no per-epoch image
I/O or backbone forward, so the tiny head trains in seconds per epoch.

Data recipe: frames from the 22 fps TRACKING dataset (stride-sampled, the
project's frozen sequence split, TEST excluded), with ``glare_mix`` of q-seq
samples swapped for their synthetic-glare twins - so ``m_glare`` sees real
burnt bodies and ``present`` sees lamps over people under heavy glare.

CPU (DataLoader workers: JPEG decode + cv2 analytic) and GPU (backbone
forward) run concurrently.

Run:  python scripts/05_preprocess_hpsi.py [--workers 12]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ultralytics"))

import torch
from torch.utils.data import DataLoader

from glaremot.config import cache_dir, data_path, load_config, weight_path
from glaremot.data.hpsi_dataset import HPsiDataset, build_image_list, collate
from glaremot.data.splits import make_splits
from glaremot.perception.h_psi import HPsiModel

KEYS = ["m_glare_analytic", "m_valid"]


@torch.no_grad()
def run_split(model, image_list, imgsz, batch, workers, device):
    ds = HPsiDataset(image_list, imgsz=imgsz)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate,
                    pin_memory=True, persistent_workers=workers > 0)
    pooled_all = []
    cat = {k: [] for k in KEYS}
    n_img = 0
    for bi, b in enumerate(dl):
        if b["rois"].shape[0] == 0:
            n_img += b["img"].shape[0]
            continue
        pooled = model.pool(b["img"].to(device, non_blocking=True),
                            b["rois"].to(device, non_blocking=True))
        pooled_all.append(pooled.cpu())
        for k in KEYS:
            cat[k].append(b[k])
        n_img += b["img"].shape[0]
        if bi % 20 == 0:
            print(f"    batch {bi}: imgs~{n_img} rois~{sum(p.shape[0] for p in pooled_all)}",
                  flush=True)
    out = {"pooled": torch.cat(pooled_all, 0)}
    for k in KEYS:
        out[k] = torch.cat(cat[k], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tap", type=int, default=16)
    ap.add_argument("--splits", nargs="+", choices=("train", "val"),
                    default=["train", "val"])
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hp = cfg.train.hpsi

    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HPsiModel(YOLO(weight_path(cfg, "detector")).model,
                      tap_index=args.tap).to(device).eval()
    print(f"backbone tap={args.tap} C={model.backbone.out_channels} "
          f"stride={model.stride} device={device}")
    out_dir = cache_dir(cfg, "hpsi")

    sp = make_splits()
    tracking = data_path(cfg, "tracking")
    # Local D:\bruh run: train h_psi only on the real tracking dataset.
    # Do not swap any sampled frame to the synthetic-glare twin.
    glare = tracking
    meta = {"in_channels": model.backbone.out_channels, "roi_size": model.head.roi_size,
            "stride": model.stride, "tap": args.tap, "imgsz": cfg.imgsz}
    for split in args.splits:
        path = os.path.join(out_dir, f"cache_{split}.pt")
        if args.skip_existing and os.path.exists(path):
            print(f"[{split}] skip existing cache: {path}", flush=True)
            continue
        lst = build_image_list(tracking, glare, sp[split],
                               frame_stride=int(hp.frame_stride),
                               glare_mix=0.0, seed=0)
        print(f"[{split}] {len(lst)} sampled frames ({len(sp[split])} seqs) ...", flush=True)
        cache = run_split(model, lst, cfg.imgsz, args.batch, args.workers, device)
        cache["meta"] = meta
        torch.save(cache, path)
        mv = cache["m_valid"]
        print(f"[{split}] saved {path}  ROIs={mv.numel()} "
              f"person={int(mv.sum())} env_neg={int((~mv).sum())}", flush=True)
    print("STEP 05 DONE ->", out_dir)


if __name__ == "__main__":
    main()
