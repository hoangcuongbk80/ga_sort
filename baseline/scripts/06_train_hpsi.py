"""Step 06 - Train the m-only h_psi head on the cached frozen-backbone ROI features.

Run scripts/05 first. The head is tiny and the cache lives on the GPU, so this
trains in seconds per epoch. The head has a single output, the observability
``m_glare``: it regresses the analytic ``m_glare`` target (computed from the image,
see perception/analytic.py) with a smooth-L1 loss on person ROIs, and reports the
``m_glare`` MAE on the val split. Checkpoint selection keeps the lowest val MAE.

The ``--lambda_present``/``--lambda_sigma`` and ``--hardneg`` options are vestigial:
the present/sigma auxiliary heads and the mined lamp-negative tier they used were
removed, so they have no effect on the m-only head (kept only for CLI compatibility).

Run:  python scripts/06_train_hpsi.py
"""
import argparse
import glob as _glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from glaremot.config import cache_dir, load_config, weight_path
from glaremot.perception.h_psi import HybridVisibilityHead, Stage1HeadLoss

KEYS = ("m_glare_analytic", "m_valid")


def load_cache(path, device):
    c = torch.load(path, weights_only=False)
    t = {"pooled": c["pooled"].to(device)}
    for k in KEYS:
        t[k] = c[k].to(device)
    return t, c["meta"]


def concat_caches(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


@torch.no_grad()
def evaluate(head, val):
    head.eval()
    out = head(val["pooled"])
    mv = val["m_valid"]
    mae = float((out["m_glare"][mv] - val["m_glare_analytic"][mv]).abs().mean()) if mv.any() else 0.0
    return {"m_MAE": mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_present", type=float, default=2.0)
    ap.add_argument("--lambda_m", type=float, default=1.0)
    ap.add_argument("--lambda_sigma", type=float, default=0.5)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--hardneg", action="store_true",
                    help="concat cache_hardneg_*.pt (step 07) into train/val")
    ap.add_argument("--hardneg_repeat", type=int, default=8,
                    help="oversample the mined lamp negatives (they are dilute vs "
                         "env negatives, so give them real gradient share)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs or int(cfg.train.hpsi.epochs)
    cdir = cache_dir(cfg, "hpsi")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, meta = load_cache(os.path.join(cdir, "cache_train.pt"), device)
    val, _ = load_cache(os.path.join(cdir, "cache_val.pt"), device)
    n_lamp_val = 0
    if args.hardneg:
        tr_paths = sorted(_glob.glob(os.path.join(cdir, "cache_hardneg_train*.pt")))
        if not tr_paths:
            raise FileNotFoundError("no cache_hardneg_train*.pt - run scripts/07 first")
        for trp in tr_paths:
            hn_tr, hn_meta = load_cache(trp, device)
            assert hn_meta["tap"] == meta["tap"] and hn_meta["imgsz"] == meta["imgsz"]
            for _ in range(max(1, args.hardneg_repeat)):
                train = concat_caches(train, hn_tr)
            vap = trp.replace("cache_hardneg_train", "cache_hardneg_val")
            n_va = 0
            if os.path.exists(vap):
                hn_va, _ = load_cache(vap, device)
                val = concat_caches(val, hn_va)   # lamp rows stay LAST
                n_va = hn_va["pooled"].shape[0]
                n_lamp_val += n_va
            print(f"hard negatives [{os.path.basename(trp)}]: "
                  f"+{hn_tr['pooled'].shape[0]} train x{args.hardneg_repeat}, +{n_va} val")
    n = train["pooled"].shape[0]
    print(f"train ROIs={n} | val ROIs={val['pooled'].shape[0]} | "
          f"C={meta['in_channels']} device={device}")

    head = HybridVisibilityHead(meta["in_channels"], roi_size=meta["roi_size"]).to(device)
    crit = Stage1HeadLoss(args.lambda_present, args.lambda_m, args.lambda_sigma,
                          focal_alpha=args.focal_alpha)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_score, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            sub = {"pooled": train["pooled"][idx]}
            for k in KEYS:
                sub[k] = train[k][idx]
            loss = crit(head(sub["pooled"]), sub)["total"]
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            m = evaluate(head, val)
            print(f"ep {ep:3d}  m_MAE={m['m_MAE']:.3f}", flush=True)
            score = -m["m_MAE"]   # lower regression error is better
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    out_path = weight_path(cfg, "hpsi")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": best_state, "meta": meta}, out_path)
    print(f"BEST score (-m_MAE)={best_score:.3f}  saved -> {out_path}")


if __name__ == "__main__":
    main()
