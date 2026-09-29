"""HOTA / IDF1 / MOTA evaluation of the tracker via TrackEval (MotChallenge2DBox).

Runs the full online tracker (no video rendering), writes MOT txt per
sequence, arranges a TrackEval workspace under ``results/eval/``, and reports
HOTA / DetA / AssA / IDF1 / MOTA / IDSW / Recall / Precision per sequence and
combined, per split.

TrackEval is imported from the installed package
(``pip install git+https://github.com/JonathonLuiten/TrackEval.git``); the
``TRACKEVAL_ROOT`` environment variable can point at a checkout instead.
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from ..config import data_path, load_config, results_dir, weight_path
from ..data.splits import make_splits
from ..tracking.tracker import GlareTrack, Perception, norm_box_to_xyxy

if os.environ.get("TRACKEVAL_ROOT"):
    sys.path.insert(0, os.environ["TRACKEVAL_ROOT"])


def track_sequence(seq_dir, percep, out_txt, max_frames=0):
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    if max_frames:
        img_paths = img_paths[:max_frames]
    if not img_paths:
        raise FileNotFoundError(f"no frames in {seq_dir}")
    tracker = GlareTrack()
    if hasattr(percep, "reset_scene"):
        percep.reset_scene()                 # per-sequence fixed-camera scene prior
    H, W = cv2.imread(img_paths[0]).shape[:2]
    rows = []
    t0 = time.time()

    # prefetch thread: decode the next frames while the GPU works on this one
    rq: "queue.Queue" = queue.Queue(maxsize=8)

    def _reader():
        for p in img_paths:
            rq.put(cv2.imread(p))
        rq.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    fi = 0
    while True:
        frame = rq.get()
        if frame is None:
            break
        fi += 1
        dets = percep(frame)
        tracks = tracker.step(dets, W, H, fi, core_fn=percep.core_mask,
                              flare=getattr(percep, "flare_frac", 0.0))
        for t in tracks:
            x1, y1, x2, y2 = norm_box_to_xyxy(t.out_box(), W, H)
            rows.append(
                f"{fi},{t.id},{max(0.0, x1):.2f},{max(0.0, y1):.2f},"
                f"{max(1.0, x2 - x1):.2f},{max(1.0, y2 - y1):.2f},"
                f"{max(t.m, 0.001):.4f},-1,-1,-1")
    fps = len(img_paths) / max(time.time() - t0, 1e-6)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    return len(img_paths), fps


def build_gt_workspace(seqs, trk_root, gt_root, max_frames=0):
    """TrackEval layout: <gt_root>/<seq>/gt/gt.txt."""
    seq_lengths = {}
    for seq in seqs:
        src = os.path.join(trk_root, seq, "mot", "gt", "gt.txt")
        dst_dir = os.path.join(gt_root, seq, "gt")
        os.makedirs(dst_dir, exist_ok=True)
        n_imgs = len(glob.glob(os.path.join(trk_root, seq, "images", "*.jpg")))
        if max_frames:
            n_imgs = min(n_imgs, max_frames)
        with open(src, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if max_frames:
            lines = [ln for ln in lines if int(ln.split(",")[0]) <= max_frames]
        with open(os.path.join(dst_dir, "gt.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        seq_lengths[seq] = n_imgs
    return seq_lengths


def run_trackeval(gt_root, trackers_root, tracker_name, seq_lengths, log_dir):
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "LOG_ON_ERROR": os.path.join(log_dir, "trackeval_error.log"),
        "OUTPUT_SUMMARY": False, "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
    })
    ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_config.update({
        "GT_FOLDER": gt_root, "TRACKERS_FOLDER": trackers_root,
        "TRACKERS_TO_EVAL": [tracker_name], "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "DO_PREPROC": False,
        "PRINT_CONFIG": False, "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "",
        "SEQ_INFO": seq_lengths, "SKIP_SPLIT_FOL": True,
    })
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_config)
    metrics = [
        trackeval.metrics.HOTA({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    ]
    results, _ = evaluator.evaluate([dataset], metrics)
    res = results["MotChallenge2DBox"][tracker_name]
    out = {}
    for seq, seq_res in res.items():
        ped = seq_res["pedestrian"]
        out[seq] = {
            "HOTA": float(np.mean(ped["HOTA"]["HOTA"])),
            "DetA": float(np.mean(ped["HOTA"]["DetA"])),
            "AssA": float(np.mean(ped["HOTA"]["AssA"])),
            "IDF1": float(ped["Identity"]["IDF1"]),
            "MOTA": float(ped["CLEAR"]["MOTA"]),
            "IDSW": int(ped["CLEAR"]["IDSW"]),
            "FP": int(ped["CLEAR"]["CLR_FP"]),
            "FN": int(ped["CLEAR"]["CLR_FN"]),
            "Rcll": float(ped["CLEAR"]["CLR_Re"]),
            "Prcn": float(ped["CLEAR"]["CLR_Pr"]),
        }
    return out


def print_table(title, res):
    """Print and return the per-sequence + combined metric table (HOTA / DetA /
    AssA / IDF1 / MOTA + IDSW / Recall / Precision / FP / FN)."""
    hdr = (f"{'sequence':<34}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'IDF1':>7}{'MOTA':>7}"
           f"{'IDSW':>6}{'Rcll':>7}{'Prcn':>7}{'FP':>7}{'FN':>8}")
    lines = [f"\n===== {title} =====", hdr, "-" * len(hdr)]
    order = [s for s in sorted(res) if s != "COMBINED_SEQ"] + \
            (["COMBINED_SEQ"] if "COMBINED_SEQ" in res else [])
    for seq in order:
        r = res[seq]
        name = "COMBINED" if seq == "COMBINED_SEQ" else seq
        if seq == "COMBINED_SEQ":
            lines.append("-" * len(hdr))
        lines.append(f"{name:<34}{r['HOTA'] * 100:>7.2f}{r['DetA'] * 100:>7.2f}"
                     f"{r['AssA'] * 100:>7.2f}{r['IDF1'] * 100:>7.2f}{r['MOTA'] * 100:>7.2f}"
                     f"{r['IDSW']:>6}{r['Rcll'] * 100:>7.2f}{r['Prcn'] * 100:>7.2f}"
                     f"{r['FP']:>7}{r['FN']:>8}")
    out = "\n".join(lines)
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test",
                    choices=["val", "test", "all", "heldout"],
                    help="'all' = every sequence, reported per split + overall; "
                         "'heldout' = val+test (val/test/combined tables)")
    ap.add_argument("--tracker_name", default="glaretrack")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--skip_track", action="store_true",
                    help="reuse existing MOT txt, only run TrackEval")
    ap.add_argument("--resume", action="store_true",
                    help="skip sequences whose MOT txt already exists")
    ap.add_argument("--data_root", default=None,
                    help="image+GT dataset root (default data.tracking). Point at the "
                         "glare bench to get full-scene metrics on the recovery twin.")
    ap.add_argument("--results_subdir", default="eval",
                    help="results/<subdir> workspace (use 'eval_recovery' with "
                         "--skip_track to score the recovery tracker txt).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = make_splits()
    if args.split == "all":
        seqs = sp["train"] + sp["val"] + sp["test"]
    elif args.split == "heldout":
        seqs = sp["val"] + sp["test"]
    else:
        seqs = sp[args.split]
    trk_root = args.data_root or data_path(cfg, "tracking")
    seqs = [s for s in seqs if os.path.isdir(os.path.join(trk_root, s))]
    print(f"{args.split} split: {len(seqs)} sequences | device={device} | "
          f"data_root={trk_root}")

    eval_root = results_dir(cfg, args.results_subdir)
    gt_root = os.path.join(eval_root, "gt")
    trackers_root = os.path.join(eval_root, "trackers")
    data_dir = os.path.join(trackers_root, args.tracker_name, "data")
    seq_lengths = build_gt_workspace(seqs, trk_root, gt_root, max_frames=args.max_frames)

    total_frames, total_time = 0, 0.0
    if not args.skip_track:
        det_conf = args.conf
        if os.environ.get("GMOT_LOWCONF_RESCUE", "0") not in ("0", "off", "false", ""):
            # Stage 1 low-conf rescue: lower the detector floor so 0.05-0.25 boxes reach
            # the tracker's sustain-only BYTE-lite tier (birth gate unchanged in tracker.py).
            det_conf = float(os.environ.get("GMOT_RESCUE_CONF", "0.05"))
            print(f"[lowconf-rescue] detector floor -> {det_conf}", flush=True)
        percep = Perception(weight_path(cfg, "detector"), weight_path(cfg, "hpsi"),
                            device, det_conf)
        for i, seq in enumerate(seqs, 1):
            out_txt = os.path.join(data_dir, f"{seq}.txt")
            if args.resume and os.path.exists(out_txt):
                print(f"[{i}/{len(seqs)}] {seq}: resume, skip", flush=True)
                continue
            n, fps = track_sequence(os.path.join(trk_root, seq), percep,
                                    out_txt, max_frames=args.max_frames)
            total_frames += n
            total_time += n / max(fps, 1e-6)
            print(f"[{i}/{len(seqs)}] {seq}: {n} frames @ {fps:.1f} FPS", flush=True)

    fps_line = ""
    if total_time > 0:
        fps_line = (f"\nEnd-to-end speed (detector + h_psi + tracker, full pipeline): "
                    f"{total_frames / total_time:.1f} FPS "
                    f"over {total_frames} frames | device={device}")

    print("\nrunning TrackEval ...", flush=True)
    report = []
    if args.split in ("all", "heldout"):
        if args.split == "all":
            groups = [("TRAIN", sp["train"]), ("VAL", sp["val"]),
                      ("TEST (real glare, held-out)", sp["test"]), ("ALL 116", seqs)]
        else:
            groups = [("VAL", sp["val"]), ("TEST", sp["test"]),
                      ("HELDOUT (val+test)", seqs)]
        for name, gseqs in groups:
            gseqs = [s for s in gseqs if s in seq_lengths]
            res = run_trackeval(gt_root, trackers_root, args.tracker_name,
                                {s: seq_lengths[s] for s in gseqs}, eval_root)
            report.append(print_table(name, res))
    else:
        res = run_trackeval(gt_root, trackers_root, args.tracker_name, seq_lengths, eval_root)
        report.append(print_table(args.split.upper(), res))
    if fps_line:
        print(fps_line)
        report.append(fps_line)

    # persist the tables so the README / a later session can read the numbers
    out_txt = os.path.join(eval_root, f"benchmark_{args.tracker_name}_{args.split}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nbenchmark report saved -> {out_txt}")


if __name__ == "__main__":
    main()
