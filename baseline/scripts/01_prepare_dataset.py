"""Step 01 - Normalize detector labels to 5-col HBB and build the train/val split.

The detector dataset may contain a mix of label formats per row:
  * 5-col  ``cls cx cy w h``                       (kept as-is)
  * 9-col  ``cls x1 y1 x2 y2 x3 y3 x4 y4`` polygon  (converted to the enclosing
    axis-aligned box - exact when the polygon is already axis-aligned)

This script rewrites every label file in place to clean 5-col HBB (a one-time
backup of the current labels is kept in ``labels_backup_pre_5col/``), then
writes ``train.txt`` / ``val.txt`` / ``dataset.yaml`` using the FROZEN
sequence-level split (glaremot/data/splits.json). The 11 real-glare TEST
sequences are excluded from both lists so the tracking benchmark stays
leak-free end to end.

Run:  python scripts/01_prepare_dataset.py [--config configs/paths.yaml]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.config import data_path, load_config
from glaremot.data.splits import make_splits


def convert_row(tokens):
    """One label row -> 5-col HBB string, or None if malformed."""
    if len(tokens) == 5:
        return " ".join(tokens)
    if len(tokens) == 9:
        cls = tokens[0]
        xs = [float(v) for v in tokens[1:9:2]]
        ys = [float(v) for v in tokens[2:9:2]]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"
    return None


def normalize_labels(root):
    lab_root = os.path.join(root, "labels")
    bak_root = os.path.join(root, "labels_backup_pre_5col")
    if not os.path.exists(bak_root):
        print(f"backing up labels -> {bak_root}")
        shutil.copytree(lab_root, bak_root)
    n_files = n_conv = n_bad = 0
    for seq in sorted(os.listdir(lab_root)):
        seq_dir = os.path.join(lab_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for fn in os.listdir(seq_dir):
            if not fn.endswith(".txt") or fn == "classes.txt":
                continue
            p = os.path.join(seq_dir, fn)
            with open(p, encoding="utf-8") as f:
                rows = [ln.split() for ln in f if ln.strip()]
            out, changed = [], False
            for t in rows:
                r = convert_row(t)
                if r is None:
                    n_bad += 1
                    continue
                if len(t) != 5:
                    changed = True
                out.append(r)
            if changed or len(out) != len(rows):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("\n".join(out) + ("\n" if out else ""))
                n_conv += 1
            n_files += 1
    print(f"labels: {n_files} files scanned, {n_conv} rewritten to 5-col, {n_bad} malformed rows dropped")


def write_split(root):
    sp = make_splits()
    img_root = os.path.join(root, "images")
    seqs_avail = set(os.listdir(img_root))
    lists = {}
    for split in ("train", "val"):
        paths = []
        for seq in sp[split]:
            seq_dir = os.path.join(img_root, seq)
            if seq not in seqs_avail:
                print(f"  [{split}] WARNING: sequence missing in detector dataset: {seq}")
                continue
            paths += [os.path.join(seq_dir, f) for f in sorted(os.listdir(seq_dir))
                      if f.lower().endswith((".jpg", ".png"))]
        lists[split] = paths
        with open(os.path.join(root, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(paths) + "\n")
        print(f"  {split}: {len(sp[split])} seqs -> {len(paths)} images")
    n_test_excluded = sum(1 for s in sp["test"] if s in seqs_avail)
    print(f"  test: {n_test_excluded} real-glare seqs EXCLUDED (held out for the MOT benchmark)")

    yaml_path = os.path.join(root, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {root.replace(os.sep, '/')}\n"
            "train: train.txt\n"
            "val: val.txt\n"
            "names:\n  0: person\nnc: 1\n"
        )
    print(f"wrote {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = data_path(cfg, "detector")
    print(f"detector dataset: {root}")
    normalize_labels(root)
    write_split(root)
    print("STEP 01 DONE")


if __name__ == "__main__":
    main()
