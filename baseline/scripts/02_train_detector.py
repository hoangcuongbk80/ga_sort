"""Step 02 - Fine-tune the YOLO person detector at imgsz=1536 (default: YOLO11m).

NOTE: the shipped checkpoint (weights/detector/best.pt) was trained on a separate
detector-training corpus (GlareMOT-Det, see configs/paths.yaml: data.detector), which
is not bundled in this release and will be published separately. This script is
provided for reference / retraining on your own data pointed at by data.detector;
it is independent of the GlareMOT tracking/recovery data used elsewhere in this repo.

Model choice (measured on RTX 5070 Ti, FP16 @1536): 11m runs 21.8 ms/frame - the full pipeline stays realtime at 22 fps source rate; 11x runs 38.1 ms and
breaks realtime. People are small (~30 px), so resolution is kept at the
native 1536 and capacity is traded instead.

Anti-overfit setup:
  * sequence-level split (step 01) - no near-duplicate frame leakage into val;
  * Ultralytics augmentation (mosaic / scale / translate / HSV / fliplr) with
    ``close_mosaic`` so the final epochs converge on undistorted frames;
  * ``degrees=0`` - people in a mine are upright; rotation augment would create
    a fake distribution;
  * early stopping on val mAP (``patience``), best.pt picked by val fitness.

GPU usage: AMP is on by default; ``batch`` and ``workers`` are read from the
config. CPU only feeds the DataLoader (``workers``).

Run:  python scripts/02_train_detector.py [--config configs/paths.yaml]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.config import data_path, load_config, resolve, weight_path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr = cfg.train.detector

    data_yaml = os.path.join(data_path(cfg, "detector"), "dataset.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found - run scripts/01_prepare_dataset.py first")

    from ultralytics import YOLO
    model = YOLO(weight_path(cfg, "pretrained_yolo"))
    project = resolve(os.path.join(cfg.results_dir, "detector_train"))

    raw_batch = float(tr.batch)
    batch = int(raw_batch) if raw_batch >= 1 else (raw_batch if raw_batch > 0 else -1)

    model.train(
        data=data_yaml,
        imgsz=cfg.imgsz,
        epochs=int(tr.epochs),
        patience=int(tr.patience),
        batch=batch,
        workers=int(tr.workers),
        device=args.device,
        cos_lr=True,
        close_mosaic=10,
        degrees=0.0,
        project=project,
        name="detector_1536",
        exist_ok=True,
    )

    best = os.path.join(project, "detector_1536", "weights", "best.pt")
    dst = weight_path(cfg, "detector")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(best, dst)
    print(f"STEP 02 DONE - best checkpoint copied to {dst}")


if __name__ == "__main__":
    main()
