from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from degradation_mot.models.networks import CalibrationNetworks
from degradation_mot.models.training import calibration_loss


def load_samples(path: Path, threshold: float) -> dict[str, np.ndarray]:
    required = {
        "degradation",
        "baseline",
        "detected",
        "normalized_residual",
        "confidence",
        "localization_eligible",
    }
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        archive_threshold = float(archive["confidence_threshold"]) if "confidence_threshold" in archive else 0.10
        if not np.isclose(archive_threshold, threshold, atol=1.0e-8, rtol=0.0):
            raise ValueError(f"{path} requires labels prepared at threshold {threshold}")
        samples = {key: archive[key] for key in required}
    detected = samples["detected"]
    eligible = samples["localization_eligible"].astype(bool)
    confidence = samples["confidence"][eligible]
    if len(detected) == 0 or not np.all(np.isin(detected, [0, 1])):
        raise ValueError(f"{path} requires nonempty binary detection labels")
    if not len(confidence) or not np.all(detected[eligible] == 1) or not np.all(
        (confidence >= threshold) & (confidence < 1.0)
    ):
        raise ValueError(f"{path} requires verified retained localization samples")
    return samples


def as_dataset(
    samples: dict[str, np.ndarray],
    degradation_mean: np.ndarray,
    degradation_std: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
) -> TensorDataset:
    degradation = (samples["degradation"] - degradation_mean) / degradation_std
    baseline = (samples["baseline"] - baseline_mean) / baseline_std
    return TensorDataset(
        torch.as_tensor(degradation, dtype=torch.float32),
        torch.as_tensor(baseline, dtype=torch.float32),
        torch.as_tensor(samples["detected"], dtype=torch.float32),
        torch.as_tensor(samples["normalized_residual"], dtype=torch.float32),
        torch.as_tensor(samples["confidence"], dtype=torch.float32),
        torch.as_tensor(samples["localization_eligible"], dtype=torch.bool),
    )


def evaluate(
    model: CalibrationNetworks, loader: DataLoader, device: torch.device, threshold: float,
) -> dict[str, float]:
    totals = {"detection": 0.0, "localization": 0.0, "confidence": 0.0}
    sample_count = 0
    localization_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            degradation, baseline, detected, residual, confidence, eligible = [
                item.to(device) for item in batch
            ]
            losses = calibration_loss(
                model(degradation, baseline),
                detected,
                residual,
                confidence,
                eligible,
                confidence_threshold=threshold,
            )
            batch_size = len(degradation)
            sample_count += batch_size
            eligible_count = int(eligible.sum())
            localization_count += eligible_count
            totals["detection"] += float(losses.detection) * batch_size
            totals["localization"] += float(losses.localization) * eligible_count
            totals["confidence"] += float(losses.confidence) * eligible_count
    metrics = {
        "detection": totals["detection"] / sample_count,
        "localization": totals["localization"] / localization_count,
        "confidence": totals["confidence"] / localization_count,
    }
    return {**metrics, "total": sum(metrics.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--hidden-layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--hidden-width", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--radial-bins", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--visibility-only", action="store_true", help="Omit photometric features from the detection head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.confidence_threshold < 1.0:
        parser.error("--confidence-threshold must lie in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train = load_samples(args.train, args.confidence_threshold)
    validation = load_samples(args.validation, args.confidence_threshold)
    architecture = {
        "degradation_dim": 11 + args.radial_bins,
        "hidden_dim": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "visibility_only": args.visibility_only,
    }
    for samples in (train, validation):
        if samples["degradation"].shape != (len(samples["detected"]), architecture["degradation_dim"]):
            raise ValueError("descriptor dimensions do not match --radial-bins")
    degradation_mean = train["degradation"].mean(axis=0)
    degradation_std = np.maximum(train["degradation"].std(axis=0), 1.0e-8)
    baseline_mean = train["baseline"].mean(axis=0)
    baseline_std = np.maximum(train["baseline"].std(axis=0), 1.0e-8)
    train_dataset = as_dataset(
        train, degradation_mean, degradation_std, baseline_mean, baseline_std
    )
    validation_dataset = as_dataset(
        validation, degradation_mean, degradation_std, baseline_mean, baseline_std
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device(args.device)
    model = CalibrationNetworks(**architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            degradation, baseline, detected, residual, confidence, eligible = [
                item.to(device) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = calibration_loss(
                model(degradation, baseline),
                detected,
                residual,
                confidence,
                eligible,
                confidence_threshold=args.confidence_threshold,
            )
            losses.total.backward()
            optimizer.step()

        validation_metrics = evaluate(model, validation_loader, device, args.confidence_threshold)
        history.append({"epoch": epoch, **validation_metrics})
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "degradation_mean": torch.as_tensor(degradation_mean),
                    "degradation_std": torch.as_tensor(degradation_std),
                    "baseline_mean": torch.as_tensor(baseline_mean),
                    "baseline_std": torch.as_tensor(baseline_std),
                    "sigma_min": (0.01, 0.01, 0.02, 0.02),
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "validation_loss": best_validation,
                    "confidence_threshold": args.confidence_threshold,
                },
                args.output,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best epoch: {best_epoch}; validation loss: {best_validation:.6f}")


if __name__ == "__main__":
    main()

