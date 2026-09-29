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


def load_samples(path: Path) -> dict[str, np.ndarray]:
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
        return {key: archive[key] for key in required}


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
    model: CalibrationNetworks, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    totals = {"total": 0.0, "detection": 0.0, "localization": 0.0, "confidence": 0.0}
    sample_count = 0
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
            )
            batch_size = len(degradation)
            sample_count += batch_size
            for key in totals:
                totals[key] += float(getattr(losses, key)) * batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train = load_samples(args.train)
    validation = load_samples(args.validation)
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
    model = CalibrationNetworks().to(device)
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
            )
            losses.total.backward()
            optimizer.step()

        validation_metrics = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, **validation_metrics})
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": {},
                    "degradation_mean": torch.as_tensor(degradation_mean),
                    "degradation_std": torch.as_tensor(degradation_std),
                    "baseline_mean": torch.as_tensor(baseline_mean),
                    "baseline_std": torch.as_tensor(baseline_std),
                    "sigma_min": (0.01, 0.01, 0.02, 0.02),
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "validation_loss": best_validation,
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

