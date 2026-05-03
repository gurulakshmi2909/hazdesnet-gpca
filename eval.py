# eval.py — GPCAHazDesNet Evaluation (Regression + Classification)

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HazeDensityDataset
from model import GPCAHazDesNet


# ══════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════

CLASS_NAMES = ["Clear", "Low", "Moderate", "Heavy"]


# ══════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════

def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor):
    mae = torch.mean(torch.abs(preds - targets)).item()
    mse = torch.mean((preds - targets) ** 2).item()
    rmse = mse ** 0.5

    vx = preds - preds.mean()
    vy = targets - targets.mean()
    pearson = (vx * vy).sum() / (
        torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson.item(),
    }


def compute_classification_metrics(preds, targets, num_classes=4):
    preds = preds.cpu()
    targets = targets.cpu()

    acc = (preds == targets).float().mean().item()

    # Confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t.long(), p.long()] += 1

    return acc, cm


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="GPCAHazDesNet Evaluation")

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default=None)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_predictions", type=str, default=None)

    return parser.parse_args()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────
    dataset = HazeDensityDataset(
        csv_file=args.test_csv,
        root_dir=args.root_dir,
        image_size=args.image_size,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────
    model = GPCAHazDesNet(num_classes=len(CLASS_NAMES)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Storage ──────────────────────────────
    all_density_preds = []
    all_density_targets = []

    all_class_preds = []
    all_class_targets = []

    all_paths = []

    # ── Inference ────────────────────────────
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            density_targets = batch["density"].to(device)
            class_targets = batch["class_id"].to(device)

            density_preds, class_logits = model(images)
            class_preds = torch.argmax(class_logits, dim=1)

            all_density_preds.append(density_preds.cpu())
            all_density_targets.append(density_targets.cpu())

            all_class_preds.append(class_preds.cpu())
            all_class_targets.append(class_targets.cpu())

            all_paths.extend(batch["image_path"])

    # ── Concatenate ──────────────────────────
    all_density_preds = torch.cat(all_density_preds)
    all_density_targets = torch.cat(all_density_targets)

    all_class_preds = torch.cat(all_class_preds)
    all_class_targets = torch.cat(all_class_targets)

    # ── Metrics ──────────────────────────────
    reg_metrics = compute_regression_metrics(
        all_density_preds, all_density_targets
    )

    acc, cm = compute_classification_metrics(
        all_class_preds, all_class_targets
    )

    # ── Print results ────────────────────────
    print("\n" + "=" * 50)
    print("📊 Regression Metrics")
    for k, v in reg_metrics.items():
        print(f"{k:<10}: {v:.6f}")

    print("\n🎯 Classification Metrics")
    print(f"Accuracy : {acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm.numpy())

    print("\nPer-class accuracy:")
    for i in range(len(CLASS_NAMES)):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc_i = correct / total if total > 0 else 0.0
        print(f"{CLASS_NAMES[i]:<10}: {acc_i:.4f}")

    print("=" * 50)

    # ── Save predictions ─────────────────────
    if args.save_predictions:
        save_path = Path(args.save_predictions)

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "image_path",
                "target_density",
                "pred_density",
                "target_class",
                "pred_class"
            ])

            for i in range(len(all_paths)):
                writer.writerow([
                    all_paths[i],
                    float(all_density_targets[i]),
                    float(all_density_preds[i]),
                    int(all_class_targets[i]),
                    int(all_class_preds[i]),
                ])

        print(f"\nPredictions saved to: {save_path}")


if __name__ == "__main__":
    main()
