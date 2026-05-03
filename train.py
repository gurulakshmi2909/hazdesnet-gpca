# train.py — GPCAHazDesNet Training 
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HazeDensityDataset
from model import GPCAHazDesNet, HazeLoss


# ══════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mixup_batch(images, densities, class_ids, alpha=0.3):
    """
    Mixup between samples in the same batch.
    Only mixes samples from ADJACENT classes (|class_a - class_b| <= 1)
    to avoid mixing Clear with Heavy which would confuse the model.
    alpha: mixup coefficient — higher = more mixing
    """
    batch_size = images.size(0)
    lam        = np.random.beta(alpha, alpha)
    lam        = max(lam, 1 - lam)   # always keep majority ≥ 0.5

    # Only mix with adjacent class samples
    mixed_images   = images.clone()
    mixed_densities = densities.clone()

    for i in range(batch_size):
        ci = class_ids[i].item()
        # Find adjacent class samples
        adjacent = [j for j in range(batch_size)
                    if j != i and abs(class_ids[j].item() - ci) <= 1]
        if not adjacent:
            continue
        j = np.random.choice(adjacent)
        mixed_images[i]    = lam * images[i] + (1 - lam) * images[j]
        mixed_densities[i] = lam * densities[i] + (1 - lam) * densities[j]
        # Keep original class label (not mixed) — avoids fractional class confusion

    return mixed_images, mixed_densities, class_ids


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor):
    mae     = torch.mean(torch.abs(preds - targets)).item()
    mse     = torch.mean((preds - targets) ** 2).item()
    rmse    = mse ** 0.5
    vx      = preds - preds.mean()
    vy      = targets - targets.mean()
    pearson = (vx * vy).sum() / (
        torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
    )
    return {"mae": mae, "mse": mse, "rmse": rmse, "pearson": pearson.item()}


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    acc   = (preds == targets).float().mean().item()
    return {"accuracy": acc}


def compute_confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    preds = torch.argmax(logits, dim=1)
    cm    = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    torch.save({
        "epoch":                epoch,
        "best_metric":          best_metric,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, save_path)


# ══════════════════════════════════════════════
#  EPOCH RUNNER
# ══════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, scheduler,
              device, train=True, use_mixup=False):
    model.train(train)

    all_density_preds   = []
    all_density_targets = []
    all_class_logits    = []
    all_class_targets   = []
    running_loss        = 0.0

    pbar = tqdm(loader, desc="train" if train else "valid", leave=False)

    for batch in pbar:
        images          = batch["image"].to(device, non_blocking=True)
        density_targets = batch["density"].to(device, non_blocking=True)
        class_targets   = batch["class_id"].to(device, non_blocking=True)

        # Apply Mixup only during training
        if train and use_mixup:
            images, density_targets, class_targets = mixup_batch(
                images, density_targets, class_targets, alpha=0.3
            )

        if train:
            optimizer.zero_grad(set_to_none=True)

            pred_density, pred_class_logits, haze_map, consistency_loss = model(
                images, return_map=True, return_consistency=True
            )

            loss = criterion(
                density_pred     = pred_density,
                density_gt       = density_targets,
                class_logits     = pred_class_logits,
                class_gt         = class_targets,
                consistency_loss = consistency_loss,
                haze_map         = haze_map,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        else:
            with torch.no_grad():
                pred_density, pred_class_logits, haze_map = model(
                    images, return_map=True
                )

                loss = criterion(
                    density_pred     = pred_density,
                    density_gt       = density_targets,
                    class_logits     = pred_class_logits,
                    class_gt         = class_targets,
                    consistency_loss = None,
                    haze_map         = haze_map,
                )

        running_loss += loss.item() * images.size(0)

        all_density_preds.append(pred_density.detach().cpu())
        all_density_targets.append(density_targets.detach().cpu())
        all_class_logits.append(pred_class_logits.detach().cpu())
        all_class_targets.append(class_targets.detach().cpu())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    if train:
        scheduler.step()

    all_density_preds   = torch.cat(all_density_preds)
    all_density_targets = torch.cat(all_density_targets)
    all_class_logits    = torch.cat(all_class_logits)
    all_class_targets   = torch.cat(all_class_targets)

    reg_metrics = compute_regression_metrics(all_density_preds, all_density_targets)
    cls_metrics = compute_classification_metrics(all_class_logits, all_class_targets)

    return {
        **reg_metrics,
        **cls_metrics,
        "loss": running_loss / len(loader.dataset),
    }


def evaluate_for_confusion_matrix(model, loader, device, num_classes):
    model.eval()
    all_logits  = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images        = batch["image"].to(device, non_blocking=True)
            class_targets = batch["class_id"].to(device, non_blocking=True)
            _, logits, _  = model(images, return_map=True)
            all_logits.append(logits.cpu())
            all_targets.append(class_targets.cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


# ══════════════════════════════════════════════
#  ARGS
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",      type=str,   required=True)
    parser.add_argument("--val_csv",        type=str,   required=True)
    parser.add_argument("--test_csv",       type=str,   required=True)
    parser.add_argument("--root_dir",       type=str,   default=None)
    parser.add_argument("--image_size",     type=int,   default=224)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--epochs",         type=int,   default=80)      # up from 50
    parser.add_argument("--warmup_epochs",  type=int,   default=5)       # LR warmup
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_dir",       type=str,   default="checkpoints")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num_classes",    type=int,   default=4)
    parser.add_argument("--reg_weight",     type=float, default=1.0)
    parser.add_argument("--cls_weight",     type=float, default=1.0)
    parser.add_argument("--cons_weight",    type=float, default=0.2)
    parser.add_argument("--smooth_weight",  type=float, default=0.05)
    parser.add_argument("--use_mixup",      action="store_true", default=True)
    parser.add_argument("--patience",       type=int,   default=15)      # early stopping
    return parser.parse_args()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    args   = parse_args()
    set_seed(args.seed)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASS_NAMES = {0: "Clear", 1: "Low", 2: "Moderate", 3: "Heavy"}
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Datasets ─────────────────────────────
    train_ds = HazeDensityDataset(csv_file=args.train_csv, root_dir=args.root_dir,
                                   image_size=args.image_size, train=True)
    val_ds   = HazeDensityDataset(csv_file=args.val_csv,   root_dir=args.root_dir,
                                   image_size=args.image_size, train=False)
    test_ds  = HazeDensityDataset(csv_file=args.test_csv,  root_dir=args.root_dir,
                                   image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    # ── Sanity check ─────────────────────────
    batch = next(iter(train_loader))
    print("images shape :", batch["image"].shape)
    print("density shape:", batch["density"].shape)
    print("class_id shape:", batch["class_id"].shape)
    print("images dtype :", batch["image"].dtype)
    print("density dtype:", batch["density"].dtype)
    print("class_id dtype:", batch["class_id"].dtype)
    print("images min/max:", batch["image"].min().item(), batch["image"].max().item())
    print("sample paths :", batch["image_path"][:3])

    # ── Class counts ─────────────────────────
    train_df     = pd.read_csv(args.train_csv)
    class_counts = train_df["class_id"].value_counts().sort_index()
    print(f"\nClass counts: {dict(class_counts)}")

    missing_ids = [i for i in range(args.num_classes) if i not in class_counts.index]
    if missing_ids:
        raise ValueError(f"Missing class_id values in train CSV: {missing_ids}")

    # ── Model ────────────────────────────────
    model = GPCAHazDesNet(num_classes=args.num_classes).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss ─────────────────────────────────
    # Clear=1.0, Low=1.0, Moderate=1.2, Heavy=2.0
    class_weights = torch.tensor([1.0, 1.0, 1.2, 2.0], dtype=torch.float32)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = HazeLoss(
        reg_weight    = args.reg_weight,
        cls_weight    = args.cls_weight,
        cons_weight   = args.cons_weight,
        smooth_weight = args.smooth_weight,
        class_weights = class_weights,
        focal_gamma   = 2.0,
        label_smoothing = 0.1,
    ).to(device)

    # ── Optimizer ────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Scheduler: Linear warmup → Cosine decay ──
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=args.warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer,
                                  T_max=args.epochs - args.warmup_epochs,
                                  eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[args.warmup_epochs])

    # ── Training loop ─────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'='*65}")
    print(f"Training for {args.epochs} epochs | Warmup: {args.warmup_epochs} | "
          f"Mixup: {args.use_mixup} | Patience: {args.patience}")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, train=True, use_mixup=args.use_mixup
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, scheduler,
            device, train=False, use_mixup=False
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_pearson={val_metrics['pearson']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save latest
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc,
                        str(Path(args.save_dir) / "latest.pt"))

        # Save best on val_acc (more meaningful than val_loss for 90% target)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc   = val_metrics["accuracy"]
            patience_count = 0
            best_path      = str(Path(args.save_dir) / "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, best_path)
            print(f"  ✅ New best val_acc={best_val_acc:.4f} → saved {best_path}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Test evaluation ───────────────────────
    print(f"\n{'='*65}")
    print("Running TEST evaluation on best model...")
    print(f"{'='*65}")

    best_path  = str(Path(args.save_dir) / "best.pt")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, scheduler,
        device, train=False, use_mixup=False
    )

    print(f"\nTEST RESULTS:")
    print(f"  Test Loss    : {test_metrics['loss']:.4f}")
    print(f"  Test MAE     : {test_metrics['mae']:.4f}")
    print(f"  Test MSE     : {test_metrics['mse']:.4f}")
    print(f"  Test RMSE    : {test_metrics['rmse']:.4f}")
    print(f"  Test Pearson : {test_metrics['pearson']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

    all_logits, all_targets = evaluate_for_confusion_matrix(
        model, test_loader, device, args.num_classes
    )
    cm = compute_confusion_matrix(all_logits, all_targets, args.num_classes)

    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nPer-class accuracy:")
    for i in range(args.num_classes):
        correct = cm[i, i].item()
        total   = cm[i].sum().item()
        acc     = correct / total if total > 0 else 0.0
        print(f"  [{i}] {CLASS_NAMES.get(i, '?'):10s}: {acc:.4f}  ({correct}/{total})")

    with open(Path(args.save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved → {Path(args.save_dir) / 'history.json'}")


if __name__ == "__main__":
    main()
