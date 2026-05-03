# predict.py — GPCAHazDesNet Inference (Single + Batch)
#
# Features:
#   - Single image OR folder inference
#   - Haze density (0–1)
#   - Class prediction + full probabilities
#   - Optional haze map saving
#   - Optional CSV export
#
# Usage:
#   python predict.py --image img.png --checkpoint checkpoints/best.pt
#   python predict.py --image_dir images/ --checkpoint checkpoints/best.pt --save_csv results.csv
#   python predict.py --image img.png --checkpoint checkpoints/best.pt --save_map map.png


import argparse
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

from model import GPCAHazDesNet


# ══════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════

CLASS_NAMES = ["Clear", "Low", "Moderate", "Heavy"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Optional: density → class mapping (for comparison/debug)
DENSITY_BOUNDARIES = [0.0, 0.25, 0.50, 0.75, 1.0]


# ══════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════

def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════

def load_checkpoint(checkpoint_path: str, device: torch.device):
    model = GPCAHazDesNet(num_classes=len(CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"Invalid checkpoint format: {checkpoint_path}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ══════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════

def predict_single(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        density, class_logits, haze_map = model(image_tensor, return_map=True)

    probs = F.softmax(class_logits, dim=1)
    class_id = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, class_id].item()) * 100.0

    return {
        "density": density.item(),
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": confidence,
        "probs": (probs[0] * 100.0).tolist(),
        "haze_map": haze_map,
    }


# ══════════════════════════════════════════════
# HAZE MAP
# ══════════════════════════════════════════════

def save_haze_map(haze_map, save_path):
    m = haze_map[0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    vutils.save_image(m, save_path)


# ══════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════

def get_images_from_dir(folder):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]


def density_to_class(density):
    for i in range(len(DENSITY_BOUNDARIES) - 1):
        if DENSITY_BOUNDARIES[i] <= density < DENSITY_BOUNDARIES[i + 1]:
            return i
    return len(DENSITY_BOUNDARIES) - 2


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="GPCAHazDesNet Inference")

    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Folder of images")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--save_map", type=str, default=None)
    parser.add_argument("--save_csv", type=str, default=None)

    return parser.parse_args()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide either --image or --image_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, ckpt = load_checkpoint(args.checkpoint, device)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {ckpt.get('epoch', 'N/A')}")

    best = ckpt.get("best_metric")
    if best is not None:
        print(f"Best val_acc: {best:.4f}")
    else:
        print("Best val_acc: N/A")

    transform = build_transform(args.image_size)

    # Prepare image list
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = get_images_from_dir(args.image_dir)

    results_all = []

    for img_path in image_paths:
        if not img_path.exists():
            print(f"Skipping (not found): {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        x = transform(image).unsqueeze(0)

        results = predict_single(model, x, device)

        # Print
        print("-" * 45)
        print(f"Image        : {img_path.name}")
        print(f"Haze Density : {results['density']:.4f}")
        print(f"Class        : {results['class_name']}")
        print(f"Confidence   : {results['confidence']:.1f}%")

        # Optional comparison
        density_class = density_to_class(results["density"])
        print(f"Density→Class: {CLASS_NAMES[density_class]}")

        # Probabilities
        for name, prob in zip(CLASS_NAMES, results["probs"]):
            print(f"{name:<10}: {prob:5.1f}%")

        # Save map (only for single image or if folder + naming)
        if args.save_map:
            if args.image:
                save_path = args.save_map
            else:
                save_path = str(Path(args.save_map) / f"{img_path.stem}_map.png")
                Path(args.save_map).mkdir(parents=True, exist_ok=True)

            save_haze_map(results["haze_map"], save_path)

        results_all.append([
            img_path.name,
            results["density"],
            results["class_name"],
            results["confidence"]
        ])

    # Save CSV
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "density", "class", "confidence"])
            writer.writerows(results_all)

        print(f"\nCSV saved to: {args.save_csv}")


if __name__ == "__main__":
    main()
