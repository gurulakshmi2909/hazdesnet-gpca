# dataset.py — HazeDensityDataset 
#
# Features:
#   - Strong training augmentation (domain-aware)
#   - Optional Test-Time Augmentation (TTA)
#   - Mixup support via get_item_raw()
#   - Class count helper for imbalance handling
#   - Robust path + error handling


from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HazeDensityDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        image_size: int = 224,
        train: bool = True,
        use_tta: bool = False,
    ):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir) if root_dir else None
        self.train = train
        self.use_tta = use_tta
        self.image_size = image_size

        # ── Validate CSV ─────────────────────────────
        required = {"image", "density", "class", "class_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(list(missing))}")

        self.image_paths = self.df["image"].astype(str).tolist()
        self.densities = self.df["density"].astype("float32").tolist()
        self.classes = self.df["class"].astype(str).tolist()
        self.class_ids = self.df["class_id"].astype("int64").tolist()

        # ── Normalization (ImageNet) ────────────────
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ── Training transform ──────────────────────
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),

                # Removed vertical flip (not realistic for scenes)

                transforms.RandomRotation(10),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.05,
                ),

                # Apply blur only sometimes (more realistic)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
                ], p=0.3),

                transforms.ToTensor(),
                self.normalize,

                transforms.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    value=0,
                ),
            ])

        # ── Validation / Test transform ─────────────
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ])

        # ── TTA transforms (clean & correct) ────────
        self.tta_transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # Center crop from slightly larger
            transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]),
        ]

    # ──────────────────────────────────────────────
    # Utility functions
    # ──────────────────────────────────────────────

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if self.root_dir and not path.is_absolute():
            path = self.root_dir / path
        return path

    def get_class_counts(self) -> Dict[int, int]:
        """Returns {class_id: count} for computing class weights."""
        counts = {}
        for cid in self.class_ids:
            counts[cid] = counts.get(cid, 0) + 1
        return dict(sorted(counts.items()))

    def get_item_raw(self, idx: int):
        """
        Returns raw PIL image + labels.
        Useful for Mixup or custom augmentation in training loop.
        """
        image_path = self._resolve_path(self.image_paths[idx])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        return (
            img,
            self.densities[idx],
            self.class_ids[idx],
            str(image_path),
        )

    # ──────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self._resolve_path(self.image_paths[idx])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        density = self.densities[idx]
        class_name = self.classes[idx]
        class_id = self.class_ids[idx]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            # ── TTA mode ──
            if self.use_tta:
                images = torch.stack([t(image) for t in self.tta_transforms])
                return {
                    "image": images,  # (T, C, H, W)
                    "density": torch.tensor(density, dtype=torch.float32),
                    "class": class_name,
                    "class_id": torch.tensor(class_id, dtype=torch.long),
                    "image_path": str(image_path),
                }

            # ── Standard mode ──
            image = self.transform(image)

        return {
            "image": image,
            "density": torch.tensor(density, dtype=torch.float32),
            "class": class_name,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "image_path": str(image_path),
        }
