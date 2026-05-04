"""
SURREALDataset — loads pre-extracted JPEG frames and joint labels.

Requires preprocess.py to have been run first.

Each sample returned by __getitem__:
  image_tensor : (3, 192, 256) float32, ImageNet-normalized
  joints_tensor: (24, 2)       float32, (x, y) in [0, 1]
                 index mapping follows SMPL_JOINT_NAMES in model.py
"""

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SURREALDataset(Dataset):
    # ImageNet statistics — required by all three backbone variants
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, index_csv: str, augment: bool = False) -> None:
        """
        Parameters
        ----------
        index_csv : str
            Path to index_train.csv or index_val.csv produced by preprocess.py.
        augment : bool
            If True, apply colour jitter + random erasing (use for the training
            split only; always False for validation).
        """
        self._load_index(index_csv)

        # ── Transforms ────────────────────────────────────────────────────────
        # ToTensor converts PIL (H×W×C uint8) → (C×H×W) float32 in [0,1].
        # Normalize shifts to ImageNet distribution expected by the backbone.
        base = [
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ]

        if augment:
            # Colour augmentations simulate real-camera variability.
            # RandomErasing drops a small rectangle to prevent over-reliance
            # on texture in any single patch.
            # Horizontal-flip augmentation is intentionally omitted: all
            # training frames are already in selfie orientation, and flipping
            # them again would require swapping left/right joint pairs.
            aug = [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.05),
            ]
            self.transform = transforms.Compose(aug + base + [
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose(base)

    # ── Index loading ─────────────────────────────────────────────────────────

    def _load_index(self, csv_path: str) -> None:
        # Read paths and joints separately so joints can be stored as a single
        # contiguous numpy array rather than 5M+ individual arrays.
        # This reduces peak RAM 
        paths: list[str] = []
        joints_flat: list[list[float]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                paths.append(row["jpeg_path"])
                joints_flat.append(
                    [float(row[f"j{j}_{ax}"]) for j in range(24) for ax in ("x", "y")]
                )
        self._paths  = paths
        self._joints = np.array(joints_flat, dtype=np.float32).reshape(-1, 24, 2)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self._paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, torch.from_numpy(self._joints[idx])
