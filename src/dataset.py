"""PyTorch dataset: returns (image, mask, label) for one MRI slice.

Image and mask go through the same albumentations transform, so they always stay aligned.
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Grayscale average of ImageNet mean/std (encoder is ImageNet-pretrained)
MEAN, STD = 0.449, 0.226


def get_transforms(train, size, seed=0):
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ], seed=seed)
    return A.Compose([A.Resize(size, size)])


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, splits_csv, split, size=256, train=False, seed=0):
        self.data_dir = Path(data_dir)
        df = pd.read_csv(splits_csv, dtype={"id": str, "pid": str})
        self.df = df[df.split == split].reset_index(drop=True)
        self.tf = get_transforms(train, size, seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = cv2.imread(str(self.data_dir / "images" / f"{row.id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.data_dir / "masks" / f"{row.id}.png"), cv2.IMREAD_GRAYSCALE)
        out = self.tf(image=image, mask=mask)
        image = (out["image"].astype(np.float32) / 255.0 - MEAN) / STD
        mask = (out["mask"] > 127).astype(np.float32)
        return {
            "image": torch.from_numpy(image)[None],  # (1, H, W)
            "mask": torch.from_numpy(mask)[None],    # (1, H, W)
            "label": torch.tensor(int(row.label)),
            "id": row.id,
        }
