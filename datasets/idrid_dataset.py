"""
datasets/idrid_dataset.py

IDRiD dataset loader for XAI evaluation (Week 2+).
Returns (tensor, label, image_id, pil_image) so XAI methods
have access to both the model input AND the original PIL image
for heatmap resizing.

Contract:
  - image_id matches IDRiD naming exactly (e.g. "IDRiD_55")
  - PIL image is original resolution, no transforms applied
  - tensor is preprocessed for model inference (512x512, normalized)
"""

import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd

# Import shared config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import (
    GRADING_TEST_DIR, TEST_CSV, TEST_IDS_JSON,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
)


# ── Inference transform (same as val_transform in week1_training.ipynb) ──────
INFERENCE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class IDRiDTestDataset(Dataset):
    """
    Loads IDRiD test images from the frozen test_image_ids.json list.

    Returns per item:
        tensor     (C, H, W) float32  — preprocessed model input
        label      int                — ground truth DR grade (0-4)
        image_id   str                — e.g. "IDRiD_55"
        pil_image  PIL.Image          — original resolution, for heatmap resizing
    """

    def __init__(self, img_dir=None, csv_path=None, test_ids_json=None):
        self.img_dir = img_dir or GRADING_TEST_DIR
        csv_path     = csv_path or TEST_CSV
        test_ids_json = test_ids_json or TEST_IDS_JSON

        # Load frozen test IDs
        with open(test_ids_json, "r") as f:
            self.test_ids = json.load(f)

        # Load labels CSV
        df = pd.read_csv(csv_path, usecols=["Image name", "Retinopathy grade"])
        df = df.rename(columns={"Image name": "image_id",
                                 "Retinopathy grade": "label"})

        # Filter to only frozen test IDs and build lookup
        self.label_map = dict(zip(df["image_id"], df["label"]))

        # Verify all test IDs have labels
        missing = [i for i in self.test_ids if i not in self.label_map]
        if missing:
            raise ValueError(f"Labels missing for test IDs: {missing}")

        self.transform = INFERENCE_TRANSFORM

    def __len__(self):
        return len(self.test_ids)

    def __getitem__(self, idx):
        image_id  = self.test_ids[idx]
        img_path  = os.path.join(self.img_dir, image_id + ".jpg")
        label     = int(self.label_map[image_id])

        pil_image = Image.open(img_path).convert("RGB")
        tensor    = self.transform(pil_image)

        return tensor, label, image_id, pil_image

    def get_original_size(self, image_id):
        """Returns (W, H) of original image — used for heatmap resizing."""
        img_path = os.path.join(self.img_dir, image_id + ".jpg")
        with Image.open(img_path) as img:
            return img.size  # PIL returns (W, H)


def get_test_loader(batch_size=1, num_workers=2):
    """
    Returns DataLoader for test set.
    batch_size=1 is recommended for XAI (each image processed individually).
    """
    dataset = IDRiDTestDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn needed because PIL images can't be stacked into a tensor
        collate_fn=_collate_with_pil,
    )


def _collate_with_pil(batch):
    """
    Custom collate: stacks tensors and labels normally,
    keeps image_ids as list of strings, pil_images as list of PIL.
    """
    tensors    = torch.stack([item[0] for item in batch])
    labels     = torch.tensor([item[1] for item in batch])
    image_ids  = [item[2] for item in batch]
    pil_images = [item[3] for item in batch]
    return tensors, labels, image_ids, pil_images
