"""
mask_loader.py: shared ground-truth lesion mask loader for IDRiD

Used by:
- shreya (fidelity_scoring.py / stratify.py) for IoU/Dice computation
- anyone who needs to freeze the fixed 81-image test ID list

Expected folder structure (official IDRiD release, zip A — Segmentation),
rooted at <data_root>:

A. Segmentation/
    1. Original Images/
        a. Training Set/
        b. Testing Set/
    2. All Segmentation Groundtruths/
        a. Training Set/
            1. Microaneurysms/
            2. Haemorrhages/
            3. Hard Exudates/
            4. Soft Exudates/
            5. Optic Disc/
        b. Testing Set/
            (same 5 subfolders)

Mask filename convention: IDRiD_<NN>_<SUFFIX>.tif
  MA = Microaneurysms, HE = Haemorrhages, EX = Hard Exudates,
  SE = Soft Exudates, OD = Optic Disc.
Not every image has every lesion type - a missing file means that lesion
is absent for that image; loader returns an all-zero mask in that case

Note: 
If our actual extracted folder names differ slightly (ordinal prefixes,
spacing, "a."/"b." vs full words), adjust LESION_SUFFIX / split strings
below — everything else stays the same
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

LESION_SUFFIX = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Hard Exudates",
    "SE": "Soft Exudates",
    "OD": "Optic Disc",
}

# DR lesion types used for fidelity scoring (OD excluded - not a DR lesion)
LESION_TYPES = ["MA", "HE", "EX", "SE"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class MaskLoader:
    def __init__(self, data_root: str, split: str = "Testing Set"):
        """
        data_root: path to '.../data/IDRiD'
        split: 'Training Set' or 'Testing Set' — must match wherever your
               81 evaluation images actually live.
        """
        self.seg_root = Path(data_root) / "A. Segmentation"
        self.img_dir = self._find_dir(self.seg_root / "1. Original Images", split)
        self.gt_root = self._find_dir(self.seg_root / "2. All Segmentation Groundtruths", split)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

    @staticmethod
    def _find_dir(parent: Path, split: str) -> Path:
        """Tolerates ordinal prefixes like 'a. Training Set' vs 'Training Set'."""
        exact = parent / split
        if exact.exists():
            return exact
        matches = list(parent.glob(f"*{split}*")) if parent.exists() else []
        return matches[0] if matches else exact

    def _lesion_folder(self, lesion: str) -> Path:
        name = LESION_SUFFIX[lesion]
        matches = list(self.gt_root.glob(f"*{name}*"))
        return matches[0] if matches else self.gt_root / name

    def _mask_path(self, image_id: str, lesion: str) -> Path:
        return self._lesion_folder(lesion) / f"{image_id}_{lesion}.tif"

    def _image_path(self, image_id: str) -> Path:
        for ext in IMG_EXTS:
            p = self.img_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Original image not found for {image_id} in {self.img_dir}")

    def get_image_shape(self, image_id: str) -> Tuple[int, int]:
        """Returns (H, W)."""
        with Image.open(self._image_path(image_id)) as im:
            w, h = im.size
            return (h, w)

    def load_lesion_mask(
        self, image_id: str, lesion: str, target_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Returns binary (H, W) uint8 mask. All-zero if this lesion is absent for the image."""
        path = self._mask_path(image_id, lesion)
        shape = target_shape or self.get_image_shape(image_id)
        if not path.exists():
            return np.zeros(shape, dtype=np.uint8)
        mask = np.array(Image.open(path).convert("L"))
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != shape:
            resized = Image.fromarray(mask * 255).resize((shape[1], shape[0]), Image.NEAREST)
            mask = (np.array(resized) > 0).astype(np.uint8)
        return mask

    def load_all_masks(
        self, image_id: str, lesion_types: List[str] = LESION_TYPES
    ) -> Dict[str, np.ndarray]:
        """Returns {lesion_type: (H,W) binary mask} for one image, all same shape."""
        shape = self.get_image_shape(image_id)
        return {l: self.load_lesion_mask(image_id, l, shape) for l in lesion_types}

    def discover_image_ids(self) -> List[str]:
        """Scans the original-images dir, returns sorted list of IDRiD image IDs present."""
        ids = []
        for f in sorted(self.img_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                ids.append(f.stem)
        return ids


def freeze_test_ids(data_root: str, split: str, out_path: str = "test_image_ids.json") -> List[str]:
    """
    Run ONCE, by ONE person. Locks the exact image-ID list everyone uses.
    Commit the resulting JSON to the repo — nobody regenerates their own.
    """
    loader = MaskLoader(data_root, split)
    ids = loader.discover_image_ids()
    with open(out_path, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"Froze {len(ids)} image IDs -> {out_path}")
    return ids


def load_frozen_test_ids(path: str = "test_image_ids.json") -> List[str]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Usage: python mask_loader.py <data_root> <split>
    # example:   python mask_loader.py data/IDRiD "Testing Set"
    root = sys.argv[1] if len(sys.argv) > 1 else "data/IDRiD"
    split = sys.argv[2] if len(sys.argv) > 2 else "Testing Set"
    freeze_test_ids(root, split)