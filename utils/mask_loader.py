"""
mask_loader.py: shared ground-truth lesion mask loader for IDRiD.

Used by:
- Shreyashree (fidelity_scoring.py / stratify.py) for IoU/Dice computation
- Anyone who needs to freeze the fixed test-image ID list

ACTUAL Drive folder layout for this project:

<data_root>/                       (= .../data/IDRiD)
└── segmentation/
    ├── images/
    │   ├── train/   IDRiD_01.jpg ... IDRiD_54.jpg
    │   └── test/    IDRiD_55.jpg ... IDRiD_81.jpg
    └── masks/
        ├── train/
        │   ├── 1. Microaneurysms/   IDRiD_01_MA.tif
        │   ├── 2. Haemorrhages/     IDRiD_01_HE.tif
        │   ├── 3. Hard Exudates/    IDRiD_01_EX.tif
        │   ├── 4. Soft Exudates/    IDRiD_01_SE.tif
        │   └── 5. Optic Disc/       IDRiD_01_OD.tif
        └── test/
            (same 5 subfolders, IDRiD_55_MA.tif etc.)

Not every image has every lesion type - a missing mask file means that
lesion is absent for that image; loader returns an all-zero mask.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

LESION_FOLDER = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}

# DR lesion types used for fidelity scoring (OD excluded — not a DR lesion)
LESION_TYPES = ["MA", "HE", "EX", "SE"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class MaskLoader:
    def __init__(self, data_root: str, split: str = "test"):
        """
        data_root: path to '.../data/IDRiD'
        split: 'train' or 'test' (lowercase, matches your Drive folder names)
        """
        self.split = split
        self.img_dir = Path(data_root) / "segmentation" / "images" / split
        self.mask_dir = Path(data_root) / "segmentation" / "masks" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")

    def _mask_path(self, image_id: str, lesion: str) -> Path:
        return self.mask_dir / LESION_FOLDER[lesion] / f"{image_id}_{lesion}.tif"

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
        """Scans the images dir for this split, returns sorted list of IDRiD image IDs."""
        ids = []
        for f in sorted(self.img_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                ids.append(f.stem)
        # numeric sort (IDRiD_55, IDRiD_56, ... not lexicographic)
        ids.sort(key=lambda s: int(s.split("_")[-1]))
        return ids


def freeze_test_ids(data_root: str, split: str = "test", out_path: str = "test_image_ids.json") -> List[str]:
    """
    Shreyashree runs this. 
    Locks the exact image-ID list everyone uses.
    Commit the resulting JSON to the repo root - nobody regenerates their own.
    """
    loader = MaskLoader(data_root, split)
    ids = loader.discover_image_ids()
    with open(out_path, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"Froze {len(ids)} image IDs -> {out_path}")
    print(ids)
    return ids


def load_frozen_test_ids(path: str = "test_image_ids.json") -> List[str]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Usage: python mask_loader.py <data_root> <split>
    # example: python mask_loader.py "/content/drive/MyDrive/.../data/IDRiD" test
    root = sys.argv[1] if len(sys.argv) > 1 else "data/IDRiD"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    freeze_test_ids(root, split)