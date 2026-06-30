"""
eval/check_corner_notch.py
Scans all 27 IDRiD test images to check how many have the triangular
corner-notch artifact (seen in IDRiD_56, IDRiD_57) that survives the
current FOV-masking fix in roi_mask_utils.get_fundus_mask.

This notch is NOT pure black -- it's a low-brightness/low-saturation
camera-clip region in-frame, distinct from the black background that
get_fundus_mask already handles. This script flags candidate images
so we know the scope of the problem before patching the mask logic.

usage in Colab:
    from check_corner_notch import scan_corner_notches
    flagged = scan_corner_notches()
"""

import os
import json
import numpy as np
import cv2

from roi_mask_utils import get_fundus_mask

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/IDRiD")
ORIG_IMG_DIR = os.path.join(DATA_ROOT, "segmentation/images/test")

CORNER_FRACTION = 0.15  # check top N% x N% of each corner region


def get_corner_regions(shape, frac=CORNER_FRACTION):
    """Returns slices for the 4 corners of an image."""
    h, w = shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return {
        "top_left": (slice(0, ch), slice(0, cw)),
        "top_right": (slice(0, ch), slice(w - cw, w)),
        "bottom_left": (slice(h - ch, h), slice(0, cw)),
        "bottom_right": (slice(h - ch, h), slice(w - cw, w)),
    }


def scan_corner_notches(test_ids=None, save_path=None):
    """
    For each test image: load the fundus mask, check each corner region.
    A corner is "notched" if it has a mix of masked (0) and unmasked (1)
    pixels within the corner crop -- i.e. the fundus boundary cuts diagonally
    through that corner rather than being a clean black background,
    AND that boundary pixel region has non-trivial brightness (camera clip,
    not pure black) that could still register as heatmap signal.
    """
    if test_ids is None:
        with open(os.path.join(PROJECT_ROOT, "test_image_ids.json")) as f:
            test_ids = sorted(json.load(f))

    flagged = []
    for img_id in test_ids:
        img_path = os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"SKIP {img_id}: image not found")
            continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fov_mask = get_fundus_mask(img_path)
        if fov_mask.shape != gray.shape[:2]:
            fov_mask = cv2.resize(fov_mask, (gray.shape[1], gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        corners = get_corner_regions(gray.shape)
        for corner_name, (rs, cs) in corners.items():
            mask_crop = fov_mask[rs, cs]
            gray_crop = gray[rs, cs]

            frac_unmasked = mask_crop.mean()  # fraction of corner inside FOV mask
            # a "clean" corner is either ~0 (all background) or ~1 (fully inside fundus,
            # e.g. small image / fundus fills frame). A notch shows partial coverage
            # AND non-trivial brightness in the masked-out portion (i.e. it's not pure black).
            if 0.05 < frac_unmasked < 0.95:
                masked_out = gray_crop[mask_crop == 0]
                if masked_out.size > 0 and masked_out.mean() > 15:  # not pure black
                    flagged.append({
                        "image_id": img_id,
                        "corner": corner_name,
                        "frac_unmasked": round(float(frac_unmasked), 3),
                        "masked_out_mean_brightness": round(float(masked_out.mean()), 1),
                    })

    print(f"Scanned {len(test_ids)} images. Flagged {len(flagged)} (image, corner) notch candidates.")
    if flagged:
        ids = sorted(set(f["image_id"] for f in flagged))
        print("Affected image_ids:", ids)
        for f in flagged:
            print(f"  {f['image_id']} - {f['corner']}: "
                  f"frac_unmasked={f['frac_unmasked']}, "
                  f"masked_out_brightness={f['masked_out_mean_brightness']}")

    if save_path:
        import pandas as pd
        pd.DataFrame(flagged).to_csv(save_path, index=False)
        print("Saved:", save_path)

    return flagged


if __name__ == "__main__":
    scan_corner_notches()