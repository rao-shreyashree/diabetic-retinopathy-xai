"""
eval/check_gradcam_overlay.py

quick visual sanity check: 
confirms the FOV-masking fix removed the border-artifact (corner activation) from EfficientNet-B4 GradCAM heatmaps.

plots, for a few sample images: 
original fundus image | raw heatmap | FOV-masked heatmap (post-fix) overlaid on the image

usage in Colab (after %cd into eval/ or sys.path.append(eval/)):
    from check_gradcam_overlay import check_overlay
    check_overlay(n_samples=3)
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/IDRiD")
HEATMAP_ROOT = os.path.join(PROJECT_ROOT, "results/heatmaps")
ORIG_IMG_DIR = os.path.join(DATA_ROOT, "segmentation/images/test")


def check_overlay(model="efficientnetb4", method="gradcam", n_samples=3, test_ids=None):
    heatmap_folder = os.path.join(HEATMAP_ROOT, model, method)

    if test_ids is None:
        with open(os.path.join(PROJECT_ROOT, "test_image_ids.json")) as f:
            test_ids = sorted(json.load(f))[:n_samples]
    else:
        test_ids = test_ids[:n_samples]

    fig, axes = plt.subplots(len(test_ids), 3, figsize=(12, 4 * len(test_ids)))
    if len(test_ids) == 1:
        axes = axes.reshape(1, -1)

    for row, img_id in enumerate(test_ids):
        matches = [f for f in os.listdir(heatmap_folder) if img_id in f and f.endswith(".npy")]
        if len(matches) != 1:
            print(f"SKIP {img_id}: expected 1 heatmap, found {len(matches)}")
            continue

        heatmap = np.load(os.path.join(heatmap_folder, matches[0]))
        img_path = os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg")
        orig = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        fov_mask = get_fundus_mask(img_path)
        if fov_mask.shape != heatmap.shape:
            fov_mask = cv2.resize(fov_mask, (heatmap.shape[1], heatmap.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        heatmap_clean = apply_fov_mask_and_renormalize(heatmap, fov_mask)

        orig_resized = cv2.resize(orig, (heatmap.shape[1], heatmap.shape[0]))

        axes[row, 0].imshow(orig_resized)
        axes[row, 0].set_title(f"{img_id} - original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(orig_resized)
        axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[row, 1].set_title("raw heatmap (pre-fix) - check corners")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(orig_resized)
        axes[row, 2].imshow(heatmap_clean, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("FOV-masked heatmap (post-fix)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_overlay()