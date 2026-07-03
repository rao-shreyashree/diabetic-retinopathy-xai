"""
eval/generate_overlays.py

generates overlay visualizations:
heatmap (jet, alpha=0.45) on fundus image with lesion mask boundaries drawn on top.

one figure per image = 3 columns (gradcam / lime / shap) x N lesion rows

usage in Colab:
    import sys
    sys.path.append(".../eval")
    from generate_overlays import run_overlay_generation

    run_overlay_generation(
        model_name="efficientnetb4",
        image_ids=CANDIDATES,   # list of image id strings
    )

CLI:
    !python generate_overlays.py --model efficientnetb4
"""

import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize
from fidelity_scoring import (
    PROJECT_ROOT, HEATMAP_ROOT, ORIG_IMG_DIR, MASK_ROOT, RESULTS_DIR,
    LESION_DIRS, load_lesion_mask,
)

METHODS = ["gradcam", "lime", "shap"]
OVERLAY_DIR = os.path.join(PROJECT_ROOT, "results", "overlays")

CANDIDATES = [
    # (image_id, label)
    ("IDRiD_55", "Grade 2 – best"),
    ("IDRiD_57", "Grade 2 – failure"),
    ("IDRiD_66", "Grade 3 – best"),
    ("IDRiD_61", "Grade 3 – mid"),
    ("IDRiD_68", "Grade 3/4 – cross-method failure"),
    ("IDRiD_67", "Grade 4 – best"),
    ("IDRiD_69", "Grade 4 – mid"),
    ("IDRiD_65", "Grade 4 – failure"),
]

LESION_COLORS = {
    "MA": (255, 0,   0), # red
    "HE": (0,   255, 0), # green
    "EX": (0,   0,   255),  # blue
    "SE": (255, 255, 0), # yellow
}


def load_heatmap(model_name: str, method: str, img_id: str):
    folder = os.path.join(HEATMAP_ROOT, model_name, method)
    matches = [f for f in os.listdir(folder) if img_id in f and f.endswith(".npy")]
    if not matches:
        return None
    return np.load(os.path.join(folder, matches[0]))


def make_overlay(fundus_rgb: np.ndarray, heatmap_clean: np.ndarray, alpha: float = 0.45):
    """Blend jet-colored heatmap onto fundus image."""
    h, w = fundus_rgb.shape[:2]
    hmap_resized = cv2.resize(heatmap_clean, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = Normalize(vmin=0, vmax=1)
    jet = plt.cm.jet(norm(hmap_resized))[:, :, :3]          # H x W x 3, float [0,1]
    jet_uint8 = (jet * 255).astype(np.uint8)
    overlay = cv2.addWeighted(fundus_rgb, 1 - alpha, jet_uint8, alpha, 0)
    return overlay


def draw_lesion_contours(overlay: np.ndarray, img_id: str, heatmap_shape):
    """Draw lesion boundary contours on overlay image for all present lesions."""
    h, w = overlay.shape[:2]
    for code, color in LESION_COLORS.items():
        mask = load_lesion_mask(img_id, code, heatmap_shape)
        if mask is None:
            continue
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def run_overlay_generation(
    model_name: str = "efficientnetb4",
    image_ids=None, # list of (img_id, label) tuples or just img_ids
    methods=None,
    save_dir: str = None,
):
    if methods is None:
        methods = METHODS
    if image_ids is None:
        image_ids = CANDIDATES
    # normalise: accept plain strings or (id, label) tuples
    candidates = []
    for item in image_ids:
        if isinstance(item, tuple):
            candidates.append(item)
        else:
            candidates.append((item, item))

    out_dir = save_dir or os.path.join(OVERLAY_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    for img_id, label in candidates:
        fundus_path = os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(fundus_path):
            print(f"SKIP {img_id}: fundus image not found")
            continue

        fundus_bgr = cv2.imread(fundus_path)
        fundus_rgb = cv2.cvtColor(fundus_bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5))
        fig.suptitle(f"{img_id}  |  {label}", fontsize=13, fontweight="bold")

        # col 0: clean fundus with lesion contours only
        fundus_with_contours = fundus_rgb.copy()
        draw_lesion_contours(fundus_with_contours, img_id, (fundus_rgb.shape[0], fundus_rgb.shape[1]))
        axes[0].imshow(fundus_with_contours)
        axes[0].set_title("Fundus + GT masks", fontsize=10)
        axes[0].axis("off")

        for col, method in enumerate(methods, start=1):
            heatmap = load_heatmap(model_name, method, img_id)
            if heatmap is None:
                axes[col].text(0.5, 0.5, "No heatmap", ha="center", va="center")
                axes[col].axis("off")
                axes[col].set_title(method.upper(), fontsize=10)
                continue

            fov_mask = get_fundus_mask(fundus_path)
            if fov_mask.shape != heatmap.shape:
                fov_mask = cv2.resize(
                    fov_mask, (heatmap.shape[1], heatmap.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            heatmap_clean = apply_fov_mask_and_renormalize(heatmap, fov_mask)

            overlay = make_overlay(fundus_rgb, heatmap_clean)
            draw_lesion_contours(overlay, img_id, heatmap.shape)

            axes[col].imshow(overlay)
            axes[col].set_title(method.upper(), fontsize=10)
            axes[col].axis("off")

        # shared legend
        legend_patches = [
            mpatches.Patch(color=tuple(c/255 for c in col), label=code)
            for code, col in LESION_COLORS.items()
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

        out_path = os.path.join(out_dir, f"{img_id}_overlay.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone. {len(candidates)} figures in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnetb4")
    args = parser.parse_args()
    run_overlay_generation(model_name=args.model)