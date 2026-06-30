"""
eval/fidelity_scoring_otsu.py
OTSU-THRESHOLD VARIANT of fidelity_scoring.py
computes IoU/Dice fidelity scores between XAI saliency heatmaps and 
pixel-level lesion ground-truth masks (IDRiD segmentation test split)

stratifies by PREDICTED grade (not ground-truth grade)
CFS measures whether the heatmap explains what the model decided
not whether the model was clinically correct

we MUST keep this file in the same folder as roi_mask_utils.py

usage in Colab notebook:
    import sys
    sys.path.append("/content/drive/MyDrive/.../diabetic-retinopathy-xai/eval")
    from fidelity_scoring_otsu import run_fidelity_scoring

    df = run_fidelity_scoring(
        heatmap_folder=".../results/heatmaps/resnet50/gradcam",
        model_name="resnet50",
        method_name="gradcam",
    )

usage as a standalone script (after %cd into the eval/ folder in Colab):
    !python fidelity_scoring_otsu.py --model resnet50 --method gradcam

NOTE: this is the OTSU variant. The original percentile(75) version is
kept as fidelity_scoring.py — do not delete, both are referenced in repo.
"""

import os
import glob
import json
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import cv2

from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/IDRiD")
HEATMAP_ROOT = os.path.join(PROJECT_ROOT, "results/heatmaps")
ORIG_IMG_DIR = os.path.join(DATA_ROOT, "segmentation/images/test")
MASK_ROOT = os.path.join(DATA_ROOT, "segmentation/masks/test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

LESION_DIRS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

THRESH_METHOD = "otsu" # data-driven threshold via Otsu's method
THRESH_VALUE  = None # unused for otsu, kept for signature compatibility

# core functions
def binarize_heatmap(heatmap: np.ndarray, method: str = "otsu", value: float = None) -> np.ndarray:
    """
    Threshold a normalized heatmap into a binary attention mask using Otsu's method.
    Otsu finds the threshold minimizing intra-class variance over the histogram
    of nonzero heatmap values — data-driven, no arbitrary percentile cutoff.
    Falls back to all-zero mask if heatmap is degenerate (flat / empty after FOV mask).
    """
    nonzero = heatmap[heatmap > 0]
    if nonzero.size == 0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    if method == "otsu":
        # scale nonzero values to 0-255 uint8 for cv2 Otsu
        # apply same scaling to full heatmap
        lo, hi = nonzero.min(), nonzero.max()
        if hi <= lo:
            # degenerate: heatmap is flat, Otsu undefined -> no attended region
            return np.zeros_like(heatmap, dtype=np.uint8)
        scaled = np.clip((heatmap - lo) / (hi - lo), 0, 1)
        scaled_u8 = (scaled * 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(
            scaled_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = (scaled_u8 >= otsu_thresh).astype(np.uint8)
        # zero out background pixels that were never part of the heatmap (heatmap<=0)
        binary[heatmap <= 0] = 0
        return binary
    elif method == "percentile":
        thresh = np.percentile(nonzero, value)
        return (heatmap >= thresh).astype(np.uint8)
    elif method == "fixed":
        return (heatmap >= value).astype(np.uint8)
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def load_lesion_mask(img_id: str, lesion_code: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Load ground-truth lesion mask for a given image + lesion type.
    Returns None if this lesion type is absent for this image (valid IDRiD
    case - must be excluded from scoring, NOT counted as IoU=0).
    """
    folder = os.path.join(MASK_ROOT, LESION_DIRS[lesion_code])
    matches = glob.glob(os.path.join(folder, f"{img_id}_*"))
    if not matches:
        return None
    mask = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Standard IoU and Dice coefficient between two binary masks."""
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else np.nan
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else np.nan
    return iou, dice


def run_fidelity_scoring(
    heatmap_folder: str,
    model_name: str,
    method_name: str,
    test_ids: Optional[List[str]] = None,
    predictions_csv: Optional[str] = None,
    thresh_method: str = THRESH_METHOD,
    thresh_value: float = THRESH_VALUE,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scores all heatmaps in `heatmap_folder` against lesion masks.
    Returns a long-format DataFrame: one row per (image, lesion_type).
    """
    if test_ids is None:
        with open(os.path.join(PROJECT_ROOT, "test_image_ids.json")) as f:
            test_ids = sorted(json.load(f))

    if predictions_csv is None:
        predictions_csv = os.path.join(HEATMAP_ROOT, "predictions.csv")
    predictions = pd.read_csv(predictions_csv)

    records = []
    for img_id in test_ids:

        matches = [
            f for f in os.listdir(heatmap_folder)
            if img_id in f and f.endswith(".npy")
        ]

        if len(matches) != 1:
            print(f"SKIP {img_id}: expected 1 heatmap, found {len(matches)}")
            continue

        hmap_path = os.path.join(heatmap_folder, matches[0])
        if not os.path.exists(hmap_path):
            print(f"SKIP {img_id}: heatmap not found at {hmap_path}")
            continue

        heatmap = np.load(hmap_path)
        fov_mask = get_fundus_mask(os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg"))
        if fov_mask.shape != heatmap.shape:
            fov_mask = cv2.resize(
                fov_mask, (heatmap.shape[1], heatmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        heatmap_clean = apply_fov_mask_and_renormalize(heatmap, fov_mask)
        pred_mask = binarize_heatmap(heatmap_clean, thresh_method, thresh_value)
        row = predictions[
            (predictions["image_id"] == img_id) & (predictions["model"] == model_name)
        ]
        if row.empty:
            print(f"SKIP {img_id}: no prediction row for model={model_name}")
            continue
        predicted_grade = int(row["predicted_grade"].values[0])

        for lesion_code in LESION_DIRS:
            gt_mask = load_lesion_mask(img_id, lesion_code, heatmap.shape)
            if gt_mask is None:
                continue  # lesion absent = excluded, not a fidelity failure

            iou, dice = compute_iou_dice(pred_mask, gt_mask)
            records.append({
                "image_id": img_id,
                "model": model_name,
                "method": method_name,
                "lesion_type": lesion_code,
                "predicted_grade": predicted_grade,
                "iou": iou,
                "dice": dice,
            })

    results_df = pd.DataFrame(records)
    print(f"Scored {len(results_df)} (image, lesion_type) pairs "
          f"for model={model_name}, method={method_name}")

    if save_path:
        results_df.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return results_df


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--method", default="gradcam")
    parser.add_argument("--thresh_method", default=THRESH_METHOD)
    parser.add_argument("--thresh_value", type=float, default=None)
    args = parser.parse_args()

    heatmap_folder = os.path.join(HEATMAP_ROOT, args.model, args.method)
    save_path = os.path.join(
        RESULTS_DIR, f"fidelity_scores_{args.model}_{args.method}_otsu.csv"
    )

    df = run_fidelity_scoring(
        heatmap_folder=heatmap_folder,
        model_name=args.model,
        method_name=args.method,
        thresh_method=args.thresh_method,
        thresh_value=args.thresh_value,
        save_path=save_path,
    )

    print("\nStratified summary (by predicted_grade x lesion_type):")
    print(df.groupby(["predicted_grade", "lesion_type"])[["iou", "dice"]].agg(["mean", "std", "count"]))