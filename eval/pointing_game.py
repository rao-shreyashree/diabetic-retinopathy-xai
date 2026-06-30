"""
eval/pointing_game.py

pointing game metric: 
for each (image, lesion_type) pair where the lesion is present, we check whether the heatmap's argmax 
(peak attention pixel, after FOV masking/renormalization) falls inside that lesion's GT mask

hit  = 1 if argmax pixel inside gt_mask
miss = 0 otherwise
NaN  = lesion absent for that image (excluded, same convention as fidelity_scoring.py)

stratified by predicted_grade x lesion_type, same as CFS IoU/Dice
we MUST keep this file in the same folder as fidelity_scoring.py and roi_mask_utils.py

usage:
    from pointing_game import run_pointing_game
    df = run_pointing_game(
        heatmap_folder=".../results/heatmaps/resnet50/gradcam",
        model_name="resnet50",
        method_name="gradcam",
    )

CLI:
    !python pointing_game.py --model resnet50 --method gradcam
"""

import os
import json
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import cv2

from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize
from fidelity_scoring import (
    PROJECT_ROOT, HEATMAP_ROOT, ORIG_IMG_DIR, MASK_ROOT, RESULTS_DIR,
    LESION_DIRS, load_lesion_mask,
)


def get_argmax_point(heatmap_clean: np.ndarray) -> Tuple[int, int]:
    """Returns (row, col) of the peak attention pixel."""
    return np.unravel_index(np.argmax(heatmap_clean), heatmap_clean.shape)


def is_hit(argmax_point: Tuple[int, int], gt_mask: np.ndarray) -> int:
    r, c = argmax_point
    return int(gt_mask[r, c] > 0)


def run_pointing_game(
    heatmap_folder: str,
    model_name: str,
    method_name: str,
    test_ids: Optional[List[str]] = None,
    predictions_csv: Optional[str] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
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

        heatmap = np.load(os.path.join(heatmap_folder, matches[0]))
        fov_mask = get_fundus_mask(os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg"))
        if fov_mask.shape != heatmap.shape:
            fov_mask = cv2.resize(
                fov_mask, (heatmap.shape[1], heatmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        heatmap_clean = apply_fov_mask_and_renormalize(heatmap, fov_mask)

        row = predictions[
            (predictions["image_id"] == img_id) & (predictions["model"] == model_name)
        ]
        if row.empty:
            print(f"SKIP {img_id}: no prediction row for model={model_name}")
            continue
        predicted_grade = int(row["predicted_grade"].values[0])
        argmax_point = get_argmax_point(heatmap_clean)

        for lesion_code in LESION_DIRS:
            gt_mask = load_lesion_mask(img_id, lesion_code, heatmap.shape)
            if gt_mask is None:
                continue  # lesion absent, excluded -> same convention as fidelity_scoring.py

            hit = is_hit(argmax_point, gt_mask)
            records.append({
                "image_id": img_id,
                "model": model_name,
                "method": method_name,
                "lesion_type": lesion_code,
                "predicted_grade": predicted_grade,
                "hit": hit,
            })

    results_df = pd.DataFrame(records)
    n_hits = results_df["hit"].sum() if len(results_df) else 0
    print(f"Pointing game: {n_hits}/{len(results_df)} hits "
          f"for model={model_name}, method={method_name} "
          f"(accuracy={n_hits/len(results_df):.3f})" if len(results_df) else "no rows scored")

    if save_path:
        results_df.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--method", default="gradcam")
    args = parser.parse_args()

    heatmap_folder = os.path.join(HEATMAP_ROOT, args.model, args.method)
    save_path = os.path.join(
        RESULTS_DIR, f"pointing_game_{args.model}_{args.method}.csv"
    )

    df = run_pointing_game(
        heatmap_folder=heatmap_folder,
        model_name=args.model,
        method_name=args.method,
        save_path=save_path,
    )

    print("\nStratified pointing game accuracy (predicted_grade x lesion_type):")
    print(df.groupby(["predicted_grade", "lesion_type"])["hit"].agg(["mean", "count"]))