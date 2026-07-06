"""
eval/cfs_severity_weighted.py
Severity-weighted Composite Fidelity Score. Standard CFS averages IoU/Dice
equally across lesion types. This weights lesion types by clinical severity
(SE > HE > EX > MA) so failing to localize a more sight-threatening lesion
penalizes the score more.

Weights are editable in SEVERITY_WEIGHTS below (currently ordinal, not
size-normalized) -- swap in size/difficulty-based weights if that's the
intent instead.

usage in Colab notebook:
    import sys
    sys.path.append("/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai/eval")
    from cfs_severity_weighted import run_severity_weighted_cfs
    df = run_severity_weighted_cfs(model_name="efficientnetb4", method_name="gradcam")

usage as standalone script:
    !python cfs_severity_weighted.py --model efficientnetb4 --method gradcam
"""

import os
import argparse
from typing import Optional

import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
SCORES_DIR = os.path.join(PROJECT_ROOT, "results/scores/fidelity")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results/scores/fidelity")

# clinical severity ordering: SE (sight-threatening) > HE > EX > MA (earliest/mildest)
SEVERITY_WEIGHTS = {
    "SE": 4,
    "HE": 3,
    "EX": 2,
    "MA": 1,
}


def run_severity_weighted_cfs(
    model_name: str,
    method_name: str,
    weights: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads fidelity_scores_{model}_{method}.csv (long format, one row per
    image x lesion_type) and computes a per-image severity-weighted IoU/Dice,
    then aggregates by predicted_grade.
    """
    weights = weights or SEVERITY_WEIGHTS

    path = os.path.join(SCORES_DIR, f"fidelity_scores_{model_name}_{method_name}.csv")
    df = pd.read_csv(path)
    df["weight"] = df["lesion_type"].map(weights)

    if df["weight"].isna().any():
        missing = df.loc[df["weight"].isna(), "lesion_type"].unique()
        raise ValueError(f"No severity weight defined for lesion types: {missing}")

    # per-image weighted average 
    # only over lesion types present for that image, consistent with fidelity_scoring.py's exclusion of absent lesions
    def weighted_avg(g, col):
        return (g[col] * g["weight"]).sum() / g["weight"].sum()

    per_image = (
        df.groupby(["image_id", "model", "method", "predicted_grade"])
        .apply(lambda g: pd.Series({
            "cfs_weighted_iou": weighted_avg(g, "iou"),
            "cfs_weighted_dice": weighted_avg(g, "dice"),
            "n_lesion_types": g["lesion_type"].nunique(),
        }))
        .reset_index()
    )

    print(f"Severity-weighted CFS for model={model_name}, method={method_name} "
          f"(n_images={len(per_image)}):")
    print(per_image.groupby("predicted_grade")[["cfs_weighted_iou", "cfs_weighted_dice"]]
          .agg(["mean", "std", "count"]))

    if save_path:
        per_image.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return per_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnetb4")
    parser.add_argument("--method", default="gradcam")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(
        SAVE_DIR, f"cfs_severity_weighted_{args.model}_{args.method}.csv"
    )

    run_severity_weighted_cfs(
        model_name=args.model,
        method_name=args.method,
        save_path=save_path,
    )