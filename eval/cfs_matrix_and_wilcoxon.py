"""
eval/cfs_matrix_and_wilcoxon.py
Builds the 3x3x3 fidelity matrix (method x lesion_type x predicted_grade)
and runs Wilcoxon signed-rank tests for method comparisons.

inputs: the 6 fidelity_scores_{model}_{method}_otsu.csv files

design:
- descriptive matrix: 
    mean/std/count of IoU & Dice, per (method, lesion_type, predicted_grade), separately per model
    Grade axis will show 1-2-3-4 (whatever predicted grades are present), NOT a forced 0-4 
- significance: 
    Wilcoxon signed-rank, paired by image_id, computed at the lesion_type level only (grade collapsed) due to small per-cell n.
    Pairwise: gradcam-vs-lime, gradcam-vs-shap, lime-vs-shap, per lesion_type, per model.
    Holm correction applied across the 12 tests (4 lesion types x 3 pairs) per model.

usage in Colab:
    import sys
    sys.path.append(f"{PROJECT_ROOT}/eval")
    from cfs_matrix_and_wilcoxon import run_all
    run_all(results_dir=RESULTS_DIR)
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

METHODS = ["gradcam", "lime", "shap"]
LESIONS = ["MA", "HE", "EX", "SE"]
MODELS = ["efficientnetb4", "resnet50"]


def load_all(results_dir, suffix="_otsu"):
    """Load and concat all 6 model x method CSVs into one long df."""
    frames = []
    for model in MODELS:
        for method in METHODS:
            path = os.path.join(results_dir, f"fidelity_scores_{model}_{method}{suffix}.csv")
            if not os.path.exists(path):
                print(f"WARNING missing: {path}")
                continue
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    return df


def build_descriptive_matrix(df, save_path=None):
    """
    3 (method) x 3-4 (lesion_type) x N (predicted_grade) matrix.
    Separate per model. 
    Mean/std/count of IoU and Dice.
    """
    matrix = (
        df.groupby(["model", "method", "lesion_type", "predicted_grade"])[["iou", "dice"]]
        .agg(["mean", "std", "count"])
    )
    matrix.columns = ["_".join(c) for c in matrix.columns]
    matrix = matrix.reset_index()

    # flag underpowered cells (n<5) for downstream caution - descriptive only, not excluded
    matrix["underpowered"] = matrix["iou_count"] < 5

    if save_path:
        matrix.to_csv(save_path, index=False)
        print("Saved descriptive matrix:", save_path)

    return matrix


def run_wilcoxon_per_lesion(df, model, metric="iou", alpha=0.05):
    """
    Pairwise Wilcoxon signed-rank between methods, paired by image_id, computed within each lesion_type (grade collapsed). 
    Holm-corrected across all lesion x pair combinations for this model.
    """
    sub = df[df["model"] == model]
    pairs = list(itertools.combinations(METHODS, 2))

    results = []
    for lesion in LESIONS:
        lesion_df = sub[sub["lesion_type"] == lesion]
        # pivot to image_id x method for this lesion_type
        wide = lesion_df.pivot_table(index="image_id", columns="method", values=metric)

        for m1, m2 in pairs:
            if m1 not in wide.columns or m2 not in wide.columns:
                results.append({
                    "model": model, "lesion_type": lesion,
                    "method_a": m1, "method_b": m2,
                    "n": 0, "statistic": np.nan, "p_raw": np.nan,
                    "note": "missing method column",
                })
                continue

            paired = wide[[m1, m2]].dropna()
            n = len(paired)

            if n < 5:
                results.append({
                    "model": model, "lesion_type": lesion,
                    "method_a": m1, "method_b": m2,
                    "n": n, "statistic": np.nan, "p_raw": np.nan,
                    "note": "n<5, underpowered, excluded from test",
                })
                continue

            diff = paired[m1] - paired[m2]
            if (diff == 0).all():
                results.append({
                    "model": model, "lesion_type": lesion,
                    "method_a": m1, "method_b": m2,
                    "n": n, "statistic": np.nan, "p_raw": np.nan,
                    "note": "all differences zero, wilcoxon undefined",
                })
                continue

            try:
                stat, p = wilcoxon(paired[m1], paired[m2])
            except ValueError as e:
                results.append({
                    "model": model, "lesion_type": lesion,
                    "method_a": m1, "method_b": m2,
                    "n": n, "statistic": np.nan, "p_raw": np.nan,
                    "note": f"wilcoxon failed: {e}",
                })
                continue

            results.append({
                "model": model, "lesion_type": lesion,
                "method_a": m1, "method_b": m2,
                "n": n, "statistic": stat, "p_raw": p,
                "note": "",
            })

    res_df = pd.DataFrame(results)

    # Holm correction across all valid tests for this model (note == "")
    valid_mask = res_df["note"] == ""
    if valid_mask.sum() > 0:
        reject, p_corrected, _, _ = multipletests(
            res_df.loc[valid_mask, "p_raw"], alpha=alpha, method="holm"
        )
        res_df.loc[valid_mask, "p_holm"] = p_corrected
        res_df.loc[valid_mask, "significant_holm"] = reject
    res_df["p_holm"] = res_df.get("p_holm", np.nan)
    res_df["significant_holm"] = res_df.get("significant_holm", False)

    return res_df


def run_all(results_dir, suffix="_otsu", metric="iou", save=True):
    df = load_all(results_dir, suffix=suffix)

    matrix = build_descriptive_matrix(
        df,
        save_path=os.path.join(results_dir, "cfs_descriptive_matrix.csv") if save else None,
    )

    wilcoxon_results = []
    for model in MODELS:
        wilcoxon_results.append(run_wilcoxon_per_lesion(df, model, metric=metric))
    wilcoxon_df = pd.concat(wilcoxon_results, ignore_index=True)

    if save:
        wilcoxon_path = os.path.join(results_dir, f"wilcoxon_{metric}_results.csv")
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print("Saved wilcoxon results:", wilcoxon_path)

    print("\n--- Descriptive matrix (head) ---")
    print(matrix.head(10))
    print("\n--- Wilcoxon results ---")
    print(wilcoxon_df)

    return matrix, wilcoxon_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--metric", default="iou", choices=["iou", "dice"])
    args = parser.parse_args()
    run_all(args.results_dir, metric=args.metric)