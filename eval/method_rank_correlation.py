"""
eval/method_rank_correlation.py

spearman rank correlation of per-image IoU across XAI methods (GradCAM/LIME/SHAP/AttnRollout), per model, aggregated across lesion types
it answers: do methods fail on the same images (high correlation) or different images (low/negative correlation)?

usage in Colab notebook:
    import sys
    sys.path.append("/content/drive/MyDrive/.../diabetic-retinopathy-xai/eval")
    from method_rank_correlation import run_rank_correlation
    df = run_rank_correlation(model_name="efficientnetb4")

usage as standalone script:
    !python method_rank_correlation.py --model efficientnetb4
    !python method_rank_correlation.py --model resnet50
"""

import os
import argparse
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
SCORES_DIR = os.path.join(PROJECT_ROOT, "results/scores/fidelity")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results/scores/summary")

METHODS_BY_MODEL = {
    "efficientnetb4": ["gradcam", "lime", "shap"],
    "resnet50": ["gradcam", "lime", "shap"],
}


def load_method_iou(model_name: str, method: str) -> pd.DataFrame:
    """Load per-image mean IoU (averaged across lesion types) for one model+method."""
    path = os.path.join(SCORES_DIR, f"fidelity_scores_{model_name}_{method}.csv")
    df = pd.read_csv(path)
    # average across lesion_type -> one IoU per image (some images missing some lesions, that's fine)
    per_image = df.groupby("image_id")["iou"].mean().rename(method)
    return per_image


def run_rank_correlation(
    model_name: str,
    methods: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes pairwise Spearman correlation of per-image mean IoU between all method pairs
    for a given model. Returns long-format df: one row per method pair.
    """
    if methods is None:
        methods = METHODS_BY_MODEL[model_name]

    series_by_method = {m: load_method_iou(model_name, m) for m in methods}
    wide = pd.concat(series_by_method.values(), axis=1)
    wide.columns = methods
    wide = wide.dropna(how="any")  # only images scored by all methods

    records = []
    for method_a, method_b in combinations(methods, 2):
        rho, p = spearmanr(wide[method_a], wide[method_b])
        records.append({
            "model": model_name,
            "method_a": method_a,
            "method_b": method_b,
            "n": len(wide),
            "spearman_rho": rho,
            "p_value": p,
        })

    results_df = pd.DataFrame(records)
    print(f"Rank correlation for model={model_name} (n_images={len(wide)}):")
    print(results_df[["method_a", "method_b", "spearman_rho", "p_value"]].to_string(index=False))

    if save_path:
        results_df.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnetb4")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"method_rank_correlation_{args.model}.csv")

    run_rank_correlation(model_name=args.model, save_path=save_path)