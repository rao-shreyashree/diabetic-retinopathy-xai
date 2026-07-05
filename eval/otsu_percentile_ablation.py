"""
eval/otsu_percentile_ablation.py

ablation: does heatmap binarization method (percentile-75 vs Otsu) change fidelity conclusions? 
we compare fidelity_scores_{model}_{method}.csv (percentile) against fidelity_scores_{model}_{method}_otsu.csv (otsu) per model/method

usage in Colab notebook:
    import sys
    sys.path.append("/content/drive/MyDrive/.../diabetic-retinopathy-xai/eval")
    from otsu_percentile_ablation import run_ablation_all
    df = run_ablation_all()

usage as standalone script:
    !python otsu_percentile_ablation.py
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
SCORES_DIR = os.path.join(PROJECT_ROOT, "results/scores/fidelity")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results/scores/summary")

# otsu variants only exist for these combos
MODEL_METHOD_COMBOS = [
    ("efficientnetb4", "gradcam"),
    ("efficientnetb4", "lime"),
    ("efficientnetb4", "shap"),
    ("resnet50", "gradcam"),
    ("resnet50", "lime"),
    ("resnet50", "shap"),
]


def compare_one(model_name: str, method_name: str) -> Optional[dict]:
    perc_path = os.path.join(SCORES_DIR, f"fidelity_scores_{model_name}_{method_name}.csv")
    otsu_path = os.path.join(SCORES_DIR, f"fidelity_scores_{model_name}_{method_name}_otsu.csv")

    if not (os.path.exists(perc_path) and os.path.exists(otsu_path)):
        print(f"SKIP {model_name}/{method_name}: missing percentile or otsu csv")
        return None

    perc = pd.read_csv(perc_path)
    otsu = pd.read_csv(otsu_path)

    # per-image mean IoU/Dice (avg across lesion types present), matched on image_id
    perc_img = perc.groupby("image_id")[["iou", "dice"]].mean()
    otsu_img = otsu.groupby("image_id")[["iou", "dice"]].mean()
    merged = perc_img.join(otsu_img, lsuffix="_percentile", rsuffix="_otsu", how="inner")

    iou_stat, iou_p = wilcoxon(merged["iou_percentile"], merged["iou_otsu"])
    dice_stat, dice_p = wilcoxon(merged["dice_percentile"], merged["dice_otsu"])

    return {
        "model": model_name,
        "method": method_name,
        "n": len(merged),
        "iou_mean_percentile": merged["iou_percentile"].mean(),
        "iou_mean_otsu": merged["iou_otsu"].mean(),
        "iou_wilcoxon_stat": iou_stat,
        "iou_wilcoxon_p": iou_p,
        "dice_mean_percentile": merged["dice_percentile"].mean(),
        "dice_mean_otsu": merged["dice_otsu"].mean(),
        "dice_wilcoxon_stat": dice_stat,
        "dice_wilcoxon_p": dice_p,
    }


def run_ablation_all(
    combos: Optional[List[Tuple[str, str]]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    combos = combos or MODEL_METHOD_COMBOS
    records = [r for r in (compare_one(m, meth) for m, meth in combos) if r is not None]
    results_df = pd.DataFrame(records)

    print("Otsu vs percentile ablation:")
    print(results_df[["model", "method", "iou_mean_percentile", "iou_mean_otsu",
                       "iou_wilcoxon_p", "dice_mean_percentile", "dice_mean_otsu",
                       "dice_wilcoxon_p"]].to_string(index=False))

    if save_path:
        results_df.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return results_df


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "otsu_percentile_ablation.csv")
    run_ablation_all(save_path=save_path)