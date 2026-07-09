"""
eval/failure_case_analysis.py

picks the N worst-IoU images (lowest fidelity) for a model/method, for qualitative overlay writeup
selection is quantitative (not cherry-picked) so we need to pair the output with generate_overlays.py

usage:
    !python failure_case_analysis.py --model vit_b16 --method attention_rollout --n 4
"""

import os
import argparse
from typing import Optional

import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
FIDELITY_DIR = os.path.join(PROJECT_ROOT, "results/scores/fidelity")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results/scores/summary")


def run_failure_cases(
    model_name: str,
    method_name: str,
    n: int = 4,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    path = os.path.join(FIDELITY_DIR, f"fidelity_scores_{model_name}_{method_name}.csv")
    df = pd.read_csv(path)

    per_image = (
        df.groupby(["image_id", "predicted_grade"])["iou"]
        .mean()
        .reset_index()
        .rename(columns={"iou": "mean_iou"})
        .sort_values("mean_iou")
        .reset_index(drop=True)
    )
    per_image.insert(0, "rank", per_image.index + 1)  # rank 1 = worst
    per_image.insert(0, "method", method_name)
    per_image.insert(0, "model", model_name)

    worst = per_image.head(n)
    print(f"Worst {n} images by mean IoU, model={model_name}, method={method_name}:")
    print(worst.to_string(index=False))
    print("\nimage_ids for generate_overlays.py:", list(worst["image_id"]))

    if save_path:
        per_image.to_csv(save_path, index=False)  # full ranked list
        print("Saved:", save_path)

    return per_image


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    for model in ["efficientnetb4", "resnet50"]:
        for method in ["gradcam", "lime", "shap"]:
            run_failure_cases(
                model_name=model,
                method_name=method,
                n=4,
                save_path=os.path.join(
                    SAVE_DIR,
                    f"failure_cases_{model}_{method}.csv",
                ),
            )