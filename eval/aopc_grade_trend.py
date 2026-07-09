"""
eval/aopc_grade_trend.py

we check if faithfulness (AOPC deletion/insertion AUC) degrade with DR severity?
and group AOPC scores by predicted_grade
the file handles both column-naming conventions

usage:
    !python aopc_grade_trend.py --model vit_b16 --method attention_rollout
    !python aopc_grade_trend.py --model efficientnetb4 --method gradcam
"""

import os
import argparse
from typing import Optional

import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
AOPC_DIR = os.path.join(PROJECT_ROOT, "results/scores/aopc")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results/scores/aopc")

COL_ALIASES = {
    "deletion_auc": "deletion_auc",
    "aopc_deletion": "deletion_auc",
    "insertion_auc": "insertion_auc",
    "aopc_insertion": "insertion_auc",
}


def run_grade_trend(model_name: str, method_name: str, save_path: Optional[str] = None) -> pd.DataFrame:
    path = os.path.join(AOPC_DIR, f"aopc_{model_name}_{method_name}.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={c: COL_ALIASES[c] for c in df.columns if c in COL_ALIASES})

    trend = (
        df.groupby("predicted_grade")[["deletion_auc", "insertion_auc"]]
        .agg(["mean", "std", "count"])
    )
    trend.columns = ["_".join(c) for c in trend.columns]
    trend = trend.reset_index()
    trend.insert(0, "method", method_name)
    trend.insert(0, "model", model_name)

    print(f"Grade-wise AOPC trend for model={model_name}, method={method_name}:")
    print(trend.to_string(index=False))

    if save_path:
        trend.to_csv(save_path, index=False)
        print("Saved:", save_path)

    return trend


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    for model in ["efficientnetb4", "resnet50"]:
        for method in ["gradcam", "lime", "shap"]:
            run_grade_trend(
                model_name=model,
                method_name=method,
                save_path=os.path.join(
                    SAVE_DIR,
                    f"grade_trend_{model}_{method}.csv",
                ),
            )
