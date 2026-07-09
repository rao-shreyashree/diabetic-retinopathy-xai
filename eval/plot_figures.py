"""
eval/plot_figures.py
Two paper figures:
1. ViT AOPC grade trend (deletion + insertion AUC vs predicted_grade)
2. Method rank correlation heatmap (Spearman rho, per model)

usage:
    !python plot_figures.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AOPC_DIR = os.path.join(PROJECT_ROOT, "results", "scores", "aopc")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "results", "scores", "summary")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def plot_vit_aopc_grade_trend(save=True):
    df = pd.read_csv(os.path.join(SUMMARY_DIR, "grade_trend_vit_b16_attention_rollout.csv"))
    df = df.sort_values("predicted_grade")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(df["predicted_grade"], df["deletion_auc_mean"], yerr=df["deletion_auc_std"],
                marker="o", label="Deletion AUC", capsize=3)
    ax.errorbar(df["predicted_grade"], df["insertion_auc_mean"], yerr=df["insertion_auc_std"],
                marker="s", label="Insertion AUC", capsize=3)
    ax.set_xlabel("Predicted DR Grade")
    ax.set_ylabel("AOPC AUC")
    ax.set_title("ViT-B/16 Attention Rollout: AOPC vs Predicted Grade")
    ax.set_xticks(df["predicted_grade"])
    ax.legend()
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "vit_aopc_grade_trend.png")
        fig.savefig(path, dpi=300)
        print("Saved:", path)
    return fig


def plot_rank_correlation_heatmap(model_name: str, save=True):
    path = os.path.join(SUMMARY_DIR, f"method_rank_correlation_{model_name}.csv")
    df = pd.read_csv(path)

    methods = sorted(set(df["method_a"]) | set(df["method_b"]))
    mat = pd.DataFrame(1.0, index=methods, columns=methods)
    for _, row in df.iterrows():
        mat.loc[row["method_a"], row["method_b"]] = row["spearman_rho"]
        mat.loc[row["method_b"], row["method_a"]] = row["spearman_rho"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Method Rank Correlation (Spearman rho) - {model_name}")
    fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, f"rank_correlation_heatmap_{model_name}.png")
        fig.savefig(path, dpi=300)
        print("Saved:", path)
    return fig


if __name__ == "__main__":
    plot_vit_aopc_grade_trend()
    plot_rank_correlation_heatmap("efficientnetb4")
    plot_rank_correlation_heatmap("resnet50")
    plt.show()