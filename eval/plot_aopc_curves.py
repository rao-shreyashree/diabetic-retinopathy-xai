import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"
SCORES_DIR = os.path.join(PROJECT_ROOT, "results/scores")

models = ["efficientnetb4", "resnet50"]
methods = ["gradcam", "lime", "shap"]
colors = {"gradcam": "red", "lime": "blue", "shap": "green"}
steps = np.linspace(0, 1, 21)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
titles = ["EfficientNet-B4 Deletion", "EfficientNet-B4 Insertion",
          "ResNet-50 Deletion", "ResNet-50 Insertion"]

for i, model in enumerate(models):
    for j, mode in enumerate(["deletion", "insertion"]):
        ax = axes[i][j]
        ax.set_title(titles[i*2 + j])
        for method in methods:
            npz_path = os.path.join(SCORES_DIR, f"aopc_curves_{model}_{method}.npz")
            data = np.load(npz_path, allow_pickle=True)
            curves = data[mode].item()

            raw = np.stack(list(curves.values()))
            # normalizing each image curve by its step-0 value
            normalized = raw / (raw[:, 0:1] + 1e-8)
            mean_curve = np.mean(normalized, axis=0)

            ax.plot(steps, mean_curve, label=method.upper(), color=colors[method])
        ax.set_xlabel("Fraction of pixels perturbed")
        ax.set_ylabel("Mean model confidence")
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(PROJECT_ROOT, "results/figures/aopc_curves.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", out)