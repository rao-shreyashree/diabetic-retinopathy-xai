"""
eval/aopc.py

computes AOPC (Area Over the Perturbation Curve) for insertion and deletion
perturbation order: most-relevant-first (descending heatmap value)
replacement: Gaussian blur baseline (sigma=10)
steps: 20 (5% of pixels perturbed per step)

outputs per (image, model, method):
  - deletion_auc, insertion_auc
  - per-step confidence arrays (for plotting curves)

saves to: results/scores/aopc_{model}_{method}.csv
curves (for plotting): results/scores/aopc_curves_{model}_{method}.npz

usage in Colab:
    import sys
    sys.path.append(".../eval")
    from aopc import run_aopc
    df = run_aopc(model_name="efficientnetb4", method_name="gradcam")

CLI:
    !python aopc.py --model efficientnetb4 --method gradcam
"""

import os
import json
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize
from fidelity_scoring import (
    PROJECT_ROOT, HEATMAP_ROOT, ORIG_IMG_DIR, RESULTS_DIR, LESION_DIRS,
)

SCORES_DIR = os.path.join(PROJECT_ROOT, "results", "scores")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
N_STEPS = 20
GAUSSIAN_SIGMA = 10
IMG_SIZE = 512

CHECKPOINT_MAP = {
    "efficientnetb4": os.path.join(CHECKPOINT_DIR, "efficientnet_b4_best.pth"),
    "resnet50":       os.path.join(CHECKPOINT_DIR, "resnet50_best.pth"),
}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(model_name: str, device: torch.device):
    if model_name == "efficientnetb4":
        import timm
        model = timm.create_model("efficientnet_b4", num_classes=5, pretrained=False)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(CHECKPOINT_MAP[model_name], map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def make_blur_baseline(img_rgb: np.ndarray) -> np.ndarray:
    """Gaussian-blurred version of image as the 'uninformative' baseline."""
    ksize = int(6 * GAUSSIAN_SIGMA + 1) | 1  # ensure odd
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), GAUSSIAN_SIGMA)


def get_sorted_pixel_indices(heatmap_clean: np.ndarray):
    """Returns flat indices sorted most-relevant-first."""
    return np.argsort(heatmap_clean.ravel())[::-1]


def perturb_image(
    img_resized: np.ndarray, # H x W x 3, uint8
    baseline: np.ndarray, # H x W x 3, uint8
    sorted_indices: np.ndarray,
    step: int,
    n_steps: int,
    mode: str, # "deletion" or "insertion"
) -> np.ndarray:
    """
    Returns perturbed image at given step.
    deletion: replace top-k% most relevant pixels with baseline
    insertion: start from baseline, reveal top-k% most relevant pixels
    """
    total_pixels = img_resized.shape[0] * img_resized.shape[1]
    k = int((step / n_steps) * total_pixels)
    top_k = sorted_indices[:k]

    if mode == "deletion":
        perturbed = img_resized.copy()
        perturbed_flat = perturbed.reshape(-1, 3)
        baseline_flat = baseline.reshape(-1, 3)
        perturbed_flat[top_k] = baseline_flat[top_k]
        return perturbed_flat.reshape(img_resized.shape)
    else:  # insertion
        perturbed = baseline.copy()
        perturbed_flat = perturbed.reshape(-1, 3)
        img_flat = img_resized.reshape(-1, 3)
        perturbed_flat[top_k] = img_flat[top_k]
        return perturbed_flat.reshape(img_resized.shape)


@torch.no_grad()
def get_confidence(model, img_rgb: np.ndarray, predicted_grade: int, device) -> float:
    """Returns softmax confidence for the predicted class."""
    tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    return probs[0, predicted_grade].item()


def compute_aopc_curves(
    model,
    img_rgb: np.ndarray,
    baseline: np.ndarray,
    heatmap_clean: np.ndarray,
    predicted_grade: int,
    device,
    n_steps: int = N_STEPS,
):
    """Returns deletion_curve, insertion_curve — each of length n_steps+1."""
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    baseline_resized = cv2.resize(baseline, (IMG_SIZE, IMG_SIZE))
    hmap_resized = cv2.resize(heatmap_clean, (IMG_SIZE, IMG_SIZE))
    sorted_indices = get_sorted_pixel_indices(hmap_resized)

    deletion_curve, insertion_curve = [], []
    for step in range(n_steps + 1):
        del_img = perturb_image(img_resized, baseline_resized, sorted_indices, step, n_steps, "deletion")
        ins_img = perturb_image(img_resized, baseline_resized, sorted_indices, step, n_steps, "insertion")
        deletion_curve.append(get_confidence(model, del_img, predicted_grade, device))
        insertion_curve.append(get_confidence(model, ins_img, predicted_grade, device))

    return np.array(deletion_curve), np.array(insertion_curve)


def auc_from_curve(curve: np.ndarray) -> float:
    """Trapezoidal AUC, normalized to [0,1]."""
    return float(np.trapz(curve) / (len(curve) - 1))


def run_aopc(
    model_name: str,
    method_name: str,
    test_ids: Optional[List[str]] = None,
    predictions_csv: Optional[str] = None,
    save_dir: str = SCORES_DIR,
    n_steps: int = N_STEPS,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(model_name, device)

    if test_ids is None:
        with open(os.path.join(PROJECT_ROOT, "test_image_ids.json")) as f:
            test_ids = sorted(json.load(f))

    if predictions_csv is None:
        predictions_csv = os.path.join(PROJECT_ROOT, "results", "heatmaps", "predictions.csv")
    predictions = pd.read_csv(predictions_csv)

    heatmap_folder = os.path.join(HEATMAP_ROOT, model_name, method_name)
    os.makedirs(save_dir, exist_ok=True)

    records = []
    all_del_curves, all_ins_curves = {}, {}

    for img_id in test_ids:
        matches = [f for f in os.listdir(heatmap_folder) if img_id in f and f.endswith(".npy")]
        if len(matches) != 1:
            print(f"SKIP {img_id}: expected 1 heatmap, found {len(matches)}")
            continue

        heatmap = np.load(os.path.join(heatmap_folder, matches[0]))
        fundus_path = os.path.join(ORIG_IMG_DIR, f"{img_id}.jpg")
        img_bgr = cv2.imread(fundus_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        fov_mask = get_fundus_mask(fundus_path)
        if fov_mask.shape != heatmap.shape:
            fov_mask = cv2.resize(fov_mask, (heatmap.shape[1], heatmap.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        heatmap_clean = apply_fov_mask_and_renormalize(heatmap, fov_mask)

        row = predictions[
            (predictions["image_id"] == img_id) & (predictions["model"] == model_name)
        ]
        if row.empty:
            print(f"SKIP {img_id}: no prediction row")
            continue
        predicted_grade = int(row["predicted_grade"].values[0])

        baseline = make_blur_baseline(img_rgb)
        del_curve, ins_curve = compute_aopc_curves(
            model, img_rgb, baseline, heatmap_clean, predicted_grade, device, n_steps
        )
        del_auc = auc_from_curve(del_curve)
        ins_auc = auc_from_curve(ins_curve)
        all_del_curves[img_id] = ins_curve
        all_ins_curves[img_id] = del_curve

        records.append({
            "image_id": img_id,
            "model": model_name,
            "method": method_name,
            "predicted_grade": predicted_grade,
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
        })
        print(f"{img_id}: del_auc={del_auc:.4f}, ins_auc={ins_auc:.4f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, f"aopc_{model_name}_{method_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    npz_path = os.path.join(save_dir, f"aopc_curves_{model_name}_{method_name}.npz")
    np.savez(npz_path, deletion=all_del_curves, insertion=all_ins_curves)
    print(f"Saved curves: {npz_path}")

    print(f"\nMean deletion AUC: {df['deletion_auc'].mean():.4f}")
    print(f"Mean insertion AUC: {df['insertion_auc'].mean():.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnetb4")
    parser.add_argument("--method", default="gradcam")
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    args = parser.parse_args()
    run_aopc(model_name=args.model, method_name=args.method, n_steps=args.n_steps)