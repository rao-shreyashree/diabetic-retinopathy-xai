"""
utils/config.py
Single source of truth for all paths and constants.
Everyone imports from here — never hardcode paths in notebooks.
"""
import os

# ── Drive root (Colab) ──────────────────────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive/Projects/diabetic retinopathy/diabetic-retinopathy-xai"

# ── IDRiD data paths ────────────────────────────────────────────────────────
IDRID_ROOT        = os.path.join(DRIVE_ROOT, "data/IDRiD")
GRADING_TRAIN_DIR = os.path.join(IDRID_ROOT, "grading/images/train")
GRADING_TEST_DIR  = os.path.join(IDRID_ROOT, "grading/images/test")
TRAIN_CSV         = os.path.join(IDRID_ROOT, "grading/labels/train.csv")
TEST_CSV          = os.path.join(IDRID_ROOT, "grading/labels/test.csv")

# Lesion mask folders (Task 1 — segmentation)
# Used by P3 (fidelity scoring) but defined here so everyone uses same names
LESION_DIRS = {
    "MA": os.path.join(IDRID_ROOT, "segmentation/masks/test/1. Microaneurysms"),
    "HE": os.path.join(IDRID_ROOT, "segmentation/masks/test/2. Haemorrhages"),
    "EX": os.path.join(IDRID_ROOT, "segmentation/masks/test/3. Hard Exudates"),
    "SE": os.path.join(IDRID_ROOT, "segmentation/masks/test/4. Soft Exudates"),
}

# ── Results paths ────────────────────────────────────────────────────────────
RESULTS_ROOT  = os.path.join(DRIVE_ROOT, "results")
CKPT_DIR      = os.path.join(RESULTS_ROOT, "checkpoints")
HEATMAP_DIR   = os.path.join(RESULTS_ROOT, "heatmaps")
SCORES_DIR    = os.path.join(RESULTS_ROOT, "scores")

# Checkpoint filenames (matches what week1_training.ipynb saved)
CKPT = {
    "efficientnetb4": os.path.join(CKPT_DIR, "efficientnet_b4_best.pth"),
    "resnet50":       os.path.join(CKPT_DIR, "resnet50_best.pth"),
}

# Heatmap output dirs — one per model per method
# Final path pattern: HEATMAP_DIR/{model}/{method}/{model}_{method}_{image_id}.npy
HEATMAP_MODELS  = ["efficientnetb4", "resnet50"]
HEATMAP_METHODS = ["gradcam", "lime", "shap"]

# Predictions CSV (shared sidecar — written once by P1, read by P2 and P3)
PREDICTIONS_CSV = os.path.join(HEATMAP_DIR, "predictions.csv")

# Frozen test image list
TEST_IDS_JSON = "/content/diabetic-retinopathy-xai/test_image_ids.json"

# ── Model hyperparameters (must match week1_training.ipynb exactly) ──────────
IMG_SIZE    = 512
NUM_CLASSES = 5
DEVICE_STR  = "cuda"   # will fall back to cpu if unavailable

# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── XAI settings ─────────────────────────────────────────────────────────────
# LIME — number of superpixels and perturbation samples
LIME_NUM_SAMPLES    = 1000   # higher = more accurate, slower
LIME_NUM_SEGMENTS   = 50     # superpixels (quickshift segmentation)
LIME_BATCH_SIZE     = 32

# GradCAM — target layer names (one per model architecture)
GRADCAM_LAYER = {
    "efficientnetb4": "conv_head",   # last conv layer in EfficientNet-B4 (timm)
    "resnet50":       "layer4",      # last residual block in ResNet-50
}
