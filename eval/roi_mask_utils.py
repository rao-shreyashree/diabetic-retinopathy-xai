"""
roi_mask_utils.py
shared FOV (field-of-view) masking utility for diabetic retinopathy heatmaps

problem: 
heatmaps normalized over the full rectangular image (including black background/corners) let border artifacts dominate 
global min-max normalization, corrupting downstream fidelity scoring (IoU/Dice against lesion masks)
this is what broke effnet_gradcam's spot-check

fix: 
mask to the circular fundus region BEFORE normalizing, so background pixels can never become the global max/min

Usage: (for Tanvi and Shravani - shared across GradCAM, LIME, SHAP wrappers + fidelity_scoring.py)
    from roi_mask_utils import get_fundus_mask, apply_fov_mask_and_renormalize
    fov_mask = get_fundus_mask(image_path)
    clean_heatmap = apply_fov_mask_and_renormalize(raw_heatmap, fov_mask)
"""

import cv2
import numpy as np


def get_fundus_mask(image_path: str, threshold: int = 10) -> np.ndarray:
    """
    Binary mask (1=fundus, 0=background) of the circular retinal disc,
    derived from the original fundus image.

    Args:
        image_path: path to the original .jpg fundus image
        threshold: grayscale intensity below which pixels are treated as
                   background (black border). Default 10 works for IDRiD.

    Returns:
        float32 array, shape (H, W), values in {0.0, 1.0}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: no contour found, treat whole image as FOV
        return np.ones_like(gray, dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(gray)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)

    return (filled > 0).astype(np.float32)


def apply_fov_mask_and_renormalize(
    heatmap: np.ndarray,
    fov_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Zero out background, then redo per-image min-max normalization using
    ONLY values inside the FOV. Prevents border artifacts from dominating
    the global min/max (the effnet_gradcam corner-blob bug).

    Args:
        heatmap: raw saliency map, shape (H, W)
        fov_mask: binary mask from get_fundus_mask(), same shape as heatmap
        eps: numerical stability constant

    Returns:
        float32 array, shape (H, W), values in [0, 1] inside FOV, 0 outside
    """
    if heatmap.shape != fov_mask.shape:
        raise ValueError(
            f"Shape mismatch: heatmap={heatmap.shape}, fov_mask={fov_mask.shape}. "
            f"Resize fov_mask to match heatmap before calling this function."
        )

    masked = heatmap * fov_mask
    fov_vals = masked[fov_mask.astype(bool)]

    if fov_vals.size == 0:
        return masked.astype(np.float32)

    vmin, vmax = fov_vals.min(), fov_vals.max()
    if vmax - vmin < eps:
        # flat heatmap inside FOV
        return masked.astype(np.float32)

    normalized = (masked - vmin) / (vmax - vmin + eps)
    return (normalized * fov_mask).astype(np.float32)


if __name__ == "__main__":
    # self-test with a synthetic example
    fake_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(fake_img, (50, 50), 40, 255, -1)
    fake_heatmap = np.random.rand(100, 100).astype(np.float32)
    fake_heatmap[0:10, 0:10] = 999.0  # simulated border artifact

    mask = (fake_img > 10).astype(np.float32)
    clean = apply_fov_mask_and_renormalize(fake_heatmap, mask)

    assert clean[0, 0] == 0.0, "Border artifact not zeroed"
    assert np.isclose(clean[mask.astype(bool)].max(), 1.0), "FOV max not ~1.0"
    print("Self-test passed.")
