"""
xai/shap_wrapper.py

GradientSHAP wrapper for DR grading models (EfficientNet-B4, ResNet-50).

Design decisions (documented here for the writeup):
- Method: captum's GradientShap (gradient-based Shapley approximation).
  Chosen over KernelSHAP because KernelSHAP needs many perturbed forward
  passes per image to converge and is not designed for high-res CNN inputs --
  far too slow/unstable for 27 images x 2 models. GradientShap uses gradients
  (like GradCAM) plus noise + interpolation from a baseline distribution,
  giving Shapley-style attribution much faster and more stably on deep nets.
- Background distribution: a fixed random sample of Grade-0 (no DR) TRAINING
  images. Rationale: a black/zero baseline is not physiologically meaningful
  for a fundus photo (black pixels are just outside the retinal field of
  view, not "absence of disease"). Using Grade-0 images as baseline means
  attribution answers "what does this pixel contribute relative to healthy
  retinal tissue" -- clinically interpretable, ties directly to the lesion-
  localization framing of the thesis.
- Background sample is FIXED (same N images, same seed) across all test
  images and both models, not resampled per image. This is standard practice
  for GradientShap and far more stable than resampling per-call.
- n_samples kept LOW (diagnostic measurements showed n_samples=3 -> 51s,
  n_samples=5 -> 166s on 8GB CPU-only hardware -- a worse-than-linear
  "memory thrashing" signature, not raw compute cost. A single model
  forward+backward pass alone takes only ~4 seconds, confirming the
  bottleneck is captum's internal batch expansion / memory management,
  not the model itself. Resolution downsampling was tried and made things
  WORSE (added interpolation overhead without addressing the actual
  bottleneck) -- reverted. Final approach: keep n_samples and background
  size small and accept the per-image runtime, since that's what this
  hardware can sustain without crashing.
- Preprocessing matches week1_training.ipynb val_transform EXACTLY:
  Resize((512, 512)) -> ToTensor -> Normalize(ImageNet mean/std). No
  augmentation at inference time.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from captum.attr import GradientShap
from PIL import Image

# ---------------------------------------------------------------------------
# Preprocessing constants -- MUST match week1_training.ipynb val_transform
# ---------------------------------------------------------------------------
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Output contract constants -- MUST match the locked Week 2 contract
# ---------------------------------------------------------------------------
METHOD_NAME = "shap"
VALID_MODELS = {"efficientnetb4", "resnet50"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image_tensor(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Loads an image, applies val_transform, returns:
      - tensor of shape (1, 3, IMG_SIZE, IMG_SIZE)
      - original (H, W) for later upsampling back
    """
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)  # (H, W)
    tensor = val_transform(image).unsqueeze(0)  # (1, 3, 512, 512)
    return tensor, original_size


def build_background_tensor(
    grade0_image_paths: List[str],
    n_background: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds the FIXED background distribution for GradientShap.

    grade0_image_paths: list of file paths to Grade-0 (no DR) TRAINING images.
                         Must be passed in by caller -- this module does not
                         know about labels/CSVs, keeps it decoupled from the
                         dataset class.
    n_background: how many images to sample. COMMITTED VALUE: 3 -- only
                  configuration empirically tested as stable on this
                  hardware. Larger background sizes were not tested after
                  n_samples=5 already showed memory-pressure slowdown;
                  increasing background size multiplies memory cost further,
                  so 3 was kept as the safe, tested floor.

    Returns: tensor of shape (n_background, 3, IMG_SIZE, IMG_SIZE)
    """
    rng = random.Random(seed)
    if len(grade0_image_paths) < n_background:
        raise ValueError(
            f"Requested {n_background} background images but only "
            f"{len(grade0_image_paths)} Grade-0 images available."
        )
    sampled_paths = rng.sample(grade0_image_paths, n_background)

    tensors = []
    for p in sampled_paths:
        t, _ = load_image_tensor(p)
        tensors.append(t)
    background = torch.cat(tensors, dim=0).to(device)
    return background


class SHAPWrapper:
    """
    Model-agnostic GradientSHAP wrapper. Works with any nn.Module that
    outputs (N, NUM_CLASSES) logits -- i.e. both EfficientNet-B4 and
    ResNet-50 checkpoints, no model-specific code needed.
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        device: str = "cpu",
        n_samples: int = 3,
        stdevs: float = 0.1,
    ):
        """
        model: trained model in eval() mode, already loaded with checkpoint
        background: fixed background tensor from build_background_tensor()
        n_samples: number of randomly sampled points along the path between
                   baseline and input, per GradientShap call. COMMITTED
                   VALUE: 3. Empirically tested on this hardware (8GB RAM,
                   CPU-only): n_samples=3 -> 51s/image (stable), n_samples=5
                   -> 166s/image (disproportionate slowdown, memory pressure
                   signature). 3 was chosen as the stable, reproducible
                   setting -- see module docstring for full justification.
        stdevs: noise added to background samples for smoothing. Default
                from captum docs, rarely needs changing.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.background = background.to(device)
        self.n_samples = n_samples
        self.stdevs = stdevs
        self.explainer = GradientShap(self.model)

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> Tuple[int, float]:
        """Returns (predicted_class, confidence) for logging to predictions.csv."""
        logits = self.model(input_tensor.to(self.device))
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        return predicted_class.item(), confidence.item()

    def explain(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Runs GradientShap on a single image.

        input_tensor: (1, 3, IMG_SIZE, IMG_SIZE), already preprocessed
        target_class: which class to explain. Per the locked contract,
                       this should be the model's PREDICTED grade, not
                       ground truth. If None, predicts internally.

        Returns: (IMG_SIZE, IMG_SIZE) float32 heatmap, NOT yet normalized
                 or resized -- normalization/resize/saving handled by the
                 notebook driver loop, not this method, so this stays a
                 pure "explain one image" function.
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        if target_class is None:
            target_class, _ = self.predict(input_tensor)

        attributions = self.explainer.attribute(
            input_tensor,
            baselines=self.background,
            target=target_class,
            n_samples=self.n_samples,
            stdevs=self.stdevs,
        )
        # attributions shape: (1, 3, H, W) -- collapse channel dim
        # sum across channels (captum convention for multi-channel attribution)
        heatmap = attributions.sum(dim=1).squeeze(0)  # (H, W)
        heatmap = heatmap.detach().cpu().numpy().astype(np.float32)
        return heatmap


def normalize_and_resize(
    heatmap: np.ndarray, original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Applies the LOCKED output contract:
    - resize to ORIGINAL image dims (not model input size)
    - min-max normalize to [0, 1] -- LAST step, so the final array is
      guaranteed to hit exactly 0.0 and 1.0
    - no per-method normalization differences

    BUG FIX (caught in team review): previous version normalized to [0,1]
    THEN converted to uint8 and resized with bilinear interpolation. Bilinear
    interpolation averages neighboring pixels, which smooths out the exact
    peak value -- so after resizing, max was ~0.97-0.99, not exactly 1.0,
    while GradCAM/LIME (which normalize AFTER resizing) hit exactly 1.0.
    Fix: resize the RAW heatmap first (in float, no lossy uint8 round-trip),
    THEN do min-max normalization as the final step. This guarantees the
    final saved array always has min=0.0 and max=1.0 exactly, matching
    GradCAM/LIME's behavior.

    original_size: (H, W) of the ORIGINAL image, from load_image_tensor()
    """
    # Take absolute value first -- SHAP attributions can be negative
    # (negative = "this pixel pushed AWAY from predicted class"). For a
    # saliency map / fidelity comparison against binary lesion masks, we
    # care about magnitude of influence, not sign. Document this choice
    # in the writeup -- it's a real decision, not implicit.
    heatmap = np.abs(heatmap)

    # Resize FIRST, in float precision, no lossy uint8 conversion.
    # Use PIL's float-mode resize via numpy -> PIL "F" mode (32-bit float),
    # which avoids the uint8 round-trip that caused the original bug.
    target_h, target_w = original_size
    heatmap_img = Image.fromarray(heatmap.astype(np.float32), mode="F")
    heatmap_resized_img = heatmap_img.resize((target_w, target_h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized_img).astype(np.float32)

    # Normalize LAST -- guarantees exact [0, 1] bounds on the final array
    h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
    if h_max - h_min < 1e-8:
        # Degenerate case: completely flat attribution (shouldn't normally
        # happen, but guard against divide-by-zero). Treat as failure --
        # caller should skip this file per the "missing file = NaN" rule.
        raise ValueError("Degenerate heatmap: max == min, cannot normalize.")
    heatmap_final = (heatmap_resized - h_min) / (h_max - h_min)

    return heatmap_final.astype(np.float32)


def save_heatmap(
    heatmap: np.ndarray, model_name: str, image_id: str, output_root: str
) -> str:
    """
    Saves to results/heatmaps/{model}/shap/{model}_shap_{image_id}.npy
    per the locked naming/location contract.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name must be one of {VALID_MODELS}, got {model_name}")

    out_dir = Path(output_root) / model_name / METHOD_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{METHOD_NAME}_{image_id}.npy"
    np.save(out_path, heatmap)
    return str(out_path)