"""
xai/lime_wrapper.py

LIME wrapper for DR image explanation.
Uses lime_image with quickshift segmentation (faster + better for retinal images
than SLIC because quickshift respects color boundaries — lesions have distinct
color profiles from background tissue).

Output contract: same as GradCAM — (H_orig, W_orig) float32 in [0, 1]
Explains PREDICTED class.
"""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from lime import lime_image
from skimage.segmentation import quickshift

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, LIME_NUM_SAMPLES, LIME_NUM_SEGMENTS, LIME_BATCH_SIZE


# Same transform as inference (without normalization — LIME handles raw uint8)
_RESIZE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class LIMEWrapper:
    """
    LIME explanation for image classification models.

    LIME perturbs the image by masking superpixels, runs the model on
    each perturbation, and fits a local linear model to find which
    superpixels most influenced the prediction.

    Usage:
        lime = LIMEWrapper(model, device)
        heatmap = lime.generate(pil_image)
        # heatmap is (H, W) float32 in [0, 1] at original resolution
    """

    def __init__(self, model, device=None,
                 num_samples=None,
                 num_segments=None,
                 batch_size=None):
        """
        Args:
            model:        nn.Module, loaded with weights, eval mode
            device:       torch.device
            num_samples:  number of perturbations (default from config)
            num_segments: number of superpixels (default from config)
            batch_size:   inference batch size for perturbations
        """
        self.model       = model
        self.device      = device or next(model.parameters()).device
        self.num_samples = num_samples or LIME_NUM_SAMPLES
        self.num_segments = num_segments or LIME_NUM_SEGMENTS
        self.batch_size  = batch_size or LIME_BATCH_SIZE
        self.model.eval()

        self.explainer = lime_image.LimeImageExplainer(verbose=False)

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function that LIME calls with perturbed images.

        Args:
            images: (N, H, W, 3) uint8 numpy array — LIME's format

        Returns:
            probs: (N, num_classes) float32 softmax probabilities
        """
        all_probs = []
        for i in range(0, len(images), self.batch_size):
            batch_np = images[i:i + self.batch_size]
            tensors  = []
            for img_np in batch_np:
                pil  = Image.fromarray(img_np.astype(np.uint8))
                tens = _RESIZE_TRANSFORM(pil)
                tensors.append(tens)

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def generate(self, original_pil: Image.Image,
                 target_class: int = None) -> tuple:
        """
        Generates LIME heatmap for one image.

        Args:
            original_pil:  PIL.Image at original resolution
            target_class:  class to explain. If None, uses predicted class.

        Returns:
            heatmap:    np.ndarray (H, W) float32, normalized [0, 1]
                        at ORIGINAL image resolution
            pred_class: int — predicted class
        """
        # Get predicted class first
        tensor  = _RESIZE_TRANSFORM(original_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits     = self.model(tensor)
            pred_class = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # LIME needs resized image as uint8 numpy (H, W, 3)
        pil_resized = original_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np      = np.array(pil_resized)  # (512, 512, 3) uint8

        # Segmentation function using quickshift
        def segmentation_fn(image):
            return quickshift(
                image,
                kernel_size=4,       # spatial kernel size
                max_dist=200,        # max color distance for merging
                ratio=0.2,           # balances color vs space
                convert2lab=True,    # LAB color space — better for retinal images
            )

        # Run LIME
        explanation = self.explainer.explain_instance(
            img_np,
            self.predict_fn_wrapper,
            top_labels=5,
            hide_color=0,            # masked regions → black
            num_samples=self.num_samples,
            segmentation_fn=segmentation_fn,
            batch_size=self.batch_size,
            random_seed=42,
        )

        # Extract heatmap for target class
        # get_image_and_mask returns a mask where each superpixel gets a weight
        # We use the raw local explanation weights instead for a continuous map
        segments = explanation.segments    # (IMG_SIZE, IMG_SIZE) int — superpixel IDs
        local_exp = explanation.local_exp  # dict: class → [(seg_id, weight), ...]

        if target_class not in local_exp:
            # Model was very confident about one class — LIME only returns top_labels
            # Fall back to top-1 label
            target_class = list(local_exp.keys())[0]

        # Build continuous heatmap from superpixel weights
        heatmap_resized = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for seg_id, weight in local_exp[target_class]:
            heatmap_resized[segments == seg_id] = weight

        # Keep only positive contributions (same philosophy as GradCAM ReLU)
        heatmap_resized = np.maximum(heatmap_resized, 0)

        # Resize to original image dimensions
        orig_w, orig_h = original_pil.size
        heatmap = self._resize_heatmap(heatmap_resized, (orig_h, orig_w))

        # Min-max normalize
        heatmap = self._normalize(heatmap)

        return heatmap.astype(np.float32), pred_class

    def predict_fn_wrapper(self, images):
        """Wrapper exposed to LIME — identical to _predict_fn."""
        return self._predict_fn(images)

    @staticmethod
    def _resize_heatmap(cam: np.ndarray, target_size: tuple) -> np.ndarray:
        target_h, target_w = target_size
        cam_pil = Image.fromarray(cam.astype(np.float32), mode='F')
        cam_pil = cam_pil.resize((target_w, target_h), Image.BILINEAR)
        return np.array(cam_pil)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_val) / (max_val - min_val)
