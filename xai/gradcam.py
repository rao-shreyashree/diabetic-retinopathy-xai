"""
xai/gradcam.py

Hook-based GradCAM wrapper.
Works with any nn.Module — pass the model and the target layer name.
Follows the Week 2 output contract exactly:
  - output shape: (H, W) matching original image dims
  - float32, min-max normalized to [0, 1]
  - explains PREDICTED class, not ground truth
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Computes GradCAM heatmap for a given model and target layer.

    Usage:
        cam = GradCAM(model, target_layer_name="layer4")
        heatmap = cam.generate(tensor_input, original_pil_image)
        # heatmap is (H, W) float32 in [0, 1], resized to original image dims
    """

    def __init__(self, model, target_layer_name: str, device=None):
        """
        Args:
            model:             nn.Module, already loaded with weights, in eval mode
            target_layer_name: name of the layer to hook (e.g. "layer4", "conv_head")
                               use model.named_modules() to find the right name
            device:            torch.device — inferred from model if not given
        """
        self.model  = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self._activations = None
        self._gradients   = None

        # Find and hook the target layer
        target_layer = self._find_layer(target_layer_name)
        self._register_hooks(target_layer)

    def _find_layer(self, name: str):
        """Finds layer by name in model's named_modules."""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        # If exact match fails, try partial match and warn
        candidates = [n for n, _ in self.model.named_modules() if name in n]
        if candidates:
            print(f"[GradCAM] Exact layer '{name}' not found. "
                  f"Using closest match: '{candidates[-1]}'")
            for n, module in self.model.named_modules():
                if n == candidates[-1]:
                    return module
        raise ValueError(
            f"[GradCAM] Layer '{name}' not found in model. "
            f"Available layers:\n" +
            "\n".join(n for n, _ in self.model.named_modules() if n)
        )

    def _register_hooks(self, layer):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor_input: torch.Tensor,
                 original_pil: Image.Image,
                 target_class: int = None) -> np.ndarray:
        """
        Generates GradCAM heatmap for one image.

        Args:
            tensor_input:   (1, C, H, W) or (C, H, W) — preprocessed model input
            original_pil:   PIL.Image at original resolution — for final resize
            target_class:   class to explain. If None, uses predicted class.

        Returns:
            heatmap: np.ndarray (H_orig, W_orig) float32, normalized [0, 1]
                     H_orig, W_orig = original_pil.size[::-1]
        """
        self.model.eval()

        # Ensure batch dim
        if tensor_input.dim() == 3:
            tensor_input = tensor_input.unsqueeze(0)
        tensor_input = tensor_input.to(self.device)
        tensor_input.requires_grad_(False)

        # Forward pass
        self.model.zero_grad()
        output = self.model(tensor_input)           # (1, num_classes)
        logits = output[0]                          # (num_classes,)

        # Predicted class (what the model actually said)
        pred_class = logits.argmax().item()
        if target_class is None:
            target_class = pred_class

        # Backward pass on the target class score
        self.model.zero_grad()
        class_score = logits[target_class]
        class_score.backward()

        # activations: (1, C, h, w) — feature maps at target layer
        # gradients:   (1, C, h, w) — gradients of class score w.r.t. activations
        activations = self._activations[0]   # (C, h, w)
        gradients   = self._gradients[0]     # (C, h, w)

        # Global average pool gradients → weights per channel
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)  # (h, w)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().numpy()

        # Resize to original image dimensions
        orig_w, orig_h = original_pil.size   # PIL gives (W, H)
        cam_resized = self._resize_heatmap(cam, (orig_h, orig_w))

        # Min-max normalize to [0, 1] — contract requirement
        heatmap = self._normalize(cam_resized)

        return heatmap.astype(np.float32), pred_class

    @staticmethod
    def _resize_heatmap(cam: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resizes cam (h, w) to target_size (H, W) using PIL bilinear interpolation.
        This is softer than cv2 and doesn't require an extra dependency.
        """
        target_h, target_w = target_size
        cam_pil = Image.fromarray(cam.astype(np.float32), mode='F')
        cam_pil = cam_pil.resize((target_w, target_h), Image.BILINEAR)
        return np.array(cam_pil)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]. Returns zeros if map is flat."""
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_val) / (max_val - min_val)


def build_model_for_xai(model_name: str, ckpt_path: str, device: torch.device):
    """
    Loads a model with weights from checkpoint.
    Uses same architecture definitions as week1_training.ipynb.

    Args:
        model_name: "efficientnetb4" or "resnet50"
        ckpt_path:  path to .pth file (state_dict)
        device:     torch.device

    Returns:
        model in eval mode on device
    """
    import timm
    import torchvision.models as tv_models
    import torch.nn as nn

    if model_name == "efficientnetb4":
        model = timm.create_model("efficientnet_b4",
                                   pretrained=False,
                                   num_classes=5)
    elif model_name == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose 'efficientnetb4' or 'resnet50'.")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
