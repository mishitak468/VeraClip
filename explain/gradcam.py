"""
veraclip/explain/gradcam.py
GradCAM on the ViT vision encoder.

Why it works here:
  ViT encodes an image as a grid of patch tokens. GradCAM differentiates
  the inconsistency score w.r.t. the last encoder layer's activations,
  then maps those gradients back to the 16×16 patch grid → spatial heatmap.

  High-activation patches = "the model thinks THIS region is inconsistent
  with the caption."  That's what the UI shows as the red overlay.

Usage:
    gcam    = CLIPGradCAM(backbone, head)
    heatmap = gcam.generate(pixel_values, text_emb)   # (224, 224)
    overlay = gcam.overlay(image_np, heatmap)          # (224, 224, 3) uint8
"""

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class _VisionWrapper(torch.nn.Module):
    """
    Minimal wrapper so GradCAM can call a single forward() that returns
    a 1-D score. Receives frozen text_emb via closure at construction time.
    """

    def __init__(self, backbone, head, text_emb: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.head     = head
        self.text_emb = text_emb  # frozen during GradCAM differentiation

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        img_emb = self.backbone.encode_image(pixel_values)
        score   = self.head(img_emb, self.text_emb.expand(img_emb.size(0), -1))
        return score.unsqueeze(-1)   # GradCAM expects (B, num_classes)


def _reshape_transform(tensor, height: int = 16, width: int = 16) -> torch.Tensor:
    """
    ViT hidden states: (B, num_patches+1, D).
    Drop CLS (index 0), reshape to (B, D, H, W) as GradCAM expects.
    ViT-L/14 at 224px: 224/14 = 16 → 16×16 = 256 patches.
    """
    patches = tensor[:, 1:, :]                             # drop CLS  → (B, 256, D)
    B, N, D = patches.shape
    return patches.reshape(B, height, width, D).permute(0, 3, 1, 2)   # (B, D, H, W)


