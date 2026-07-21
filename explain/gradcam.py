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


class CLIPGradCAM:

    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head     = head

    def _get_target_layer(self):
        """Last ViT encoder LayerNorm — highest-level spatial features."""
        return self.backbone.clip.vision_model.encoder.layers[-1].layer_norm2

    def generate(
        self,
        pixel_values: torch.Tensor,   # (1, 3, 224, 224)
        text_emb: torch.Tensor,       # (1, D)
    ) -> np.ndarray:
        """
        Returns (224, 224) float32 heatmap, values in [0, 1].
        High values = regions causally driving the inconsistency score.
        """
        wrapper = _VisionWrapper(self.backbone, self.head, text_emb)
        cam     = GradCAM(
            model=wrapper,
            target_layers=[self._get_target_layer()],
            reshape_transform=_reshape_transform,
        )
        grayscale = cam(
            input_tensor=pixel_values,
            targets=[ClassifierOutputTarget(0)],
        )
        return grayscale[0]   # (224, 224)

    @staticmethod
    def overlay(
        image_np: np.ndarray,    # (H, W, 3) float32 in [0, 1]
        heatmap: np.ndarray,     # (H, W)    float32 in [0, 1]
    ) -> np.ndarray:
        """Returns (H, W, 3) uint8 RGB blended overlay."""
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        return show_cam_on_image(image_np, heatmap_resized, use_rgb=True)
