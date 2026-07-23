"""
veraclip/explain/visualize.py
Utilities for rendering and saving GradCAM / attention outputs.
Used by the Gradio app and by eval/report.py.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
from pathlib import Path


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert float32 [0,1] heatmap to (H,W,3) uint8 RGB using JET colormap.
    Blue = low activation, Red = high activation.
    """
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def blend_heatmap(
    image_rgb: np.ndarray,     # (H, W, 3) float32 in [0, 1]
    heatmap: np.ndarray,       # (H', W') float32 in [0, 1]
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Resize heatmap to match image, then alpha-blend.
    Returns (H, W, 3) uint8.
    """
    h, w = image_rgb.shape[:2]
    hmap  = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    color = heatmap_to_rgb(hmap).astype(np.float32) / 255.0
    blended = (1 - alpha) * image_rgb + alpha * color
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def attention_to_image(
    patch_attn: np.ndarray,    # (num_patches,) e.g. 196 for ViT-L/14
    image_size: int = 224,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Reshape flat patch attention to 2-D, then upscale to image_size × image_size.
    Returns (image_size, image_size) float32 in [0, 1].
    """
    grid_size = image_size // patch_size   # 16 for ViT-L/14
    # Trim or pad to grid_size² in case of rounding
    n = grid_size * grid_size
    attn = patch_attn[:n]
    grid = attn.reshape(grid_size, grid_size)
    return cv2.resize(grid, (image_size, image_size), interpolation=cv2.INTER_CUBIC)


def save_explanation_figure(
    image_rgb: np.ndarray,    # (H, W, 3) uint8
    heatmap: np.ndarray,      # (H', W') float32
    save_path: str,
    verdict: str = "",
    score: float = 0.0,
) -> None:
    """
    Saves a 3-panel figure: original | GradCAM heatmap | overlay.
    Used by eval/report.py to generate per-sample visualizations.
    """
    img_float = image_rgb.astype(np.float32) / 255.0
    overlay   = blend_heatmap(img_float, heatmap)
    hmap_rgb  = heatmap_to_rgb(cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0])))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb);  axes[0].set_title("Original image");   axes[0].axis("off")
    axes[1].imshow(hmap_rgb);   axes[1].set_title("GradCAM heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay");           axes[2].axis("off")

    title = f"Verdict: {verdict.upper()}  |  Score: {score:.3f}" if verdict else f"Score: {score:.3f}"
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
