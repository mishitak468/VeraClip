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


