"""
veraclip/explain/attn_rollout.py
Two explainability signals:

1. attention_rollout(model, pixel_values)
   → (num_patches,) float array  — which image patches the ViT attended to most.
   Algorithm: Abnar & Zuidema 2020 (https://arxiv.org/abs/2005.00928)
   Propagates attention across all layers; avoids the "last-layer only" blind spot.

2. token_importance(text_emb, tokens)
   → list[{token, score}]        — which caption words pulled hardest on the text embedding.
   Gradient-free proxy: L2 norm of each position's contribution is a reasonable
   first approximation without running a full integrated-gradients pass.
"""

import torch
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Image-side: attention rollout over ViT layers
# ---------------------------------------------------------------------------

def attention_rollout(
    backbone,
    pixel_values: torch.Tensor,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Args:
        backbone:      CLIPBackbone (with or without LoRA)
        pixel_values:  (1, 3, 224, 224) preprocessed tensor
        discard_ratio: fraction of lowest-attention tokens to zero out per layer

    Returns:
        (num_patches,) float32 array, values in [0, 1].
        For ViT-L/14: num_patches = 14×14 = 196.
    """
    attentions: list[torch.Tensor] = []

    def _hook(module, input, output):
        # HuggingFace CLIP self-attention returns (context_layer, attn_weights)
        # attn_weights shape: (B, num_heads, seq_len, seq_len)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attentions.append(output[1].detach().cpu())

    hooks = []
    for layer in backbone.clip.vision_model.encoder.layers:
        hooks.append(
            layer.self_attn.register_forward_hook(_hook)
        )

    with torch.no_grad():
        backbone.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
        )

    for h in hooks:
        h.remove()

    # Fallback: if no attention weights captured (some PEFT configs suppress them)
    if not attentions:
        num_patches = (pixel_values.shape[-1] // 14) ** 2   # ViT-L/14 → 256
        return np.ones(num_patches, dtype=np.float32)

    # Rollout algorithm
    result = torch.eye(attentions[0].size(-1))

    for attn in attentions:
        # Average across heads → (seq_len, seq_len)
        attn_avg = attn.mean(dim=1)[0]

        # Discard low-attention connections
        flat      = attn_avg.view(-1)
        threshold = torch.quantile(flat, discard_ratio)
        attn_avg[attn_avg < threshold] = 0.0

        # Add residual identity and re-normalize rows
        A      = (attn_avg + torch.eye(attn_avg.size(0))) / 2.0
        A      = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        result = torch.matmul(A, result)

    # Row 0 = CLS token; columns 1: = patch tokens
    mask = result[0, 1:].numpy().astype(np.float32)
    denom = mask.max() - mask.min() + 1e-8
    return (mask - mask.min()) / denom


# ---------------------------------------------------------------------------
# Text-side: token importance via embedding norm
# ---------------------------------------------------------------------------

