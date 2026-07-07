"""
veraclip/model/clip_base.py
Thin wrapper around HuggingFace CLIP.
Exposes separate image and text encoders that return L2-normalized embeddings.
LoRA is applied on top of this class in lora_adapter.py.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPBackbone(nn.Module):
    """
    Wraps openai/clip-vit-large-patch14.
    embed_dim = 768 (ViT-L/14 projection dimension).

    forward() returns (img_emb, txt_emb) both shape (B, embed_dim), L2-normalized.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip      = CLIPModel.from_pretrained(model_name)
        self.embed_dim = self.clip.config.projection_dim  # 768

    # ------------------------------------------------------------------
    # Individual encoders (used during GradCAM + attention rollout)
    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, 224, 224)
        Returns:      (B, embed_dim) L2-normalized
        """
        out  = self.clip.vision_model(pixel_values=pixel_values)
        cls  = out.last_hidden_state[:, 0]          # CLS token
        proj = self.clip.visual_projection(cls)
        return proj / proj.norm(dim=-1, keepdim=True)

