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

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids:      (B, seq_len)
        attention_mask: (B, seq_len)
        Returns:        (B, embed_dim) L2-normalized
        """
        out  = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0]
        proj = self.clip.text_projection(cls)
        return proj / proj.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Convenience: returns both in one shot (used in train / eval loops)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_emb  = self.encode_image(pixel_values)
        text_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, text_emb

    # ------------------------------------------------------------------
    # Helper for GradCAM — exposes the vision model directly
    # ------------------------------------------------------------------

    def get_vision_encoder(self):
        return self.clip.vision_model

    def get_visual_projection(self):
        return self.clip.visual_projection
