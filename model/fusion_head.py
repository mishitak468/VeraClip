"""
veraclip/model/fusion_head.py
MLP classification head that sits on top of the CLIP embeddings.

Input:  cosine similarity scalar + concatenated (img_emb, txt_emb)
Output: inconsistency score in [0, 1]
        0 = consistent (caption matches image)
        1 = inconsistent (likely misinformation)

Architecture:
    [cosine | img_emb | txt_emb]  →  Linear(1+2D, 512)  →  LN  →  GELU  →  Dropout
                                  →  Linear(512, 256)           →  GELU  →  Dropout
                                  →  Linear(256, 1)             →  Sigmoid
"""

import torch
import torch.nn as nn


class InconsistencyHead(nn.Module):

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        input_dim = 1 + 2 * embed_dim   # cosine scalar + two embeddings

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

