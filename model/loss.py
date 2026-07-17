"""
veraclip/model/loss.py
Combined loss:
  1. BCE with label smoothing — standard binary classification signal
  2. Contrastive margin loss — pushes consistent pairs to high cosine sim,
     inconsistent pairs to low cosine sim (enforces embedding geometry)

Total loss = BCE + alpha * contrastive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MisinfoLoss(nn.Module):

    def __init__(
        self,
        label_smoothing: float = 0.1,
        margin: float = 0.3,
        alpha: float = 0.5,
    ):
        """
        Args:
            label_smoothing: prevents overconfidence (0.1 is standard)
            margin:          minimum cosine distance between classes
            alpha:           weight of contrastive term relative to BCE
        """
        super().__init__()
        self.bce             = nn.BCELoss()
        self.label_smoothing = label_smoothing
        self.margin          = margin
        self.alpha           = alpha

    def smooth(self, labels: torch.Tensor) -> torch.Tensor:
        """Soft targets: 0 → 0.05, 1 → 0.95 (with smoothing=0.1)."""
        return labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def contrastive(
        self,
        cosine_sim: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        For consistent pairs (label=0):   cosine should be high (+target direction)
        For inconsistent pairs (label=1): cosine should be low  (-target direction)
        Penalises any pair that is within `margin` of the wrong side.
        """
        target = 1.0 - 2.0 * labels            # consistent → +1, inconsistent → -1
        return F.relu(self.margin - target * cosine_sim).mean()

    def forward(
        self,
        scores: torch.Tensor,    # (B,) head output in [0,1]
        img_emb: torch.Tensor,   # (B, D) L2-normalized
        txt_emb: torch.Tensor,   # (B, D) L2-normalized
        labels: torch.Tensor,    # (B,) float 0 or 1
    ) -> tuple[torch.Tensor, float, float]:
        """
        Returns:
            total_loss:      scalar tensor (differentiable)
            bce_val:         float for logging
            contrastive_val: float for logging
        """
        cosine_sim   = (img_emb * txt_emb).sum(dim=-1)
        smooth_labels = self.smooth(labels)

        bce_loss     = self.bce(scores, smooth_labels)
        contra_loss  = self.contrastive(cosine_sim, labels)
        total        = bce_loss + self.alpha * contra_loss

        return total, bce_loss.item(), contra_loss.item()
