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

