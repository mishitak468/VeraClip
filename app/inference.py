"""
veraclip/app/inference.py
Single-pair inference pipeline.
Called by app.py for every Gradio submission.

Returns a fully typed result dict that app.py renders into the UI.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor

from explain.gradcam      import CLIPGradCAM
from explain.attn_rollout import attention_rollout, token_importance
from explain.visualize    import blend_heatmap

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

CONSISTENT_THRESHOLD   = 0.35
INCONSISTENT_THRESHOLD = 0.60


class MisinfoInference:
    """
    Stateful inference wrapper.
    Instantiated once at Gradio startup, reused for every request.

    Args:
        backbone: trained CLIPBackbone (with LoRA)
        head:     trained InconsistencyHead
        device:   torch.device
    """

    def __init__(self, backbone, head, device: torch.device):
        self.backbone  = backbone
        self.head      = head
        self.device    = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.gcam      = CLIPGradCAM(backbone, head)

        self.backbone.eval()
        self.head.eval()

