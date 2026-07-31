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

    def predict(self, image: Image.Image, caption: str) -> dict:
        """
        Args:
            image:   PIL Image (any size, any mode)
            caption: raw caption string

        Returns dict:
            score          float [0,1]     — inconsistency score
            cosine_sim     float [-1,1]    — raw CLIP cosine similarity
            verdict        str             — 'consistent' | 'uncertain' | 'inconsistent'
            overlay        np.ndarray      — (224,224,3) uint8 GradCAM overlay
            heatmap        np.ndarray      — (224,224)   float32 raw heatmap
            patch_attn     list[float]     — (num_patches,) attention rollout
            token_scores   list[dict]      — [{token, score}, ...] sorted desc
            z_score        float           — how many std devs from baseline cosine
        """
        # ── Preprocess ───────────────────────────────────────────────────────
        img_224  = image.convert("RGB").resize((224, 224))
        img_np   = np.array(img_224).astype(np.float32) / 255.0   # (224,224,3) in [0,1]
        pv       = CLIP_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)

        tok = self.processor(
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        ii = tok["input_ids"].to(self.device)
        am = tok["attention_mask"].to(self.device)

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            img_emb, txt_emb = self.backbone(pv, ii, am)
            score             = self.head(img_emb, txt_emb).item()
            cosine_sim        = (img_emb * txt_emb).sum().item()

        # ── Verdict ──────────────────────────────────────────────────────────
        if score >= INCONSISTENT_THRESHOLD:
            verdict = "inconsistent"
        elif score <= CONSISTENT_THRESHOLD:
            verdict = "consistent"
        else:
            verdict = "uncertain"

        # ── GradCAM ──────────────────────────────────────────────────────────
        heatmap = self.gcam.generate(pv, txt_emb)          # (224, 224)
        overlay = blend_heatmap(img_np, heatmap)            # (224, 224, 3) uint8

        # ── Attention rollout ────────────────────────────────────────────────
        patch_attn = attention_rollout(self.backbone, pv)   # (num_patches,)

        # ── Token importance ─────────────────────────────────────────────────
        tokens      = caption.split()
        tok_scores  = token_importance(txt_emb, tokens, self.backbone, ii, am)

        # ── Approximate z-score (vs. random-pair cosine baseline ≈ 0.0, std ≈ 0.25)
        z_score = (cosine_sim - 0.0) / 0.25

        return {
            "score":        round(score, 4),
            "cosine_sim":   round(cosine_sim, 4),
            "z_score":      round(z_score, 2),
            "verdict":      verdict,
            "overlay":      overlay,
            "heatmap":      heatmap,
            "patch_attn":   patch_attn.tolist(),
            "token_scores": tok_scores,
        }
