"""
veraclip/app/app.py
Gradio web interface for VeraClip.
Run: python -m app.app   (or: make app)

Loads the trained checkpoint at startup, then handles real-time inference
via the Gradio UI. Works locally and on HuggingFace Spaces.
"""

import os
import yaml
import torch
import numpy as np
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

from model.clip_base    import CLIPBackbone
from model.lora_adapter import apply_lora
from model.fusion_head  import InconsistencyHead
from app.inference      import MisinfoInference

load_dotenv()

# ── Model loading (done once at startup) ─────────────────────────────────────

CONFIG_PATH = os.getenv("CONFIG_PATH", "model/config.yaml")
CKPT_PATH   = os.getenv("CKPT_PATH",   "model/checkpoints/best_model.pt")

cfg    = yaml.safe_load(open(CONFIG_PATH))
mc     = cfg["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading VeraClip on {device}...")
backbone = CLIPBackbone(mc["clip_model_name"]).to(device)
backbone = apply_lora(backbone, mc)
head     = InconsistencyHead(
    embed_dim=backbone.embed_dim,
    hidden_dim=mc["fusion_hidden_dim"],
    dropout=mc["dropout"],
).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
backbone.load_state_dict(ckpt["backbone_state"])
head.load_state_dict(ckpt["head_state"])
print(f"Checkpoint loaded (epoch {ckpt.get('epoch','?')}, val AUC {ckpt.get('val_auc', '?')})")

inferencer = MisinfoInference(backbone, head, device)

# ── Inference handler ─────────────────────────────────────────────────────────

def analyze(image: Image.Image, caption: str):
    if image is None:
        return None, "**Error:** Please upload an image.", "", ""
    if not caption or not caption.strip():
        return None, "**Error:** Please enter a caption to verify.", "", ""

    result = inferencer.predict(image, caption)

    overlay_pil = Image.fromarray(result["overlay"])

    # Verdict markdown
    icons   = {"inconsistent": "🔴", "consistent": "🟢", "uncertain": "🟡"}
    icon    = icons[result["verdict"]]
    verdict_md = (
        f"### {icon} Verdict: **{result['verdict'].upper()}**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Inconsistency score | `{result['score']}` |\n"
        f"| CLIP cosine similarity | `{result['cosine_sim']}` |\n"
        f"| Anomaly z-score | `{result['z_score']}σ` |\n\n"
        f"*Score → 1: caption likely misrepresents the image. "
        f"Score → 0: caption appears consistent.*"
    )

    # Attention token breakdown
    top5 = result["token_scores"][:5]
    tok_rows = "\n".join([f"| `{t['token']}` | {t['score']} |" for t in top5])
    attn_md = (
        f"### Top attention tokens\n\n"
        f"| Token | Importance |\n"
        f"|-------|------------|\n"
        f"{tok_rows}\n\n"
        f"High-importance tokens are the primary contradiction signals "
        f"the model used to reach its verdict."
    )

    return overlay_pil, verdict_md, attn_md, str(result["score"])


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
**VeraClip** detects whether a social media caption accurately describes its accompanying image.

Upload any image and paste the caption that appeared with it. The model highlights which
image regions and caption words drove the inconsistency verdict using GradCAM + attention rollout.

Fine-tuned on [VERITE](https://huggingface.co/datasets/Ftheodorakis/VERITE) · Architecture: CLIP-ViT-L/14 + LoRA (r=16)
"""
