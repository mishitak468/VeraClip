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

