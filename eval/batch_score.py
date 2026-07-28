"""
veraclip/eval/batch_score.py
Scores all 10,000 NewsCLIPpings pairs and writes a CSV.
This is what generates the resume stat:
  "Processed and scored 10,000+ image-caption pairs during evaluation"

Run: python -m eval.batch_score

Outputs:
  eval/results/newsclipper_scores.csv  — id, score, label, caption_snippet
  eval/results/newsclipper_summary.json — aggregate statistics
"""

import csv
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from torchvision import transforms
from transformers import CLIPProcessor
from torch.cuda.amp import autocast

from model.clip_base   import CLIPBackbone
from model.fusion_head import InconsistencyHead

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])


def load_model(cfg: dict, device: torch.device):
    mc       = cfg["model"]
    backbone = CLIPBackbone(mc["clip_model_name"]).to(device)
    head     = InconsistencyHead(embed_dim=backbone.embed_dim, hidden_dim=mc["fusion_hidden_dim"]).to(device)
    ckpt     = torch.load("model/checkpoints/best_model.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval(); head.eval()
    return backbone, head


