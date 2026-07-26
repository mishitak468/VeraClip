"""
veraclip/eval/evaluate.py
Runs the trained model on the VERITE test split and saves metrics.
Run: python -m eval.evaluate

Outputs:
  eval/results/test_metrics.json   — full metric dict
  eval/results/score_distribution.png — histogram of scores by class
"""

import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import CLIPProcessor
from tqdm import tqdm

from model.clip_base   import CLIPBackbone
from model.fusion_head import InconsistencyHead
from data.dataset      import MisinfoDataset
from eval.metrics      import compute_all_metrics, find_best_threshold

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    mc = cfg["model"]
    backbone = CLIPBackbone(mc["clip_model_name"]).to(device)
    head     = InconsistencyHead(
        embed_dim=backbone.embed_dim,
        hidden_dim=mc["fusion_hidden_dim"],
        dropout=mc["dropout"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval(); head.eval()
    print(f"Loaded checkpoint from {ckpt_path}  (epoch {ckpt.get('epoch','?')})")
    return backbone, head


def plot_score_distribution(labels, scores, save_path: str) -> None:
    consistent   = [s for s, l in zip(scores, labels) if l == 0]
    inconsistent = [s for s, l in zip(scores, labels) if l == 1]
    plt.figure(figsize=(8, 4))
    plt.hist(consistent,   bins=50, alpha=0.6, label="Consistent (label=0)",   color="#3B8BD4")
    plt.hist(inconsistent, bins=50, alpha=0.6, label="Inconsistent (label=1)", color="#E24B4A")
    plt.xlabel("Inconsistency score"); plt.ylabel("Count")
    plt.title("Score distribution by class — VeraClip VERITE test set")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


