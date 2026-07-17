"""
veraclip/model/train.py
Full training loop. Run: python -m model.train

What this does:
  1. Loads CLIP-ViT-L/14 + applies LoRA adapters
  2. Attaches InconsistencyHead on top
  3. Trains with BCE + contrastive loss, cosine LR schedule, fp16
  4. Saves best checkpoint by validation AUC to model/checkpoints/best_model.pt
  5. Logs everything to WandB
"""

import os
import yaml
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

from model.clip_base    import CLIPBackbone
from model.lora_adapter import apply_lora
from model.fusion_head  import InconsistencyHead
from model.loss         import MisinfoLoss
from data.dataset       import MisinfoDataset

load_dotenv()


def load_config(path: str = "model/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


