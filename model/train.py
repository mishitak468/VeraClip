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


def build_model(mc: dict, device: torch.device):
    backbone = CLIPBackbone(mc["clip_model_name"]).to(device)
    backbone = apply_lora(backbone, mc)
    head     = InconsistencyHead(
        embed_dim=backbone.embed_dim,
        hidden_dim=mc["fusion_hidden_dim"],
        dropout=mc["dropout"],
    ).to(device)
    return backbone, head


def run_epoch(backbone, head, loader, criterion, optimizer, scaler, scheduler, device, tc, train=True):
    backbone.train(train)
    head.train(train)

    total_loss = 0.0
    all_scores, all_labels = [], []

    ctx = torch.enable_grad if train else torch.no_grad

    with ctx():
        for batch in tqdm(loader, desc="  train" if train else "  val", leave=False):
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["label"].to(device)

            with autocast(enabled=tc["mixed_precision"]):
                img_emb, txt_emb = backbone(pv, ii, am)
                scores           = head(img_emb, txt_emb)
                loss, bce, contra = criterion(scores, img_emb, txt_emb, lb)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()),
                    tc["max_grad_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels.extend(lb.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc      = roc_auc_score(all_labels, all_scores)
    acc      = ((torch.tensor(all_scores) > 0.5).float().numpy() == all_labels).mean()
    return avg_loss, auc, acc


