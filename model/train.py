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


def train():
    cfg = load_config()
    mc, tc, dc, lc = cfg["model"], cfg["training"], cfg["data"], cfg["logging"]

    wandb.init(project=lc["wandb_project"], config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone, head = build_model(mc, device)
    processor      = CLIPProcessor.from_pretrained(mc["clip_model_name"])
    criterion      = MisinfoLoss(label_smoothing=tc["label_smoothing"])

    train_ds = MisinfoDataset(dc["processed_dir"], "train",      processor.tokenizer, dc["max_caption_length"])
    val_ds   = MisinfoDataset(dc["processed_dir"], "validation", processor.tokenizer, dc["max_caption_length"])

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,  num_workers=dc["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tc["batch_size"], shuffle=False, num_workers=dc["num_workers"], pin_memory=True)

    params    = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tc["warmup_steps"],
        num_training_steps=len(train_loader) * tc["num_epochs"],
    )
    scaler = GradScaler(enabled=tc["mixed_precision"] and torch.cuda.is_available())

    ckpt_dir = Path(lc["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, tc["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{tc['num_epochs']}")

        tr_loss, tr_auc, tr_acc = run_epoch(
            backbone, head, train_loader, criterion,
            optimizer, scaler, scheduler, device, tc, train=True
        )
        va_loss, va_auc, va_acc = run_epoch(
            backbone, head, val_loader, criterion,
            optimizer, scaler, scheduler, device, tc, train=False
        )

        print(f"  train  loss={tr_loss:.4f}  auc={tr_auc:.4f}  acc={tr_acc:.4f}")
        print(f"  val    loss={va_loss:.4f}  auc={va_auc:.4f}  acc={va_acc:.4f}")

        wandb.log({
            "epoch":      epoch,
            "train/loss": tr_loss, "train/auc": tr_auc, "train/acc": tr_acc,
            "val/loss":   va_loss, "val/auc":   va_auc, "val/acc":   va_acc,
        })

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({
                "epoch":           epoch,
                "backbone_state":  backbone.state_dict(),
                "head_state":      head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc":         va_auc,
                "val_acc":         va_acc,
                "config":          cfg,
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ New best saved  (AUC={va_auc:.4f})")

        if epoch % lc["save_every_n_epochs"] == 0:
            torch.save({
                "epoch":          epoch,
                "backbone_state": backbone.state_dict(),
                "head_state":     head.state_dict(),
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    wandb.finish()
    print(f"\nDone. Best validation AUC: {best_auc:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train()
