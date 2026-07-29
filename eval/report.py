"""
veraclip/eval/report.py
Generates a visual HTML report of the worst false positives and false negatives.
Useful for qualitative error analysis and the README.
Run: python -m eval.report
"""

import json
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from torch.cuda.amp import autocast
from PIL import Image

from model.clip_base    import CLIPBackbone
from model.fusion_head  import InconsistencyHead
from data.dataset       import MisinfoDataset
from explain.gradcam    import CLIPGradCAM
from explain.visualize  import save_explanation_figure

REPORTS_DIR = Path("eval/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_report(n_samples: int = 20, config_path: str = "model/config.yaml") -> None:
    cfg    = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc     = cfg["model"]

    backbone = CLIPBackbone(mc["clip_model_name"]).to(device)
    head     = InconsistencyHead(embed_dim=backbone.embed_dim, hidden_dim=mc["fusion_hidden_dim"]).to(device)
    ckpt     = torch.load("model/checkpoints/best_model.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval(); head.eval()

    processor = CLIPProcessor.from_pretrained(mc["clip_model_name"])
    gcam      = CLIPGradCAM(backbone, head)
    ds        = MisinfoDataset(cfg["data"]["processed_dir"], "test", processor.tokenizer)
    loader    = DataLoader(ds, batch_size=1, shuffle=False)

    errors = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 200:   # check first 200 to find interesting errors
                break
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["label"].item()

            with autocast():
                img_emb, txt_emb = backbone(pv, ii, am)
                score = head(img_emb, txt_emb).item()

            pred = 1 if score > 0.5 else 0
            if pred != int(lb):
                errors.append({
                    "idx":     i,
                    "label":   int(lb),
                    "pred":    pred,
                    "score":   score,
                    "caption": batch["caption"][0],
                    "pv":      pv,
                    "img_emb": img_emb,
                    "txt_emb": txt_emb,
                })

    print(f"Found {len(errors)} errors in first 200 samples")
    errors.sort(key=lambda x: abs(x["score"] - 0.5))   # most confident errors first

    for j, err in enumerate(errors[:n_samples]):
        # Reconstruct image from tensor for visualization
        pv_np   = err["pv"][0].cpu().numpy().transpose(1, 2, 0)
        # Un-normalize for display
        mean    = np.array([0.48145466, 0.4578275,  0.40821073])
        std     = np.array([0.26862954, 0.26130258, 0.27577711])
        img_rgb = np.clip((pv_np * std + mean), 0, 1)

        heatmap = gcam.generate(err["pv"], err["txt_emb"])
        verdict = "FP" if err["label"] == 0 and err["pred"] == 1 else "FN"

        save_explanation_figure(
            (img_rgb * 255).astype(np.uint8),
            heatmap,
            save_path=str(REPORTS_DIR / f"{verdict}_{j:03d}.png"),
            verdict=f"{verdict} — score={err['score']:.3f}",
            score=err["score"],
        )

    print(f"Saved {min(n_samples, len(errors))} error visualizations to {REPORTS_DIR}/")


if __name__ == "__main__":
    run_report()
