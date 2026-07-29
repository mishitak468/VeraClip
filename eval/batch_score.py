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


def run_batch_score(config_path: str = "model/config.yaml") -> list[dict]:
    cfg       = yaml.safe_load(open(config_path))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head = load_model(cfg, device)
    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model_name"])

    ds      = load_from_disk("./data/raw/newsclipper")
    results = []

    for i, item in enumerate(tqdm(ds, desc="Scoring NewsCLIPpings")):
        try:
            img = item["image"].convert("RGB")
            pv  = CLIP_TRANSFORM(img).unsqueeze(0).to(device)

            tok = processor(
                text=item["caption"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            ii = tok["input_ids"].to(device)
            am = tok["attention_mask"].to(device)

            with torch.no_grad(), autocast():
                img_emb, txt_emb = backbone(pv, ii, am)
                score            = head(img_emb, txt_emb).item()

            results.append({
                "id":             i,
                "score":          round(score, 4),
                "label":          item.get("falsified", item.get("label", -1)),
                "caption_snippet": item["caption"][:100].replace("\n", " "),
                "error":          "",
            })
        except Exception as e:
            results.append({
                "id": i, "score": -1.0, "label": -1,
                "caption_snippet": "", "error": str(e),
            })

    # Write CSV
    csv_path = RESULTS_DIR / "newsclipper_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "score", "label", "caption_snippet", "error"])
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    valid   = [r for r in results if r["score"] >= 0]
    flagged = [r for r in valid  if r["score"] > 0.5]
    summary = {
        "total_scored":          len(valid),
        "flagged_inconsistent":  len(flagged),
        "flag_rate_pct":         round(100 * len(flagged) / max(len(valid), 1), 1),
        "mean_score":            round(sum(r["score"] for r in valid) / max(len(valid), 1), 4),
        "errors":                len(results) - len(valid),
    }

    with open(RESULTS_DIR / "newsclipper_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== NewsCLIPpings Batch Score Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nCSV saved to {csv_path}")

    return results


if __name__ == "__main__":
    run_batch_score()
