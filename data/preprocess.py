"""
veraclip/data/preprocess.py
Converts raw VERITE images + captions into tensor-cached HuggingFace datasets.
Run: python -m data.preprocess
"""

import os
from pathlib import Path
from datasets import load_from_disk
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()

RAW_DIR  = Path(os.getenv("DATA_DIR",   "./data/raw"))
PROC_DIR = Path(os.getenv("OUTPUT_DIR", "./data/processed"))
PROC_DIR.mkdir(parents=True, exist_ok=True)

# CLIP ViT-L/14 normalization constants (OpenAI)
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# Map VERITE label strings to binary: 0 = consistent, 1 = inconsistent
LABEL_MAP = {
    "pristine":       0,
    "miscaptioned":   1,
    "out-of-context": 1,
}


def process_example(example: dict) -> dict:
    """Transform one VERITE row into model-ready tensors."""
    img    = example["image"].convert("RGB")
    tensor = CLIP_TRANSFORM(img)
    label  = LABEL_MAP.get(example.get("label", "pristine"), 0)
    return {
        "pixel_values": tensor.numpy(),   # (3, 224, 224) float32
        "caption":      example["caption"],
        "label":        label,            # 0 or 1
        "image_id":     str(example.get("id", "")),
    }


def preprocess_split(split: str) -> None:
    verite = load_from_disk(str(RAW_DIR / "verite"))
    if split not in verite:
        print(f"  Split '{split}' not in dataset — skipping")
        return

    raw = verite[split]
    print(f"Processing VERITE {split} ({len(raw)} examples)...")

    processed = raw.map(
        process_example,
        remove_columns=raw.column_names,
        num_proc=4,
        desc=f"  {split}",
    )
    out_path = PROC_DIR / f"verite_{split}"
    processed.save_to_disk(str(out_path))
    print(f"  Saved {len(processed)} rows → {out_path}")


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        preprocess_split(split)
    print("\nPreprocessing complete.")
