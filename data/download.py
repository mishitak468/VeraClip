"""
veraclip/data/download.py
Downloads VERITE (fine-tuning) and NewsCLIPpings (evaluation) from HuggingFace.
Run: python -m data.download
"""

import os
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(os.getenv("DATA_DIR", "./data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_verite():
    """
    VERITE: ~3,600 real-world image-caption pairs.
    Labels: 'pristine' (consistent) | 'miscaptioned' | 'out-of-context' (both = inconsistent).
    HuggingFace: Ftheodorakis/VERITE
    """
    print("Downloading VERITE dataset...")
    ds = load_dataset(
        "Ftheodorakis/VERITE",
        token=os.getenv("HF_TOKEN"),
    )
    save_path = RAW_DIR / "verite"
    ds.save_to_disk(str(save_path))

    for split, data in ds.items():
        print(f"  {split}: {len(data)} examples")

    print(f"Saved to {save_path}\n")
    return ds


