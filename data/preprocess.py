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
