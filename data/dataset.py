"""
veraclip/data/dataset.py
PyTorch Dataset that reads from the preprocessed HuggingFace cache.
Used by both train.py and evaluate.py.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class MisinfoDataset(Dataset):
    """
    Wraps a preprocessed VERITE split.

    Args:
        data_dir:   path to ./data/processed/
        split:      'train' | 'validation' | 'test'
        tokenizer:  CLIPTokenizer instance (from CLIPProcessor)
        max_length: max caption token length (77 for CLIP)
    """

