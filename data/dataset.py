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

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer=None,
        max_length: int = 77,
    ):
        path = Path(data_dir) / f"verite_{split}"
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed split not found at {path}. "
                "Run `python -m data.preprocess` first."
            )
        self.data      = load_from_disk(str(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        pixel_values = torch.tensor(item["pixel_values"], dtype=torch.float32)
        label        = torch.tensor(item["label"],        dtype=torch.float32)

        if self.tokenizer is None:
            return {
                "pixel_values": pixel_values,
                "label":        label,
                "caption":      item["caption"],
            }

        tokens = self.tokenizer(
            item["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values":  pixel_values,
            "input_ids":     tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label":         label,
            "caption":       item["caption"],
        }
