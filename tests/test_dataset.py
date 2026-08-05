"""
veraclip/tests/test_dataset.py
Tests for data/dataset.py — mocks the preprocessed dataset so
no files need to be downloaded to run CI.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from datasets import Dataset
from data.dataset import MisinfoDataset


def make_fake_dataset(n: int = 10) -> Dataset:
    """Create an in-memory fake dataset that mimics the preprocessed schema."""
    return Dataset.from_dict({
        "pixel_values": [np.random.randn(3, 224, 224).astype(np.float32) for _ in range(n)],
        "caption":      [f"Test caption number {i}" for i in range(n)],
        "label":        [i % 2 for i in range(n)],
        "image_id":     [str(i) for i in range(n)],
    })


@pytest.fixture
def fake_data_dir(tmp_path):
    """Write a fake preprocessed dataset to a temp directory."""
    ds = make_fake_dataset(20)
    ds.save_to_disk(str(tmp_path / "verite_train"))
    return str(tmp_path)


class TestMisinfoDataset:

    def test_len(self, fake_data_dir):
        ds = MisinfoDataset(fake_data_dir, "train")
        assert len(ds) == 20

    def test_item_keys_without_tokenizer(self, fake_data_dir):
        ds   = MisinfoDataset(fake_data_dir, "train")
        item = ds[0]
        assert "pixel_values" in item
        assert "label" in item
        assert "caption" in item

    def test_pixel_values_shape(self, fake_data_dir):
        ds   = MisinfoDataset(fake_data_dir, "train")
        item = ds[0]
        assert item["pixel_values"].shape == (3, 224, 224)

    def test_label_is_binary(self, fake_data_dir):
        ds = MisinfoDataset(fake_data_dir, "train")
        for i in range(len(ds)):
            assert ds[i]["label"].item() in [0.0, 1.0]

    def test_missing_split_raises(self, fake_data_dir):
        with pytest.raises(FileNotFoundError):
            MisinfoDataset(fake_data_dir, "nonexistent_split")
