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
