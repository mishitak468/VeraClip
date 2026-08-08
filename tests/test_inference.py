"""
veraclip/tests/test_inference.py
Integration test for the full inference pipeline.
Uses random weights (no checkpoint needed) — just confirms the pipeline
runs end-to-end without crashing.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from model.clip_base   import CLIPBackbone
from model.fusion_head import InconsistencyHead
from app.inference     import MisinfoInference

SMALL_MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
