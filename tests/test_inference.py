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
def inferencer():
    device   = torch.device("cpu")
    backbone = CLIPBackbone(SMALL_MODEL).to(device)
    head     = InconsistencyHead(embed_dim=backbone.embed_dim, hidden_dim=128).to(device)
    return MisinfoInference(backbone, head, device)


class TestMisinfoInference:

    def test_predict_returns_dict(self, inferencer):
        img     = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        caption = "A crowd gathers in downtown Chicago for a protest."
        result  = inferencer.predict(img, caption)
        assert isinstance(result, dict)

