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

    def test_required_keys(self, inferencer):
        img    = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = inferencer.predict(img, "Heavy flooding in New York.")
        for key in ["score", "cosine_sim", "verdict", "overlay", "heatmap", "patch_attn", "token_scores"]:
            assert key in result, f"Missing key: {key}"

    def test_score_in_range(self, inferencer):
        img    = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = inferencer.predict(img, "Sample caption.")
        assert 0.0 <= result["score"] <= 1.0

    def test_verdict_is_valid(self, inferencer):
        img    = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = inferencer.predict(img, "Another test caption here.")
        assert result["verdict"] in {"consistent", "uncertain", "inconsistent"}

    def test_overlay_shape(self, inferencer):
        img    = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = inferencer.predict(img, "Test.")
        assert result["overlay"].shape == (224, 224, 3)
        assert result["overlay"].dtype == np.uint8

    def test_token_scores_sorted(self, inferencer):
        img    = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = inferencer.predict(img, "Wildfire smoke over California in summer 2024.")
        scores = [t["score"] for t in result["token_scores"]]
        assert scores == sorted(scores, reverse=True)
