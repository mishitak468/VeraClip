"""
veraclip/tests/test_gradcam.py
Tests for explain/ — uses ViT-B/32 for speed.
"""

import pytest
import torch
import numpy as np
from model.clip_base      import CLIPBackbone
from model.fusion_head    import InconsistencyHead
from explain.attn_rollout import attention_rollout, token_importance
from explain.visualize    import heatmap_to_rgb, blend_heatmap, attention_to_image

SMALL_MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def backbone():
    return CLIPBackbone(SMALL_MODEL)

@pytest.fixture(scope="module")
def head(backbone):
    return InconsistencyHead(embed_dim=backbone.embed_dim, hidden_dim=128)


class TestAttentionRollout:

    def test_output_length(self, backbone):
        pv   = torch.randn(1, 3, 224, 224)
        mask = attention_rollout(backbone, pv)
        # ViT-B/32: 224/32 = 7 → 49 patches
        assert mask.shape[0] == 49

    def test_values_in_range(self, backbone):
        pv   = torch.randn(1, 3, 224, 224)
        mask = attention_rollout(backbone, pv)
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


class TestTokenImportance:

    def test_returns_list(self):
        emb    = torch.randn(1, 512)
        tokens = ["heavy", "flooding", "in", "Mumbai"]
        result = token_importance(emb, tokens)
        assert isinstance(result, list)
        assert len(result) == len(tokens)

    def test_has_token_and_score_keys(self):
        emb    = torch.randn(1, 512)
        tokens = ["test", "caption"]
        result = token_importance(emb, tokens)
        for item in result:
            assert "token" in item and "score" in item

    def test_sorted_descending(self):
        emb    = torch.randn(1, 512)
        tokens = ["a", "b", "c", "d", "e"]
        result = token_importance(emb, tokens)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestVisualize:

    def test_heatmap_to_rgb_shape(self):
        hmap = np.random.rand(224, 224).astype(np.float32)
        rgb  = heatmap_to_rgb(hmap)
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8

    def test_blend_heatmap_shape(self):
        img  = np.random.rand(224, 224, 3).astype(np.float32)
        hmap = np.random.rand(224, 224).astype(np.float32)
        out  = blend_heatmap(img, hmap)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_attention_to_image_shape(self):
        patches = np.random.rand(49).astype(np.float32)   # ViT-B/32
        out     = attention_to_image(patches, image_size=224, patch_size=32)
        assert out.shape == (224, 224)
