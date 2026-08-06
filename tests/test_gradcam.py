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


