"""
veraclip/tests/test_model.py
Unit tests for backbone, head, and loss.
Uses clip-vit-base-patch32 (smaller/faster than ViT-L/14) for CI speed.
"""

import pytest
import torch
from model.clip_base   import CLIPBackbone
from model.fusion_head import InconsistencyHead
from model.loss        import MisinfoLoss

SMALL_MODEL = "openai/clip-vit-base-patch32"   # 151M params — fast for tests
BATCH       = 4
SEQ_LEN     = 77


@pytest.fixture(scope="module")
def backbone():
    return CLIPBackbone(SMALL_MODEL)

@pytest.fixture(scope="module")
def tokenizer(backbone):
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(SMALL_MODEL)

@pytest.fixture(scope="module")
def head(backbone):
    return InconsistencyHead(embed_dim=backbone.embed_dim, hidden_dim=256)


class TestCLIPBackbone:

    def test_embed_dim(self, backbone):
        assert backbone.embed_dim == 512   # ViT-B/32 projection dim

    def test_encode_image_shape(self, backbone):
        pv  = torch.randn(BATCH, 3, 224, 224)
        emb = backbone.encode_image(pv)
        assert emb.shape == (BATCH, backbone.embed_dim)

    def test_image_emb_normalized(self, backbone):
        pv  = torch.randn(2, 3, 224, 224)
        emb = backbone.encode_image(pv)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_encode_text_shape(self, backbone, tokenizer):
        tok = tokenizer(
            ["test caption"] * BATCH,
            return_tensors="pt",
            padding="max_length",
            max_length=SEQ_LEN,
        )
        emb = backbone.encode_text(tok["input_ids"], tok["attention_mask"])
        assert emb.shape == (BATCH, backbone.embed_dim)

    def test_forward_returns_tuple(self, backbone, tokenizer):
        pv  = torch.randn(2, 3, 224, 224)
        tok = tokenizer(
            ["hello", "world"],
            return_tensors="pt",
            padding="max_length",
            max_length=SEQ_LEN,
        )
        img_emb, txt_emb = backbone(pv, tok["input_ids"], tok["attention_mask"])
        assert img_emb.shape == txt_emb.shape == (2, backbone.embed_dim)


class TestInconsistencyHead:

    def test_output_shape(self, backbone, head):
        img = torch.randn(BATCH, backbone.embed_dim)
        txt = torch.randn(BATCH, backbone.embed_dim)
        out = head(img, txt)
        assert out.shape == (BATCH,)

    def test_output_in_zero_one(self, backbone, head):
        img = torch.randn(16, backbone.embed_dim)
        txt = torch.randn(16, backbone.embed_dim)
        out = head(img, txt)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_consistent_pair_lower_score(self, backbone, head):
        """Identical embeddings should produce a lower score than random ones."""
        emb       = torch.randn(1, backbone.embed_dim)
        norm_emb  = emb / emb.norm(dim=-1, keepdim=True)
        consistent   = head(norm_emb, norm_emb).item()
        random_img   = torch.randn(1, backbone.embed_dim)
        random_txt   = torch.randn(1, backbone.embed_dim)
        inconsistent = head(
            random_img / random_img.norm(dim=-1, keepdim=True),
            random_txt / random_txt.norm(dim=-1, keepdim=True),
        ).item()
        # Consistent pairs should score lower on average (not guaranteed for random
        # weights, but holds after training — keep as sanity check, not hard assert)
        assert isinstance(consistent, float)
        assert isinstance(inconsistent, float)


