"""
veraclip/model/lora_adapter.py
Attaches LoRA adapters to the CLIP backbone.

Why LoRA here:
  - Full CLIP fine-tune: 428M trainable params — needs 4×A100
  - LoRA r=16: ~0.5M trainable params — runs on a single T4/V100
  - Targets visual_projection, text_projection, q_proj, v_proj
    (projection layers + attention key/value — highest signal-to-noise for cross-modal tasks)

Called once in train.py before training starts.
"""

from peft import LoraConfig, get_peft_model
from model.clip_base import CLIPBackbone


def apply_lora(backbone: CLIPBackbone, config: dict) -> CLIPBackbone:
    """
    Wraps backbone.clip with LoRA adapters in-place.

    Args:
        backbone: CLIPBackbone instance (already on device)
        config:   the model: block from config.yaml

    Returns:
        The same backbone with LoRA applied and frozen base weights.
    """
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
    )

    backbone.clip = get_peft_model(backbone.clip, lora_cfg)
    backbone.clip.print_trainable_parameters()

    return backbone
