from __future__ import annotations

from calculator_bench.config import (
    ACEREASON_7B,
    DEEPSCALER_1_5B,
    OPENREASONING_1_5B,
    QWEN_1_5B,
    QWEN_7B,
)
from calculator_bench.models.hf_causal import HfCausalBackend


def get_backend(model_id: str):
    if model_id in (QWEN_1_5B, QWEN_7B, DEEPSCALER_1_5B):
        return HfCausalBackend(model_id, prompt_style="qwen")
    if model_id == ACEREASON_7B:
        return HfCausalBackend(model_id, prompt_style="acereason")
    if model_id == OPENREASONING_1_5B:
        return HfCausalBackend(model_id, prompt_style="openreasoning")
    raise ValueError(f"Unknown model_id: {model_id}")
