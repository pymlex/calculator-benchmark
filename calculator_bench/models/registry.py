from __future__ import annotations

from calculator_bench.config import (
    ACEREASON_7B,
    MATHGLM_2B,
    QWEN_1_5B,
    QWEN_7B,
)
from calculator_bench.models.hf_causal import HfCausalBackend
from calculator_bench.models.mathglm_sat import MathGLMBackend


def get_backend(model_id: str):
    if model_id in (QWEN_1_5B, QWEN_7B):
        return HfCausalBackend(model_id, prompt_style="qwen")
    if model_id == ACEREASON_7B:
        return HfCausalBackend(model_id, prompt_style="acereason")
    if model_id == MATHGLM_2B:
        return MathGLMBackend()
    raise ValueError(f"Unknown model_id: {model_id}")
