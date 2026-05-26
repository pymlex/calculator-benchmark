from __future__ import annotations

import os
from pathlib import Path


DATASET_ID = os.environ.get("CALC_BENCH_DATASET", "pymlex/calculator")
MAX_NEW_TOKENS = int(os.environ.get("CALC_BENCH_MAX_NEW_TOKENS", "4096"))
SEED = int(os.environ.get("CALC_BENCH_SEED", "3407"))
RUN_DIR = Path(os.environ.get("CALC_BENCH_RUN_DIR", "results/run"))

MATHGLM_CHECKPOINT_DIR = Path(
    os.environ.get("MATHGLM_CHECKPOINT_DIR", "checkpoints/mathglm-2b")
)

QWEN_1_5B = "Qwen/Qwen2.5-Math-1.5B-Instruct"
QWEN_7B = "Qwen/Qwen2.5-Math-7B-Instruct"
ACEREASON_7B = "nvidia/AceReason-Nemotron-1.1-7B"
OPENREASONING_1_5B = "nvidia/OpenReasoning-Nemotron-1.5B"
DEEPSCALER_1_5B = "agentica-org/DeepScaleR-1.5B-Preview"
MATHGLM_2B = "THUDM/MathGLM-2B"

ALL_MODEL_IDS = [
    QWEN_1_5B,
    QWEN_7B,
    ACEREASON_7B,
    OPENREASONING_1_5B,
    DEEPSCALER_1_5B,
]

NEW_MODEL_IDS = [
    OPENREASONING_1_5B,
    DEEPSCALER_1_5B,
]

DEFAULT_BATCH_SIZES = {
    QWEN_1_5B: 20,
    QWEN_7B: 8,
    ACEREASON_7B: 8,
    OPENREASONING_1_5B: 20,
    DEEPSCALER_1_5B: 20,
}

MATHGLM_MAX_SEQUENCE_LENGTH = int(
    os.environ.get("MATHGLM_MAX_SEQUENCE_LENGTH", "1024")
)

GITHUB_REPO = "pymlex/calculator-benchmark"
HF_DATASET_REPO = "pymlex/calculator"
