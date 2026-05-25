from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from calculator_bench.config import (
    MATHGLM_2B,
    MATHGLM_CHECKPOINT_DIR,
    MATHGLM_MAX_SEQUENCE_LENGTH,
)
from calculator_bench.metrics import (
    extract_gold_answer,
    extract_pred_answer,
    overall_accuracy,
    weighted_score,
)


def expression_to_mathglm_input(expression: str) -> str:
    raw = expression.replace(" ", "")
    if "=" not in raw:
        raw += "="
    return raw


def read_mathglm_model_config(checkpoint_dir: Path) -> dict:
    cfg_path = checkpoint_dir / "model_config.json"
    text = cfg_path.read_text(encoding="utf-8")
    return json.loads(text)


class MathGLMBackend:
    def __init__(self, checkpoint_dir: Path | None = None):
        self.model_id = MATHGLM_2B
        self.checkpoint_dir = Path(
            checkpoint_dir or MATHGLM_CHECKPOINT_DIR
        )
        self._model = None
        self._args = None
        self._strategy = None
        self._icetk = None

    def load(self) -> None:
        from SwissArmyTransformer import get_args
        from SwissArmyTransformer.generation.autoregressive_sampling import (
            filling_sequence,
        )
        from SwissArmyTransformer.generation.sampling_strategies import (
            BaseStrategy,
        )
        from SwissArmyTransformer.model import CachedAutoregressiveModel
        from icetk import icetk

        cfg = read_mathglm_model_config(self.checkpoint_dir)
        num_layers = int(cfg.get("num_layers", 40))
        hidden_size = int(cfg.get("hidden_size", 2048))
        num_attention_heads = int(cfg.get("num_attention_heads", 32))
        vocab_size = int(cfg.get("vocab_size", 61667))

        argv = [
            "--mode",
            "inference",
            "--distributed-backend",
            "nccl",
            f"--max-sequence-length",
            str(MATHGLM_MAX_SEQUENCE_LENGTH),
            "--fp16",
            "--model-parallel-size",
            "1",
            "--num-layers",
            str(num_layers),
            "--hidden-size",
            str(hidden_size),
            "--num-attention-heads",
            str(num_attention_heads),
            "--vocab-size",
            str(vocab_size),
            "--temperature",
            "0.0",
            "--top_k",
            "1",
            "--batch-size",
            "1",
            "--max-inference-batch-size",
            "8",
        ]
        self._args = get_args(argv)
        self._args.device = torch.device("cuda")

        model, _ = CachedAutoregressiveModel.from_pretrained(
            self._args,
            str(self.checkpoint_dir),
        )
        self._model = model
        self._icetk = icetk
        end_tokens = [icetk.encode("SEP")[-1]]
        self._strategy = BaseStrategy(
            temperature=0.0,
            top_k=1,
            end_tokens=end_tokens,
        )
        icetk.text_tokenizer.discourage_tokens(["▁[", "▁("])
        icetk.text_tokenizer.discourage_tokens(
            ["▁[", "▁(", "▁+", "▁=", "▁*", "▁-"]
        )
        icetk.text_tokenizer.discourage_ids(range(125653, 130000))
        self._filling_sequence = filling_sequence

    def unload(self) -> None:
        del self._model
        self._model = None
        torch.cuda.empty_cache()

    def generate_one(self, expression: str) -> str:
        raw_text = expression_to_mathglm_input(expression)
        seq = self._icetk.encode(raw_text)
        if seq[0] == 20005:
            seq = seq[1:]
        max_len = self._args.max_sequence_length
        seq += [-1] * (max_len - len(seq))
        seq_tensor = torch.cuda.LongTensor(seq, device=self._args.device)
        output = self._filling_sequence(
            self._model,
            seq_tensor.clone(),
            batch_size=1,
            strategy=self._strategy,
            log_attention_weights=None,
        )[0]
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        unfinished = len(output)
        if -1 in output:
            unfinished = output.index(-1)
        end_tokens = [self._icetk.encode("SEP")[-1]]
        if output[unfinished - 1] in end_tokens:
            unfinished -= 1
        return self._icetk.decode(output[:unfinished])

    def evaluate(
        self,
        dataset,
        batch_size: int,
        max_new_tokens: int | None = None,
    ) -> tuple[pd.DataFrame, float, float]:
        rows = []
        correct = 0
        for row in tqdm(dataset):
            expression = row["expression"]
            prompt_text = row["prompt"]
            pred_text = self.generate_one(expression)
            pred = extract_pred_answer(pred_text)
            gold = extract_gold_answer(row["answer"])
            is_correct = int(pred == gold)
            correct += is_correct
            rows.append(
                {
                    "prompt": prompt_text,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "correct": is_correct,
                    "pred_text": pred_text,
                    "steps": int(row["steps"]),
                }
            )
        df = pd.DataFrame(rows)
        df["model_id"] = self.model_id
        return df, overall_accuracy(df), weighted_score(df)
