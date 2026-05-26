from __future__ import annotations

import time
from typing import Literal

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from calculator_bench.config import MAX_NEW_TOKENS
from calculator_bench.metrics import (
    extract_gold_answer,
    extract_pred_answer,
    overall_accuracy,
    weighted_score,
)


QWEN_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
ACEREASON_SYSTEM = (
    "You are a helpful and harmless assistant. You should think step-by-step."
)
ACEREASON_USER_SUFFIX = "Please place your final answer inside \\boxed{}."
OPENREASONING_MATH = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{}.\n\n{user}"
)


class HfCausalBackend:
    def __init__(
        self,
        model_id: str,
        prompt_style: Literal["qwen", "acereason", "openreasoning"],
    ):
        self.model_id = model_id
        self.prompt_style = prompt_style
        self._model = None
        self._tokenizer = None
        self._bf16 = (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )

    def load(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self._bf16 else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    def build_prompt(self, prompt: str) -> str:
        messages: list[dict[str, str]]
        if self.prompt_style == "qwen":
            messages = [
                {"role": "system", "content": QWEN_SYSTEM},
                {"role": "user", "content": prompt.strip()},
            ]
        elif self.prompt_style == "openreasoning":
            messages = [
                {
                    "role": "user",
                    "content": OPENREASONING_MATH.format(
                        user=prompt.strip()
                    ),
                },
            ]
        else:
            user_text = f"{prompt.strip()}\n\n{ACEREASON_USER_SUFFIX}"
            messages = [
                {"role": "system", "content": ACEREASON_SYSTEM},
                {"role": "user", "content": user_text},
            ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list[str]:
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        decoded = []
        for i in range(outputs.size(0)):
            text = self._tokenizer.decode(
                outputs[i][prompt_len:],
                skip_special_tokens=False,
            )
            decoded.append(text)
        return decoded

    def evaluate(
        self,
        dataset,
        batch_size: int,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> tuple[pd.DataFrame, float, float]:
        rows = []
        correct = 0
        n = len(dataset)
        for start in tqdm(range(0, n, batch_size)):
            batch = dataset.select(
                range(start, min(start + batch_size, n))
            )
            prompts = [
                self.build_prompt(x) for x in batch["prompt"]
            ]
            decoded = self.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
            )
            for prompt_text, gold_answer_value, pred_text, steps in zip(
                batch["prompt"],
                batch["answer"],
                decoded,
                batch["steps"],
            ):
                pred = extract_pred_answer(pred_text)
                gold = extract_gold_answer(gold_answer_value)
                is_correct = int(pred == gold)
                correct += is_correct
                rows.append(
                    {
                        "prompt": prompt_text,
                        "gold_answer": gold,
                        "pred_answer": pred,
                        "correct": is_correct,
                        "pred_text": pred_text,
                        "steps": int(steps),
                    }
                )
        df = pd.DataFrame(rows)
        df["model_id"] = self.model_id
        return df, overall_accuracy(df), weighted_score(df)
