from __future__ import annotations

import re

import numpy as np
import pandas as pd


number_re = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
boxed_re = re.compile(r"\\boxed\{([^}]+)\}", re.S)
answer_tag_re = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.S)


def canonicalize_answer(text: str) -> str:
    s = str(text).strip().replace(",", "").replace("$", "")
    s = s.replace(" ", "")
    if s.endswith(".0"):
        s = s[:-2]
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_gold_answer(answer_value: str) -> str:
    return canonicalize_answer(answer_value)


def extract_pred_answer(text: str) -> str:
    tag_matches = answer_tag_re.findall(text)
    if tag_matches:
        return canonicalize_answer(tag_matches[-1].strip())
    boxed_matches = boxed_re.findall(text)
    if boxed_matches:
        return canonicalize_answer(boxed_matches[-1].strip())
    nums = number_re.findall(text.replace(",", ""))
    if nums:
        return canonicalize_answer(nums[-1])
    return canonicalize_answer(text)


def overall_accuracy(df: pd.DataFrame) -> float:
    return float(df["correct"].mean())


def weighted_score(df: pd.DataFrame, n_steps: int = 15) -> float:
    numer = 0.0
    denom = 0.0
    for step in range(1, n_steps + 1):
        subset = df[df["steps"] == step]
        if len(subset) == 0:
            continue
        mean_correct = float(subset["correct"].mean())
        numer += mean_correct * (step ** 2)
        denom += step ** 2
    return numer / denom if denom > 0 else 0.0


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in df["model_id"].unique():
        mdf = df[df["model_id"] == model_id]
        rows.append(
            {
                "model_id": model_id,
                "overall_acc": overall_accuracy(mdf),
                "weighted_score": weighted_score(mdf),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["weighted_score", "overall_acc"],
        ascending=False,
    )


def safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "__")
