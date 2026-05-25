from __future__ import annotations

from datasets import load_dataset

from calculator_bench.config import DATASET_ID


def load_test_split(dataset_id: str = DATASET_ID):
    raw = load_dataset(dataset_id)
    return raw["test"]
