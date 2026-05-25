from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from calculator_bench.config import (
    DEFAULT_BATCH_SIZES,
    MAX_NEW_TOKENS,
    RUN_DIR,
)
from calculator_bench.data import load_test_split
from calculator_bench.metrics import safe_model_filename
from calculator_bench.models.registry import get_backend


def run_models(
    model_ids: list[str],
    run_dir: Path | None = None,
    batch_sizes: dict[str, int] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> pd.DataFrame:
    out_dir = run_dir or RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
    test_ds = load_test_split()
    all_dfs: list[pd.DataFrame] = []

    for model_id in model_ids:
        print("=" * 80)
        print("Loading", model_id)
        started = time.time()
        backend = get_backend(model_id)
        backend.load()
        batch_size = batch_sizes.get(model_id, 8)
        df, overall_acc, wscore = backend.evaluate(
            test_ds,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        backend.unload()
        safe = safe_model_filename(model_id)
        df.to_csv(out_dir / f"{safe}.csv", index=False)
        print(
            f"Done {model_id}: overall_acc={overall_acc:.6f} "
            f"weighted_score={wscore:.6f} elapsed={time.time() - started:.1f}s"
        )
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)
