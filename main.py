from __future__ import annotations

import argparse
import json
from pathlib import Path

import pynvml
import torch

from calculator_bench.config import (
    ALL_MODEL_IDS,
    MAX_NEW_TOKENS,
    NEW_MODEL_IDS,
    RUN_DIR,
)
from calculator_bench.evaluate import run_models
from calculator_bench.metrics import build_summary, safe_model_filename
from calculator_bench.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on pymlex/calculator test split",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=NEW_MODEL_IDS,
        help="HuggingFace or registry model ids",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(RUN_DIR),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all four benchmark models",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild plots from existing CSV files in run-dir",
    )
    return parser.parse_args()


def load_csv_results(run_dir):
    import pandas as pd

    frames = []
    for path in run_dir.glob("*.csv"):
        if path.name == "summary.csv":
            continue
        df = pd.read_csv(path)
        if "model_id" not in df.columns:
            name = path.stem.replace("__", "/")
            if name.count("/") == 0:
                name = path.stem
            df["model_id"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: none")

    if args.plots_only:
        combined = load_csv_results(run_dir)
    else:
        model_ids = ALL_MODEL_IDS if args.all_models else args.models
        combined = run_models(
            model_ids,
            run_dir=run_dir,
            max_new_tokens=args.max_new_tokens,
        )

    assets_dir = run_dir.parent / "assets"
    len_stats = generate_all_plots(combined, assets_dir)
    summary = build_summary(combined)
    metrics = {
        "models": summary.to_dict(orient="records"),
        "length_stats": len_stats.to_dict(orient="records"),
    }
    metrics_path = run_dir.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("Wrote", metrics_path)


if __name__ == "__main__":
    main()
