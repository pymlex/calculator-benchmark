from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, login

from calculator_bench.config import HF_DATASET_REPO, RUN_DIR
from calculator_bench.metrics import build_summary, safe_model_filename
from calculator_bench.plots import generate_all_plots


GITHUB_RAW = (
    "https://raw.githubusercontent.com/pymlex/calculator-benchmark/main"
)


def build_dataset_readme(
    summary_rows: list[dict],
    github_repo: str,
) -> str:
    table_lines = [
        "| model_id | overall_acc | weighted_score |",
        "|---|---|---|",
    ]
    for row in summary_rows:
        table_lines.append(
            f"| {row['model_id']} | {row['overall_acc']:.6f} | "
            f"{row['weighted_score']:.6f} |"
        )
    table = "\n".join(table_lines)
    return f"""---
license: gpl-3.0
pretty_name: Calculator
---

# Arithmetic Expression Dataset

Source code and reproducible evaluation pipeline:
[calculator-benchmark](https://github.com/{github_repo}) on GitHub.

## Evaluation across four models

Models: `Qwen2.5-Math-1.5B`, `Qwen2.5-Math-7B`,
`AceReason-Nemotron-1.1-7B`, `MathGLM-2B`.

Weighted score:

$$
\\text{{weighted\\_score}} = \\frac{{\\sum_{{s=1}}^{{15}} (\\text{{mean}}(\\text{{correct}}_s) \\cdot s^2)}}{{\\sum_{{s=1}}^{{15}} s^2}}
$$

{table}

![Accuracy by steps]({GITHUB_RAW}/results/assets/teacher_step_comparison.png)

![Response length distribution]({GITHUB_RAW}/results/assets/pred_len_all_models.png)

Per-model CSV files are attached in this dataset repository root.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--token-env",
        type=str,
        default="HF_TOKEN",
    )
    args = parser.parse_args()

    import os
    import pandas as pd

    token = os.environ.get(args.token_env)
    if token:
        login(token=token)

    frames = []
    for path in args.run_dir.glob("*.csv"):
        if path.name == "summary.csv":
            continue
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    assets_dir = args.run_dir.parent / "assets"
    generate_all_plots(combined, assets_dir)
    summary = build_summary(combined)

    api = HfApi()
    readme = build_dataset_readme(
        summary.to_dict(orient="records"),
        "pymlex/calculator-benchmark",
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Update four-model benchmark report",
    )

    for model_id in combined["model_id"].unique():
        safe = safe_model_filename(model_id)
        csv_path = args.run_dir / f"{safe}.csv"
        if not csv_path.exists():
            subset = combined[combined["model_id"] == model_id]
            csv_path = args.run_dir / f"{safe}.csv"
            subset.to_csv(csv_path, index=False)
        api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo=f"{safe}.csv",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload results for {model_id}",
        )

    for png in assets_dir.glob("*.png"):
        api.upload_file(
            path_or_fileobj=str(png),
            path_in_repo=f"exports/{png.name}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload plot {png.name}",
        )

    metrics = {
        "summary": summary.to_dict(orient="records"),
    }
    metrics_path = args.run_dir.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo="exports/metrics.json",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Upload metrics.json",
    )
    print("Uploaded README, CSV files, and plots to", HF_DATASET_REPO)


if __name__ == "__main__":
    main()
