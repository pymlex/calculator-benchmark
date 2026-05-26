from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path

from huggingface_hub import hf_hub_download

from calculator_bench.config import HF_DATASET_REPO, QWEN_1_5B, QWEN_7B
from calculator_bench.metrics import safe_model_filename


def fetch_qwen_baselines(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for model_id in (QWEN_1_5B, QWEN_7B):
        remote = f"{safe_model_filename(model_id)}.csv"
        local = run_dir / remote
        if local.exists():
            continue
        cached = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=remote,
        )
        local.write_bytes(Path(cached).read_bytes())
        print("Downloaded", local)


if __name__ == "__main__":
    from calculator_bench.config import RUN_DIR

    fetch_qwen_baselines(RUN_DIR)
