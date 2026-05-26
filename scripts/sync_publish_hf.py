from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
from pathlib import Path

from scripts.fetch_baseline_csv import fetch_qwen_baselines
from scripts.push_hf_dataset import main as push_hf_main


def git_pull(repo_root: Path) -> None:
    subprocess.run(
        ["git", "pull", "origin", "main"],
        cwd=repo_root,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pull results from GitHub and publish to HuggingFace dataset",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.skip_pull:
        git_pull(args.repo_root)

    run_dir = args.repo_root / "results" / "run"
    fetch_qwen_baselines(run_dir)

    import sys

    argv = sys.argv
    sys.argv = [
        "push_hf_dataset.py",
        "--run-dir",
        str(run_dir),
    ]
    push_hf_main()
    sys.argv = argv


if __name__ == "__main__":
    main()
