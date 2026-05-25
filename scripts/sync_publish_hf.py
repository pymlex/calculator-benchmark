from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_baseline_csv import fetch_qwen_baselines


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

    subprocess.run(
        [
            sys.executable,
            str(args.repo_root / "scripts" / "push_hf_dataset.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=args.repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
