from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from calculator_bench.metrics import build_summary, safe_model_filename


def _model_column_order(combined: pd.DataFrame) -> list[str]:
    summary = build_summary(combined)
    ranked = summary.sort_values(
        ["overall_acc", "weighted_score"],
        ascending=False,
    )["model_id"].tolist()
    present = set(combined["model_id"].unique())
    return [m for m in ranked if m in present]


def plot_step_accuracy(
    combined: pd.DataFrame,
    out_path: Path,
    title: str = "Exact match by step on test split",
) -> None:
    pivot = combined.pivot_table(
        index="steps",
        columns="model_id",
        values="correct",
        aggfunc="mean",
    ).sort_index()
    column_order = _model_column_order(combined)
    pivot = pivot[column_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("steps")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.5)
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pred_len_by_steps(
    combined: pd.DataFrame,
    model_id: str,
    out_path: Path,
) -> None:
    mdf = combined[combined["model_id"] == model_id].copy()
    mdf["pred_text_len"] = mdf["pred_text"].astype(str).str.len()
    steps_list = sorted(mdf["steps"].unique())
    colors = cm.plasma([s / max(steps_list) for s in steps_list])
    plt.figure(figsize=(10, 4))
    for i, step in enumerate(steps_list):
        subset = mdf[mdf["steps"] == step]
        if not subset.empty:
            plt.hist(
                subset["pred_text_len"],
                alpha=0.3,
                label=f"Steps {step}",
                density=True,
                color=colors[i],
                bins=20,
            )
    plt.title(f"Distribution of pred_text length by steps — {model_id}")
    plt.xlabel("Length of pred_text")
    plt.ylabel("Density")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pred_len_all_models(
    combined: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 4))
    len_stats_rows = []
    for model_id in combined["model_id"].unique():
        mdf = combined[combined["model_id"] == model_id]
        lengths = mdf["pred_text"].astype(str).str.len()
        plt.hist(
            lengths,
            alpha=0.5,
            label=model_id,
            bins=110,
            density=True,
        )
        mode_val = (
            lengths.mode().iloc[0]
            if not lengths.mode().empty
            else None
        )
        len_stats_rows.append(
            {
                "model_id": model_id,
                "median": lengths.median(),
                "mode": mode_val,
                "variance": lengths.var(),
            }
        )
    plt.title("Overall distribution of pred_text length (all models)")
    plt.xlabel("Length of pred_text")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize=7)
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return pd.DataFrame(len_stats_rows)


def generate_all_plots(
    combined: pd.DataFrame,
    assets_dir: Path,
) -> pd.DataFrame:
    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_step_accuracy(
        combined,
        assets_dir / "teacher_step_comparison.png",
    )
    for model_id in combined["model_id"].unique():
        plot_pred_len_by_steps(
            combined,
            model_id,
            assets_dir
            / f"pred_len_by_steps_{safe_model_filename(model_id)}.png",
        )
    len_stats = plot_pred_len_all_models(
        combined,
        assets_dir / "pred_len_all_models.png",
    )
    summary = build_summary(combined)
    summary.to_csv(assets_dir / "summary.csv", index=False)
    return len_stats
