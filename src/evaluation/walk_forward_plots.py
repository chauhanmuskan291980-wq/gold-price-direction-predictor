from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_metric_by_fold_plot(
    fold_results: list[dict[str, Any]],
    metric_name: str,
    title: str,
    y_label: str,
    output_path: Path,
    reference_line: float | None = None,
) -> None:
    """
    Save one metric across all walk-forward folds.
    """
    if not fold_results:
        raise ValueError(
            "Cannot create a plot from empty fold results."
        )

    folds = [
        int(result["fold"])
        for result in fold_results
    ]

    values = [
        result[metric_name]
        for result in fold_results
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        folds,
        values,
        marker="o",
        linewidth=2,
        label=metric_name.replace(
            "_",
            " ",
        ).title(),
    )

    if reference_line is not None:
        plt.axhline(
            y=reference_line,
            linestyle="--",
            linewidth=1,
            label=(
                f"Reference: "
                f"{reference_line:.2f}"
            ),
        )

    plt.title(title)
    plt.xlabel("Walk-forward fold")
    plt.ylabel(y_label)
    plt.xticks(folds)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def save_returns_by_fold_plot(
    fold_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Compare strategy and buy-and-hold returns per fold.
    """
    if not fold_results:
        raise ValueError(
            "Cannot create a return plot from empty "
            "fold results."
        )

    folds = [
        int(result["fold"])
        for result in fold_results
    ]

    strategy_returns = [
        float(result["strategy_return"])
        for result in fold_results
    ]

    buy_and_hold_returns = [
        float(result["buy_and_hold_return"])
        for result in fold_results
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        folds,
        strategy_returns,
        marker="o",
        linewidth=2,
        label="Model strategy",
    )

    plt.plot(
        folds,
        buy_and_hold_returns,
        marker="o",
        linewidth=2,
        label="Buy and hold",
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        label="Zero return",
    )

    plt.title(
        "Strategy vs Buy-and-Hold Return by Fold"
    )
    plt.xlabel("Walk-forward fold")
    plt.ylabel("Cumulative return")
    plt.xticks(folds)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def generate_walk_forward_plots(
    fold_results: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """
    Generate every required walk-forward chart.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_metric_by_fold_plot(
        fold_results=fold_results,
        metric_name="accuracy",
        title="Accuracy Across Walk-Forward Folds",
        y_label="Accuracy",
        output_path=(
            output_directory
            / "accuracy_by_fold.png"
        ),
        reference_line=0.50,
    )

    save_metric_by_fold_plot(
        fold_results=fold_results,
        metric_name="roc_auc",
        title="ROC-AUC Across Walk-Forward Folds",
        y_label="ROC-AUC",
        output_path=(
            output_directory
            / "roc_auc_by_fold.png"
        ),
        reference_line=0.50,
    )

    save_metric_by_fold_plot(
        fold_results=fold_results,
        metric_name="win_rate",
        title="Win Rate Across Walk-Forward Folds",
        y_label="Win rate",
        output_path=(
            output_directory
            / "win_rate_by_fold.png"
        ),
        reference_line=0.50,
    )

    save_returns_by_fold_plot(
        fold_results=fold_results,
        output_path=(
            output_directory
            / "returns_by_fold.png"
        ),
    )