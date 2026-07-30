from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.aggregate_metrics import (
    aggregate_walk_forward_results,
)
from src.evaluation.walk_forward_split import (
    generate_walk_forward_folds,
)
from src.models.evaluate import (
    get_period,
    save_json,
)
from src.models.train import (
    load_training_data,
)
from src.models.walk_forward import (
    evaluate_fold,
)


CONFIG_PATH = Path("config/walk_forward_grid.yaml")

GRID_ARTIFACT_DIR = Path("artifacts/grid_search")

GRID_REPORT_PATH = GRID_ARTIFACT_DIR / "grid_report.json"

GRID_SUMMARY_PATH = GRID_ARTIFACT_DIR / "grid_summary.csv"


def load_grid_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the grid-search YAML configuration."""

    if not path.exists():
        raise FileNotFoundError(f"Grid configuration was not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Grid configuration must contain a YAML mapping.")

    required_keys = {
        "data_path",
        "models",
        "windows",
        "permutation",
    }

    missing_keys = required_keys - config.keys()

    if missing_keys:
        raise ValueError(f"Missing grid configuration values: {sorted(missing_keys)}")

    models = config["models"]

    if not isinstance(models, list) or not models:
        raise ValueError("'models' must be a non-empty list.")

    allowed_models = {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }

    for model_name in models:
        if model_name not in allowed_models:
            raise ValueError(f"Unsupported model: {model_name}")

    windows = config["windows"]

    if not isinstance(windows, list) or not windows:
        raise ValueError("'windows' must be a non-empty list.")

    required_window_keys = {
        "name",
        "train_window",
        "test_window",
        "step_size",
        "purge_gap",
    }

    for window in windows:
        if not isinstance(window, dict):
            raise TypeError("Every window configuration must be a mapping.")

        missing_window_keys = required_window_keys - window.keys()

        if missing_window_keys:
            raise ValueError(
                f"Missing window configuration values: {sorted(missing_window_keys)}"
            )

        for key in [
            "train_window",
            "test_window",
            "step_size",
        ]:
            value = window[key]

            if not isinstance(value, int):
                raise TypeError(f"Window '{key}' must be an integer.")

            if value <= 0:
                raise ValueError(f"Window '{key}' must be greater than zero.")

        purge_gap = window["purge_gap"]

        if not isinstance(purge_gap, int):
            raise TypeError("'purge_gap' must be an integer.")

        if purge_gap < 0:
            raise ValueError("'purge_gap' cannot be negative.")

    return config


def create_cell_id(
    model_name: str,
    window: dict[str, Any],
) -> str:
    """Create a stable identifier for one grid cell."""

    return (
        f"{model_name}__"
        f"{window['name']}__"
        f"train{window['train_window']}_"
        f"test{window['test_window']}_"
        f"step{window['step_size']}"
    )


def run_grid_cell(
    data: pd.DataFrame,
    model_name: str,
    window: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one model and window combination."""

    folds = list(
        generate_walk_forward_folds(
            len(data),
            train_window=int(window["train_window"]),
            test_window=int(window["test_window"]),
            step_size=int(window["step_size"]),
            purge_gap=int(window["purge_gap"]),
        )
    )

    if not folds:
        raise ValueError(
            "No folds were generated for cell: "
            f"{create_cell_id(model_name, window)}"
        )

    # Store the exact train/test indexes used by every fold.
    fold_boundaries = [
        {
            "fold": fold.fold_number,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
        }
        for fold in folds
    ]

    fold_results: list[dict[str, Any]] = []

    if verbose:
        print(
            "\nRunning grid cell:",
            create_cell_id(
                model_name,
                window,
            ),
        )

    for fold in folds:
        if verbose:
            print(
                f"  Running fold {fold.fold_number}..."
            )

        result = evaluate_fold(
            data=data,
            fold=fold,
            model_name=model_name,
        )

        fold_results.append(result)

    aggregate = aggregate_walk_forward_results(
        fold_results
    )

    return {
        "cell_id": create_cell_id(
            model_name,
            window,
        ),
        "model": model_name,
        "window": {
            "name": str(window["name"]),
            "train_window": int(
                window["train_window"]
            ),
            "test_window": int(
                window["test_window"]
            ),
            "step_size": int(
                window["step_size"]
            ),
            "purge_gap": int(
                window["purge_gap"]
            ),
        },
        "fold_count": len(fold_results),

        # Used to compare real and permuted grid folds.
        "fold_boundaries": fold_boundaries,

        "folds": fold_results,
        "aggregate": aggregate,
    }

def run_grid(
    data: pd.DataFrame,
    config: dict[str, Any],
    verbose : bool = True,
) -> list[dict[str, Any]]:
    """Run all model and windows combinations."""

    grid_results: list[dict[str, Any]] = []

    models = config["models"]
    windows = config["windows"]

    for model_name in models:
        for window in windows:
            cell_result = run_grid_cell(
                data=data,
                model_name=str(model_name),
                window=window,
                verbose=verbose
            )

            grid_results.append(cell_result)

    return grid_results


def create_summary_row(
    cell: dict[str, Any],
) -> dict[str, Any]:
    """
    Create one flat CSV summary row from a grid cell.
    """

    fold_results = cell["folds"]

    if not fold_results:
        raise ValueError(f"Grid cell contains no fold results: {cell['cell_id']}")

    window = cell["window"]

    accuracy_values = pd.Series(
        [result["accuracy"] for result in fold_results],
        dtype="float64",
    )

    roc_auc_values = pd.Series(
        [result["roc_auc"] for result in fold_results],
        dtype="float64",
    ).dropna()

    win_rate_values = pd.Series(
        [result["win_rate"] for result in fold_results],
        dtype="float64",
    )

    strategy_return_values = pd.Series(
        [result["strategy_return"] for result in fold_results],
        dtype="float64",
    )

    worst_fold_result = min(
        fold_results,
        key=lambda result: result["strategy_return"],
    )

    folds_beating_buy_and_hold = sum(
        bool(result["beats_buy_and_hold"]) for result in fold_results
    )

    beat_buy_hold_percentage = folds_beating_buy_and_hold / len(fold_results) * 100.0

    median_roc_auc: float | None

    if roc_auc_values.empty:
        median_roc_auc = None
    else:
        median_roc_auc = float(roc_auc_values.median())

    return {
        "cell_id": cell["cell_id"],
        "model": cell["model"],
        "window_name": window["name"],
        "train_window": window["train_window"],
        "test_window": window["test_window"],
        "step_size": window["step_size"],
        "purge_gap": window["purge_gap"],
        "fold_count": len(fold_results),
        "median_accuracy": float(accuracy_values.median()),
        "median_roc_auc": median_roc_auc,
        "median_win_rate": float(win_rate_values.median()),
        "median_strategy_return": float(strategy_return_values.median()),
        "worst_fold": int(worst_fold_result["fold"]),
        "worst_strategy_return": float(worst_fold_result["strategy_return"]),
        "folds_beating_buy_and_hold": (folds_beating_buy_and_hold),
        "beat_buy_hold_percentage": float(beat_buy_hold_percentage),
    }


def rank_summary(
    grid_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Rank all grid cells by median strategy return.
    """

    summary_rows = [create_summary_row(cell) for cell in grid_results]

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        by="median_strategy_return",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        range(
            1,
            len(summary) + 1,
        ),
    )

    return summary


def main() -> None:
    """Run the complete configuration grid."""

    config = load_grid_config()

    data_path = Path(str(config["data_path"]))

    data = load_training_data(data_path)

    grid_results = run_grid(
        data=data,
        config=config,
        verbose=True,
    )

    summary = rank_summary(grid_results)

    GRID_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        GRID_SUMMARY_PATH,
        index=False,
    )

    best_cell_id = str(summary.iloc[0]["cell_id"])

    report = {
        "configuration": {
            "config_path": CONFIG_PATH.as_posix(),
            "models": config["models"],
            "windows": config["windows"],
        },
        "dataset": {
            "path": data_path.as_posix(),
            "total_rows": int(len(data)),
            "period": get_period(data),
        },
        "cell_count": len(grid_results),
        "ranking_metric": ("median_strategy_return"),
        "best_cell_id": best_cell_id,
        "cells": grid_results,
    }

    save_json(
        report,
        GRID_REPORT_PATH,
    )

    print("\nConfiguration grid completed.")

    print(
        "Total grid cells:",
        len(grid_results),
    )

    print(
        "Best cell:",
        best_cell_id,
    )

    print(
        "Grid report:",
        GRID_REPORT_PATH,
    )

    print(
        "Grid summary:",
        GRID_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
