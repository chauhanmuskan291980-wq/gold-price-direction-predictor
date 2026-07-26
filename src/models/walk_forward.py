from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.aggregate_metrics import (
    aggregate_walk_forward_results,
)
from src.evaluation.walk_forward_split import (
    WalkForwardFold,
    generate_walk_forward_folds,
)
from src.models.evaluate import (
    calculate_strategy_metrics,
    evaluate_model,
    get_period,
    prepare_test_period_data,
    save_json,
)
from src.models.train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_data,
    train_model,
)

CONFIG_PATH = Path(
    "config/walk_forward.yaml"
)

WALK_FORWARD_REPORT_PATH = Path(
    "artifacts/walk_forward/"
    "walk_forward_report.json"
)

FOLD_RESULTS_PATH = Path(
    "artifacts/walk_forward/"
    "fold_results.csv"
)


def load_walk_forward_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load and validate walk-forward settings from YAML.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Walk-forward configuration file "
            f"was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Walk-forward configuration must "
            "contain a YAML mapping."
        )

    required_keys = {
        "model",
        "data_path",
        "train_window",
        "test_window",
        "step_size",
        "purge_gap",
    }

    missing_keys = (
        required_keys - config.keys()
    )

    if missing_keys:
        raise ValueError(
            "Missing walk-forward configuration "
            f"values: {sorted(missing_keys)}"
        )

    positive_integer_keys = [
        "train_window",
        "test_window",
        "step_size",
    ]

    for key in positive_integer_keys:
        value = config[key]

        if not isinstance(value, int):
            raise TypeError(
                f"'{key}' must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"'{key}' must be greater than zero."
            )

    purge_gap = config["purge_gap"]

    if not isinstance(purge_gap, int):
        raise TypeError(
            "'purge_gap' must be an integer."
        )

    if purge_gap < 0:
        raise ValueError(
            "'purge_gap' cannot be negative."
        )

    if not isinstance(config["model"], str):
        raise TypeError(
            "'model' must be a string."
        )

    if not isinstance(config["data_path"], str):
        raise TypeError(
            "'data_path' must be a string."
        )

    return config



def prepare_fold_data(
    data: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Extract training and testing data for one walk-forward fold.
    """
    train_data = data.iloc[
        fold.train_start:fold.train_end
    ].copy()

    test_data = data.iloc[
        fold.test_start:fold.test_end
    ].copy()

    X_train = train_data[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_data[
        TARGET_COLUMN
    ].copy()

    X_test = test_data[
        FEATURE_COLUMNS
    ].copy()

    y_test = test_data[
        TARGET_COLUMN
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_fold(
    data: pd.DataFrame,
    fold: WalkForwardFold,
    model_name: str,
) -> dict[str, Any]:
    """
    Train and evaluate a fresh model on one walk-forward fold.
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_fold_data(
        data=data,
        fold=fold,
    )

    # A completely fresh model is created for this fold.
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
    )

    classification_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    test_period_data = prepare_test_period_data(
        data=data,
        X_test=X_test,
        y_test=y_test,
    )

    strategy_metrics, _trades = (
        calculate_strategy_metrics(
            model=model,
            X_test=X_test,
            test_data=test_period_data,
        )
    )

    strategy_return = float(strategy_metrics[
        "cumulative_strategy_return"
    ])

    buy_and_hold_return = float(strategy_metrics[
        "buy_and_hold_return"
    ])

    strategy_excess_return = (
    strategy_return
    - buy_and_hold_return
    )

    beats_buy_and_hold = (
    strategy_return
    > buy_and_hold_return
    )

    return {
    "fold": fold.fold_number,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_period": get_period(
        data.loc[X_train.index]
    ),
    "testing_period": get_period(
        data.loc[X_test.index]
    ),

    # Flat values used by aggregation, CSV, and plots.
    "accuracy": float(
        classification_metrics["accuracy"]
    ),
    "roc_auc": float(
        classification_metrics["roc_auc"]
    ),
    "balanced_accuracy": float(
        classification_metrics[
            "balanced_accuracy"
        ]
    ),
    "win_rate": float(
        strategy_metrics["win_rate"]
    ),
    "strategy_return": strategy_return,
    "buy_and_hold_return": (
        buy_and_hold_return
    ),
    "strategy_excess_return": (
        strategy_excess_return
    ),
    "beats_buy_and_hold": (
        beats_buy_and_hold
    ),

    # Keep the detailed nested metrics.
    "classification_metrics": (
        classification_metrics
    ),
    "strategy_metrics": strategy_metrics,
}


def print_fold_result(
    result: dict[str, Any],
) -> None:
    """
    Print a readable summary for one evaluated fold.
    """
    classification = result[
        "classification_metrics"
    ]

    strategy = result[
        "strategy_metrics"
    ]

    print("\nWalk-forward fold completed")
    print("-" * 40)

    print(
        f"Fold: {result['fold']}"
    )

    print(
        f"Training rows: "
        f"{result['training_rows']}"
    )

    print(
        f"Testing rows: "
        f"{result['testing_rows']}"
    )

    print(
        "Training period:",
        result["training_period"],
    )

    print(
        "Testing period:",
        result["testing_period"],
    )

    print("\nClassification metrics:")

    print(
        f"Accuracy: "
        f"{classification['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{classification['roc_auc']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{classification['balanced_accuracy']:.4f}"
    )

    print("\nStrategy metrics:")

    print(
        f"Win rate: "
        f"{strategy['win_rate']:.4f}"
    )

    print(
        f"Strategy return: "
        f"{strategy['cumulative_strategy_return']:.4%}"
    )

    print(
        f"Buy-and-hold return: "
        f"{strategy['buy_and_hold_return']:.4%}"
    )

    print(
        f"Excess return: "
        f"{result['strategy_excess_return']:.4%}"
    )

    print(
        "Beats buy-and-hold:",
        result["beats_buy_and_hold"],
    )


def main() -> None:
    """
    Run the complete walk-forward evaluation pipeline.
    """
    config = load_walk_forward_config()

    model_name = str(
        config["model"]
    )

    data_path = Path(
        config["data_path"]
    )

    train_window = int(
        config["train_window"]
    )

    test_window = int(
        config["test_window"]
    )

    step_size = int(
        config["step_size"]
    )

    purge_gap = int(
        config["purge_gap"]
    )

    data = load_training_data(
        data_path
    )

    folds = generate_walk_forward_folds(
        total_rows=len(data),
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        purge_gap=purge_gap,
    )

    fold_results: list[
        dict[str, Any]
    ] = []

    for fold in folds:
        print(
            f"\nRunning fold "
            f"{fold.fold_number}..."
        )

        result = evaluate_fold(
            data=data,
            fold=fold,
            model_name=model_name,
        )

        fold_results.append(result)

        print_fold_result(result)

    if not fold_results:
        raise ValueError(
            "No walk-forward folds were generated. "
            "Check the YAML configuration values."
        )

    aggregate_results = (
        aggregate_walk_forward_results(
            fold_results
        )
    )

    report = {
        "model": model_name,
        "configuration": {
            "config_path": str(CONFIG_PATH),
            "train_window": train_window,
            "test_window": test_window,
            "step_size": step_size,
            "purge_gap": purge_gap,
        },
        "dataset": {
            "path": str(data_path),
            "total_rows": int(len(data)),
            "period": get_period(data),
        },
        "folds": fold_results,
        "aggregate": aggregate_results,
    }

    save_json(
        report,
        WALK_FORWARD_REPORT_PATH,
    )

    print(
        f"\nCompleted "
        f"{len(fold_results)} "
        "walk-forward folds."
    )

    print(
        "Walk-forward report saved to:",
        WALK_FORWARD_REPORT_PATH,
    )

if __name__ == "__main__":
    main()