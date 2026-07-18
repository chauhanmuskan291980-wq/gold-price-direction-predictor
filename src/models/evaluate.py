from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.build_features import FEATURE_COLUMNS
from src.models.train import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_model_pipeline,
    load_feature_data,
    prepare_training_data,
)

CLOSE_COLUMN = "close"
NEXT_RETURN_COLUMN = "next_return"


def calculate_next_returns(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the return from the current candle close
    to the next candle close.
    """
    next_close = data[CLOSE_COLUMN].shift(-1)

    next_returns = (
        next_close / data[CLOSE_COLUMN]
    ) - 1.0

    return next_returns


def create_chronological_split(
    data: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically and purge one boundary row.

    The last training target uses the following candle's close.
    Removing the row immediately before the test period prevents
    the training target from depending on the first test candle.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    split_index = int(
        len(data) * (1 - test_size)
    )

    if split_index < 2:
        raise ValueError(
            "Not enough rows for the requested split."
        )

    train_data = data.iloc[
        : split_index - 1
    ].copy()

    test_data = data.iloc[
        split_index:
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "The chronological split produced "
            "an empty train or test dataset."
        )

    return train_data, test_data


def calculate_classification_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    if y_true.nunique() >= 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def calculate_strategy_results(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Calculate a naive trade-every-signal strategy.

    Prediction 1:
        Long position.

    Prediction 0:
        Short position.
    """
    results = test_data[
        [
            TIMESTAMP_COLUMN,
            CLOSE_COLUMN,
            NEXT_RETURN_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    results["prediction"] = predictions

    results["position"] = np.where(
        results["prediction"] == 1,
        1.0,
        -1.0,
    )

    results["strategy_return"] = (
        results["position"]
        * results[NEXT_RETURN_COLUMN]
    )

    results["strategy_win"] = (
        results["strategy_return"] > 0
    )

    results["cumulative_strategy_return"] = (
        1.0 + results["strategy_return"]
    ).cumprod() - 1.0

    results["cumulative_buy_and_hold_return"] = (
        1.0 + results[NEXT_RETURN_COLUMN]
    ).cumprod() - 1.0

    strategy_metrics = {
        "number_of_trades": float(
            len(results)
        ),
        "winning_trades": float(
            results["strategy_win"].sum()
        ),
        "losing_trades": float(
            (~results["strategy_win"]).sum()
        ),
        "win_rate": float(
            results["strategy_win"].mean()
        ),
        "cumulative_strategy_return": float(
            results[
                "cumulative_strategy_return"
            ].iloc[-1]
        ),
        "buy_and_hold_return": float(
            results[
                "cumulative_buy_and_hold_return"
            ].iloc[-1]
        ),
        "average_trade_return": float(
            results["strategy_return"].mean()
        ),
        "best_trade_return": float(
            results["strategy_return"].max()
        ),
        "worst_trade_return": float(
            results["strategy_return"].min()
        ),
    }

    return results, strategy_metrics


def save_cumulative_return_chart(
    strategy_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the cumulative-return comparison chart."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.plot(
        strategy_results[TIMESTAMP_COLUMN],
        strategy_results[
            "cumulative_strategy_return"
        ],
        label="Model strategy",
    )

    axis.plot(
        strategy_results[TIMESTAMP_COLUMN],
        strategy_results[
            "cumulative_buy_and_hold_return"
        ],
        label="Buy and hold",
    )

    axis.axhline(
        y=0,
        linewidth=1,
    )

    axis.set_title(
        "XAUUSD Test-Period Cumulative Returns"
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Cumulative return")
    axis.legend()
    axis.grid(True)

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)


def save_json_results(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )


def evaluate_model(
    data: pd.DataFrame,
    test_size: float = 0.20,
    threshold: float = 0.50,
    random_state: int = 42,
    max_iterations: int = 1_000,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """Train and evaluate the model on an untouched test period."""
    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    evaluation_data = data.copy()

    evaluation_data[NEXT_RETURN_COLUMN] = (
        calculate_next_returns(
            evaluation_data
        )
    )

    evaluation_data = evaluation_data.dropna(
        subset=[
            NEXT_RETURN_COLUMN,
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
            CLOSE_COLUMN,
        ]
    ).reset_index(drop=True)

    train_data, test_data = (
        create_chronological_split(
            evaluation_data,
            test_size=test_size,
        )
    )

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[
        TARGET_COLUMN
    ].astype(int)

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[
        TARGET_COLUMN
    ].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one class."
        )

    model = build_model_pipeline(
        random_state=random_state,
        max_iterations=max_iterations,
    )

    model.fit(
        x_train,
        y_train,
    )

    probabilities_up = model.predict_proba(
        x_test
    )[:, 1]

    predictions = (
        probabilities_up >= threshold
    ).astype(int)

    classification_metrics = (
        calculate_classification_metrics(
            y_true=y_test,
            predictions=predictions,
            probabilities=probabilities_up,
        )
    )

    strategy_results, strategy_metrics = (
        calculate_strategy_results(
            test_data=test_data,
            predictions=predictions,
        )
    )

    results: dict[str, Any] = {
        "model": "LogisticRegression",
        "threshold": threshold,
        "test_size": test_size,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "dataset": {
            "total_rows": len(
                evaluation_data
            ),
            "training_rows": len(
                train_data
            ),
            "testing_rows": len(
                test_data
            ),
            "purged_boundary_rows": 1,
            "training_period": {
                "start": train_data[
                    TIMESTAMP_COLUMN
                ].min().isoformat(),
                "end": train_data[
                    TIMESTAMP_COLUMN
                ].max().isoformat(),
            },
            "testing_period": {
                "start": test_data[
                    TIMESTAMP_COLUMN
                ].min().isoformat(),
                "end": test_data[
                    TIMESTAMP_COLUMN
                ].max().isoformat(),
            },
        },
        "classification_metrics": (
            classification_metrics
        ),
        "strategy_metrics": strategy_metrics,
        "limitations": [
            (
                "The strategy ignores transaction costs, "
                "spread, slippage, and execution latency."
            ),
            (
                "Every model prediction is treated as a "
                "long or short trade."
            ),
            (
                "The reported performance is historical "
                "and does not guarantee future results."
            ),
        ],
    }

    return results, strategy_results


def print_results(
    results: dict[str, Any],
) -> None:
    """Print final evaluation results."""
    dataset = results["dataset"]
    classification = results[
        "classification_metrics"
    ]
    strategy = results["strategy_metrics"]

    print("\nFinal test-period evaluation")
    print("----------------------------")

    print(
        f"Training rows: {dataset['training_rows']}"
    )
    print(
        f"Testing rows:  {dataset['testing_rows']}"
    )
    print(
        "Purged boundary rows: "
        f"{dataset['purged_boundary_rows']}"
    )

    print("\nClassification metrics")
    print("----------------------")
    print(
        f"Accuracy:          "
        f"{classification['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{classification['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision:         "
        f"{classification['precision']:.4f}"
    )
    print(
        f"Recall:            "
        f"{classification['recall']:.4f}"
    )
    print(
        f"F1 score:          "
        f"{classification['f1_score']:.4f}"
    )
    print(
        f"ROC-AUC:           "
        f"{classification['roc_auc']:.4f}"
    )

    print("\nNaive strategy metrics")
    print("----------------------")
    print(
        f"Trades:            "
        f"{int(strategy['number_of_trades'])}"
    )
    print(
        f"Win rate:          "
        f"{strategy['win_rate']:.4f}"
    )
    print(
        "Strategy return:   "
        f"{strategy['cumulative_strategy_return']:.4%}"
    )
    print(
        "Buy-and-hold:      "
        f"{strategy['buy_and_hold_return']:.4%}"
    )
    print(
        "Average trade:     "
        f"{strategy['average_trade_return']:.6%}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Gold direction classifier "
            "on an untouched chronological test period."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/gold_features.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation_metrics.json"
        ),
    )

    parser.add_argument(
        "--chart",
        type=Path,
        default=Path(
            "artifacts/cumulative_returns.png"
        ),
    )

    parser.add_argument(
        "--trades-output",
        type=Path,
        default=Path(
            "artifacts/test_period_trades.csv"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:
    """Run final model evaluation."""
    args = parse_arguments()

    raw_data = load_feature_data(
        args.input
    )

    prepared_data = prepare_training_data(
        raw_data
    )

    results, strategy_results = evaluate_model(
        data=prepared_data,
        test_size=args.test_size,
        threshold=args.threshold,
    )

    save_json_results(
        results,
        args.output,
    )

    save_cumulative_return_chart(
        strategy_results,
        args.chart,
    )

    args.trades_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_results.to_csv(
        args.trades_output,
        index=False,
    )

    print_results(results)

    print(
        "\nEvaluation metrics saved to: "
        f"{args.output}"
    )
    print(
        "Cumulative-return chart saved to: "
        f"{args.chart}"
    )
    print(
        "Test-period trades saved to: "
        f"{args.trades_output}"
    )


if __name__ == "__main__":
    main()