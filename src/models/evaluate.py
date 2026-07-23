from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
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

from src.models.train import (
    chronological_split,
    load_training_data,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/gold_features.csv")
MODEL_DIR = Path("artifacts/models")

# Keep Logistic Regression as the official model because the existing
# evaluation report uses Logistic Regression.
SELECTED_MODEL = "logistic_regression"

EVALUATION_METRICS_PATH = Path(
    "artifacts/evaluation_metrics.json"
)
MODEL_METRICS_PATH = Path(
    "artifacts/model_metrics.json"
)
MODEL_COMPARISON_PATH = Path(
    "artifacts/model_comparison.csv"
)
TRADES_PATH = Path(
    "artifacts/test_period_trades.csv"
)
CUMULATIVE_RETURNS_PATH = Path(
    "artifacts/cumulative_returns.png"
)

TEST_SIZE = 0.20
PURGED_BOUNDARY_ROWS = 1
PREDICTION_THRESHOLD = 0.50

FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]

TARGET_COLUMN = "target"


# ---------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------

def safe_roc_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Calculate ROC-AUC safely.

    ROC-AUC cannot be calculated when the test set contains
    only one target class.
    """
    if y_true.nunique() < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calculate classification metrics for one trained model.
    """
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(
            y_true=y_test,
            probabilities=probabilities,
        ),
    }


def evaluate_all_models(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """
    Load and evaluate every trained model.

    Returns:
        A dictionary containing model metrics.
        A dictionary containing loaded model objects.
    """
    metrics: dict[str, dict[str, float]] = {}
    models: dict[str, Any] = {}

    model_paths = sorted(
        MODEL_DIR.glob("*.joblib")
    )

    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found in {MODEL_DIR}. "
            "Run the training pipeline first."
        )

    for model_path in model_paths:
        model_name = model_path.stem
        model = joblib.load(model_path)

        models[model_name] = model

        metrics[model_name] = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

    return metrics, models


# ---------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------

def prepare_test_period_data(
    data: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Prepare test-period rows for strategy evaluation.

    The model predicts the direction of the next period. Therefore,
    the strategy return must use the following period's return:

        future_return = return_1.shift(-1)

    Using the current row's return would create incorrect time
    alignment between the prediction and realised return.
    """
    if "return_1" not in data.columns:
        raise KeyError(
            "The dataset must contain a 'return_1' column "
            "for strategy evaluation."
        )

    test_data = data.loc[
        X_test.index
    ].copy()

    test_data["actual_target"] = (
        y_test.loc[X_test.index]
        .astype(int)
    )

    # Prediction made at time t is evaluated using return at t + 1.
    test_data["future_return"] = (
        data["return_1"]
        .shift(-1)
        .loc[X_test.index]
    )

    if "timestamp" in test_data.columns:
        test_data["timestamp"] = pd.to_datetime(
            test_data["timestamp"],
            utc=True,
            errors="coerce",
        )

    return test_data


def calculate_strategy_metrics(
    model: Any,
    X_test: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:
    """
    Calculate long/short strategy metrics.

    Prediction 1:
        Take a long position.

    Prediction 0:
        Take a short position.

    Transaction costs, slippage and spread are not included.
    """
    predictions = model.predict(
        X_test
    ).astype(int)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    trades = test_data.copy()

    trades["prediction"] = predictions
    trades["probability_up"] = probabilities

    # Convert binary prediction to trading position:
    # 1 -> long (+1)
    # 0 -> short (-1)
    trades["position"] = np.where(
        trades["prediction"] == 1,
        1,
        -1,
    )

    # The final row may not have a following-period return.
    trades = trades.dropna(
        subset=["future_return"]
    ).copy()

    trades["strategy_return"] = (
        trades["position"]
        * trades["future_return"]
    )

    trades["trade_won"] = (
        trades["strategy_return"] > 0
    )

    trades["cumulative_strategy_return"] = (
        1 + trades["strategy_return"]
    ).cumprod() - 1

    trades["cumulative_buy_and_hold_return"] = (
        1 + trades["future_return"]
    ).cumprod() - 1

    number_of_trades = int(
        len(trades)
    )

    winning_trades = int(
        trades["trade_won"].sum()
    )

    losing_trades = int(
        number_of_trades - winning_trades
    )

    win_rate = (
        winning_trades / number_of_trades
        if number_of_trades > 0
        else 0.0
    )

    cumulative_strategy_return = (
        float(
            trades[
                "cumulative_strategy_return"
            ].iloc[-1]
        )
        if number_of_trades > 0
        else 0.0
    )

    buy_and_hold_return = (
        float(
            trades[
                "cumulative_buy_and_hold_return"
            ].iloc[-1]
        )
        if number_of_trades > 0
        else 0.0
    )

    strategy_metrics = {
        "number_of_trades": number_of_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": float(win_rate),
        "cumulative_strategy_return": (
            cumulative_strategy_return
        ),
        "buy_and_hold_return": (
            buy_and_hold_return
        ),
        "average_trade_return": float(
            trades["strategy_return"].mean()
        )
        if number_of_trades > 0
        else 0.0,
        "best_trade_return": float(
            trades["strategy_return"].max()
        )
        if number_of_trades > 0
        else 0.0,
        "worst_trade_return": float(
            trades["strategy_return"].min()
        )
        if number_of_trades > 0
        else 0.0,
    }

    return strategy_metrics, trades


# ---------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------

def get_period(
    data: pd.DataFrame,
) -> dict[str, str | None]:
    """
    Return the start and end timestamps for a dataset period.
    """
    if "timestamp" not in data.columns:
        return {
            "start": None,
            "end": None,
        }

    timestamps = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    ).dropna()

    if timestamps.empty:
        return {
            "start": None,
            "end": None,
        }

    return {
        "start": timestamps.iloc[0].isoformat(),
        "end": timestamps.iloc[-1].isoformat(),
    }


def create_evaluation_report(
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    classification_metrics: dict[str, float],
    strategy_metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Build the official detailed evaluation report.
    """
    train_data = data.loc[
        X_train.index
    ]

    test_data = data.loc[
        X_test.index
    ]

    return {
        "model": SELECTED_MODEL,
        "threshold": PREDICTION_THRESHOLD,
        "test_size": TEST_SIZE,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "dataset": {
            "total_rows": int(
                len(data)
            ),
            "training_rows": int(
                len(X_train)
            ),
            "testing_rows": int(
                len(X_test)
            ),
            "purged_boundary_rows": (
                PURGED_BOUNDARY_ROWS
            ),
            "training_period": get_period(
                train_data
            ),
            "testing_period": get_period(
                test_data
            ),
        },
        "classification_metrics": {
            "accuracy": classification_metrics[
                "accuracy"
            ],
            "balanced_accuracy": (
                classification_metrics[
                    "balanced_accuracy"
                ]
            ),
            "precision": classification_metrics[
                "precision"
            ],
            "recall": classification_metrics[
                "recall"
            ],
            "f1_score": classification_metrics[
                "f1"
            ],
            "roc_auc": classification_metrics[
                "roc_auc"
            ],
        },
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
                "The reported performance is historical and "
                "does not guarantee future results."
            ),
        ],
    }


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Save a dictionary as formatted JSON.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            allow_nan=False,
        )


def save_model_comparison(
    metrics: dict[str, dict[str, float]],
) -> None:
    """
    Save classification metrics for every trained model.
    """
    comparison = pd.DataFrame.from_dict(
        metrics,
        orient="index",
    )

    comparison.index.name = "model"

    comparison = comparison.reset_index()

    MODEL_COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )


def save_trades(
    trades: pd.DataFrame,
) -> None:
    """
    Save test-period predictions and trading returns.
    """
    columns = [
        column
        for column in [
            "timestamp",
            "actual_target",
            "prediction",
            "probability_up",
            "position",
            "future_return",
            "strategy_return",
            "trade_won",
            "cumulative_strategy_return",
            "cumulative_buy_and_hold_return",
        ]
        if column in trades.columns
    ]

    TRADES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades[columns].to_csv(
        TRADES_PATH,
        index=False,
    )


def save_cumulative_returns_plot(
    trades: pd.DataFrame,
) -> None:
    """
    Save cumulative strategy and buy-and-hold return curves.
    """
    if trades.empty:
        raise ValueError(
            "Cannot generate cumulative-return plot "
            "because no valid trades were created."
        )

    if "timestamp" in trades.columns:
        x_axis = trades["timestamp"]
        x_label = "Timestamp"
    else:
        x_axis = range(len(trades))
        x_label = "Test observation"

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        x_axis,
        trades[
            "cumulative_strategy_return"
        ],
        label="Model strategy",
    )

    plt.plot(
        x_axis,
        trades[
            "cumulative_buy_and_hold_return"
        ],
        label="Buy and hold",
    )

    plt.axhline(
        y=0,
        linewidth=1,
    )

    plt.title(
        "Cumulative Returns on Test Period"
    )
    plt.xlabel(x_label)
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    CUMULATIVE_RETURNS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CUMULATIVE_RETURNS_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """
    Evaluate all models and regenerate every official artifact.
    """
    data = load_training_data(
        DATA_PATH
    )

    (
        X_train,
        X_test,
        _,
        y_test,
    ) = chronological_split(data)

    all_metrics, models = evaluate_all_models(
        X_test=X_test,
        y_test=y_test,
    )

    if SELECTED_MODEL not in models:
        available_models = ", ".join(
            sorted(models)
        )

        raise FileNotFoundError(
            f"Selected model '{SELECTED_MODEL}' was not found. "
            f"Available models: {available_models}"
        )

    selected_model = models[
        SELECTED_MODEL
    ]

    test_period_data = (
        prepare_test_period_data(
            data=data,
            X_test=X_test,
            y_test=y_test,
        )
    )

    (
        strategy_metrics,
        trades,
    ) = calculate_strategy_metrics(
        model=selected_model,
        X_test=X_test,
        test_data=test_period_data,
    )

    evaluation_report = (
        create_evaluation_report(
            data=data,
            X_train=X_train,
            X_test=X_test,
            classification_metrics=(
                all_metrics[
                    SELECTED_MODEL
                ]
            ),
            strategy_metrics=(
                strategy_metrics
            ),
        )
    )

    # Save official selected-model report.
    save_json(
        evaluation_report,
        EVALUATION_METRICS_PATH,
    )

    # Save all-model classification results.
    save_json(
        all_metrics,
        MODEL_METRICS_PATH,
    )

    save_model_comparison(
        all_metrics
    )

    save_trades(
        trades
    )

    save_cumulative_returns_plot(
        trades
    )

    for model_name, scores in all_metrics.items():
        print(
            f"\nModel: {model_name}"
        )

        for metric_name, value in scores.items():
            print(
                f"{metric_name}: {value:.4f}"
            )

    print(
        f"\nOfficial model: {SELECTED_MODEL}"
    )

    print("\nStrategy metrics:")

    for metric_name, value in strategy_metrics.items():
        if isinstance(value, float):
            print(
                f"{metric_name}: {value:.6f}"
            )
        else:
            print(
                f"{metric_name}: {value}"
            )

    print(
        "\nEvaluation metrics saved to:",
        EVALUATION_METRICS_PATH,
    )

    print(
        "All model metrics saved to:",
        MODEL_METRICS_PATH,
    )

    print(
        "Comparison saved to:",
        MODEL_COMPARISON_PATH,
    )

    print(
        "Trades saved to:",
        TRADES_PATH,
    )

    print(
        "Cumulative-return chart saved to:",
        CUMULATIVE_RETURNS_PATH,
    )


if __name__ == "__main__":
    main()