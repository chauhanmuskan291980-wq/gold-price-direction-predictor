import json
from pathlib import Path
from typing import Any

import joblib
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

MODEL_DIR = Path("artifacts/models")
METRICS_PATH = Path(
    "artifacts/metrics/model_metrics.json"
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series[Any],
) -> dict[str, float]:
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
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }


def evaluate_all_models(
    X_test: pd.DataFrame,
    y_test: pd.Series[Any],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    for model_path in MODEL_DIR.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)

        metrics[model_name] = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

    return metrics


def save_metrics(
    metrics: dict[str, dict[str, float]],
) -> None:
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )


def main() -> None:
    data = load_training_data(
        "data/processed/gold_features.csv"
    )

    _, X_test, _, y_test = chronological_split(
        data
    )

    metrics = evaluate_all_models(
        X_test=X_test,
        y_test=y_test,
    )

    save_metrics(metrics)

    comparison = pd.DataFrame.from_dict(
        metrics,
        orient="index",
    )

    comparison.index.name = "model"

    comparison.to_csv(
        METRICS_PATH.parent
        / "model_comparison.csv"
    )

    for model_name, scores in metrics.items():
        print(f"\nModel: {model_name}")

        for metric_name, value in scores.items():
            print(
                f"{metric_name}: {value:.4f}"
            )

    print(
        "\nMetrics saved to:",
        METRICS_PATH,
    )

    print(
        "Comparison saved to:",
        METRICS_PATH.parent
        / "model_comparison.csv",
    )


if __name__ == "__main__":
    main()