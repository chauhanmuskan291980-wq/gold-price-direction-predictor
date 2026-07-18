from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.features.build_features import FEATURE_COLUMNS
from src.models.train import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_model_pipeline,
    load_feature_data,
    prepare_training_data,
)

MetricValue = float | None
Metrics = dict[str, MetricValue]


def safe_roc_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float | None:
    """
    Calculate ROC-AUC when both target classes exist.

    ROC-AUC cannot be calculated when a test fold contains
    only one target class.
    """
    if y_true.nunique() < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> Metrics:
    """Calculate classification metrics for one fold."""
    metrics: Metrics = {
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
        "roc_auc": None,
    }

    if probabilities is not None:
        metrics["roc_auc"] = safe_roc_auc(
            y_true,
            probabilities,
        )

    return metrics


def require_metric(
    metrics: Metrics,
    metric_name: str,
) -> float:
    """
    Return a metric value that must not be None.

    ROC-AUC may legitimately be None, but metrics such as
    accuracy and balanced accuracy should always be available.
    """
    value = metrics.get(metric_name)

    if value is None:
        raise ValueError(f"Metric could not be calculated: {metric_name}")

    return value


def validate_with_time_series_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 1,
    threshold: float = 0.50,
    random_state: int = 42,
    max_iterations: int = 1_000,
) -> list[dict[str, Any]]:
    """
    Evaluate Logistic Regression and a dummy baseline
    across expanding chronological folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if gap < 1:
        raise ValueError("gap must be at least 1 to prevent target-boundary leakage.")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].astype(int)
    timestamps = data[TIMESTAMP_COLUMN]

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
    )

    fold_results: list[dict[str, Any]] = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(features),
        start=1,
    ):
        x_train = features.iloc[train_indices]
        x_test = features.iloc[test_indices]

        y_train = target.iloc[train_indices]
        y_test = target.iloc[test_indices]

        train_timestamps = timestamps.iloc[train_indices]
        test_timestamps = timestamps.iloc[test_indices]

        if y_train.nunique() < 2:
            raise ValueError(
                f"Fold {fold_number} training data contains only one class."
            )

        model = build_model_pipeline(
            random_state=random_state,
            max_iterations=max_iterations,
        )

        model.fit(
            x_train,
            y_train,
        )

        model_probabilities = model.predict_proba(x_test)[:, 1]

        model_predictions = (model_probabilities >= threshold).astype(int)

        model_metrics = calculate_metrics(
            y_true=y_test,
            predictions=model_predictions,
            probabilities=model_probabilities,
        )

        dummy_model = DummyClassifier(
            strategy="most_frequent",
        )

        dummy_model.fit(
            x_train,
            y_train,
        )

        dummy_predictions = dummy_model.predict(x_test)

        dummy_metrics = calculate_metrics(
            y_true=y_test,
            predictions=dummy_predictions,
        )

        model_accuracy = require_metric(
            model_metrics,
            "accuracy",
        )
        baseline_accuracy = require_metric(
            dummy_metrics,
            "accuracy",
        )

        model_balanced_accuracy = require_metric(
            model_metrics,
            "balanced_accuracy",
        )
        baseline_balanced_accuracy = require_metric(
            dummy_metrics,
            "balanced_accuracy",
        )

        fold_result: dict[str, Any] = {
            "fold": fold_number,
            "training_rows": len(x_train),
            "testing_rows": len(x_test),
            "gap_rows": gap,
            "training_period": {
                "start": (train_timestamps.min().isoformat()),
                "end": (train_timestamps.max().isoformat()),
            },
            "testing_period": {
                "start": (test_timestamps.min().isoformat()),
                "end": (test_timestamps.max().isoformat()),
            },
            "training_class_distribution": {
                str(key): int(value)
                for key, value in (y_train.value_counts().sort_index().items())
            },
            "testing_class_distribution": {
                str(key): int(value)
                for key, value in (y_test.value_counts().sort_index().items())
            },
            "model_metrics": model_metrics,
            "baseline_metrics": dummy_metrics,
            "accuracy_improvement": (model_accuracy - baseline_accuracy),
            "balanced_accuracy_improvement": (
                model_balanced_accuracy - baseline_balanced_accuracy
            ),
        }

        fold_results.append(fold_result)

    return fold_results


def summarize_metric(
    fold_results: list[dict[str, Any]],
    model_key: str,
    metric_name: str,
) -> dict[str, float | None]:
    """Calculate mean and standard deviation for a metric."""
    values: list[float] = []

    for fold in fold_results:
        metric_value = fold[model_key][metric_name]

        if metric_value is not None:
            values.append(float(metric_value))

    if not values:
        return {
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }

    standard_deviation = stdev(values) if len(values) > 1 else 0.0

    return {
        "mean": float(mean(values)),
        "standard_deviation": float(standard_deviation),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def build_validation_summary(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate cross-validation results."""
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    model_summary = {
        metric: summarize_metric(
            fold_results,
            "model_metrics",
            metric,
        )
        for metric in metric_names
    }

    baseline_summary = {
        metric: summarize_metric(
            fold_results,
            "baseline_metrics",
            metric,
        )
        for metric in metric_names
    }

    model_wins = 0
    baseline_wins = 0
    ties = 0

    for fold in fold_results:
        model_accuracy = require_metric(
            fold["model_metrics"],
            "accuracy",
        )
        baseline_accuracy = require_metric(
            fold["baseline_metrics"],
            "accuracy",
        )

        if model_accuracy > baseline_accuracy:
            model_wins += 1
        elif model_accuracy < baseline_accuracy:
            baseline_wins += 1
        else:
            ties += 1

    return {
        "number_of_folds": len(fold_results),
        "model_summary": model_summary,
        "baseline_summary": baseline_summary,
        "fold_comparison": {
            "model_wins": model_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
        },
    }


def save_validation_results(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Save validation results as JSON."""
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


def print_fold_results(
    fold_results: list[dict[str, Any]],
) -> None:
    """Print model and baseline results per fold."""
    print("\nTime-series validation")
    print("----------------------")

    for fold in fold_results:
        model_metrics: Metrics = fold["model_metrics"]
        baseline_metrics: Metrics = fold["baseline_metrics"]

        model_accuracy = require_metric(
            model_metrics,
            "accuracy",
        )
        baseline_accuracy = require_metric(
            baseline_metrics,
            "accuracy",
        )
        model_balanced_accuracy = require_metric(
            model_metrics,
            "balanced_accuracy",
        )
        baseline_balanced_accuracy = require_metric(
            baseline_metrics,
            "balanced_accuracy",
        )

        print(f"\nFold {fold['fold']}")
        print(f"Training rows: {fold['training_rows']}")
        print(f"Testing rows:  {fold['testing_rows']}")
        print(f"Model accuracy:    {model_accuracy:.4f}")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"Model balanced accuracy:    {model_balanced_accuracy:.4f}")
        print(f"Baseline balanced accuracy: {baseline_balanced_accuracy:.4f}")

        model_roc_auc = model_metrics["roc_auc"]

        if model_roc_auc is not None:
            print(f"Model ROC-AUC:     {model_roc_auc:.4f}")


def print_summary(
    summary: dict[str, Any],
) -> None:
    """Print aggregate validation results."""
    model = summary["model_summary"]
    baseline = summary["baseline_summary"]
    comparison = summary["fold_comparison"]

    model_accuracy = model["accuracy"]["mean"]
    baseline_accuracy = baseline["accuracy"]["mean"]

    model_balanced_accuracy = model["balanced_accuracy"]["mean"]
    baseline_balanced_accuracy = baseline["balanced_accuracy"]["mean"]

    if model_accuracy is None:
        raise ValueError("Model mean accuracy is unavailable.")

    if baseline_accuracy is None:
        raise ValueError("Baseline mean accuracy is unavailable.")

    if model_balanced_accuracy is None:
        raise ValueError("Model mean balanced accuracy is unavailable.")

    if baseline_balanced_accuracy is None:
        raise ValueError("Baseline mean balanced accuracy is unavailable.")

    print("\nAggregate results")
    print("-----------------")

    print(f"Model mean accuracy:        {model_accuracy:.4f}")
    print(f"Baseline mean accuracy:     {baseline_accuracy:.4f}")
    print(f"Model balanced accuracy:    {model_balanced_accuracy:.4f}")
    print(f"Baseline balanced accuracy: {baseline_balanced_accuracy:.4f}")

    roc_auc_mean = model["roc_auc"]["mean"]

    if roc_auc_mean is not None:
        print(f"Model mean ROC-AUC:         {roc_auc_mean:.4f}")

    print(f"\nModel wins:    {comparison['model_wins']}")
    print(f"Baseline wins: {comparison['baseline_wins']}")
    print(f"Ties:          {comparison['ties']}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Gold direction model using "
            "purged time-series cross-validation."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/gold_features.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/time_series_validation.json"),
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:
    """Run purged time-series validation."""
    args = parse_arguments()

    raw_data = load_feature_data(args.input)

    prepared_data = prepare_training_data(raw_data)

    print(f"Dataset rows: {len(prepared_data):,}")
    print(f"Time-series splits: {args.splits}")
    print(f"Gap between train and test: {args.gap} row(s)")

    fold_results = validate_with_time_series_splits(
        data=prepared_data,
        n_splits=args.splits,
        gap=args.gap,
        threshold=args.threshold,
    )

    summary = build_validation_summary(fold_results)

    results: dict[str, Any] = {
        "model": "LogisticRegression",
        "baseline": ("DummyClassifier-most_frequent"),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "threshold": args.threshold,
        "gap": args.gap,
        "folds": fold_results,
        "summary": summary,
    }

    print_fold_results(fold_results)
    print_summary(summary)

    save_validation_results(
        results,
        args.output,
    )

    print(f"\nValidation results saved to: {args.output}")


if __name__ == "__main__":
    main()
