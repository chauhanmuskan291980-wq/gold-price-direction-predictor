from pathlib import Path

import pandas as pd
import pytest

from src.models.train import (
    prepare_training_data,
)
from src.models.validate import (
    build_validation_summary,
    calculate_metrics,
    save_validation_results,
    validate_with_time_series_splits,
)


def create_validation_data(
    rows: int = 180,
) -> pd.DataFrame:
    """Create chronological sample feature data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "return_1": [
                ((index % 7) - 3) / 1_000
                for index in range(rows)
            ],
            "ma_gap": [
                ((index % 11) - 5) / 1_000
                for index in range(rows)
            ],
            "volatility_10": [
                0.001
                + (index % 6) / 10_000
                for index in range(rows)
            ],
            "candle_body_ratio": [
                0.1 + (index % 8) / 10
                for index in range(rows)
            ],
            "rsi_14": [
                30.0 + (index % 40)
                for index in range(rows)
            ],
            "target": [
                index % 2
                for index in range(rows)
            ],
        }
    )


def test_validation_creates_requested_folds() -> None:
    """Validation should return the requested folds."""
    data = prepare_training_data(
        create_validation_data()
    )

    results = validate_with_time_series_splits(
        data=data,
        n_splits=5,
        gap=1,
    )

    assert len(results) == 5


def test_validation_preserves_time_order() -> None:
    """Training periods must finish before testing."""
    data = prepare_training_data(
        create_validation_data()
    )

    results = validate_with_time_series_splits(
        data=data,
        n_splits=3,
        gap=1,
    )

    for fold in results:
        training_end = pd.Timestamp(
            fold["training_period"]["end"]
        )

        testing_start = pd.Timestamp(
            fold["testing_period"]["start"]
        )

        assert training_end < testing_start


def test_validation_compares_baseline() -> None:
    """Every fold should contain model and baseline metrics."""
    data = prepare_training_data(
        create_validation_data()
    )

    results = validate_with_time_series_splits(
        data=data,
        n_splits=3,
        gap=1,
    )

    for fold in results:
        assert "model_metrics" in fold
        assert "baseline_metrics" in fold
        assert "accuracy_improvement" in fold


def test_invalid_gap_is_rejected() -> None:
    """A zero gap should be rejected."""
    data = prepare_training_data(
        create_validation_data()
    )

    with pytest.raises(
        ValueError,
        match="gap",
    ):
        validate_with_time_series_splits(
            data=data,
            n_splits=3,
            gap=0,
        )


def test_calculate_metrics_returns_values() -> None:
    """Metrics should be between zero and one."""
    y_true = pd.Series([0, 1, 0, 1])
    predictions = pd.Series([0, 1, 1, 1]).to_numpy()
    probabilities = pd.Series(
        [0.2, 0.8, 0.7, 0.9]
    ).to_numpy()

    metrics = calculate_metrics(
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
    )

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_build_validation_summary() -> None:
    """Fold results should produce an aggregate summary."""
    data = prepare_training_data(
        create_validation_data()
    )

    fold_results = (
        validate_with_time_series_splits(
            data=data,
            n_splits=3,
            gap=1,
        )
    )

    summary = build_validation_summary(
        fold_results
    )

    assert summary["number_of_folds"] == 3
    assert "model_summary" in summary
    assert "baseline_summary" in summary


def test_save_validation_results(
    tmp_path: Path,
) -> None:
    """Validation results should be saved as JSON."""
    output_path = (
        tmp_path / "validation.json"
    )

    save_validation_results(
        {
            "model": "LogisticRegression",
            "number_of_folds": 5,
        },
        output_path,
    )

    assert output_path.exists()