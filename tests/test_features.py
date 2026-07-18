from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    build_features,
    calculate_rsi,
    load_processed_data,
    save_feature_data,
)


def create_sample_data(
    rows: int = 40,
) -> pd.DataFrame:
    """Create sample processed Gold data."""
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )

    close_prices = [
        2600.0 + index
        for index in range(rows)
    ]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [
                price - 1
                for price in close_prices
            ],
            "high": [
                price + 3
                for price in close_prices
            ],
            "low": [
                price - 3
                for price in close_prices
            ],
            "close": close_prices,
            "volume": [
                100 + index
                for index in range(rows)
            ],
            "target": [
                index % 2
                for index in range(rows)
            ],
        }
    )


def test_calculate_rsi_returns_same_length() -> None:
    """RSI output should match the input length."""
    data = create_sample_data()

    result = calculate_rsi(data["close"])

    assert len(result) == len(data)


def test_build_features_creates_expected_columns() -> None:
    """All expected feature columns should be created."""
    data = create_sample_data()

    result = build_features(data)

    for feature in FEATURE_COLUMNS:
        assert feature in result.columns


def test_feature_dataset_has_no_missing_values() -> None:
    """Final feature columns should not contain nulls."""
    data = create_sample_data()

    result = build_features(data)

    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_feature_dataset_is_sorted() -> None:
    """Feature data should remain chronological."""
    data = create_sample_data().iloc[::-1]

    result = build_features(data)

    assert result["timestamp"].is_monotonic_increasing


def test_short_dataset_is_rejected() -> None:
    """Datasets without enough rolling rows should fail."""
    data = create_sample_data(rows=5)

    with pytest.raises(
        ValueError,
        match="No valid rows remained",
    ):
        build_features(data)


def test_load_missing_file() -> None:
    """Missing processed files should raise an error."""
    missing_path = Path(
        "data/processed/missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        load_processed_data(missing_path)


def test_save_feature_data(
    tmp_path: Path,
) -> None:
    """Feature dataset should be saved as CSV."""
    data = create_sample_data()

    featured = build_features(data)

    output_path = (
        tmp_path / "gold_features.csv"
    )

    save_feature_data(
        featured,
        output_path,
    )

    assert output_path.exists()

    saved_data = pd.read_csv(output_path)

    for feature in FEATURE_COLUMNS:
        assert feature in saved_data.columns