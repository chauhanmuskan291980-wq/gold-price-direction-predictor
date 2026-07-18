from pathlib import Path

import pandas as pd
import pytest

from src.data.download import (
    normalize_columns,
    save_raw_data,
    validate_downloaded_data,
)


def create_sample_data() -> pd.DataFrame:
    """Create sample Yahoo-style Gold OHLC data."""
    return pd.DataFrame(
        {
            "Datetime": pd.date_range(
                start="2026-01-01",
                periods=3,
                freq="h",
                tz="UTC",
            ),
            "Open": [2600.0, 2605.0, 2610.0],
            "High": [2610.0, 2615.0, 2620.0],
            "Low": [2595.0, 2600.0, 2605.0],
            "Close": [2605.0, 2610.0, 2615.0],
            "Volume": [100, 120, 110],
        }
    )


def test_normalize_columns() -> None:
    """Yahoo-style columns should be normalized."""
    sample = create_sample_data()

    result = normalize_columns(sample)

    assert result.columns.tolist() == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert result["timestamp"].is_monotonic_increasing


def test_validation_accepts_valid_data() -> None:
    """Valid OHLC data should pass validation."""
    sample = normalize_columns(create_sample_data())

    validate_downloaded_data(sample)


def test_validation_rejects_duplicate_timestamp() -> None:
    """Duplicate hourly timestamps should be rejected."""
    sample = normalize_columns(create_sample_data())

    sample.loc[1, "timestamp"] = sample.loc[0, "timestamp"]

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        validate_downloaded_data(sample)


def test_validation_rejects_invalid_high() -> None:
    """High must not be lower than the candle close."""
    sample = normalize_columns(create_sample_data())

    sample.loc[0, "high"] = 2500.0

    with pytest.raises(
        ValueError,
        match="invalid high",
    ):
        validate_downloaded_data(sample)


def test_save_raw_data(tmp_path: Path) -> None:
    """Normalized data should be written to CSV."""
    sample = normalize_columns(create_sample_data())

    output_path = tmp_path / "gold_hourly.csv"

    save_raw_data(
        sample,
        output_path,
    )

    assert output_path.exists()

    saved_data = pd.read_csv(output_path)

    assert len(saved_data) == 3
    assert saved_data.columns.tolist() == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
