from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    clean_ohlc_data,
    create_direction_target,
    load_raw_data,
    save_processed_data,
    validate_required_columns,
)


def create_sample_data() -> pd.DataFrame:
    """Create valid hourly Gold OHLCV test data."""
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
            ],
            "open": [
                2600.0,
                2605.0,
                2601.0,
                2610.0,
            ],
            "high": [
                2610.0,
                2612.0,
                2615.0,
                2618.0,
            ],
            "low": [
                2595.0,
                2600.0,
                2598.0,
                2605.0,
            ],
            "close": [
                2605.0,
                2601.0,
                2610.0,
                2608.0,
            ],
            "volume": [
                100,
                110,
                120,
                105,
            ],
        }
    )


def test_validate_required_columns() -> None:
    """Valid data should contain every required column."""
    data = create_sample_data()

    validate_required_columns(data)


def test_missing_required_column_is_rejected() -> None:
    """Missing required columns should raise an error."""
    data = create_sample_data().drop(columns=["close"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_required_columns(data)


def test_clean_data_removes_duplicate_timestamps() -> None:
    """Duplicate timestamps should be removed."""
    data = create_sample_data()

    duplicate = data.iloc[[0]].copy()

    data = pd.concat(
        [data, duplicate],
        ignore_index=True,
    )

    cleaned = clean_ohlc_data(data)

    assert len(cleaned) == 4
    assert not cleaned["timestamp"].duplicated().any()


def test_clean_data_removes_invalid_prices() -> None:
    """Rows with impossible OHLC prices should be removed."""
    data = create_sample_data()

    data.loc[0, "high"] = 2500.0

    cleaned = clean_ohlc_data(data)

    assert len(cleaned) == 3


def test_clean_data_sorts_timestamps() -> None:
    """Cleaned data should be chronologically sorted."""
    data = create_sample_data().iloc[::-1]

    cleaned = clean_ohlc_data(data)

    assert cleaned["timestamp"].is_monotonic_increasing


def test_create_direction_target() -> None:
    """Target should represent the next close direction."""
    cleaned = clean_ohlc_data(create_sample_data())

    result = create_direction_target(cleaned)

    assert len(result) == 3
    assert result["target"].tolist() == [
        0,
        1,
        0,
    ]


def test_next_close_is_not_in_final_dataset() -> None:
    """Future close should not remain as a model feature."""
    cleaned = clean_ohlc_data(create_sample_data())

    result = create_direction_target(cleaned)

    assert "next_close" not in result.columns


def test_load_raw_data_missing_file() -> None:
    """Missing input files should raise an error."""
    missing_path = Path("data/raw/file-that-does-not-exist.csv")

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)


def test_save_processed_data(
    tmp_path: Path,
) -> None:
    """Processed data should be saved as CSV."""
    cleaned = clean_ohlc_data(create_sample_data())

    processed = create_direction_target(cleaned)

    output_path = tmp_path / "gold_processed.csv"

    save_processed_data(
        processed,
        output_path,
    )

    assert output_path.exists()

    saved_data = pd.read_csv(output_path)

    assert len(saved_data) == 3
    assert "target" in saved_data.columns
