from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration used to download hourly Gold market data."""

    symbol: str = "GC=F"
    period: str = "60d"
    interval: str = "1h"
    output_path: Path = Path("data/raw/gold_hourly.csv")


def download_gold_data(config: DownloadConfig) -> pd.DataFrame:
    """
    Download hourly Gold futures data from Yahoo Finance.

    Yahoo Finance uses the symbol GC=F for Gold futures.
    """
    data = yf.download(
        tickers=config.symbol,
        period=config.period,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "Yahoo Finance returned an empty dataset. "
            "Check the symbol, interval, or internet connection."
        )

    return data


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo Finance output into a standard OHLCV structure."""
    normalized = data.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    normalized = normalized.reset_index()

    normalized.columns = [
        str(column).strip().lower().replace(" ", "_") for column in normalized.columns
    ]

    timestamp_candidates = [
        "datetime",
        "date",
        "timestamp",
    ]

    timestamp_column = next(
        (column for column in timestamp_candidates if column in normalized.columns),
        None,
    )

    if timestamp_column is None:
        raise ValueError("Could not find a timestamp column in downloaded data.")

    normalized = normalized.rename(columns={timestamp_column: "timestamp"})

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column for column in required_columns if column not in normalized.columns
    ]

    if missing_columns:
        raise ValueError(f"Downloaded data is missing columns: {missing_columns}")

    normalized = normalized[required_columns]

    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        utc=True,
    )

    normalized = normalized.sort_values("timestamp")
    normalized = normalized.reset_index(drop=True)

    return normalized


def validate_downloaded_data(data: pd.DataFrame) -> None:
    """Validate the normalized hourly OHLC dataset."""
    if data.empty:
        raise ValueError("The normalized dataset is empty.")

    if data["timestamp"].isna().any():
        raise ValueError("The dataset contains invalid timestamps.")

    if data["timestamp"].duplicated().any():
        raise ValueError("The dataset contains duplicate timestamps.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("The dataset is not sorted chronologically.")

    price_columns = ["open", "high", "low", "close"]

    if data[price_columns].isna().any().any():
        raise ValueError("The dataset contains missing OHLC prices.")

    if (data[price_columns] <= 0).any().any():
        raise ValueError("The dataset contains non-positive prices.")

    invalid_high = data["high"] < data[["open", "low", "close"]].max(axis=1)

    if invalid_high.any():
        raise ValueError("The dataset contains invalid high prices.")

    invalid_low = data["low"] > data[["open", "high", "close"]].min(axis=1)

    if invalid_low.any():
        raise ValueError("The dataset contains invalid low prices.")


def save_raw_data(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save normalized hourly market data as a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download hourly Gold market data.")

    parser.add_argument(
        "--symbol",
        default="GC=F",
        help="Yahoo Finance market symbol.",
    )

    parser.add_argument(
        "--period",
        default="60d",
        help="Historical period to download.",
    )

    parser.add_argument(
        "--interval",
        default="1h",
        help="Market candle interval.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/gold_hourly.csv"),
        help="Path where the downloaded CSV will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the hourly Gold data download pipeline."""
    args = parse_arguments()

    config = DownloadConfig(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        output_path=args.output,
    )

    print("Downloading Gold hourly data...")
    print(f"Symbol: {config.symbol}")
    print(f"Period: {config.period}")
    print(f"Interval: {config.interval}")

    raw_data = download_gold_data(config)
    normalized_data = normalize_columns(raw_data)

    validate_downloaded_data(normalized_data)

    save_raw_data(
        normalized_data,
        config.output_path,
    )

    print("Download completed successfully.")
    print(f"Rows downloaded: {len(normalized_data):,}")
    print(
        "Time range: "
        f"{normalized_data['timestamp'].min()} "
        f"to {normalized_data['timestamp'].max()}"
    )
    print(f"Saved to: {config.output_path}")


if __name__ == "__main__":
    main()
