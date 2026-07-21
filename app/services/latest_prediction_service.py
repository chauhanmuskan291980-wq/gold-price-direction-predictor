from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.features.build_features import FEATURE_COLUMNS, build_features

GOLD_TICKER = "GC=F"
DOWNLOAD_PERIOD = "60d"
DOWNLOAD_INTERVAL = "1h"


class LatestPredictionError(RuntimeError):
    """Raised when the latest hourly prediction cannot be generated."""


def download_latest_gold_data() -> pd.DataFrame:
    """Download recent hourly gold futures data."""
    data = yf.download(
        GOLD_TICKER,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )

    if data.empty:
        raise LatestPredictionError(
            "No hourly gold market data was returned by Yahoo Finance."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    # Hourly data normally contains a Datetime column.
    if "datetime" in data.columns:
        data = data.rename(columns={"datetime": "timestamp"})
    elif "date" in data.columns:
        data = data.rename(columns={"date": "timestamp"})

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise LatestPredictionError(
            f"Downloaded hourly data is missing columns: {missing}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    return data


def build_latest_feature_row() -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build the most recent hourly feature row without changing build_features."""
    raw_data = download_latest_gold_data()

    # build_features() expects a target column because it was created
    # for model training. This temporary value is only used to satisfy
    # that existing contract.
    raw_data = raw_data.copy()
    raw_data["target"] = 0

    feature_data = build_features(raw_data)

    if feature_data.empty:
        raise LatestPredictionError(
            "Feature engineering returned no usable hourly rows."
        )

    missing_features = set(FEATURE_COLUMNS).difference(
        feature_data.columns
    )

    if missing_features:
        missing = ", ".join(sorted(missing_features))
        raise LatestPredictionError(
            f"Hourly feature data is missing model features: {missing}"
        )

    feature_data["timestamp"] = pd.to_datetime(
        feature_data["timestamp"],
        utc=True,
        errors="coerce",
    )

    feature_data = (
        feature_data.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if feature_data.empty:
        raise LatestPredictionError(
            "No complete hourly feature row is available."
        )

    latest_record = feature_data.iloc[-1]
    market_timestamp = pd.Timestamp(
        latest_record["timestamp"]
    )

    feature_row = feature_data.loc[
        [feature_data.index[-1]],
        FEATURE_COLUMNS,
    ].copy()

    return market_timestamp, feature_row




def serialize_features(
    feature_row: pd.DataFrame,
) -> dict[str, float]:
    """Convert the latest hourly features to JSON-compatible values."""
    return {
        feature_name: float(feature_row.iloc[0][feature_name])
        for feature_name in FEATURE_COLUMNS
    }


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()

if __name__ == "__main__":
    try:
        timestamp, feature_row = build_latest_feature_row()

        print("\nDownload successful")
        print(f"Ticker: {GOLD_TICKER}")
        print(f"Interval: {DOWNLOAD_INTERVAL}")
        print(f"Latest candle: {timestamp}")

        print("\nLatest feature row:")
        print(feature_row)

        print("\nSerialized features:")
        print(serialize_features(feature_row))

    except LatestPredictionError as error:
        print(f"\nLatest prediction error: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {type(error).__name__}: {error}")