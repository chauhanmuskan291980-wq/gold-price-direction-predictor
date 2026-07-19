# Data Pipeline

## Purpose

The data pipeline downloads hourly Gold market data, validates it, standardizes the schema, and saves it for preprocessing and feature engineering.

## Data Source

The project downloads data from Yahoo Finance using the `yfinance` Python package.

The selected symbol is:

```text
GC=F
```

This symbol represents Gold futures. It is used as a liquid proxy for the Gold market, but it is not identical to broker-specific spot XAU/USD data.

## Pipeline Flow

```text
Yahoo Finance
    ↓
download_gold_data()
    ↓
Raw pandas DataFrame
    ↓
Column normalization
    ↓
OHLC validation
    ↓
data/raw/gold_hourly.csv
    ↓
Preprocessing
```

## File: `src/data/download.py`

### Responsibility

This file is responsible for:

* defining the download configuration
* requesting hourly Gold data
* validating that data was returned
* preparing the raw dataset for saving

It does not create features, train the model, or make predictions.

## `DownloadConfig`

The `DownloadConfig` dataclass stores the download settings.

```python
@dataclass(frozen=True)
class DownloadConfig:
    symbol: str = "GC=F"
    period: str = "60d"
    interval: str = "1h"
    output_path: Path = Path("data/raw/gold_hourly.csv")
```

### Fields

* `symbol`: Yahoo Finance ticker used to identify Gold futures.
* `period`: Amount of recent history requested.
* `interval`: Candle duration. `1h` means one-hour candles.
* `output_path`: Location where the raw CSV file is stored.

The dataclass is frozen so that configuration values cannot be accidentally modified after creation.

## `download_gold_data`

```python
def download_gold_data(config: DownloadConfig) -> pd.DataFrame:
```

This function receives a `DownloadConfig` object and returns a pandas DataFrame.

It calls:

```python
yf.download(...)
```

with the configured symbol, period, and interval.

The returned dataset contains:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume
* Datetime index

If Yahoo Finance returns no rows, the function raises a `ValueError` instead of allowing the remaining pipeline to continue with an empty dataset.

## Raw Data Shape

A typical downloaded dataset contains columns similar to:

```text
Adj Close
Close
High
Low
Open
Volume
```

Each row represents one hourly market candle.

## Limitations

* `GC=F` represents Gold futures rather than exact spot XAU/USD.
* Yahoo Finance is a free source and may contain missing or delayed candles.
* Volume values may be zero during some periods.
* Recent candles may still be incomplete.
* The data does not include bid-ask spread, commissions, or slippage.
