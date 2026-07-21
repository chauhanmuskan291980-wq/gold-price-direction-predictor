# Data Pipeline

## Purpose

The data pipeline downloads hourly Gold market data, validates it, standardizes the schema, and saves it for preprocessing and feature engineering.

The project initially experimented with approximately **3 months of data**. The final pipeline was expanded to approximately **6 months of hourly history** so that the three machine-learning models could be compared using a larger and more representative dataset.

---

## Data Source

The project downloads data from **Yahoo Finance** using the `yfinance` Python package.

```text
Symbol: GC=F
Interval: 1h
Initial period: approximately 3 months
Final period: approximately 6 months
```

`GC=F` represents Gold Futures. It is used as a liquid and freely available proxy for the Gold market, but it is not identical to broker-specific spot `XAU/USD`.

---

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
    ↓
data/processed/gold_processed.csv
    ↓
Feature engineering
    ↓
Model-ready dataset
```

---

## File: `src/data/download.py`

### Responsibility

This module is responsible for:

- Defining the download configuration
- Requesting hourly Gold Futures data
- Checking that Yahoo Finance returned data
- Normalizing returned column names
- Validating required OHLC fields
- Saving the raw dataset

It does not create features, train models, or make predictions.

---

## Download Configuration

A typical configuration is:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadConfig:
    symbol: str = "GC=F"
    period: str = "6mo"
    interval: str = "1h"
    output_path: Path = Path("data/raw/gold_hourly.csv")
```

Depending on the exact implementation and Yahoo Finance limitations, the six-month history may also be collected using a start and end date or another supported period value.

### Configuration Fields

| Field | Description |
|---|---|
| `symbol` | Yahoo Finance ticker; `GC=F` represents Gold Futures |
| `period` | Historical window requested from Yahoo Finance |
| `interval` | Candle duration; `1h` means one-hour candles |
| `output_path` | Path where the raw CSV is saved |

The dataclass is frozen to prevent accidental configuration changes after creation.

---

## Download Function

```python
def download_gold_data(config: DownloadConfig) -> pd.DataFrame:
    ...
```

The downloader calls Yahoo Finance through:

```python
yf.download(
    tickers=config.symbol,
    period=config.period,
    interval=config.interval,
    progress=False,
)
```

A date-based version may use:

```python
yf.download(
    tickers=config.symbol,
    start=start_date,
    end=end_date,
    interval=config.interval,
    progress=False,
)
```

The exact argument style should match the current implementation.

---

## Expected Raw Data

Yahoo Finance commonly returns:

- Open
- High
- Low
- Close
- Adjusted Close
- Volume
- Datetime index

After normalization, the project uses a consistent lowercase schema:

```text
timestamp
open
high
low
close
volume
```

Each row represents one hourly market candle.

---

## Empty-Data Handling

If Yahoo Finance returns no rows, the pipeline raises a descriptive error instead of continuing with an empty DataFrame.

Example:

```python
if data.empty:
    raise ValueError(
        "Yahoo Finance returned no hourly Gold data for the requested period."
    )
```

This prevents later preprocessing and training steps from failing with unclear errors.

---

## Column Normalization

Yahoo Finance may return:

- Capitalized column names
- MultiIndex columns
- An adjusted-close field
- A datetime index rather than a timestamp column

The pipeline normalizes this structure before saving.

Example logic:

```python
data = data.reset_index()
data.columns = [
    "_".join(str(part) for part in column if part)
    if isinstance(column, tuple)
    else str(column)
    for column in data.columns
]

data.columns = [
    column.strip().lower().replace(" ", "_")
    for column in data.columns
]
```

The exact normalization code may vary, but the final schema should remain predictable.

---

## Data Validation

Before the raw dataset is saved, the pipeline checks:

1. The DataFrame is not empty.
2. Required columns exist.
3. Timestamps can be parsed.
4. OHLC values are numeric.
5. Invalid or missing rows are handled.
6. Duplicate timestamps are removed or rejected.
7. Rows are sorted chronologically.
8. High-low relationships are valid.

Required columns:

```python
REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
}
```

Volume may be retained when available, but it is not required by the current five-feature model.

---

## OHLC Relationship Validation

For every valid candle:

```text
high >= open
high >= close
high >= low

low <= open
low <= close
low <= high
```

Rows that violate these relationships should be removed or should cause a validation error, depending on the implementation policy.

---

## Preprocessing

The preprocessing stage:

1. Loads the raw CSV.
2. Parses timestamps.
3. Converts OHLC columns to numeric values.
4. Removes duplicates.
5. Removes invalid rows.
6. Sorts rows chronologically.
7. Creates the next-hour target.
8. Saves the processed dataset.

Example command:

```bash
python -m src.data.preprocess
```

Output:

```text
data/processed/gold_processed.csv
```

---

## Target Creation

The target represents whether the next hourly close is higher than the current close.

```python
data["next_close"] = data["close"].shift(-1)
data["target"] = (data["next_close"] > data["close"]).astype(int)
```

| Value | Meaning |
|---:|---|
| `1` | Next close is higher |
| `0` | Next close is lower or equal |

The last row does not have a known future close and must be removed before training.

```python
data = data.dropna(subset=["next_close"])
```

The `next_close` column is then removed from model inputs to prevent leakage.

---

## Feature-Engineering Output

The processed data is passed to `src/features/build_features.py`.

The final five features are:

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]
```

These features are calculated only from current and historical observations.

Rolling indicators create missing values at the start of the dataset. Those rows are removed after all features are calculated.

A recent final run produced approximately:

```text
Input rows: 3346
Rows after feature engineering: 3327
Training rows: 2661
Testing rows: 666
```

Exact row counts can change as Yahoo Finance data is updated.

---

## How the Data Supports Three Models

The same feature dataset is used to train:

- Logistic Regression
- Random Forest
- Gradient Boosting

All models receive the same:

- Five input columns
- Target labels
- Chronological training set
- Chronological test set

This ensures that differences in evaluation results come from the algorithms rather than from different datasets.

---

## Data-Leakage Prevention

The pipeline prevents leakage by following these rules:

- Features use only current and past candles.
- Future close is used only to create the target.
- The future-price column is removed before training.
- Data is sorted chronologically.
- Training data occurs before testing data.
- Scaling is fitted only on training data.
- The final test period is not shuffled.

---

## Running the Data Pipeline

### Download data

```bash
python -m src.data.download
```

### Preprocess data

```bash
python -m src.data.preprocess
```

### Build features

```bash
python -m src.features.build_features
```

Expected files:

```text
data/
├── raw/
│   └── gold_hourly.csv
└── processed/
    ├── gold_processed.csv
    └── gold_features.csv
```

The exact processed filename should match the current codebase.

---

## Latest-Prediction Data Flow

The `GET /predict/latest` endpoint reuses the data and feature logic for inference.

```text
GET /predict/latest
    ↓
Download recent GC=F hourly candles
    ↓
Validate and normalize data
    ↓
Calculate five features
    ↓
Select latest complete feature row
    ↓
Run three model predictions
    ↓
Return individual results and ensemble vote
```

The latest candle should be treated carefully because Yahoo Finance may return an incomplete in-progress candle.

---

## Limitations

- `GC=F` represents Gold Futures rather than exact spot XAU/USD.
- Yahoo Finance may provide delayed or missing candles.
- The most recent candle may still be incomplete.
- Volume can be zero or unavailable for some periods.
- Free historical intraday data may have provider-imposed limits.
- The dataset does not include spread, commissions, slippage, or execution delay.
- Six months of hourly data is still small for financial machine learning.
- Historical market behaviour may not continue in the future.

---

## Summary

The data pipeline provides a reproducible path from Yahoo Finance to a model-ready chronological dataset:

```text
GC=F hourly data
    ↓
Validation
    ↓
Schema normalization
    ↓
Preprocessing
    ↓
Next-hour target
    ↓
Five engineered features
    ↓
Chronological training and testing
    ↓
Three-model comparison
```