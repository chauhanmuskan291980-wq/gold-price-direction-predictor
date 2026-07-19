# `docs/feature_engineering.md`

# Feature Engineering

## Purpose

The feature-engineering stage converts raw OHLCV market data into numerical signals that can be used by the machine-learning model.

Raw prices alone do not clearly describe momentum, volatility, candle strength, or trend direction. Feature engineering creates additional columns that summarize this information.

## Pipeline Position

```text
Raw Gold data
    ↓
Data preprocessing
    ↓
Feature engineering
    ↓
Model-ready dataset
```

## Main File

```text
src/features/build_features.py
```

## File Responsibility

This file is responsible for:

* reading cleaned market data
* creating model features
* creating the prediction target
* removing rows with missing feature values
* returning a model-ready DataFrame

It does not train the model or calculate final evaluation metrics.

## Input Columns

The feature builder expects cleaned columns such as:

```text
timestamp
open
high
low
close
volume
```

Each row represents one hourly Gold candle.

## Created Features

### 1. One-Candle Return

```python
data["return_1"] = data["close"].pct_change()
```

This feature measures the percentage change in closing price compared with the previous candle.

Formula:

```text
return_1 = (current_close / previous_close) - 1
```

Interpretation:

* positive value: price increased
* negative value: price decreased
* value near zero: little movement

Reason for using it:

Short-term price movement may contain momentum information that helps predict the next candle.

---

### 2. Moving-Average Gap

```python
moving_average = data["close"].rolling(window=10).mean()

data["ma_gap"] = (
    data["close"] - moving_average
) / moving_average
```

This feature compares the current closing price with its recent moving average.

Interpretation:

* positive value: price is above the recent average
* negative value: price is below the recent average
* value near zero: price is close to the recent average

Reason for using it:

The moving-average gap helps describe short-term trend strength and possible mean-reversion conditions.

---

### 3. Rolling Volatility

```python
data["volatility_10"] = (
    data["return_1"]
    .rolling(window=10)
    .std()
)
```

This feature calculates the standard deviation of recent returns.

Interpretation:

* larger value: market has been moving more aggressively
* smaller value: market has been relatively stable

Reason for using it:

The same price signal may behave differently during calm and volatile market conditions.

---

### 4. Candle Body Ratio

```python
candle_range = data["high"] - data["low"]

data["candle_body_ratio"] = (
    data["close"] - data["open"]
) / candle_range
```

This feature compares the candle body with the complete high-low range.

Interpretation:

* positive value: bullish candle
* negative value: bearish candle
* large absolute value: strong directional candle
* value near zero: weak or indecisive candle

A safe implementation should avoid division by zero:

```python
candle_range = candle_range.replace(0, pd.NA)
```

Reason for using it:

It captures the direction and strength of the current candle instead of using only its closing price.

---

### 5. Relative Strength Index

The project uses a 14-period RSI.

Conceptually:

```text
RSI = 100 - 100 / (1 + average_gain / average_loss)
```

Interpretation:

* RSI above 70: price may be overbought
* RSI below 30: price may be oversold
* RSI near 50: balanced momentum

Reason for using it:

RSI summarizes recent positive and negative price movement in a single momentum indicator.

## Target Creation

The target represents whether the next candle closes higher than the current candle.

```python
data["target"] = (
    data["close"].shift(-1) > data["close"]
).astype(int)
```

Target meaning:

```text
1 = next candle closes higher
0 = next candle closes lower or equal
```

Example:

| Current close | Next close | Target |
| ------------: | ---------: | -----: |
|          2400 |       2405 |      1 |
|          2400 |       2396 |      0 |
|          2400 |       2400 |      0 |

## Why `shift(-1)` Is Used

```python
data["close"].shift(-1)
```

moves the next row’s closing price into the current row.

This allows the current candle’s features to be connected with the next candle’s direction.

The shifted future value is used only to create the label. It must never be included as an input feature.

## Missing Values

Rolling indicators require historical rows.

For example, a 14-period RSI cannot be calculated for the earliest rows because 14 previous observations are not yet available.

These incomplete rows are removed after all features are created:

```python
data = data.dropna().reset_index(drop=True)
```

Rows should not be dropped separately after every feature because that can make alignment harder to follow.

## Feature List

The final model input contains:

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]
```

## Leakage Prevention

Feature engineering must use only information available at or before the current candle.

Safe operations include:

```text
pct_change()
rolling()
current open
current high
current low
current close
```

Unsafe feature examples include:

```text
next candle close
future return
future high
future low
statistics calculated using test data
```

The next candle is used only to create the target.

## Output

The feature-engineering stage produces a DataFrame similar to:

```text
timestamp
open
high
low
close
volume
return_1
ma_gap
volatility_10
candle_body_ratio
rsi_14
target
```

This output becomes the input for:

```text
src/models/train.py
```

## Limitations

* Technical indicators do not guarantee future price direction.
* Rolling features lose some initial rows.
* Different feature windows may produce different results.
* The features do not include macroeconomic news or order-book information.
* The model uses historical relationships that may change over time.
