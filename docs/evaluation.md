# `docs/evaluation_and_backtest.md`

# Evaluation and Backtesting

## Purpose

The evaluation stage measures both classification quality and simplified trading performance.

Classification metrics show how accurately the model predicts candle direction.

Backtesting shows how those predictions would have behaved as trading signals on historical test data.

## Pipeline Position

```text
Trained model
    â†“
Predictions on unseen test data
    â†“
Classification metrics
    â†“
Trading signal simulation
    â†“
Cumulative return curve
```

## Main Files

```text
src/evaluation/metrics.py
src/evaluation/backtest.py
```

## `metrics.py` Responsibility

This file calculates model-quality metrics such as:

* accuracy
* balanced accuracy
* precision
* recall
* F1 score
* ROC-AUC

## `backtest.py` Responsibility

This file:

* converts predictions into trading positions
* applies positions to future candle returns
* calculates win rate
* calculates cumulative strategy return
* compares the strategy with buy and hold
* creates the cumulative-return chart

## Classification Metrics

### Accuracy

Accuracy measures the percentage of correct predictions.

```text
accuracy = correct predictions / total predictions
```

Current project result:

```text
50.15%
```

Interpretation:

The model predicted the correct next-candle direction for approximately 53 out of every 100 test observations.

---

### Balanced Accuracy

Balanced accuracy gives equal importance to both classes.

Current result:

```text
50.20%
```

This is useful when upward and downward candles are not perfectly balanced.

---

### Precision

Precision measures how often an upward prediction was correct.

```text
precision =
correct upward predictions /
all upward predictions
```

Current result:

```text
47.20%
```

A lower precision means some predicted upward candles actually closed lower.

---

### Recall

Recall measures how many actual upward candles were identified.

```text
recall =
correct upward predictions /
all actual upward candles
```

Current result:

```text
50.96%
```

---

### F1 Score

F1 combines precision and recall into one metric.

Current result:

```text
49.00%
```

It is useful when both missed upward candles and false upward predictions matter.

---

### ROC-AUC

ROC-AUC evaluates how well prediction probabilities separate upward and downward candles.

Current result:

```text
51.85%
```

Interpretation:

The model has some directional ranking ability, but the result is not strong enough to claim reliable market prediction.

## Honest Interpretation

The model performs slightly better than random guessing on the selected test period.

However:

* the advantage is small
* the test period is limited
* trading costs can remove small gains
* market behavior changes over time
* results may not generalize to live trading

These metrics should be presented as a baseline, not as evidence of a production trading strategy.

## Backtesting Logic

A simple long/short signal can be created from predictions:

```python
position = prediction.map(
    {
        1: 1,
        0: -1,
    }
)
```

Meaning:

```text
1 prediction â†’ long position
0 prediction â†’ short position
```

The next-candle return is:

```python
future_return = data["close"].pct_change().shift(-1)
```

The strategy return is:

```python
strategy_return = position * future_return
```

This aligns the current prediction with the following candleâ€™s return.

## Important Alignment Rule

The prediction created from the current candle must be applied to the next candle.

Incorrect:

```text
current prediction Ã— current return
```

Correct:

```text
current prediction Ã— next-candle return
```

This prevents look-ahead bias.

## Win Rate

Win rate measures the percentage of strategy observations with positive returns.

```text
win rate =
profitable strategy candles /
total strategy candles
```

Current project result:

```text
49.63%
```

Win rate alone does not prove profitability because winning and losing moves can have different sizes.

## Cumulative Return

Cumulative strategy return can be calculated with:

```python
cumulative_strategy = (
    1 + strategy_return
).cumprod()
```

Total return:

```python
total_return = cumulative_strategy.iloc[-1] - 1
```

Current project strategy return:

```text
-7.74%
```

Current buy-and-hold return:

```text
-6.46%
```

On this test period, the model-based strategy outperformed passive holding.

This does not guarantee that it will outperform on other periods.

## Buy-and-Hold Benchmark

The benchmark represents holding the asset throughout the complete test period.

```python
buy_hold_curve = (
    1 + future_return
).cumprod()
```

Comparing the strategy against buy and hold helps determine whether model signals added value during the selected period.

## Cumulative Return Curve

The chart should include:

```text
Strategy cumulative return
Buy-and-hold cumulative return
```

Recommended output path:

```text
artifacts/cumulative_returns.png
```

Example plotting logic:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(strategy_curve, label="Model Strategy")
plt.plot(buy_hold_curve, label="Buy and Hold")
plt.title("Cumulative Return on Test Data")
plt.xlabel("Test Observation")
plt.ylabel("Growth of 1 Unit")
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.close()
```

## Simplifying Assumptions

The current backtest may not include:

* bid-ask spread
* brokerage fees
* slippage
* execution delays
* futures rollover costs
* position-size limits
* stop-loss or take-profit rules
* market liquidity constraints

Because of these assumptions, actual trading performance would likely be lower than the reported historical result.

## Leakage Prevention

The backtest must:

1. use only test-set predictions
2. maintain timestamp order
3. apply predictions to the next candle
4. avoid using future prices in feature calculations
5. avoid evaluating on the training dataset

## Current Results Summary

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 50.15% |
| Balanced Accuracy | 50.20% |
| Precision         | 47.20% |
| Recall            | 50.96% |
| F1 Score          | 49.00% |
| ROC-AUC           | 51.85% |
| Win Rate          | 49.63% |
| Strategy Return   | -7.74% |
| Buy and Hold      | -6.46% |

## Conclusion

The model showed a small predictive advantage and positive historical strategy return during the selected test period.

The results are encouraging for a baseline engineering assignment, but they are not strong enough to justify live financial use without larger datasets, walk-forward testing, cost modelling, and further validation.

