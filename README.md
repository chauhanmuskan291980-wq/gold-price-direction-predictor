# Gold Price Direction Predictor

A production-style machine-learning project that predicts whether the next
hourly Gold price candle will move upward or downward.

The project covers the complete machine-learning engineering workflow:

1. Download historical hourly Gold market data.
2. Clean and validate OHLC candles.
3. Engineer predictive technical features.
4. Create the next-hour direction target.
5. Split data chronologically to prevent look-ahead leakage.
6. Train and evaluate a classification model.
7. Backtest a naive trade-every-signal strategy.
8. Report classification metrics, win rate, and cumulative returns.
9. Expose model predictions through FastAPI.
10. Package the service using Docker.

## Prediction Target

The model predicts the direction of the next hourly candle:

- `1`: the next hourly close is higher than the current close.
- `0`: the next hourly close is equal to or lower than the current close.

## Evaluation

The project will report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Test-period win rate
- Cumulative strategy returns
- Buy-and-hold comparison

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Matplotlib
- Pytest
- Docker

## Current Status

Project initialization completed. Data collection is the next development stage.

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv