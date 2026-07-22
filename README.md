# Gold Price Direction Predictor

A production-style machine learning project that predicts whether the next hourly Gold Futures candle (`GC=F`) is likely to move **up** or **down/flat**.

The project includes an end-to-end workflow for data collection, validation, feature engineering, chronological model training, model comparison, trading evaluation, FastAPI inference, automated testing, Docker, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)

---

## Project Objective

The objective is to classify the direction of the next hourly candle:

- `1` — the next close is higher
- `0` — the next close is lower or equal

The project does **not** predict the exact future gold price. It predicts direction using technical features calculated only from information available at or before the current timestamp.

---

## System Design

```mermaid
flowchart TD
    A[Yahoo Finance GC=F H1 Data] --> B[Data Validation]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Chronological Train/Test Split]

    E --> F[Logistic Regression]
    E --> G[Random Forest]
    E --> H[Gradient Boosting]

    F --> I[Model Evaluation]
    G --> I
    H --> I

    I --> J[Model Comparison]
    J --> K[Trading Evaluation]
    J --> L[Saved Model Artifacts]

    L --> M[FastAPI Service]
    M --> N[Individual Predictions]
    N --> O[Ensemble Majority Vote]
    O --> P[JSON API Response]
```

### Prediction Flow

```text
Hourly market data
        ↓
Feature generation
        ↓
Logistic Regression
Random Forest
Gradient Boosting
        ↓
Majority-vote ensemble
        ↓
FastAPI response
```

---

## Why These Features Were Chosen

The model uses five lightweight and interpretable technical features.

| Feature | Purpose |
|---|---|
| `return_1` | Captures immediate short-term momentum |
| `ma_gap` | Measures the distance between price and its moving average |
| `volatility_10` | Represents recent market uncertainty |
| `candle_body_ratio` | Measures directional candle strength |
| `rsi_14` | Captures relative momentum and overbought/oversold conditions |

These features were selected because they represent complementary market behaviour:

- **Momentum** — `return_1`, `rsi_14`
- **Trend** — `ma_gap`
- **Risk and uncertainty** — `volatility_10`
- **Price action** — `candle_body_ratio`

All features are computed from current and historical candles only. Future values are never included in the model inputs.

---

## Why Three Algorithms Were Used

Three classifiers were trained using the same data, features, and chronological split.

| Model | Reason for Selection |
|---|---|
| Logistic Regression | Provides a simple, interpretable baseline |
| Random Forest | Captures non-linear relationships and feature interactions |
| Gradient Boosting | Learns sequential corrections and can model complex decision boundaries |

Using multiple models makes it possible to compare different modelling assumptions instead of relying on a single algorithm.

The API also returns an **ensemble prediction** based on majority voting across the three classifiers.

---

## Preventing Look-Ahead Leakage

Financial time-series models must preserve time order.

This project prevents leakage by using:

- Chronological train/test splitting
- Time-ordered validation
- Features based only on current and previous candles
- Removal of future-price columns before training
- A target derived from the next close only
- No random shuffling of financial observations

The training period always occurs before the final test period.

---

## Model Results

The three models were evaluated on the same unseen chronological test period.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **51.50%** | **51.56%** | **48.22%** | 52.41% | 50.23% | **52.47%** |
| Random Forest | 50.30% | 50.45% | 47.13% | 52.73% | 49.77% | 52.42% |
| Gradient Boosting | 50.30% | 51.11% | 47.58% | **63.34%** | **54.34%** | 51.55% |

### Interpretation

- Logistic Regression achieved the strongest overall accuracy and ROC-AUC.
- Gradient Boosting produced the highest recall and F1 score.
- Random Forest performed similarly but did not outperform the other models.
- Overall performance remained only slightly above random guessing, which reflects the difficulty of short-term financial-market prediction.

The project also evaluates:

- Win rate
- Strategy return
- Buy-and-hold return
- Cumulative return curve
- Trade-level predictions
 

## Cumulative Return Curve

The generated cumulative-return chart compares:

* Model-based trading strategy
* Buy-and-hold benchmark

![Cumulative Return Curve](artifacts/cumulative_returns.png)

## Repository Structure

```text
gold-price-direction-predictor/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── model_service.py
│   │   └── latest_prediction_service.py
│   ├── main.py
│   └── schemas.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── tests/
├── docs/
├── artifacts/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/chauhanmuskan291980-wq/gold-price-direction-predictor.git
cd gold-price-direction-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project pipeline

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
python -m src.models.validate
python -m src.models.evaluate
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic API information |
| `GET` | `/health` | API and model health status |
| `GET` | `/model/info` | Loaded model metadata |
| `POST` | `/predict/compare` | Predictions from all models and the ensemble |
| `GET` | `/predict/latest` | Prediction using the latest available Gold Futures data |

### Example Prediction Request

```http
POST /predict/compare
Content-Type: application/json
```

```json
{
  "return_1": 0.0012,
  "ma_gap": -0.0021,
  "volatility_10": 0.0045,
  "candle_body_ratio": 0.62,
  "rsi_14": 54.3
}
```

### Example Response

```json
{
  "predictions": {
    "logistic_regression": {
      "predicted_class": 1,
      "direction": "up",
      "confidence": 0.53
    },
    "random_forest": {
      "predicted_class": 0,
      "direction": "down_or_flat",
      "confidence": 0.51
    },
    "gradient_boosting": {
      "predicted_class": 1,
      "direction": "up",
      "confidence": 0.55
    }
  },
  "ensemble_prediction": {
    "predicted_class": 1,
    "direction": "up",
    "votes_up": 2,
    "votes_down_or_flat": 1
  }
}
```

Exact probabilities depend on the input features and trained model artifacts.

---

## Run with Docker

### Build locally

```bash
docker build -t gold-direction-api:latest .
```

### Run the container

```bash
docker run -d \
  --name gold-direction-api \
  -p 8000:8000 \
  gold-direction-api:latest
```

### Pull the published image

Docker Hub:

```bash
docker pull muskanchauhan2890/gold-direction-api:latest
```

GitHub Container Registry:

```bash
docker pull ghcr.io/chauhanmuskan291980-wq/gold-price-direction-predictor:latest
```

---

## Testing and Code Quality

Run tests:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

Run Mypy:

```bash
mypy app src
```

Run coverage:

```bash
pytest --cov=app --cov=src --cov-report=term-missing
```

The test suite covers API routes, validation, data processing, feature engineering, model training, prediction, artifact loading, and time-series validation.

---

## CI/CD

GitHub Actions automatically runs on pushes and pull requests to `main`.

The workflow installs dependencies, runs Ruff, Mypy, and Pytest, builds the Docker image, starts the container, verifies the health endpoint, and publishes container images after successful checks.

---

## Design Decisions

- Used hourly Gold Futures data because the assignment required H1 candles.
- Used chronological splitting because random splitting is unsafe for financial time series.
- Selected simple Scikit-learn models for reproducibility and interpretability.
- Used the same feature set and test period for all models to ensure a fair comparison.
- Loaded trained models once during FastAPI startup for efficient inference.
- Used a router-based API structure to keep application setup separate from endpoint logic.
- Reported both classification metrics and trading-oriented evaluation.
- Kept model limitations visible instead of overstating predictive performance.

---

## Limitations

- Financial markets are noisy and difficult to predict.
- The dataset covers a limited historical period.
- Only five technical features are used.
- Transaction costs, spread, slippage, and execution delays are excluded.
- No macroeconomic, sentiment, or news features are included.
- Yahoo Finance data may differ from institutional XAU/USD feeds.
- Market behaviour can change over time.
- The model does not guarantee profitable live trading.

---

## Future Improvements

- Use a longer historical dataset
- Add walk-forward validation
- Include transaction costs and slippage
- Add feature-importance analysis
- Calibrate model probabilities
- Add scheduled retraining and model monitoring

---

## Technical Documentation

- [Architecture](docs/architecture.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Feature Engineering](docs/feature_engineering.md)
- [Model Training](docs/model_training.md)
- [Evaluation and Backtest](docs/evaluation_and_backtest.md)
- [FastAPI Service](docs/api.md)

---

## Release

[Gold Price Direction Predictor v1.0.0](https://github.com/chauhanmuskan291980-wq/gold-price-direction-predictor/releases/tag/v1.0.0)

---

## Disclaimer

This project is intended for educational and technical demonstration purposes only. It is not financial advice and should not be used as the sole basis for trading or investment decisions.

---

## Author

**Muskan Chauhan**  
Backend Developer and Applied AI/ML Engineer

- [GitHub](https://github.com/chauhanmuskan291980-wq)
- [LinkedIn](https://www.linkedin.com/in/muskan-chauhan-0783b4325/)
