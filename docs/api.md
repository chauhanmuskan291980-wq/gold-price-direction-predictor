# `docs/api.md`

# FastAPI Prediction Service

## Purpose

The FastAPI service exposes the trained **Gold Price Direction Predictor** through a REST API.

The API loads three trained machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Clients can:

- Compare predictions from all three models using manually supplied feature values.
- Generate predictions for the latest hourly Gold Futures (`GC=F`) candle downloaded from Yahoo Finance.

The API also returns an **ensemble majority-vote prediction**.

---

## Pipeline Position

```text
Client Request
      ↓
FastAPI Endpoint
      ↓
Pydantic Validation
      ↓
Prediction Service
      ↓
Three Saved Models
      ↓
Model Comparison
      ↓
Ensemble Majority Vote
      ↓
JSON Response
```

---

## Main Files

```text
app/
├── main.py
├── schemas.py
├── api/routes.py
└── services/model_service.py
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---------|----------|---------|
| GET | `/` | API information |
| GET | `/health` | Health status |
| GET | `/model/info` | Loaded models and feature information |
| POST | `/predict/compare` | Compare predictions from all three models |
| GET | `/predict/latest` | Download latest Yahoo Finance data and predict |

---

## Prediction Service

The model service is responsible for:

- Loading all trained Joblib models
- Preserving feature order
- Creating model-ready DataFrames
- Running inference
- Returning predictions from all models
- Calculating the ensemble majority vote

Example:

```python
from pathlib import Path
import joblib

MODEL_DIR = Path("artifacts/models")

MODELS = {
    "logistic_regression": joblib.load(
        MODEL_DIR / "logistic_regression.joblib"
    ),
    "random_forest": joblib.load(
        MODEL_DIR / "random_forest.joblib"
    ),
    "gradient_boosting": joblib.load(
        MODEL_DIR / "gradient_boosting.joblib"
    ),
}
```

---

## Feature Columns

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]
```

---

## Prediction Workflow

```text
Receive request
      ↓
Validate input
      ↓
Create DataFrame
      ↓
Run Logistic Regression
      ↓
Run Random Forest
      ↓
Run Gradient Boosting
      ↓
Calculate Ensemble Vote
      ↓
Return JSON
```

---

## Prediction Request Example

```json
{
  "return_1": 0.0015,
  "ma_gap": 0.0032,
  "volatility_10": 0.0048,
  "candle_body_ratio": 0.42,
  "rsi_14": 57.8
}
```

---

## Prediction Response Example

```json
{
  "predictions": {
    "logistic_regression": {
      "predicted_class": 1,
      "direction": "up",
      "probability_up": 0.53
    },
    "random_forest": {
      "predicted_class": 0,
      "direction": "down_or_flat",
      "probability_up": 0.49
    },
    "gradient_boosting": {
      "predicted_class": 1,
      "direction": "up",
      "probability_up": 0.55
    }
  },
  "ensemble_prediction": {
    "predicted_class": 1,
    "direction": "up"
  }
}
```

---

## Latest Prediction Endpoint

```bash
curl http://127.0.0.1:8000/predict/latest
```

Workflow

```text
Yahoo Finance
      ↓
Download latest GC=F candle
      ↓
Feature engineering
      ↓
Three model predictions
      ↓
Ensemble vote
      ↓
JSON response
```

---

## Running the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
http://127.0.0.1:8000/model/info
http://127.0.0.1:8000/predict/latest
```

---

## Test Manual Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict/compare" \
-H "Content-Type: application/json" \
-d '{
  "return_1":0.0015,
  "ma_gap":0.0032,
  "volatility_10":0.0048,
  "candle_body_ratio":0.42,
  "rsi_14":57.8
}'
```

---

## Test Latest Prediction

```bash
curl http://127.0.0.1:8000/predict/latest
```

---

## Error Handling

The API returns descriptive validation errors when:

- Model artifacts are missing
- Invalid request types are supplied
- Required fields are missing
- RSI is outside the allowed range
- Input validation fails

---

## Docker

```bash
docker build -t gold-direction-api .

docker run -d \
--name gold-direction-container \
-p 8000:8000 \
gold-direction-api
```

---

## API Limitations

- Predictions are based on historical market data.
- GC=F is a proxy for spot XAU/USD.
- Yahoo Finance may return delayed or incomplete candles.
- Transaction costs and slippage are not included.
- The ensemble vote is not guaranteed to outperform every individual model.
- This project is for educational purposes and is not financial advice.