# `docs/api.md`

# FastAPI Prediction Service

## Purpose

The FastAPI service exposes the trained Gold-direction model through an HTTP API.

Instead of manually loading Python files and calling the model, another application can send candle information to an endpoint and receive a prediction.

## Pipeline Position

```text
Client request
    ↓
FastAPI endpoint
    ↓
Request validation
    ↓
Prediction service
    ↓
Saved model artifact
    ↓
JSON response
```

## Main Files

```text
app/main.py
app/schemas.py
app/api/routes.py
app/services/prediction_service.py
```

## `app/main.py`

This file creates the FastAPI application.

Typical structure:

```python
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Gold Direction Predictor API",
    version="1.0.0",
)

app.include_router(router)
```

Responsibilities:

* create the FastAPI instance
* set API metadata
* register application routes
* serve as the Uvicorn entry point

Application command:

```bash
uvicorn app.main:app --reload
```

Meaning:

```text
app.main = Python module
app = FastAPI object
--reload = restart server after code changes
```

## `app/schemas.py`

This file defines request and response structures using Pydantic.

A request schema may look like:

```python
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    return_1: float
    ma_gap: float
    volatility_10: float = Field(ge=0)
    candle_body_ratio: float
    rsi_14: float = Field(ge=0, le=100)
```

Responsibilities:

* define required fields
* validate incoming values
* reject invalid request bodies
* document the API automatically

A response schema may look like:

```python
class PredictionResponse(BaseModel):
    prediction: int
    direction: str
    probability_up: float
```

## `app/api/routes.py`

This file defines HTTP endpoints.

### Health Endpoint

```python
@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
```

Purpose:

* confirm the API is running
* support Docker health checks
* support deployment monitoring
* provide a simple test endpoint

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction Endpoint

```python
@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_direction(
    request: PredictionRequest,
) -> PredictionResponse:
    return prediction_service.predict(request)
```

Purpose:

* receive validated feature values
* pass them to the prediction service
* return a structured result

## `app/services/prediction_service.py`

This file contains model-loading and inference logic.

Responsibilities:

* load the saved Joblib artifact
* preserve the expected feature order
* convert request data into a DataFrame
* call the model
* return class and probability

Example model loading:

```python
from pathlib import Path

import joblib

MODEL_PATH = Path("artifacts/model.joblib")

model = joblib.load(MODEL_PATH)
```

Example feature order:

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]
```

Maintaining the same order used during training is essential.

## Prediction Logic

Example input conversion:

```python
features = pd.DataFrame(
    [
        {
            "return_1": request.return_1,
            "ma_gap": request.ma_gap,
            "volatility_10": request.volatility_10,
            "candle_body_ratio": request.candle_body_ratio,
            "rsi_14": request.rsi_14,
        }
    ],
    columns=FEATURE_COLUMNS,
)
```

Prediction:

```python
prediction = int(model.predict(features)[0])
```

Probability:

```python
probability_up = float(
    model.predict_proba(features)[0, 1]
)
```

Direction label:

```python
direction = "up" if prediction == 1 else "down"
```

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

## Prediction Response Example

```json
{
  "prediction": 1,
  "direction": "up",
  "probability_up": 0.5634
}
```

Interpretation:

```text
prediction = 1
predicted direction = upward
estimated probability = 56.34%
```

This is a model estimate, not a guarantee.

## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model first so that the artifact exists.

Start the server:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides Swagger documentation.

## Test the Health Endpoint

Using PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/health" `
  -Method Get
```

Using curl:

```bash
curl http://127.0.0.1:8000/health
```

## Test the Prediction Endpoint

PowerShell example:

```powershell
$body = @{
    return_1 = 0.0015
    ma_gap = 0.0032
    volatility_10 = 0.0048
    candle_body_ratio = 0.42
    rsi_14 = 57.8
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Curl example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "return_1": 0.0015,
    "ma_gap": 0.0032,
    "volatility_10": 0.0048,
    "candle_body_ratio": 0.42,
    "rsi_14": 57.8
  }'
```

## Error Handling

The API should return a clear error when:

* the model artifact is missing
* the model cannot be loaded
* required request fields are absent
* RSI is outside the valid range
* the request contains an invalid data type

Example validation error:

```json
{
  "detail": [
    {
      "loc": ["body", "rsi_14"],
      "msg": "Input should be less than or equal to 100"
    }
  ]
}
```

## Docker Usage

Build the image:

```bash
docker build -t gold-direction-api .
```

Run the container:

```bash
docker run \
  --rm \
  -p 8000:8000 \
  gold-direction-api
```

Then access:

```text
http://localhost:8000/docs
```

## API Limitations

* The endpoint expects already engineered feature values.
* It does not automatically download live Gold candles.
* The prediction is based on a simple historical model.
* The probability is not a calibrated trading guarantee.
* The API does not place trades.
* The service should not be treated as financial advice.
