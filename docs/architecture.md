# Gold Price Direction Predictor — System Architecture

## 1. Purpose

This document explains the architecture and end-to-end execution flow of the **Gold Price Direction Predictor**.

The system predicts whether the next hourly Gold candle will move:

- `UP`
- `DOWN_OR_FLAT`

The project performs binary classification. It does not predict the exact future Gold price.

The implementation contains two main subsystems:

1. A machine-learning experimentation and evaluation pipeline
2. A FastAPI inference service

The project started with approximately **3 months of hourly data and one Logistic Regression model**. It was later expanded to approximately **6 months of hourly data**, and three models were trained under the same chronological evaluation setup:

- Logistic Regression
- Random Forest
- Gradient Boosting

---

## 2. Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data source | Yahoo Finance |
| Data package | `yfinance` |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| API | FastAPI |
| Server | Uvicorn |
| Model persistence | Joblib |
| Testing | Pytest |
| Static analysis | Ruff and Mypy |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## 3. Data Source

The pipeline downloads hourly Gold Futures data from Yahoo Finance.

```text
Symbol: GC=F
Interval: 1h
Final history window: approximately 6 months
```

`GC=F` represents Gold Futures. It is used as a freely available and liquid proxy for the Gold market, but it is not identical to broker-specific spot `XAU/USD`.

---

## 4. High-Level Architecture

```mermaid
flowchart LR
    A[Yahoo Finance GC=F] --> B[Hourly Data Download]
    B --> C[Validation and Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Chronological Train-Test Split]

    E --> F1[Logistic Regression]
    E --> F2[Random Forest]
    E --> F3[Gradient Boosting]

    F1 --> G[Model Comparison]
    F2 --> G
    F3 --> G

    G --> H[Classification Evaluation]
    G --> I[Trading Backtest]
    H --> J[Saved Metrics and Artifacts]
    I --> J

    J --> K[FastAPI Service]
    K --> L1[POST /predict/compare]
    K --> L2[GET /predict/latest]
```

---

## 5. Main Project Components

```text
src/
├── data/
│   ├── download.py
│   └── preprocess.py
├── features/
│   └── build_features.py
├── models/
│   ├── train.py
│   ├── validate.py
│   ├── evaluate.py
│   └── predict.py
└── evaluation/

app/
├── main.py
├── schemas.py
└── services/

data/
├── raw/
└── processed/

artifacts/
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
├── model_comparison.csv
├── evaluation_metrics.json
├── cumulative_returns.png
└── test_period_trades.csv

tests/
docs/
Dockerfile
requirements.txt
pyproject.toml
README.md
```

---

## 6. End-to-End Application Flow

```mermaid
flowchart TD
    A[Start] --> B[Download Hourly GC=F Data]
    B --> C{Valid Dataset?}

    C -- No --> D[Raise Validation Error]
    C -- Yes --> E[Normalize and Clean Data]

    E --> F[Create Next-Hour Target]
    F --> G[Build Five Technical Features]
    G --> H[Remove Unusable Rolling Rows]

    H --> I[Chronological 80/20 Split]
    I --> J1[Train Logistic Regression]
    I --> J2[Train Random Forest]
    I --> J3[Train Gradient Boosting]

    J1 --> K[Generate Test Predictions]
    J2 --> K
    J3 --> K

    K --> L[Compare Classification Metrics]
    L --> M[Run Trading Evaluation]
    M --> N[Save Models, Metrics and Charts]

    N --> O[Start FastAPI]
    O --> P[Load Three Model Artifacts]
    P --> Q1[Manual Feature Comparison]
    P --> Q2[Latest Yahoo Finance Prediction]
    Q1 --> R[Structured JSON Response]
    Q2 --> R
```

---

# Part 1: Data Pipeline

## 7. Data Download Sequence

```mermaid
sequenceDiagram
    participant User
    participant Download as download.py
    participant Yahoo as Yahoo Finance
    participant Validator
    participant Storage as Raw CSV

    User->>Download: Run download command
    Download->>Yahoo: Request GC=F hourly data
    Yahoo-->>Download: Return OHLCV candles
    Download->>Validator: Validate returned data

    alt Valid data
        Validator-->>Download: Validation passed
        Download->>Storage: Save gold_hourly.csv
        Storage-->>User: Raw data saved
    else Invalid or empty data
        Validator-->>Download: Validation failed
        Download-->>User: Raise descriptive error
    end
```

The raw dataset contains:

```text
timestamp
open
high
low
close
volume
```

Depending on Yahoo Finance output, adjusted-close columns may also be returned and normalized during preprocessing.

---

## 8. Dataset Evolution

The project was developed in multiple stages.

| Stage | Dataset | Models | Purpose |
|---|---|---|---|
| Initial baseline | Approximately 3 months | Logistic Regression | Establish a simple, interpretable baseline |
| Expanded experiment | Approximately 6 months | Logistic Regression | Test whether additional history improves stability |
| Final comparison | Approximately 6 months | Logistic Regression, Random Forest, Gradient Boosting | Compare linear and nonlinear classifiers fairly |

All final models use the same:

- Hourly dataset
- Five engineered features
- Chronological train/test split
- Target definition
- Test period

This makes the model comparison consistent and reproducible.

---

## 9. Data Validation

Before saving or processing the data, the application checks that:

- Required columns exist
- The dataset is not empty
- Timestamps are valid
- Timestamps are unique or safely deduplicated
- Records are chronologically ordered
- OHLC values are numeric
- `high` is not lower than `open`, `close`, or `low`
- `low` is not higher than `open`, `close`, or `high`
- Missing or invalid rows are handled

```mermaid
flowchart TD
    A[Raw Yahoo Finance Data] --> B{Required Columns Present?}
    B -- No --> X[Reject Dataset]
    B -- Yes --> C{Valid Timestamps?}
    C -- No --> X
    C -- Yes --> D[Remove Duplicate Timestamps]
    D --> E{Valid OHLC Relationships?}
    E -- No --> X
    E -- Yes --> F[Sort Chronologically]
    F --> G[Validated Dataset]
```

---

## 10. Target Creation

The target represents the direction of the next hourly closing price.

```python
target = (next_close > current_close).astype(int)
```

| Target | Meaning |
|---:|---|
| `1` | The next hourly close is higher |
| `0` | The next hourly close is lower or equal |

The future close is used only to create the historical label. It is never included in the model input features.

---

# Part 2: Feature Engineering

## 11. Feature Engineering Flow

```mermaid
flowchart TD
    A[Processed OHLC Data] --> B[return_1]
    A --> C[ma_gap]
    A --> D[volatility_10]
    A --> E[candle_body_ratio]
    A --> F[rsi_14]

    B --> G[Combine Features]
    C --> G
    D --> G
    E --> G
    F --> G

    G --> H[Remove Missing Rolling Rows]
    H --> I[Final Feature Dataset]
```

---

## 12. Model Features

The final models use five engineered features.

| Feature | Meaning | Reason for inclusion |
|---|---|---|
| `return_1` | Previous one-hour percentage return | Captures immediate momentum |
| `ma_gap` | Distance between current close and moving average | Represents trend position |
| `volatility_10` | Rolling standard deviation of recent returns | Captures changing market uncertainty |
| `candle_body_ratio` | Candle body relative to high-low range | Measures intrabar directional strength |
| `rsi_14` | 14-period Relative Strength Index | Represents recent momentum balance |

All features use only current and historical information.

---

# Part 3: Model Training

## 13. Training Architecture

```mermaid
flowchart LR
    A[Feature Dataset] --> B[Chronological Split]
    B --> C[Training Dataset]
    B --> D[Unseen Test Dataset]

    C --> E1[StandardScaler + Logistic Regression]
    C --> E2[Random Forest]
    C --> E3[Gradient Boosting]

    E1 --> F1[logistic_regression.joblib]
    E2 --> F2[random_forest.joblib]
    E3 --> F3[gradient_boosting.joblib]

    F1 --> G[Model Comparison]
    F2 --> G
    F3 --> G
    D --> G
```

---

## 14. Why Chronological Splitting Is Required

Random splitting is inappropriate for financial time-series data because it can place future observations in the training set.

The project follows:

```text
Older observations → Training set
Newer observations → Testing set
```

The split is approximately:

```text
80% training
20% testing
```

A boundary row may be purged to reduce leakage between adjacent training and testing periods.

---

## 15. Models

### 15.1 Logistic Regression

Logistic Regression is used as the interpretable baseline.

The pipeline applies feature scaling before classification:

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)
```

### 15.2 Random Forest

Random Forest is included to test whether an ensemble of decision trees can learn nonlinear relationships between the five features and the next-candle direction.

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
```

Tree-based models do not require feature scaling.

### 15.3 Gradient Boosting

Gradient Boosting builds decision trees sequentially, with each new estimator attempting to correct errors made by previous estimators.

```python
GradientBoostingClassifier(
    random_state=42,
)
```

It is included to test a second nonlinear learning approach.

---

## 16. Training Sequence

```mermaid
sequenceDiagram
    participant User
    participant Train as train.py
    participant Dataset as Feature Dataset
    participant LR as Logistic Regression
    participant RF as Random Forest
    participant GB as Gradient Boosting
    participant Artifacts

    User->>Train: Run python -m src.models.train
    Train->>Dataset: Load time-ordered feature data
    Dataset-->>Train: Return features and target
    Train->>Train: Perform chronological split

    Train->>LR: Fit baseline pipeline
    Train->>RF: Fit random forest
    Train->>GB: Fit gradient boosting

    LR-->>Artifacts: Save logistic_regression.joblib
    RF-->>Artifacts: Save random_forest.joblib
    GB-->>Artifacts: Save gradient_boosting.joblib

    Artifacts-->>User: Training completed
```

---

# Part 4: Validation and Evaluation

## 17. Time-Series Validation

Validation folds preserve chronological order.

```text
Fold 1: Train on early history → validate on later history
Fold 2: Train on larger history → validate on a later window
Fold 3: Train on larger history → validate on the final validation window
```

At no point does a model learn from observations that occur after its validation period.

---

## 18. Model Performance Comparison

The three models were evaluated on the same unseen chronological test period.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **50.45%** | 50.50% | 47.49% | 51.27% | 49.31% | 51.80% |
| Random Forest | **51.50%** | **51.63%** | **48.56%** | 53.82% | 51.06% | **52.06%** |
| Gradient Boosting | 49.55% | 50.64% | 47.47% | **68.79%** | **56.18%** | 49.50% |

### Interpretation

- Logistic Regression achieved the highest accuracy, balanced accuracy, and ROC-AUC.
- Gradient Boosting achieved the highest recall and F1 score.
- Random Forest produced a similar ROC-AUC to Logistic Regression but lower overall accuracy.
- All results remain close to 50%, which reflects the difficulty of predicting short-term financial-market direction.

These results must not be interpreted as guaranteed profitability.

---

## 19. Evaluation Sequence

```mermaid
sequenceDiagram
    participant Evaluate as evaluate.py
    participant Models as Three Saved Models
    participant Test as Unseen Test Data
    participant Metrics
    participant Artifacts

    Evaluate->>Models: Load all model artifacts
    Evaluate->>Test: Load chronological test period
    Evaluate->>Models: Generate classes and probabilities
    Models-->>Evaluate: Return predictions

    Evaluate->>Metrics: Calculate metrics for each model
    Evaluate->>Artifacts: Save comparison CSV or JSON
    Evaluate->>Artifacts: Save trade-level results
    Evaluate->>Artifacts: Save cumulative-return chart
```

---

## 20. Trading Evaluation

The evaluation module may convert model predictions into a simplified directional strategy.

```mermaid
flowchart TD
    A[Model Prediction] --> B{Predicted Class}
    B -- 1 --> C[Long Direction]
    B -- 0 --> D[Down or Defensive Direction]
    C --> E[Calculate Strategy Return]
    D --> E
    E --> F[Win Rate]
    E --> G[Cumulative Return]
    E --> H[Buy-and-Hold Comparison]
```

The backtest does not include:

- Bid-ask spread
- Brokerage or commissions
- Slippage
- Execution delay
- Market-impact costs

---

## 21. Generated Artifacts

```text
artifacts/
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
├── model_comparison.csv
├── evaluation_metrics.json
├── cumulative_returns.png
└── test_period_trades.csv
```

| Artifact | Purpose |
|---|---|
| `logistic_regression.joblib` | Stores the scaled Logistic Regression pipeline |
| `random_forest.joblib` | Stores the trained Random Forest |
| `gradient_boosting.joblib` | Stores the trained Gradient Boosting model |
| `model_comparison.csv` | Stores metrics for all three models |
| `evaluation_metrics.json` | Stores detailed evaluation information |
| `cumulative_returns.png` | Compares strategy and benchmark returns |
| `test_period_trades.csv` | Stores test-period predictions and returns |

---

# Part 5: FastAPI Prediction Service

## 22. API Architecture

```mermaid
flowchart LR
    A[Client or Swagger] --> B[FastAPI Endpoint]
    B --> C[Pydantic Validation]
    C --> D[Model Service]

    D --> E1[Logistic Regression]
    D --> E2[Random Forest]
    D --> E3[Gradient Boosting]

    E1 --> F[Prediction Comparison]
    E2 --> F
    E3 --> F

    F --> G[Ensemble Majority Vote]
    G --> H[Structured JSON Response]
    H --> A
```

---

## 23. API Startup Sequence

```mermaid
sequenceDiagram
    participant Uvicorn
    participant API as app/main.py
    participant Service as Model Service
    participant Artifacts as Model Artifacts

    Uvicorn->>API: Start application
    API->>Service: Initialize service
    Service->>Artifacts: Load three saved models

    alt All required models are available
        Artifacts-->>Service: Return trained models
        Service-->>API: Models loaded
        API-->>Uvicorn: Application ready
    else One or more artifacts are missing
        Artifacts-->>Service: Loading error
        Service-->>API: Unhealthy model status
    end
```

---

## 24. API Endpoints

| Method | Endpoint | Responsibility |
|---|---|---|
| `GET` | `/` | Returns API information |
| `GET` | `/health` | Returns service and model-loading status |
| `GET` | `/model/info` | Returns model names, metadata, and feature names |
| `POST` | `/predict/compare` | Compares predictions from all three models using supplied features |
| `GET` | `/predict/latest` | Downloads recent Yahoo Finance data, builds features, and predicts the latest candle |

---

## 25. Manual Model Comparison Sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as POST /predict/compare
    participant Schema
    participant Service
    participant Models as Three Models

    Client->>API: Send five feature values
    API->>Schema: Validate request

    alt Invalid input
        Schema-->>Client: HTTP 422
    else Valid input
        Schema-->>API: Validated feature vector
        API->>Service: Compare models
        Service->>Models: Run predict and predict_proba
        Models-->>Service: Return three predictions
        Service->>Service: Calculate majority vote
        Service-->>API: Comparison result
        API-->>Client: JSON response
    end
```

Example request:

```json
{
  "return_1": 0.0012,
  "ma_gap": -0.0021,
  "volatility_10": 0.0045,
  "candle_body_ratio": 0.62,
  "rsi_14": 54.3
}
```

Example response structure:

```json
{
  "predictions": {
    "logistic_regression": {
      "predicted_class": 1,
      "direction": "up",
      "probability_up": 0.53,
      "probability_down": 0.47
    },
    "random_forest": {
      "predicted_class": 0,
      "direction": "down_or_flat",
      "probability_up": 0.49,
      "probability_down": 0.51
    },
    "gradient_boosting": {
      "predicted_class": 1,
      "direction": "up",
      "probability_up": 0.55,
      "probability_down": 0.45
    }
  },
  "ensemble_prediction": {
    "predicted_class": 1,
    "direction": "up"
  }
}
```

Exact response keys and probability values depend on the current application schema and trained artifacts.

---

## 26. Latest Prediction Sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as GET /predict/latest
    participant Yahoo as Yahoo Finance
    participant Features
    participant Models as Three Models

    Client->>API: Request latest prediction
    API->>Yahoo: Download recent GC=F hourly candles
    Yahoo-->>API: Return market data
    API->>Features: Build latest five features
    Features-->>API: Return model-ready feature row
    API->>Models: Run three predictions
    Models-->>API: Return classes and probabilities
    API->>API: Calculate ensemble vote
    API-->>Client: Latest prediction response
```

This endpoint demonstrates the complete inference workflow:

```text
Yahoo Finance
    ↓
Latest Hourly Candles
    ↓
Validation and Feature Engineering
    ↓
Three Model Predictions
    ↓
Ensemble Majority Vote
    ↓
JSON Response
```

---

# Part 6: Docker and Testing

## 27. Docker Architecture

```mermaid
flowchart LR
    A[Host Computer] -->|Port 8000| B[Docker Container]

    subgraph B[gold-direction-api]
        C[Python]
        D[FastAPI]
        E[Uvicorn]
        F[Application Source]
        G[Three Saved Models]
    end

    E --> D
    D --> F
    F --> G
```

Build and run:

```bash
docker build -t gold-direction-api .
docker run -d --name gold-direction-container -p 8000:8000 gold-direction-api
```

---

## 28. Testing Strategy

The test suite should verify:

- Data download validation
- Data preprocessing
- Feature generation
- Chronological splitting
- Training all three models
- Saving all three artifacts
- Metric calculation
- Model comparison
- API health
- Model metadata
- `/predict/compare`
- `/predict/latest`
- Invalid request handling

Because the implementation has changed from one model to three models, tests that expect a single artifact or `/predict` endpoint should also be updated.

---

# Part 7: Execution Order

## 29. Recommended Command Sequence

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
python -m src.models.validate
python -m src.models.evaluate
uvicorn app.main:app --reload
```

Open:

```text
Swagger:          http://127.0.0.1:8000/docs
Health:           http://127.0.0.1:8000/health
Model information:http://127.0.0.1:8000/model/info
Latest prediction:http://127.0.0.1:8000/predict/latest
```

---

# Part 8: Design Decisions

## 30. Why Three Models Are Compared

The three models represent different learning approaches:

| Model | Learning approach |
|---|---|
| Logistic Regression | Linear and interpretable baseline |
| Random Forest | Bagged nonlinear decision trees |
| Gradient Boosting | Sequential nonlinear boosting |

Using the same dataset and test period makes it possible to assess whether model complexity provides a meaningful improvement.

The final results show that no single model dominates every metric:

- Logistic Regression performs best on accuracy and ROC-AUC.
- Gradient Boosting performs best on recall and F1.
- Random Forest remains competitive but does not lead the comparison.

---

## 31. Why an Ensemble Vote Is Returned

A majority vote summarizes the three model outputs.

For example:

```text
Logistic Regression → UP
Random Forest       → DOWN_OR_FLAT
Gradient Boosting   → UP

Ensemble            → UP
```

The ensemble is a comparison aid. It is not proof that the prediction is correct or profitable.

---

## 32. Why FastAPI Is Used

FastAPI provides:

- Automatic request validation
- Interactive Swagger documentation
- Structured JSON responses
- Strong Python typing
- Easy integration with scikit-learn
- Straightforward Docker deployment

---

# Part 9: Limitations

## 33. Technical Limitations

- Yahoo Finance may return delayed or missing candles.
- `GC=F` is Gold Futures, not exact broker-specific spot XAU/USD.
- The dataset is still relatively small.
- Only five engineered features are used.
- The latest candle may be incomplete.
- Hyperparameters are limited and may not be optimized.
- Automatic retraining and model-drift monitoring are not implemented.
- The API does not maintain a prediction-history database.

---

## 34. Financial Limitations

- Gold markets are noisy and difficult to predict.
- Metrics close to 50% indicate a weak directional edge.
- Historical relationships may not continue.
- Transaction costs, spread, slippage, and latency are not included.
- Backtest performance does not guarantee live profitability.
- This project is for technical demonstration, not financial advice.

---

# Part 10: Future Architecture

## 35. Possible Extensions

```mermaid
flowchart LR
    A[Scheduled Market Collector] --> B[Feature Service]
    B --> C[Prediction API]
    C --> D[Prediction Database]
    C --> E[Dashboard]
    C --> F[Notification Service]

    G[Model Monitoring] --> H{Performance Degraded?}
    H -- Yes --> I[Retraining Pipeline]
    I --> J[Model Registry]
    J --> C
```

Potential future improvements:

- Walk-forward optimization
- Hyperparameter search
- XGBoost, LightGBM, or CatBoost
- Probability calibration
- Feature-importance reporting
- Additional technical indicators
- Macroeconomic and news features
- Prediction history
- Scheduled retraining
- Model-drift monitoring
- Cloud deployment
- Dashboard and notifications

---

## 36. Final Architecture Summary

```text
Yahoo Finance GC=F
    ↓
Validation and Preprocessing
    ↓
Next-Hour Target Creation
    ↓
Five-Feature Engineering
    ↓
Chronological Train-Test Split
    ↓
Logistic Regression
Random Forest
Gradient Boosting
    ↓
Model Comparison and Backtest
    ↓
Saved Models and Evaluation Artifacts
    ↓
FastAPI
    ↓
/predict/compare and /predict/latest
    ↓
Docker Deployment
```

The architecture separates data acquisition, feature engineering, training, evaluation, serving, testing, and deployment. This makes the project easier to understand, reproduce, test, and extend.