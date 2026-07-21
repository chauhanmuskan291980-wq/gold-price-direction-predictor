# Model Training

## Purpose

The model-training stage teaches three classifiers to predict whether the next hourly Gold candle will close higher or lower.

The project evolved through three stages:

1. Approximately **3 months of hourly data** with Logistic Regression
2. Approximately **6 months of hourly data** with Logistic Regression
3. Approximately **6 months of hourly data** with three-model comparison

The final models are:

- Logistic Regression
- Random Forest
- Gradient Boosting

Logistic Regression remains the interpretable baseline, while Random Forest and Gradient Boosting test whether nonlinear tree-based approaches improve performance.

---

## Pipeline Position

```text
Engineered features
    ↓
Chronological train-test split
    ↓
Train Logistic Regression
Train Random Forest
Train Gradient Boosting
    ↓
Generate test predictions
    ↓
Compare evaluation metrics
    ↓
Save three model artifacts
```

---

## Main File

```text
src/models/train.py
```

Supporting files may include:

```text
src/models/validate.py
src/models/evaluate.py
src/models/predict.py
```

---

## File Responsibility

The training module is responsible for:

- Loading the feature dataset
- Selecting model input columns
- Separating features and target
- Performing a chronological split
- Building three classifiers
- Fitting all models
- Generating test predictions and probabilities
- Saving model artifacts
- Saving training metadata
- Returning split and model information

HTTP requests are handled separately by the FastAPI application.

---

## Model Inputs

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]

TARGET_COLUMN = "target"
```

Feature matrix:

```python
X = data[FEATURE_COLUMNS]
```

Target vector:

```python
y = data[TARGET_COLUMN]
```

`X` contains the five engineered feature values.

`y` contains the known historical next-candle direction:

```text
1 = next close is higher
0 = next close is lower or equal
```

---

## Time-Based Split

Financial observations must not be randomly shuffled.

The project uses:

```text
Older observations → Training set
Newer observations → Test set
```

Example:

```python
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()
```

The approximate split is:

```text
80% training
20% testing
```

A recent run produced:

```text
Rows after feature engineering: 3327
Training rows: 2661
Testing rows: 666
```

Exact counts may vary when Yahoo Finance returns updated data.

---

## Why Random Splitting Is Avoided

Random splitting can mix future observations into training data.

That would create an unrealistic evaluation because a live system cannot train on future candles.

Correct order:

```text
Past → Training
Future → Testing
```

The final test period must remain unseen during fitting.

---

# Models

## 1. Logistic Regression

Logistic Regression is used as the baseline because it is:

- Simple
- Fast to train
- Suitable for binary classification
- Easy to reproduce
- Interpretable
- Able to return class probabilities

Because the input features use different numerical scales, Logistic Regression is stored in a scikit-learn pipeline with `StandardScaler`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logistic_regression = Pipeline(
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

Train:

```python
logistic_regression.fit(X_train, y_train)
```

Probabilities:

```python
logistic_probabilities = logistic_regression.predict_proba(X_test)[:, 1]
```

---

## 2. Random Forest

Random Forest is an ensemble of decision trees trained on different samples and feature combinations.

It is included because it can model nonlinear relationships and feature interactions without requiring feature scaling.

```python
from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
```

Train:

```python
random_forest.fit(X_train, y_train)
```

Probabilities:

```python
random_forest_probabilities = random_forest.predict_proba(X_test)[:, 1]
```

Possible hyperparameters may differ in the current code. The documentation should match the exact values used in `train.py`.

---

## 3. Gradient Boosting

Gradient Boosting builds decision trees sequentially.

Each new tree attempts to correct prediction errors from the previous ensemble.

```python
from sklearn.ensemble import GradientBoostingClassifier


gradient_boosting = GradientBoostingClassifier(
    random_state=42,
)
```

Train:

```python
gradient_boosting.fit(X_train, y_train)
```

Probabilities:

```python
gradient_boosting_probabilities = (
    gradient_boosting.predict_proba(X_test)[:, 1]
)
```

Gradient Boosting does not require `StandardScaler`.

---

## Recommended Model Factory

A reusable factory can keep model creation consistent:

```python
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
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
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }
```

For Python 3.10 compatibility, add:

```python
from __future__ import annotations
```

at the top of modules that use modern annotations evaluated at runtime.

---

## Training All Models

```python
models = build_models()
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model
```

Predictions:

```python
predictions = {
    model_name: model.predict(X_test)
    for model_name, model in trained_models.items()
}
```

Probabilities:

```python
probabilities = {
    model_name: model.predict_proba(X_test)[:, 1]
    for model_name, model in trained_models.items()
}
```

---

## Type-Annotation Compatibility

On some Python 3.10 and pandas combinations, annotations such as:

```python
y_test: pd.Series[Any]
```

may raise:

```text
TypeError: 'type' object is not subscriptable
```

Recommended fix:

```python
from __future__ import annotations
```

Place it at the first line of the module.

A simpler alternative is:

```python
y_test: pd.Series
```

The future import is preferred because it postpones runtime evaluation of annotations.

---

## Saving Model Artifacts

Each trained model is saved separately.

```python
from pathlib import Path

import joblib


MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for model_name, model in trained_models.items():
    output_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, output_path)
```

Generated artifacts:

```text
artifacts/models/
├── logistic_regression.joblib
├── random_forest.joblib
└── gradient_boosting.joblib
```

This replaces the older single-model artifact structure.

---

## Saving Metadata

Training metadata can be saved as JSON.

```python
metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "split_index": split_index,
    "models": list(trained_models),
    "data_source": "Yahoo Finance",
    "symbol": "GC=F",
    "interval": "1h",
}
```

Useful metadata includes:

- Feature names
- Target name
- Training rows
- Testing rows
- Split index
- Model names
- Data symbol
- Data interval
- Training timestamp
- Library versions

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

Example metric function:

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
    }
```

---

## Final Model Comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **51.50%** | **51.56%** | **48.22%** | 52.41% | 50.23% | **52.47%** |
| Random Forest | 50.30% | 50.45% | 47.13% | 52.73% | 49.77% | 52.42% |
| Gradient Boosting | 50.30% | 51.11% | 47.58% | **63.34%** | **54.34%** | 51.55% |

### Results Interpretation

- **Logistic Regression** achieved the highest accuracy, balanced accuracy, and ROC-AUC.
- **Gradient Boosting** achieved the highest recall and F1 score.
- **Random Forest** achieved a similar ROC-AUC to Logistic Regression but lower overall accuracy.
- No model produced a strong predictive advantage.
- Performance near 50% should be interpreted carefully because short-term financial direction is highly noisy.

---

## Saving the Comparison Table

```python
import pandas as pd


comparison_rows = []

for model_name, model_metrics in metrics_by_model.items():
    comparison_rows.append(
        {
            "model": model_name,
            **model_metrics,
        }
    )

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(
    "artifacts/model_comparison.csv",
    index=False,
)
```

Expected CSV columns:

```text
model
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
```

---

## Model Selection

There is no single best model across every metric.

For this project:

- Logistic Regression is the strongest general baseline.
- Gradient Boosting is strongest when recall and F1 are prioritized.
- Random Forest provides an additional nonlinear comparison.

The API therefore returns:

1. Logistic Regression prediction
2. Random Forest prediction
3. Gradient Boosting prediction
4. Ensemble majority vote

This is more transparent than hiding the differences behind one selected model.

---

## Ensemble Majority Vote

```python
model_classes = [
    logistic_prediction,
    random_forest_prediction,
    gradient_boosting_prediction,
]

ensemble_class = int(sum(model_classes) >= 2)
```

Example:

```text
Logistic Regression → 1
Random Forest       → 0
Gradient Boosting   → 1
Ensemble            → 1
```

The ensemble does not guarantee higher accuracy. It summarizes agreement among the three classifiers.

---

## Reproducibility

Fixed random states are used where supported:

```python
random_state = 42
```

This improves reproducibility for:

- Logistic Regression
- Random Forest
- Gradient Boosting

The chronological split is deterministic because it depends on row order.

---

## Leakage Prevention

The training process follows these rules:

1. Sort observations by timestamp.
2. Build features using only current and historical values.
3. Use future close only to create the target.
4. Remove future-price columns from model inputs.
5. Split chronologically.
6. Fit scaling only on `X_train`.
7. Evaluate only on later unseen observations.
8. Avoid repeatedly tuning models against the final test set.

---

## Output

The training stage produces:

```text
Three trained model artifacts
Test predictions for each model
Prediction probabilities for each model
Training metadata
Chronological split information
Model comparison metrics
```

The saved models are later loaded by the FastAPI model service.

---

## API Integration

The trained models support two prediction workflows.

### Manual comparison

```http
POST /predict/compare
```

The caller supplies the five feature values. The API returns predictions from all models and an ensemble vote.

### Latest market prediction

```http
GET /predict/latest
```

The API downloads recent `GC=F` hourly data, calculates the latest features, and runs all three models.

---

## Running Training and Evaluation

Train all models:

```bash
python -m src.models.train
```

Run chronological validation:

```bash
python -m src.models.validate
```

Run final evaluation:

```bash
python -m src.models.evaluate
```

---

## Limitations

- Six months of hourly data is still a small financial dataset.
- Only five engineered features are used.
- Hyperparameter tuning is limited.
- Metrics remain close to random classification.
- Market regimes can change over time.
- Historical results do not guarantee live performance.
- Transaction costs are not included in basic classification metrics.
- The ensemble vote is not guaranteed to outperform each individual model.

---

## Summary

```text
Approximately 6 months of Yahoo Finance GC=F data
    ↓
Five engineered features
    ↓
Chronological 80/20 split
    ↓
Logistic Regression
Random Forest
Gradient Boosting
    ↓
Classification and trading evaluation
    ↓
Three saved model artifacts
    ↓
FastAPI comparison and latest prediction
```