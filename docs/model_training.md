# `docs/model_training.md`

# Model Training

## Purpose

The model-training stage teaches a classifier to predict whether the next hourly Gold candle will close higher or lower.

The project uses logistic regression as a simple and interpretable baseline model.

## Pipeline Position

```text
Engineered features
    ↓
Time-based train/test split
    ↓
Feature scaling
    ↓
Logistic Regression training
    ↓
Saved model artifact
```

## Main File

```text
src/models/train.py
```

## File Responsibility

This file is responsible for:

* loading the feature dataset
* selecting model input columns
* separating features and target
* performing a time-based split
* building the training pipeline
* fitting the model
* saving the trained artifact
* returning predictions or training information

It does not serve HTTP requests. API prediction is handled separately.

## Model Inputs

The model uses:

```python
FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]
```

The target is:

```python
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

`X` contains the values used to make predictions.

`y` contains the correct historical answers.

## Why Logistic Regression Is Used

Logistic regression is appropriate as a baseline because it is:

* simple
* fast to train
* suitable for binary classification
* easy to reproduce
* interpretable
* able to return probabilities

The model estimates the probability that:

```text
target = 1
```

meaning that the next candle closes higher.

## Time-Based Split

Financial data must not be randomly shuffled.

A time-based split should follow this structure:

```text
Older observations → training set
Newer observations → test set
```

Example:

```python
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]
```

This uses approximately:

```text
80% older data for training
20% newer data for testing
```

## Why Random Splitting Is Avoided

A random split can mix future candles into the training data while older candles appear in the test set.

That would create an unrealistic evaluation because a live model cannot learn from future information.

The correct order is:

```text
Past → train
Future → test
```

## Feature Scaling

Logistic regression performs better when numerical features have comparable scales.

The project can use:

```python
StandardScaler()
```

A typical pipeline is:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline(
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

## Why a Pipeline Is Used

The pipeline combines preprocessing and model training.

This ensures that:

* the same scaling is used during training and prediction
* the scaler is fitted only on training data
* the API does not need separate scaling logic
* the complete prediction workflow can be saved as one artifact

## Training

The model is trained using:

```python
model.fit(X_train, y_train)
```

Meaning:

```text
X_train = historical feature values
y_train = known next-candle directions
```

During training, the model learns how each feature relates to the probability of an upward candle.

## Predictions

Class predictions:

```python
predictions = model.predict(X_test)
```

Possible result:

```text
0
1
1
0
```

Probability predictions:

```python
probabilities = model.predict_proba(X_test)[:, 1]
```

The second probability column represents the probability of class `1`.

Example:

```text
0.62 = 62% estimated probability of an upward next candle
```

This does not mean the prediction is certain.

## Saving the Model

The trained pipeline is saved using Joblib:

```python
import joblib

joblib.dump(model, output_path)
```

Example artifact path:

```text
artifacts/model.joblib
```

The saved object should include both:

```text
StandardScaler
LogisticRegression
```

because they are part of the same pipeline.

## Saving Metadata

Useful model metadata may include:

```python
metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "split_index": split_index,
}
```

This helps the API and reviewer understand how the artifact was produced.

## Reproducibility

The training process should use a fixed random state when supported:

```python
random_state=42
```

This reduces unnecessary differences between repeated runs.

The time-based split itself is deterministic because it depends on row order.

## Leakage Prevention

The following rules are important:

1. Sort observations by timestamp before splitting.
2. Create the target without adding future values to model inputs.
3. Fit the scaler using only `X_train`.
4. Evaluate only on later unseen observations.
5. Do not tune the model using final test results repeatedly.

Using a scikit-learn pipeline ensures that the scaler learns only from training data when `fit()` is called on `X_train`.

## Output

The training stage produces:

```text
Trained model pipeline
Test predictions
Prediction probabilities
Model artifact
Train/test split information
```

The saved artifact is later loaded by:

```text
app/services/prediction_service.py
```

## Limitations

* Logistic regression models mainly linear relationships.
* The model does not directly understand market regimes.
* Predictive performance close to 50% may still be weak.
* Historical results do not guarantee live profitability.
* Transaction costs are not included in basic model training.
