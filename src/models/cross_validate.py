from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_validate,
)

from src.models.model_factory import build_models
from src.models.train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_data,
)


def run_cross_validation() -> dict:
    data = load_training_data(
        "data/processed/gold_features.csv"
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    time_split = TimeSeriesSplit(
        n_splits=5,
        gap=1,
    )

    models = build_models()
    results = {}

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": (
            "balanced_accuracy"
        ),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for model_name, model in models.items():
        scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=time_split,
            scoring=scoring,
            n_jobs=-1,
        )

        results[model_name] = {
            "mean_accuracy": float(
                scores[
                    "test_accuracy"
                ].mean()
            ),
            "mean_balanced_accuracy": float(
                scores[
                    "test_balanced_accuracy"
                ].mean()
            ),
            "mean_f1": float(
                scores["test_f1"].mean()
            ),
            "mean_roc_auc": float(
                scores[
                    "test_roc_auc"
                ].mean()
            ),
        }

    return results