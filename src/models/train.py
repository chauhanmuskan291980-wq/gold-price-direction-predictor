from pathlib import Path

import joblib
import pandas as pd

from src.models.model_factory import build_model


FEATURE_COLUMNS =[
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14"
]

TARGET_COLUMN = "target"

ARTIFACT_DIR = Path("artifacts/models")


def load_training_data(path:str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column 
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns :{missing_columns}"
        )
    
    return data.dropna(subset=required_columns).copy()


def chronological_split(
    data:pd.DataFrame,
    train_ratio : float = 0.80
):
    split_index = int(len(data)*train_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    return X_train , X_test , y_train , y_test


def train_all_models(
        X_train:pd.DataFrame,
        y_train:pd.Series,
) -> dict:
    models = build_model()
    trained_models = {}

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name , model in models.items():
        print(f"Training {model_name}........")
        model.fit(X_train,y_train)
        model_path =(
            ARTIFACT_DIR / f"{model_name}.joblib"
        )

        joblib.dump(model , model_path)
        trained_models[model_path] = model
        print(f"Saved model to: {model_path}")
    
    return trained_models


def main() -> None:
    data = load_training_data(
        'data/processed/gold_features.csv'
    )

    X_train , X_test , y_train , y_test =(
        chronological_split(data)
    )

    train_all_models(
        X_train=X_train,
        y_train=y_train
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

if __name__ == "__main__":
    main()