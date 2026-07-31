from pathlib import Path

from src.models.walk_forward_grid import (
    create_cell_id,
    load_grid_config,
)

DATA_PATH = Path(
    "data/processed/gold_features.csv"
)


def test_grid_config_contains_models_and_windows() -> None:
    config = load_grid_config()

    assert len(config["models"]) == 3
    assert len(config["windows"]) == 3


def test_cell_id_is_stable() -> None:
    window = {
        "name": "baseline",
        "train_window": 1500,
        "test_window": 250,
        "step_size": 250,
        "purge_gap": 1,
    }

    cell_id = create_cell_id(
        "logistic_regression",
        window,
    )

    assert cell_id == (
        "logistic_regression__baseline__"
        "train1500_test250_step250"
    )