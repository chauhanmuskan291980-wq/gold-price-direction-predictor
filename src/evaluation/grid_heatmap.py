from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]

TRAIN_WINDOW_ORDER = [
    1000,
    1500,
    2000,
]


def create_median_strategy_return_heatmap(
    grid_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a heatmap showing median strategy
    return for every model and train window.
    """

    required_columns = {
        "model",
        "train_window",
        "median_strategy_return",
    }

    missing_columns = (
        required_columns
        - set(grid_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Grid summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    heatmap_data = grid_summary.pivot(
        index="model",
        columns="train_window",
        values="median_strategy_return",
    )

    heatmap_data = heatmap_data.reindex(
        index=MODEL_ORDER,
        columns=TRAIN_WINDOW_ORDER,
    )

    # Convert decimal returns to percentages.
    percentage_data = heatmap_data * 100.0

    values = percentage_data.to_numpy(
        dtype=float
    )

    maximum_absolute_value = float(
        np.nanmax(
            np.abs(values)
        )
    )

    # Prevent an invalid colour scale when
    # every value is extremely close to zero.
    colour_limit = max(
        maximum_absolute_value,
        0.1,
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    image = axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-colour_limit,
        vmax=colour_limit,
    )

    axis.set_xticks(
        range(
            len(TRAIN_WINDOW_ORDER)
        )
    )

    axis.set_xticklabels(
        TRAIN_WINDOW_ORDER
    )

    axis.set_yticks(
        range(
            len(MODEL_ORDER)
        )
    )

    readable_model_names = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ]

    axis.set_yticklabels(
        readable_model_names
    )

    axis.set_xlabel(
        "Training window"
    )

    axis.set_ylabel(
        "Model"
    )

    axis.set_title(
        "Median Strategy Return Across "
        "the Approved Configuration Grid"
    )

    for row_index in range(
        len(MODEL_ORDER)
    ):
        for column_index in range(
            len(TRAIN_WINDOW_ORDER)
        ):
            value = values[
                row_index,
                column_index,
            ]

            if np.isnan(value):
                annotation = "N/A"
            else:
                annotation = (
                    f"{value:+.4f}%"
                )

            text_colour = (
                "white"
                if abs(value)
                > colour_limit * 0.55
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=text_colour,
                fontweight="bold",
            )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
    )

    colour_bar.set_label(
        "Median strategy return (%)"
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)