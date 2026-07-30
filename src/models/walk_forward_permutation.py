from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.permutation_null import (
    permute_target,
    summarize_permutation_null,
)
from src.models.evaluate import (
    get_period,
    save_json,
)
from src.models.train import (
    load_training_data,
)
from src.models.walk_forward_grid import (
    CONFIG_PATH,
    load_grid_config,
    rank_summary,
    run_grid,
)


ARTIFACT_DIRECTORY = Path(
    "artifacts/grid_search"
)

PERMUTATION_NULL_PATH = (
    ARTIFACT_DIRECTORY
    / "permutation_null.csv"
)

PERMUTATION_REPORT_PATH = (
    ARTIFACT_DIRECTORY
    / "permutation_report.json"
)


def assert_same_fold_boundaries(
    real_grid_results: list[
        dict[str, Any]
    ],
    permuted_grid_results: list[
        dict[str, Any]
    ],
) -> None:
    """
    Assert that the real and permuted grids use
    identical fold indexes for every cell.
    """

    real_boundaries = {
        cell["cell_id"]: cell[
            "fold_boundaries"
        ]
        for cell in real_grid_results
    }

    permuted_boundaries = {
        cell["cell_id"]: cell[
            "fold_boundaries"
        ]
        for cell in permuted_grid_results
    }

    if (
        real_boundaries
        != permuted_boundaries
    ):
        raise AssertionError(
            "The permuted grid does not use "
            "the same fold boundaries as "
            "the real grid."
        )


def run_selection_adjusted_null(
    data: pd.DataFrame,
    config: dict[str, Any],
    real_grid_results: list[
        dict[str, Any]
    ],
    real_best_return: float,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Shuffle the target repeatedly, run the complete
    nine-cell grid, and record each grid's best result.
    """

    permutation_config = config[
        "permutation"
    ]

    repetitions = int(
        permutation_config[
            "repetitions"
        ]
    )

    seed = int(
        permutation_config[
            "seed"
        ]
    )

    if repetitions <= 0:
        raise ValueError(
            "Permutation repetitions must "
            "be greater than zero."
        )

    rng = np.random.default_rng(
        seed
    )

    null_records: list[
        dict[str, Any]
    ] = []

    for permutation_number in range(
        1,
        repetitions + 1,
    ):
        permuted_data = permute_target(
            data=data,
            rng=rng,
        )

        permuted_grid_results = run_grid(
            data=permuted_data,
            config=config,
            verbose=False,
        )

        assert_same_fold_boundaries(
            real_grid_results=(
                real_grid_results
            ),
            permuted_grid_results=(
                permuted_grid_results
            ),
        )

        permuted_summary = rank_summary(
            permuted_grid_results
        )

        best_null_row = (
            permuted_summary.iloc[0]
        )

        null_records.append(
            {
                "permutation": (
                    permutation_number
                ),
                "best_cell_id": str(
                    best_null_row[
                        "cell_id"
                    ]
                ),
                "maximum_median_strategy_return": float(
                    best_null_row[
                        "median_strategy_return"
                    ]
                ),
            }
        )

        if (
            permutation_number == 1
            or permutation_number
            % 25 == 0
            or permutation_number
            == repetitions
        ):
            print(
                "Completed permutation "
                f"{permutation_number}/"
                f"{repetitions}"
            )

    null_results = pd.DataFrame(
        null_records
    )

    null_summary = (
        summarize_permutation_null(
            real_best_return=(
                real_best_return
            ),
            null_results=null_results,
        )
    )

    null_summary["repetitions"] = (
        repetitions
    )

    null_summary["seed"] = seed

    null_summary["method"] = (
        "maximum median strategy return "
        "across the complete nine-cell grid"
    )

    return (
        null_results,
        null_summary,
    )


def main() -> None:
    """
    Run the real grid and the selection-adjusted
    permutation null.
    """

    config = load_grid_config()

    data_path = Path(
        str(
            config["data_path"]
        )
    )

    data = load_training_data(
        data_path
    )

    print(
        "Running the real nine-cell grid..."
    )

    real_grid_results = run_grid(
        data=data,
        config=config,
        verbose=True,
    )

    real_summary = rank_summary(
        real_grid_results
    )

    best_real_row = (
        real_summary.iloc[0]
    )

    real_best_cell_id = str(
        best_real_row[
            "cell_id"
        ]
    )

    real_best_return = float(
        best_real_row[
            "median_strategy_return"
        ]
    )

    print(
        "\nRunning the selection-adjusted "
        "permutation null..."
    )

    null_results, null_summary = (
        run_selection_adjusted_null(
            data=data,
            config=config,
            real_grid_results=(
                real_grid_results
            ),
            real_best_return=(
                real_best_return
            ),
        )
    )

    p_value = float(
        null_summary[
            "empirical_p_value"
        ]
    )

    if p_value < 0.05:
        verdict = (
            "The real best cell clears the "
            "selection-adjusted permutation "
            "noise floor, but it still requires "
            "untouched future validation."
        )
    else:
        verdict = (
            "The real best cell does not clear "
            "the selection-adjusted permutation "
            "noise floor. Its performance is "
            "consistent with configuration "
            "snooping rather than a reliable edge."
        )

    null_summary["verdict"] = verdict

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    null_results.to_csv(
        PERMUTATION_NULL_PATH,
        index=False,
    )

    report = {
        "configuration": {
            "config_path": (
                CONFIG_PATH.as_posix()
            ),
            "models": config["models"],
            "windows": config["windows"],
            "permutation": config[
                "permutation"
            ],
        },
        "dataset": {
            "path": data_path.as_posix(),
            "total_rows": int(
                len(data)
            ),
            "period": get_period(data),
        },
        "real_grid": {
            "cell_count": len(
                real_grid_results
            ),
            "best_cell_id": (
                real_best_cell_id
            ),
            "best_median_strategy_return": (
                real_best_return
            ),
        },
        "permutation_null": (
            null_summary
        ),
    }

    save_json(
        report,
        PERMUTATION_REPORT_PATH,
    )

    print(
        "\nPermutation null completed."
    )

    print(
        "Real best cell:",
        real_best_cell_id,
    )

    print(
        "Real best median return:",
        f"{real_best_return:.4%}",
    )

    print(
        "Null median:",
        f"{null_summary['null_median']:.4%}",
    )

    print(
        "Null 95th percentile:",
        f"{null_summary['null_95th_percentile']:.4%}",
    )

    print(
        "Empirical p-value:",
        f"{p_value:.6f}",
    )

    print(
        "Verdict:",
        verdict,
    )

    print(
        "Null CSV:",
        PERMUTATION_NULL_PATH,
    )

    print(
        "Permutation report:",
        PERMUTATION_REPORT_PATH,
    )


if __name__ == "__main__":
    main()