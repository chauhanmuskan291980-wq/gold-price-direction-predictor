from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_metric(
    fold_results: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    """
    Summarize one numerical metric across folds.
    """
    values = pd.Series(
        [
            result[metric_name]
            for result in fold_results
            if result[metric_name] is not None
            and not pd.isna(
                result[metric_name]
            )
        ],
        dtype=float,
    )

    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "minimum": None,
            "maximum": None,
            "worst_fold": None,
            "best_fold": None,
        }

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))

    valid_results = [
        result
        for result in fold_results
        if result[metric_name] is not None
        and not pd.isna(
            result[metric_name]
        )
    ]

    worst_result = min(
        valid_results,
        key=lambda result: result[
            metric_name
        ],
    )

    best_result = max(
        valid_results,
        key=lambda result: result[
            metric_name
        ],
    )

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "worst_fold": {
            "fold": worst_result["fold"],
            "value": float(
                worst_result[metric_name]
            ),
        },
        "best_fold": {
            "fold": best_result["fold"],
            "value": float(
                best_result[metric_name]
            ),
        },
    }


def aggregate_walk_forward_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate all required walk-forward metrics.
    """
    if not fold_results:
        raise ValueError(
            "Cannot aggregate empty fold results."
        )

    folds_beating_buy_and_hold = sum(
        bool(
            result["beats_buy_and_hold"]
        )
        for result in fold_results
    )

    total_folds = len(fold_results)

    metric_names = [
        "accuracy",
        "roc_auc",
        "balanced_accuracy",
        "win_rate",
        "strategy_return",
        "buy_and_hold_return",
        "strategy_excess_return",
    ]

    metric_summaries = {
        metric_name: summarize_metric(
            fold_results=fold_results,
            metric_name=metric_name,
        )
        for metric_name in metric_names
    }

    return {
        "total_folds": total_folds,
        "folds_beating_buy_and_hold": (
            folds_beating_buy_and_hold
        ),
        "percentage_beating_buy_and_hold": (
            100
            * folds_beating_buy_and_hold
            / total_folds
        ),
        "metrics": metric_summaries,
    }