from gate.schemas import (
    BootstrapSummary,
    NullSummary,
    Verdict,
    WalkForwardSummary,
)


def determine_verdict(
    *,
    trade_count: int,
    minimum_trades: int,
    walk_forward: WalkForwardSummary | None,
    bootstrap: BootstrapSummary | None,
    null_summary: NullSummary | None,
    winner_concentration: float | None,
    config: dict,
) -> tuple[Verdict, dict[str, bool], list[str]]:
    if trade_count < minimum_trades:
        return (
            Verdict.INSUFFICIENT_DATA,
            {"minimum_sample": False},
            ["MINIMUM_SAMPLE_NOT_MET"],
        )

    if (
        walk_forward is None
        or bootstrap is None
        or null_summary is None
        or winner_concentration is None
    ):
        raise ValueError(
            "Complete evidence is required "
            "for an edge verdict."
        )

    checks = {
        "permutation_significant": (
            null_summary.p_value
            <= config["permutation"][
                "significance_level"
            ]
        ),
        "expectancy_lower_bound_positive": (
            bootstrap.expectancy_lower_bound
            > config["verdict"][
                "minimum_expectancy_lower_bound"
            ]
        ),
        "profit_factor_lower_bound": (
            bootstrap.profit_factor_lower_bound
            > config["verdict"][
                "minimum_profit_factor_lower_bound"
            ]
        ),
        "positive_window_rate": (
            walk_forward.positive_window_rate
            >= config["verdict"][
                "minimum_positive_window_rate"
            ]
        ),
        "winner_concentration": (
            winner_concentration
            <= config["concentration"][
                "maximum_allowed_concentration"
            ]
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    verdict = (
        Verdict.EDGE
        if not failed_checks
        else Verdict.NO_EDGE
    )

    return verdict, checks, failed_checks