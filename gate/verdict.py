from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gate.bootstrap import BootstrapSummary
from gate.metrics import TradeMetricsSummary
from gate.nulls import NullSummary
from gate.schemas import Verdict, WalkForwardSummary


@dataclass(frozen=True)
class VerdictPolicy:
    """Define the thresholds used by the final verdict gate."""

    min_trade_count: int = 100
    min_window_count: int = 4
    min_observed_expectancy: float = 0.0
    min_observed_profit_factor: float = 1.0
    min_median_expectancy: float = 0.0
    min_positive_window_rate: float = 0.60
    min_bootstrap_expectancy_lower_bound: float = 0.0
    min_bootstrap_profit_factor_lower_bound: float = 1.0
    max_selection_adjusted_p_value: float = 0.05
    max_top_winner_concentration: float = 0.50
    max_losing_streak: int = 20

    def __post_init__(self) -> None:
        """Validate policy threshold values."""

        if self.min_trade_count <= 0:
            raise ValueError(
                "min_trade_count must be greater than zero."
            )

        if self.min_window_count <= 0:
            raise ValueError(
                "min_window_count must be greater than zero."
            )

        if not 0.0 <= self.min_positive_window_rate <= 1.0:
            raise ValueError(
                "min_positive_window_rate must be between "
                "zero and one."
            )

        if not 0.0 < self.max_selection_adjusted_p_value < 1.0:
            raise ValueError(
                "max_selection_adjusted_p_value must be "
                "between zero and one."
            )

        if not 0.0 < self.max_top_winner_concentration <= 1.0:
            raise ValueError(
                "max_top_winner_concentration must be "
                "greater than zero and at most one."
            )

        if self.max_losing_streak < 0:
            raise ValueError(
                "max_losing_streak must be zero or greater."
            )


DEFAULT_VERDICT_POLICY = VerdictPolicy()


@dataclass(frozen=True)
class VerdictAssessment:
    """Store the final verdict and its supporting reasons."""

    verdict: Verdict
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    insufficient_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the assessment into a JSON-compatible dictionary."""

        return {
            "verdict": self.verdict.value,
            "passed_checks": list(self.passed_checks),
            "failed_checks": list(self.failed_checks),
            "insufficient_reasons": list(
                self.insufficient_reasons
            ),
        }


def evaluate_verdict(
    *,
    trade_count: int,
    walk_forward: WalkForwardSummary | None,
    metrics: TradeMetricsSummary | None,
    bootstrap: BootstrapSummary | None,
    null_test: NullSummary | None,
    policy: VerdictPolicy = DEFAULT_VERDICT_POLICY,
) -> VerdictAssessment:
    """Evaluate all evidence and return the final gate verdict."""

    insufficient_reasons: list[str] = []

    if trade_count < policy.min_trade_count:
        insufficient_reasons.append(
            f"Completed trades ({trade_count}) are below "
            f"the minimum ({policy.min_trade_count})."
        )

    if walk_forward is None:
        insufficient_reasons.append(
            "Walk-forward evidence is unavailable."
        )
    elif walk_forward.window_count < policy.min_window_count:
        insufficient_reasons.append(
            f"Walk-forward windows "
            f"({walk_forward.window_count}) are below "
            f"the minimum ({policy.min_window_count})."
        )

    if metrics is None:
        insufficient_reasons.append(
            "Observed trade metrics are unavailable."
        )

    if bootstrap is None:
        insufficient_reasons.append(
            "Bootstrap evidence is unavailable."
        )

    if null_test is None:
        insufficient_reasons.append(
            "Selection-adjusted null evidence is unavailable."
        )

    if insufficient_reasons:
        return VerdictAssessment(
            verdict=Verdict.INSUFFICIENT_DATA,
            passed_checks=(),
            failed_checks=(),
            insufficient_reasons=tuple(
                insufficient_reasons
            ),
        )

    if (
        walk_forward is None
        or metrics is None
        or bootstrap is None
        or null_test is None
    ):
        raise RuntimeError(
            "Required evidence unexpectedly became unavailable."
        )

    checks = [
        (
            metrics.expectancy
            > policy.min_observed_expectancy,
            (
                "Observed expectancy is positive: "
                f"{metrics.expectancy:.6f}."
            ),
        ),
        (
            metrics.profit_factor
            > policy.min_observed_profit_factor,
            (
                "Observed profit factor exceeds "
                f"{policy.min_observed_profit_factor:.2f}: "
                f"{metrics.profit_factor:.6f}."
            ),
        ),
        (
            walk_forward.median_expectancy
            > policy.min_median_expectancy,
            (
                "Walk-forward median expectancy is positive: "
                f"{walk_forward.median_expectancy:.6f}."
            ),
        ),
        (
            walk_forward.positive_window_rate
            >= policy.min_positive_window_rate,
            (
                "Positive-window rate meets the minimum: "
                f"{walk_forward.positive_window_rate:.2%}."
            ),
        ),
        (
            bootstrap.expectancy_lower_bound
            > policy.min_bootstrap_expectancy_lower_bound,
            (
                "Bootstrap expectancy lower bound is positive: "
                f"{bootstrap.expectancy_lower_bound:.6f}."
            ),
        ),
        (
            bootstrap.profit_factor_lower_bound
            > policy.min_bootstrap_profit_factor_lower_bound,
            (
                "Bootstrap profit-factor lower bound exceeds "
                f"{policy.min_bootstrap_profit_factor_lower_bound:.2f}: "
                f"{bootstrap.profit_factor_lower_bound:.6f}."
            ),
        ),
        (
            null_test.selection_adjusted_p_value
            <= policy.max_selection_adjusted_p_value,
            (
                "Selection-adjusted p-value meets the maximum: "
                f"{null_test.selection_adjusted_p_value:.6f}."
            ),
        ),
        (
            null_test.observed_statistic
            > null_test.adjusted_null_percentile,
            (
                "Observed expectancy exceeds the adjusted "
                f"null threshold: {null_test.observed_statistic:.6f} "
                f"> {null_test.adjusted_null_percentile:.6f}."
            ),
        ),
        (
            metrics.top_five_percent_winner_concentration
            <= policy.max_top_winner_concentration,
            (
                "Top-5% winner concentration is within the "
                f"maximum: "
                f"{metrics.top_five_percent_winner_concentration:.2%}."
            ),
        ),
        (
            metrics.max_losing_streak
            <= policy.max_losing_streak,
            (
                "Maximum losing streak is within the limit: "
                f"{metrics.max_losing_streak}."
            ),
        ),
    ]

    passed_checks: list[str] = [
        (
            f"Completed trades meet the minimum: "
            f"{trade_count} >= {policy.min_trade_count}."
        ),
        (
            "Walk-forward window count meets the minimum: "
            f"{walk_forward.window_count} "
            f">= {policy.min_window_count}."
        ),
    ]

    failed_checks: list[str] = []

    for passed, description in checks:
        if passed:
            passed_checks.append(description)
        else:
            failed_checks.append(description)

    verdict = (
        Verdict.NO_EDGE
        if failed_checks
        else Verdict.EDGE
    )

    return VerdictAssessment(
        verdict=verdict,
        passed_checks=tuple(passed_checks),
        failed_checks=tuple(failed_checks),
        insufficient_reasons=(),
    )