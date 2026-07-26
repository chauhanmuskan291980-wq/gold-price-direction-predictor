# Walk-Forward Validation Implementation Plan

## Goal

Extend the existing Gold Price Direction Predictor with a reproducible,
config-driven rolling-origin walk-forward evaluation harness.

## Requirements

- Configurable train window
- Configurable test window
- Configurable step size
- Purged boundary between train and test
- Fresh model training for every fold
- Per-fold accuracy
- Per-fold ROC-AUC
- Per-fold win rate
- Per-fold strategy return
- Per-fold buy-and-hold return
- Aggregate median and IQR
- Worst fold identification
- Percentage of folds beating buy-and-hold
- JSON report
- Metric plots
- One-command reproduction
- Tests preventing look-ahead leakage

## Implementation Stages

1. Review and reuse the existing evaluation pipeline.
2. Add walk-forward configuration.
3. Implement fold generation.
4. Add fold-boundary leakage tests.
5. Run the first fold end-to-end.
6. Extend evaluation to all folds.
7. Generate aggregate statistics.
8. Generate JSON and plots.
9. Document the real results.