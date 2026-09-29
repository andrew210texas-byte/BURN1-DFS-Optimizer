# BURN1 Backtest Workspace

This package is intentionally isolated from the production optimizer.

## Structure

- `data/dk/` — local DraftKings-compatible historical slate data location
- `data/fd/` — local FanDuel-compatible historical slate data location
- `configs/` — backtest strategy and configuration files
- `results/` — locally generated raw lineup outputs
- `reports/` — locally generated scoring and summary reports
- `experiments/` — controlled optimization and exposure experiments
- `providers/` — external-data provider adapters

Provider-derived slate datasets, generated results, reports, experiment outputs, and trained model binaries are intentionally excluded from the public repository.

## Backtesting Rules

- Never use post-lock information as optimizer input.
- Actual fantasy points are used only for scoring completed backtests.
- Historical salary/slate data must come from a lawful source.
- Production optimizer code is not modified by backtest experiments.
- Historical results from a single slate should not be treated as evidence of general predictive performance.