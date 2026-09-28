# BURN1 Backtest Workspace

This folder is intentionally isolated from the production optimizer.

Structure:

backtest/
  data/
    dk/       Historical DraftKings slate files
    fd/       Historical FanDuel slate files
  configs/    Backtest strategy/config files
  results/    Raw lineup and scoring outputs
  reports/    Summary reports

Rules:
- Never use post-lock information as optimizer input.
- Actual fantasy points are used only for scoring completed backtests.
- Historical salary/slate data must come from a lawful source.
- Production optimizer code is not modified by backtest experiments.
