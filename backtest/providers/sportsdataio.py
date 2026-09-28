"""
SportsDataIO adapter skeleton.

IMPORTANT:
This module performs NO API requests yet.

Its purpose is to isolate SportsDataIO-specific field mappings from
the BURN1 backtesting engine. API access will be added only after the
appropriate data-access/licensing terms are confirmed.
"""

from pathlib import Path

import pandas as pd

from backtest.models import REQUIRED_BACKTEST_COLUMNS


def validate_canonical_frame(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_BACKTEST_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Historical slate is missing required BURN1 columns: "
            + ", ".join(missing)
        )


def load_canonical_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(path)

    validate_canonical_frame(df)

    return df
