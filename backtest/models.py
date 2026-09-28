from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BacktestPlayer:
    """
    Canonical BURN1 historical DFS player record.

    This is provider-neutral. Any licensed data source must be
    transformed into this format before entering the backtester.
    """

    season: int
    week: int
    site: str
    slate_id: str

    player_id: str
    player_name: str

    team: str
    opponent: str
    position: str

    salary: int

    game_id: Optional[str] = None
    roster_positions: Optional[str] = None

    projection: Optional[float] = None
    projected_ownership: Optional[float] = None

    actual_points: Optional[float] = None


REQUIRED_BACKTEST_COLUMNS = [
    "season",
    "week",
    "site",
    "slate_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "salary",
]


OPTIONAL_BACKTEST_COLUMNS = [
    "game_id",
    "roster_positions",
    "projection",
    "projected_ownership",
    "actual_points",
]
