import pandas as pd

from models.player import Player
from backtest.models import REQUIRED_BACKTEST_COLUMNS


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_BACKTEST_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Historical dataframe is missing required columns: "
            + ", ".join(missing)
        )


def _site_name(site_value: str) -> str:
    normalized = str(site_value).strip().lower()

    if normalized in {"dk", "draftkings"}:
        return "DraftKings"

    if normalized in {"fd", "fanduel"}:
        return "FanDuel"

    raise ValueError(
        f"Unsupported DFS site: {site_value}"
    )


def _roster_positions(row) -> tuple[str, ...]:
    raw = row.get("roster_positions", None)

    if raw is None or pd.isna(raw):
        position = str(row["position"]).strip()

        if position == "DST":
            return ("DST",)

        return (position,)

    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw).split("/")

    positions = []

    for value in values:
        position = str(value).strip()

        if not position:
            continue

        if position == "DEF":
            position = "DST"

        positions.append(position)

    return tuple(positions)


def _float_or_default(row, column: str, default: float = 0.0) -> float:
    value = row.get(column, None)

    if value is None or pd.isna(value):
        return default

    return float(value)


def historical_frame_to_players(
    df: pd.DataFrame,
) -> list[Player]:
    """
    Convert BURN1's canonical historical dataframe into the same
    Player objects used by the live production optimizer.
    """

    _validate_columns(df)

    players = []

    for _, row in df.iterrows():
        position = str(row["position"]).strip()

        if position == "D":
            position = "DST"

        player = Player(
            name=str(row["player_name"]),
            site=_site_name(row["site"]),
            position=position,
            team=str(row["team"]),
            opponent=str(row["opponent"]),
            salary=int(row["salary"]),
            player_id=str(row["player_id"]),
            roster_positions=_roster_positions(row),
            status=(
                ""
                if "status" not in row or pd.isna(row.get("status"))
                else str(row.get("status"))
            ),
            projection=_float_or_default(
                row,
                "projection",
                0.0,
            ),
            ceiling=_float_or_default(
                row,
                "ceiling",
                0.0,
            ),
            ownership=_float_or_default(
                row,
                "projected_ownership",
                0.0,
            ),
        )

        players.append(player)

    return players
