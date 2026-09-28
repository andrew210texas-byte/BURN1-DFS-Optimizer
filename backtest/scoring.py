from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from backtest.identity_aliases import DEFAULT_PLAYER_ALIASES
from utils.player_identity import (
    normalize_player_name,
    normalize_position,
    normalize_team,
)


REQUIRED_ACTUAL_COLUMNS = {
    "player_name",
    "team",
    "position",
    "actual_dk_points",
}


def identity_name(value: object) -> str:
    return normalize_player_name(
        value,
        aliases=DEFAULT_PLAYER_ALIASES,
    )


def prepare_actuals(
    actuals_path: str | Path,
    overrides_path: str | Path | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    site: str | None = None,
) -> pd.DataFrame:
    """
    Load historical outcomes into a deterministic scoring table.

    Optional manual overrides are explicit audit records. They are not
    hidden hard-coded corrections inside scoring logic.
    """

    actuals_path = Path(actuals_path)

    actuals = pd.read_csv(actuals_path)

    missing = REQUIRED_ACTUAL_COLUMNS - set(actuals.columns)

    if missing:
        raise ValueError(
            "Historical actuals are missing required columns: "
            + ", ".join(sorted(missing))
        )

    actuals = actuals.copy()

    actuals["name_key"] = (
        actuals["player_name"]
        .map(identity_name)
    )

    actuals["team_key"] = (
        actuals["team"]
        .map(normalize_team)
    )

    actuals["position_key"] = (
        actuals["position"]
        .map(normalize_position)
    )

    actuals["actual_dk_points"] = pd.to_numeric(
        actuals["actual_dk_points"],
        errors="raise",
    )

    if overrides_path is None:
        return actuals

    overrides_path = Path(overrides_path)

    if not overrides_path.exists():
        return actuals

    overrides = pd.read_csv(overrides_path)

    if overrides.empty:
        return actuals

    if season is not None and "season" in overrides.columns:
        overrides = overrides[
            pd.to_numeric(
                overrides["season"],
                errors="coerce",
            )
            == season
        ]

    if week is not None and "week" in overrides.columns:
        overrides = overrides[
            pd.to_numeric(
                overrides["week"],
                errors="coerce",
            )
            == week
        ]

    if site is not None and "site" in overrides.columns:
        overrides = overrides[
            overrides["site"]
            .astype(str)
            .str.upper()
            .str.strip()
            == str(site).upper().strip()
        ]

    for _, override in overrides.iterrows():
        name = str(
            override["player_name"]
        )

        team = normalize_team(
            override["team"]
        )

        position = normalize_position(
            override["position"]
        )

        points = float(
            override["actual_points"]
        )

        name_key = identity_name(name)

        mask = (
            (actuals["name_key"] == name_key)
            &
            (actuals["team_key"] == team)
            &
            (actuals["position_key"] == position)
        )

        matches = int(mask.sum())

        if matches > 1:
            raise ValueError(
                "Manual outcome override matched multiple rows: "
                f"{name} / {team} / {position}"
            )

        if matches == 1:
            actuals.loc[
                mask,
                "actual_dk_points",
            ] = points

            continue

        actuals = pd.concat(
            [
                actuals,
                pd.DataFrame(
                    [
                        {
                            "player_name": name,
                            "team": team,
                            "position": position,
                            "actual_dk_points": points,
                            "name_key": name_key,
                            "team_key": team,
                            "position_key": position,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return actuals


def _lineup_players(lineup):
    """
    Support both:
      - production LineupPlayer objects
      - serialized BURN1 lineup dictionaries
    """

    if isinstance(lineup, dict):
        return lineup["players"]

    return lineup


def _extract_player(item):
    """
    Return:
        name, team, position, salary, projection
    """

    if isinstance(item, dict):
        player = item.get(
            "player",
            item,
        )

        return (
            str(player["name"]),
            normalize_team(
                player["team"]
            ),
            normalize_position(
                player["position"]
            ),
            int(player["salary"]),
            float(
                player.get(
                    "projection",
                    0.0,
                )
            ),
        )

    player = getattr(
        item,
        "player",
        item,
    )

    return (
        str(player.name),
        normalize_team(
            player.team
        ),
        normalize_position(
            player.position
        ),
        int(player.salary),
        float(player.projection),
    )


def score_lineups(
    lineups: Iterable,
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score arbitrary BURN1 lineups against historical actual outcomes.

    Returns:
        scored_lineups
        unmatched_players
    """

    scored_rows = []
    unmatched_rows = []

    for lineup_number, lineup in enumerate(
        lineups,
        start=1,
    ):
        total_actual = 0.0
        total_salary = 0
        total_projection = 0.0
        matched = 0
        names = []

        for item in _lineup_players(
            lineup
        ):
            (
                name,
                team,
                position,
                salary,
                projection,
            ) = _extract_player(item)

            names.append(name)

            total_salary += salary
            total_projection += projection

            if position == "DST":
                match = actuals[
                    (actuals["team_key"] == team)
                    &
                    (
                        actuals["position_key"]
                        == "DST"
                    )
                ]

            else:
                match = actuals[
                    (
                        actuals["name_key"]
                        == identity_name(name)
                    )
                    &
                    (
                        actuals["team_key"]
                        == team
                    )
                    &
                    (
                        actuals["position_key"]
                        == position
                    )
                ]

            if len(match) == 1:
                total_actual += float(
                    match.iloc[0][
                        "actual_dk_points"
                    ]
                )

                matched += 1

            else:
                unmatched_rows.append(
                    {
                        "lineup_number":
                            lineup_number,
                        "player_name":
                            name,
                        "team":
                            team,
                        "position":
                            position,
                        "matches_found":
                            len(match),
                    }
                )

        scored_rows.append(
            {
                "lineup_number":
                    lineup_number,
                "total_salary":
                    total_salary,
                "burn1_projection":
                    round(
                        total_projection,
                        4,
                    ),
                "actual_dk_points":
                    round(
                        total_actual,
                        2,
                    ),
                "matched_players":
                    matched,
                "players":
                    " | ".join(names),
            }
        )

    return (
        pd.DataFrame(scored_rows),
        pd.DataFrame(unmatched_rows),
    )


def summarize_scores(
    report: pd.DataFrame,
) -> dict:
    if report.empty:
        return {}

    return {
        "best_actual":
            round(
                float(
                    report[
                        "actual_dk_points"
                    ].max()
                ),
                2,
            ),
        "average_actual":
            round(
                float(
                    report[
                        "actual_dk_points"
                    ].mean()
                ),
                2,
            ),
        "median_actual":
            round(
                float(
                    report[
                        "actual_dk_points"
                    ].median()
                ),
                2,
            ),
        "worst_actual":
            round(
                float(
                    report[
                        "actual_dk_points"
                    ].min()
                ),
                2,
            ),
        "lineups_150_plus":
            int(
                (
                    report[
                        "actual_dk_points"
                    ]
                    >= 150
                ).sum()
            ),
        "lineups_160_plus":
            int(
                (
                    report[
                        "actual_dk_points"
                    ]
                    >= 160
                ).sum()
            ),
        "lineups_170_plus":
            int(
                (
                    report[
                        "actual_dk_points"
                    ]
                    >= 170
                ).sum()
            ),
        "average_projection":
            round(
                float(
                    report[
                        "burn1_projection"
                    ].mean()
                ),
                4,
            ),
        "total_projection":
            round(
                float(
                    report[
                        "burn1_projection"
                    ].sum()
                ),
                4,
            ),
        "all_lineups_matched":
            bool(
                (
                    report[
                        "matched_players"
                    ]
                    == 9
                ).all()
            ),
    }
