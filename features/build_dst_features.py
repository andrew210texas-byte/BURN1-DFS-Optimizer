from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "dst_scored_2016_2025.parquet"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "dst_features_v1_2016_2025.parquet"
)


ROLLING_STATS = [
    "actual_dk_points",
    "actual_fd_points",
    "points_allowed",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_fumbles_forced",
    "fumble_recovery_opp",
    "def_tds",
    "special_teams_tds",
    "def_safeties",
    "def_pass_defended",
    "def_tackles_for_loss",
]


CURRENT_CONTEXT = [
    "is_home",
    "is_neutral",
    "team_rest",
    "opponent_rest",
    "rest_differential",
    "team_spread",
    "total_line",
    "team_implied_points",
    "opponent_implied_points",
]


def shifted_ewm(series):
    return (
        series
        .shift(1)
        .ewm(
            halflife=2.5,
            adjust=False,
        )
        .mean()
    )


def build_features():
    print()
    print("=" * 80)
    print("DST FEATURE ENGINEERING V1")
    print("=" * 80)

    print(f"\nLoading:\n{INPUT_FILE}")

    data = pd.read_parquet(INPUT_FILE)

    print(f"\nInput rows:    {len(data):,}")
    print(f"Input columns: {len(data.columns):,}")

    required = [
        "season",
        "week",
        "game_id",
        "gameday",
        "team",
        "opponent",
        "actual_dk_points",
        "actual_fd_points",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required DST columns: {missing}"
        )

    data["gameday"] = pd.to_datetime(
        data["gameday"]
    )

    data = (
        data
        .sort_values(
            [
                "team",
                "gameday",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    if data.duplicated(
        ["season", "game_id", "team"]
    ).any():
        raise ValueError(
            "Duplicate DST team-game rows detected."
        )

    for column in ROLLING_STATS:
        if column not in data.columns:
            raise ValueError(
                f"Missing rolling DST source: {column}"
            )

    grouped = data.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    for column in ROLLING_STATS:
        data[f"{column}_lag1"] = (
            grouped[column]
            .shift(1)
        )

        data[f"{column}_roll3"] = (
            grouped[column]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=3,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

        data[f"{column}_ewm"] = (
            grouped[column]
            .transform(shifted_ewm)
        )

        data[
            f"{column}_season_to_date"
        ] = (
            data
            .groupby(
                ["team", "season"],
                sort=False,
            )[column]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .expanding(
                        min_periods=1
                    )
                    .mean()
                )
            )
        )

    # ---------------------------------------------------------
    # OPPONENT OFFENSIVE HISTORY
    #
    # For a defense facing Team X, these features describe what
    # opposing defenses have scored/produced against Team X
    # BEFORE the current game.
    # ---------------------------------------------------------

    opponent_view = data[
        [
            "season",
            "week",
            "game_id",
            "gameday",
            "team",
            "opponent",
            "actual_dk_points",
            "actual_fd_points",
            "def_sacks",
            "def_interceptions",
            "fumble_recovery_opp",
            "points_allowed",
        ]
    ].copy()

    opponent_view = opponent_view.rename(
        columns={
            "team": "defense_team",
            "opponent": "offense_team",
        }
    )

    opponent_view = (
        opponent_view
        .sort_values(
            [
                "offense_team",
                "gameday",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    opponent_group = opponent_view.groupby(
        "offense_team",
        sort=False,
        group_keys=False,
    )

    opponent_sources = [
        "actual_dk_points",
        "actual_fd_points",
        "def_sacks",
        "def_interceptions",
        "fumble_recovery_opp",
        "points_allowed",
    ]

    for column in opponent_sources:
        opponent_view[
            f"opp_allowed_{column}_roll3"
        ] = (
            opponent_group[column]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=3,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

        opponent_view[
            f"opp_allowed_{column}_ewm"
        ] = (
            opponent_group[column]
            .transform(shifted_ewm)
        )

    opponent_feature_columns = [
        column
        for column in opponent_view.columns
        if column.startswith("opp_allowed_")
    ]

    opponent_lookup = opponent_view[
        [
            "season",
            "week",
            "game_id",
            "defense_team",
        ]
        + opponent_feature_columns
    ].rename(
        columns={
            "defense_team": "team",
        }
    )

    data = data.merge(
        opponent_lookup,
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # RECENCY
    # ---------------------------------------------------------

    data["days_since_last_game"] = (
        data
        .groupby("team")["gameday"]
        .diff()
        .dt.days
    )

    data["prior_games"] = (
        data
        .groupby("team")
        .cumcount()
    )

    data["prior_games_this_season"] = (
        data
        .groupby(
            ["team", "season"]
        )
        .cumcount()
    )

    # ---------------------------------------------------------
    # SIMPLE ENVIRONMENT FLAG
    # ---------------------------------------------------------

    if "roof" in data.columns:
        roof = (
            data["roof"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        data["is_indoor"] = (
            roof.str.contains(
                "dome|closed|indoor"
            )
        ).astype(int)
    else:
        data["is_indoor"] = 0

    # ---------------------------------------------------------
    # TARGET NAMES
    # ---------------------------------------------------------

    data["dk_dst_points_current_rules"] = (
        data["actual_dk_points"]
    )

    data["fd_dst_points_current_rules"] = (
        data["actual_fd_points"]
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if len(data) != 5278:
        raise ValueError(
            f"DST row-count validation FAILED: {len(data)}"
        )

    if data.duplicated(
        ["season", "game_id", "team"]
    ).any():
        raise ValueError(
            "DST uniqueness validation FAILED."
        )

    if data[
        "dk_dst_points_current_rules"
    ].isna().any():
        raise ValueError(
            "DK DST target contains nulls."
        )

    if data[
        "fd_dst_points_current_rules"
    ].isna().any():
        raise ValueError(
            "FD DST target contains nulls."
        )

    # First archived appearance for each franchise must not
    # magically contain a prior-game lag.
    first_rows = (
        data
        .sort_values(
            ["team", "gameday", "game_id"]
        )
        .groupby("team")
        .head(1)
    )

    if first_rows[
        "actual_dk_points_lag1"
    ].notna().any():
        raise ValueError(
            "DST temporal leakage validation FAILED."
        )

    feature_columns = [
        column
        for column in data.columns
        if (
            column.endswith(
                (
                    "_lag1",
                    "_roll3",
                    "_ewm",
                    "_season_to_date",
                )
            )
            or column.startswith(
                "opp_allowed_"
            )
            or column in (
                CURRENT_CONTEXT
                + [
                    "days_since_last_game",
                    "prior_games",
                    "prior_games_this_season",
                    "is_indoor",
                ]
            )
        )
    ]

    data = (
        data
        .sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("VALIDATION")
    print("-" * 80)
    print(
        f"Rows:                     {len(data):,} PASSED"
    )
    print(
        "Team-game uniqueness:     PASSED"
    )
    print(
        "Target completeness:      PASSED"
    )
    print(
        "First-game leakage check: PASSED"
    )
    print(
        f"DST predictor candidates: {len(feature_columns):,}"
    )

    print()
    print("FEATURE COVERAGE")
    print("-" * 80)

    for column in [
        "actual_dk_points_roll3",
        "actual_dk_points_ewm",
        "def_sacks_roll3",
        "def_interceptions_roll3",
        "points_allowed_roll3",
        "opp_allowed_actual_dk_points_roll3",
        "opp_allowed_def_sacks_roll3",
        "days_since_last_game",
    ]:
        count = data[column].notna().sum()
        pct = count / len(data) * 100

        print(
            f"{column:<45} "
            f"{count:>5,}/{len(data):,} "
            f"{pct:>6.2f}%"
        )

    print()
    print(f"Saved:\n{OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("DST FEATURE ENGINEERING V1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    build_features()
