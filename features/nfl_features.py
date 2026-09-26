import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "position_group",
    "season",
    "week",
    "game_id",
    "gameday",
    "gametime",
    "team",
    "opponent",
    "is_home",
    "is_neutral",
    "team_rest",
    "opponent_rest",
    "rest_differential",
    "team_spread",
    "total_line",
    "team_implied_points",
    "opponent_implied_points",
    "roof",
    "surface",
    "actual_dk_points",
    "actual_fd_points",
]


HISTORY_COLUMNS = [
    # Passing opportunity / production
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_air_yards",
    "passing_first_downs",
    "passing_epa",
    "passing_cpoe",
    "pass_attempt_share_game",
    "pass_epa_per_attempt",

    # Rushing opportunity / production
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "rush_share_game",
    "rush_epa_per_carry",
    "yards_per_carry",

    # Receiving opportunity / production
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "calculated_target_share_game",
    "air_yards_share",
    "wopr",
    "racr",
    "receiving_epa_per_target",
    "yards_per_target",
    "adot",

    # Combined opportunity
    "opportunities",

    # Historical DFS production
    "actual_dk_points",
    "actual_fd_points",
]


PRIOR_ANCHOR_COLUMNS = [
    "attempts",
    "carries",
    "targets",
    "target_share",
    "air_yards_share",
    "wopr",
    "rush_share_game",
    "opportunities",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "passing_cpoe",
    "actual_dk_points",
    "actual_fd_points",
]


def safe_divide(numerator, denominator):
    """
    Divide two Series while returning NaN when the denominator is zero.
    """
    return numerator.div(
        denominator.replace(0, np.nan)
    )


def prepare_source_data(source):
    """
    Normalize chronological order and create same-game measurements.

    These same-game measurements are intermediate historical observations.
    They are not model predictors until they are shifted backward in time.
    """
    data = source.copy()

    data["gameday"] = pd.to_datetime(
        data["gameday"]
    )

    data = (
        data.sort_values(
            [
                "player_id",
                "gameday",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    team_game = data.groupby(
        [
            "game_id",
            "team",
        ],
        sort=False,
    )

    team_carries = team_game[
        "carries"
    ].transform("sum")

    team_attempts = team_game[
        "attempts"
    ].transform("sum")

    team_targets = team_game[
        "targets"
    ].transform("sum")

    data["rush_share_game"] = safe_divide(
        data["carries"],
        team_carries,
    )

    data["pass_attempt_share_game"] = safe_divide(
        data["attempts"],
        team_attempts,
    )

    data["calculated_target_share_game"] = safe_divide(
        data["targets"],
        team_targets,
    )

    data["opportunities"] = (
        data["carries"].fillna(0)
        + data["targets"].fillna(0)
    )

    data["pass_epa_per_attempt"] = safe_divide(
        data["passing_epa"],
        data["attempts"],
    )

    data["rush_epa_per_carry"] = safe_divide(
        data["rushing_epa"],
        data["carries"],
    )

    data["receiving_epa_per_target"] = safe_divide(
        data["receiving_epa"],
        data["targets"],
    )

    data["yards_per_carry"] = safe_divide(
        data["rushing_yards"],
        data["carries"],
    )

    data["yards_per_target"] = safe_divide(
        data["receiving_yards"],
        data["targets"],
    )

    data["adot"] = safe_divide(
        data["receiving_air_yards"],
        data["targets"],
    )

    return data


def build_player_history_features(data):
    """
    Build leakage-safe historical player features.

    Game t sees only observations from games before game t.
    """
    result = {}

    for column in HISTORY_COLUMNS:

        grouped = data.groupby(
            "player_id",
            sort=False,
        )[column]

        result[f"{column}_lag1"] = (
            grouped.shift(1)
        )

        result[f"{column}_roll3"] = (
            grouped.transform(
                lambda series: (
                    series.shift(1)
                    .rolling(
                        window=3,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

        result[f"{column}_ewm"] = (
            grouped.transform(
                lambda series: (
                    series.shift(1)
                    .ewm(
                        halflife=2.5,
                        adjust=False,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

        result[
            f"{column}_season_to_date"
        ] = (
            data.groupby(
                [
                    "player_id",
                    "season",
                ],
                sort=False,
            )[column]
            .transform(
                lambda series: (
                    series.shift(1)
                    .expanding(
                        min_periods=1
                    )
                    .mean()
                )
            )
        )

    return pd.DataFrame(
        result,
        index=data.index,
    )


def add_prior_season_anchors(
    features,
    data,
):
    """
    Attach each player's previous-season averages.

    Example:
    A player's 2024 averages become available to his 2025 rows.
    """
    prior_season = (
        data.groupby(
            [
                "player_id",
                "season",
            ],
            as_index=False,
        )[PRIOR_ANCHOR_COLUMNS]
        .mean()
    )

    prior_season["season"] = (
        prior_season["season"] + 1
    )

    prior_season = prior_season.rename(
        columns={
            column: f"{column}_prior_season"
            for column in PRIOR_ANCHOR_COLUMNS
        }
    )

    return features.merge(
        prior_season,
        on=[
            "player_id",
            "season",
        ],
        how="left",
        validate="many_to_one",
    )


def add_opponent_history(
    features,
    data,
):
    """
    Build leakage-safe positional fantasy production allowed by defenses.

    The target game is shifted out before rolling or season-to-date
    opponent statistics are calculated.
    """
    allowed = (
        data.groupby(
            [
                "game_id",
                "gameday",
                "season",
                "opponent",
                "position",
            ],
            as_index=False,
        )
        .agg(
            dk_points_allowed=(
                "actual_dk_points",
                "sum",
            ),
            fd_points_allowed=(
                "actual_fd_points",
                "sum",
            ),
        )
        .rename(
            columns={
                "opponent": "defense",
            }
        )
    )

    allowed = (
        allowed.sort_values(
            [
                "defense",
                "position",
                "gameday",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    allowed_group = allowed.groupby(
        [
            "defense",
            "position",
        ],
        sort=False,
    )

    allowed[
        "opp_pos_dk_allowed_roll5"
    ] = (
        allowed_group[
            "dk_points_allowed"
        ]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    allowed[
        "opp_pos_fd_allowed_roll5"
    ] = (
        allowed_group[
            "fd_points_allowed"
        ]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    allowed[
        "opp_pos_dk_allowed_season_to_date"
    ] = (
        allowed.groupby(
            [
                "defense",
                "position",
                "season",
            ],
            sort=False,
        )["dk_points_allowed"]
        .transform(
            lambda series: (
                series.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    allowed[
        "opp_pos_fd_allowed_season_to_date"
    ] = (
        allowed.groupby(
            [
                "defense",
                "position",
                "season",
            ],
            sort=False,
        )["fd_points_allowed"]
        .transform(
            lambda series: (
                series.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )
    )

    allowed_for_join = allowed[
        [
            "game_id",
            "defense",
            "position",
            "opp_pos_dk_allowed_roll5",
            "opp_pos_fd_allowed_roll5",
            "opp_pos_dk_allowed_season_to_date",
            "opp_pos_fd_allowed_season_to_date",
        ]
    ]

    features = features.merge(
        allowed_for_join,
        left_on=[
            "game_id",
            "opponent",
            "position",
        ],
        right_on=[
            "game_id",
            "defense",
            "position",
        ],
        how="left",
        validate="many_to_one",
    )

    return features.drop(
        columns=["defense"]
    )


def build_offensive_features(
    source,
    allow_missing_targets=False,
):
    """
    Build the complete V1 offensive feature table.

    Historical training rows contain observed DFS outcomes.

    Live inference rows may intentionally have missing DFS outcomes because
    the target game has not happened yet. Those rows are permitted only when
    allow_missing_targets=True.

    All historical predictors remain shifted backward in time, so an
    inference row can consume prior games without using its own outcome.
    """
    data = source.copy()

    if not allow_missing_targets:
        missing_targets = (
            data[
                [
                    "actual_dk_points",
                    "actual_fd_points",
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        if missing_targets != 0:
            raise ValueError(
                "Historical feature build received "
                f"{missing_targets:,} rows with missing DFS targets."
            )

    data = prepare_source_data(data)

    features = data[
        BASE_COLUMNS
    ].copy()

    features = features.rename(
        columns={
            "actual_dk_points":
                "dk_points_current_rules",
            "actual_fd_points":
                "fd_points_current_rules",
        }
    )

    player_group = data.groupby(
        "player_id",
        sort=False,
    )

    features[
        "days_since_last_game"
    ] = (
        player_group["gameday"]
        .diff()
        .dt.days
    )

    features["prior_games"] = (
        player_group.cumcount()
    )

    features[
        "prior_games_this_season"
    ] = (
        data.groupby(
            [
                "player_id",
                "season",
            ],
            sort=False,
        )
        .cumcount()
    )

    history_features = (
        build_player_history_features(
            data
        )
    )

    features = pd.concat(
        [
            features,
            history_features,
        ],
        axis=1,
    )

    features = add_prior_season_anchors(
        features,
        data,
    )

    features = add_opponent_history(
        features,
        data,
    )

    roof_text = (
        features["roof"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    features["is_indoor"] = (
        roof_text.isin(
            [
                "dome",
                "closed",
                "indoors",
                "indoor",
            ]
        )
    )

    features = (
        features.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
                "player_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return features
