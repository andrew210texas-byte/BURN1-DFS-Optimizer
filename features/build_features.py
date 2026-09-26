from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_player_games_scored_2016_2025.parquet"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_features_v1_2016_2025.parquet"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_divide(numerator, denominator):
    """Divide two Series while returning NaN when the denominator is zero."""
    return numerator.div(denominator.replace(0, np.nan))


def build_player_history_features(data, columns):
    """
    Build leakage-safe player-history features.

    Every feature is shifted by one game before it reaches the target row.
    Therefore, game t can only see games that happened before game t.
    """
    result = {}

    for column in columns:
        grouped = data.groupby("player_id", sort=False)[column]

        # Most recent previous game.
        result[f"{column}_lag1"] = grouped.shift(1)

        # Average of the player's previous three appearances.
        result[f"{column}_roll3"] = grouped.transform(
            lambda series: (
                series.shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )
        )

        # Exponentially weighted history.
        # Recent games matter more than older games.
        result[f"{column}_ewm"] = grouped.transform(
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

        # Current-season average using ONLY games before the target game.
        result[f"{column}_season_to_date"] = (
            data.groupby(
                ["player_id", "season"],
                sort=False,
            )[column]
            .transform(
                lambda series: (
                    series.shift(1)
                    .expanding(min_periods=1)
                    .mean()
                )
            )
        )

    return pd.DataFrame(result, index=data.index)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("STEP 7 - OFFENSIVE FEATURE ENGINEERING V1")
print("=" * 80)

print(f"\nLoading:\n{INPUT_FILE}")

data = pd.read_parquet(INPUT_FILE)

print(f"\nLoaded {len(data):,} offensive player-game records.")
print(f"Source columns: {len(data.columns)}")


# ---------------------------------------------------------------------
# SORT CHRONOLOGICALLY
# ---------------------------------------------------------------------

data["gameday"] = pd.to_datetime(data["gameday"])

data = (
    data.sort_values(
        ["player_id", "gameday", "game_id"],
        kind="stable",
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# DERIVE SAME-GAME HISTORICAL MEASUREMENTS
#
# IMPORTANT:
# These measurements are NOT used on their own as predictors.
# They become predictors only after lagging/rolling them backward in time.
# ---------------------------------------------------------------------

team_game = data.groupby(
    ["game_id", "team"],
    sort=False,
)

team_carries = team_game["carries"].transform("sum")
team_attempts = team_game["attempts"].transform("sum")
team_targets = team_game["targets"].transform("sum")

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

# Carries plus targets represents basic fantasy opportunity.
data["opportunities"] = (
    data["carries"].fillna(0)
    + data["targets"].fillna(0)
)

# Position-relevant efficiency rates.
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


# ---------------------------------------------------------------------
# BASE TARGET ROW
#
# These are either identifiers, genuinely current-game context, or labels.
# Same-game football performance statistics are NOT included here.
# ---------------------------------------------------------------------

base_columns = [
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

features = data[base_columns].copy()

# Make target semantics explicit.
features = features.rename(
    columns={
        "actual_dk_points": "dk_points_current_rules",
        "actual_fd_points": "fd_points_current_rules",
    }
)


# ---------------------------------------------------------------------
# RECENCY / SAMPLE-SIZE FEATURES
# ---------------------------------------------------------------------

player_group = data.groupby(
    "player_id",
    sort=False,
)

features["days_since_last_game"] = (
    player_group["gameday"]
    .diff()
    .dt.days
)

features["prior_games"] = player_group.cumcount()

features["prior_games_this_season"] = (
    data.groupby(
        ["player_id", "season"],
        sort=False,
    )
    .cumcount()
)


# ---------------------------------------------------------------------
# PLAYER HISTORY FEATURES
#
# Opportunity
# Production
# Efficiency
#
# All are shifted to t-1 before use.
# ---------------------------------------------------------------------

history_columns = [
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

history_features = build_player_history_features(
    data,
    history_columns,
)

features = pd.concat(
    [features, history_features],
    axis=1,
)


# ---------------------------------------------------------------------
# PRIOR-SEASON PLAYER ANCHORS
#
# These help early-season predictions without using future games from
# the current season.
# ---------------------------------------------------------------------

prior_anchor_columns = [
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

prior_season = (
    data.groupby(
        ["player_id", "season"],
        as_index=False,
    )[prior_anchor_columns]
    .mean()
)

# A player's 2024 averages become available to his 2025 rows.
prior_season["season"] = prior_season["season"] + 1

prior_season = prior_season.rename(
    columns={
        column: f"{column}_prior_season"
        for column in prior_anchor_columns
    }
)

features = features.merge(
    prior_season,
    on=["player_id", "season"],
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------------------
# OPPONENT POSITIONAL DFS HISTORY
#
# Example:
# How many DK points had a defense allowed to WRs BEFORE today's game?
#
# The current game is shifted out before the rolling value is calculated.
# ---------------------------------------------------------------------

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
    ["defense", "position"],
    sort=False,
)

allowed["opp_pos_dk_allowed_roll5"] = (
    allowed_group["dk_points_allowed"]
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

allowed["opp_pos_fd_allowed_roll5"] = (
    allowed_group["fd_points_allowed"]
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

allowed["opp_pos_dk_allowed_season_to_date"] = (
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
            .expanding(min_periods=1)
            .mean()
        )
    )
)

allowed["opp_pos_fd_allowed_season_to_date"] = (
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
            .expanding(min_periods=1)
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

features = features.drop(
    columns=["defense"]
)


# ---------------------------------------------------------------------
# SIMPLE ENVIRONMENT FLAGS
#
# We keep roof/surface because those are structurally known game context.
# Temperature and wind are NOT included in V1 yet because our historical
# source does not establish a consistent pre-lock forecast snapshot.
# ---------------------------------------------------------------------

roof_text = (
    features["roof"]
    .fillna("")
    .astype(str)
    .str.lower()
)

features["is_indoor"] = roof_text.isin(
    [
        "dome",
        "closed",
        "indoors",
        "indoor",
    ]
)


# ---------------------------------------------------------------------
# FINAL SORT
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

print("\nRunning Step 7 validations...")

if len(features) != len(data):
    raise ValueError(
        "Feature row-count validation FAILED: "
        f"expected {len(data):,}, found {len(features):,}."
    )

duplicate_count = (
    features.duplicated(
        subset=[
            "player_id",
            "season",
            "week",
            "game_id",
        ]
    )
    .sum()
)

if duplicate_count != 0:
    raise ValueError(
        "Feature uniqueness validation FAILED: "
        f"{duplicate_count:,} duplicate player-game rows found."
    )

target_nulls = (
    features[
        [
            "dk_points_current_rules",
            "fd_points_current_rules",
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)

if target_nulls != 0:
    raise ValueError(
        "Target validation FAILED: "
        f"{target_nulls:,} rows have missing DFS targets."
    )

# The first archived game for each player must not magically contain
# a previous-game target value.
first_games = (
    features["prior_games"] == 0
)

leaked_first_game_dk = (
    features.loc[
        first_games,
        "actual_dk_points_lag1",
    ]
    .notna()
    .sum()
)

if leaked_first_game_dk != 0:
    raise ValueError(
        "Temporal leakage validation FAILED: "
        "first archived player-games contain lagged DK values."
    )

print("Feature row-count check: PASSED")
print("Feature uniqueness check: PASSED")
print("DFS target null check: PASSED")
print("First-game temporal leakage check: PASSED")


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

features.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print(f"\nSaved:\n{OUTPUT_FILE}")

print("\nFeature dataset summary:")
print(f"Rows:    {len(features):,}")
print(f"Columns: {len(features.columns):,}")

print(
    "Players with prior-season anchors: "
    f"{features['actual_dk_points_prior_season'].notna().sum():,}"
)

print(
    "Rows with opponent positional roll-5 history: "
    f"{features['opp_pos_dk_allowed_roll5'].notna().sum():,}"
)

print(
    "Rows with at least one prior archived game: "
    f"{(features['prior_games'] > 0).sum():,}"
)

print("\nTarget columns:")
print("  dk_points_current_rules")
print("  fd_points_current_rules")

print("\nIMPORTANT:")
print(
    "Same-game football production is not used as a predictor. "
    "Historical performance reaches each row only through lagged, "
    "rolling, EWMA, season-to-date, prior-season, or opponent-history "
    "features."
)

print("\n" + "=" * 80)
print("STEP 7 FEATURE ENGINEERING V1 COMPLETE")
print("=" * 80)
