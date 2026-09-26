from pathlib import Path

import nflreadpy as nfl
import polars as pl


# Find the folder where this Python file lives.
BASE_DIR = Path(__file__).resolve().parent

# Define where untouched historical NFL datasets will be stored.
RAW_DIR = BASE_DIR / "raw"

# Define where cleaned, standardized datasets will be stored.
PROCESSED_DIR = BASE_DIR / "processed"

# These are the offensive positions used in our DraftKings
# and FanDuel NFL optimizer.
OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE"]

# Historical archive used by the DFS modeling pipeline.
START_SEASON = 2016
END_SEASON = 2025

# Canonical franchise abbreviations used throughout the processed pipeline.
# Raw source files remain untouched. Historical city abbreviations are mapped
# to the current franchise identity so relocations do not split team history.
FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
}


def normalize_franchise_columns(data, columns):
    """Normalize historical team abbreviations to canonical franchise IDs."""

    expressions = []

    for column in columns:
        if column in data.columns:
            expressions.append(
                pl.col(column).replace(FRANCHISE_ALIASES).alias(column)
            )

    if not expressions:
        return data

    return data.with_columns(expressions)


def verify_directories():
    """Verify that the historical NFL data directories exist."""

    print("Historical NFL Data Pipeline")
    print("----------------------------")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Processed data directory: {PROCESSED_DIR}")
    print(f"Raw directory exists: {RAW_DIR.exists()}")
    print(f"Processed directory exists: {PROCESSED_DIR.exists()}")


def save_raw_dataset(data, filename):
    """Save an untouched source dataset as a Parquet file."""

    output_path = RAW_DIR / filename

    data.write_parquet(output_path)

    print(f"Saved raw dataset: {output_path.name}")

    return output_path


def save_processed_dataset(data, filename):
    """Save a cleaned/model-ready dataset as a Parquet file."""

    output_path = PROCESSED_DIR / filename

    data.write_parquet(output_path)

    print(f"Saved processed dataset: {output_path.name}")

    return output_path


def load_offensive_player_stats(season):
    """Load raw player data and return regular-season QB/RB/WR/TE records."""

    print(f"\nLoading {season} NFL player statistics...")

    # Load the complete source dataset before applying DFS-specific filters.
    raw_data = nfl.load_player_stats([season])

    # Preserve the complete source dataset exactly as loaded.
    save_raw_dataset(
        raw_data,
        f"player_stats_{season}.parquet",
    )

    # Create a filtered dataset for our offensive DFS model.
    data = raw_data.filter(pl.col("season_type") == "REG")

    data = data.filter(
        pl.col("position").is_in(OFFENSIVE_POSITIONS)
    )

    # Normalize historical franchise abbreviations after raw preservation.
    data = normalize_franchise_columns(data, ["team"])

    print(f"Loaded {data.height:,} offensive player-game records.")

    return data


def validate_player_game_uniqueness(data):
    """Make sure each player has only one record per NFL game."""

    duplicates = (
        data.group_by(["player_id", "season", "week", "game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height == 0:
        print("Player-game uniqueness check: PASSED")
    else:
        raise ValueError(
            f"Player-game uniqueness check FAILED: "
            f"{duplicates.height} duplicate records found."
        )


def load_team_stats(season):
    """Load raw team data and return regular-season team-game records."""

    print(f"\nLoading {season} NFL team statistics...")

    # Load the complete source team dataset.
    raw_data = nfl.load_team_stats([season])

    # Preserve the complete source dataset before filtering.
    save_raw_dataset(
        raw_data,
        f"team_stats_{season}.parquet",
    )

    # Create the regular-season team dataset used by the DST model.
    data = raw_data.filter(pl.col("season_type") == "REG")

    # Normalize historical franchise abbreviations after raw preservation.
    data = normalize_franchise_columns(data, ["team"])

    print(f"Loaded {data.height:,} team-game records.")

    return data


def validate_team_game_uniqueness(data):
    """Make sure each NFL team has only one record per game."""

    duplicates = (
        data.group_by(["team", "season", "week", "game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height == 0:
        print("Team-game uniqueness check: PASSED")
    else:
        raise ValueError(
            "Team-game uniqueness check FAILED: "
            f"{duplicates.height} duplicate records found."
        )


def load_schedule_data(season):
    """Load raw schedule data and return regular-season games."""

    print(f"\nLoading {season} NFL schedule data...")

    # Load the complete schedule/results/environment dataset.
    raw_data = nfl.load_schedules([season])

    # Preserve the complete source dataset before filtering.
    save_raw_dataset(
        raw_data,
        f"schedules_{season}.parquet",
    )

    # Create the regular-season schedule dataset used by our pipeline.
    data = raw_data.filter(pl.col("game_type") == "REG")

    # Normalize historical franchise abbreviations after raw preservation.
    data = normalize_franchise_columns(
        data,
        ["home_team", "away_team"],
    )

    print(f"Loaded {data.height:,} regular-season games.")

    return data


def validate_schedule_uniqueness(data):
    """Make sure every NFL game appears only once in the schedule data."""

    duplicates = (
        data.group_by(["game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height == 0:
        print("Schedule game uniqueness check: PASSED")
    else:
        raise ValueError(
            "Schedule game uniqueness check FAILED: "
            f"{duplicates.height} duplicate games found."
        )


def validate_team_schedule_matches(team_stats, schedule_data):
    """Make sure every team-game record matches the correct scheduled game."""

    schedule_teams = schedule_data.select(
        ["game_id", "home_team", "away_team"]
    )

    joined = team_stats.join(
        schedule_teams,
        on="game_id",
        how="left",
    )

    unmatched_games = joined.filter(
        pl.col("home_team").is_null()
        | pl.col("away_team").is_null()
    )

    team_mismatches = joined.filter(
        pl.col("home_team").is_not_null()
        & pl.col("away_team").is_not_null()
        & (pl.col("team") != pl.col("home_team"))
        & (pl.col("team") != pl.col("away_team"))
    )

    if unmatched_games.height == 0 and team_mismatches.height == 0:
        print("Team-to-schedule matching check: PASSED")
    else:
        raise ValueError(
            "Team-to-schedule matching check FAILED: "
            f"{unmatched_games.height} unmatched games and "
            f"{team_mismatches.height} team mismatches found."
        )


def build_team_game_context(schedule_data):
    """Convert one schedule row per game into one context row per team."""

    print("\nBuilding team-game context dataset...")

    # Build the away-team perspective of every game.
    away_context = schedule_data.select(
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "gametime",
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.lit(False).alias("is_home"),
            (pl.col("location") == "Neutral").alias("is_neutral"),
            pl.col("away_rest").alias("team_rest"),
            pl.col("home_rest").alias("opponent_rest"),
            (
                pl.col("away_rest") - pl.col("home_rest")
            ).alias("rest_differential"),
            pl.col("away_moneyline").alias("team_moneyline"),
            pl.col("home_moneyline").alias("opponent_moneyline"),
            pl.col("spread_line").alias("team_spread"),
            "total_line",
            (
                (pl.col("total_line") - pl.col("spread_line")) / 2
            ).alias("team_implied_points"),
            (
                (pl.col("total_line") + pl.col("spread_line")) / 2
            ).alias("opponent_implied_points"),
            "roof",
            "surface",
            "temp",
            "wind",
        ]
    )

    # Build the home-team perspective of every game.
    home_context = schedule_data.select(
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "gametime",
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.lit(True).alias("is_home"),
            (pl.col("location") == "Neutral").alias("is_neutral"),
            pl.col("home_rest").alias("team_rest"),
            pl.col("away_rest").alias("opponent_rest"),
            (
                pl.col("home_rest") - pl.col("away_rest")
            ).alias("rest_differential"),
            pl.col("home_moneyline").alias("team_moneyline"),
            pl.col("away_moneyline").alias("opponent_moneyline"),
            (-pl.col("spread_line")).alias("team_spread"),
            "total_line",
            (
                (pl.col("total_line") + pl.col("spread_line")) / 2
            ).alias("team_implied_points"),
            (
                (pl.col("total_line") - pl.col("spread_line")) / 2
            ).alias("opponent_implied_points"),
            "roof",
            "surface",
            "temp",
            "wind",
        ]
    )

    # Stack the two perspectives into one team-game dataset.
    context = pl.concat(
        [away_context, home_context],
        how="vertical",
    )

    # Sort it so the dataset is easy for us to inspect.
    context = context.sort(
        ["season", "week", "game_id", "team"]
    )

    print(f"Built {context.height:,} team-game context records.")

    return context


def validate_team_game_context(context, schedule_data):
    """Validate the structure and implied-point math of the context table."""

    expected_rows = schedule_data.height * 2

    if context.height != expected_rows:
        raise ValueError(
            "Team-game context row-count check FAILED: "
            f"expected {expected_rows}, found {context.height}."
        )

    duplicates = (
        context.group_by(["game_id", "team"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "Team-game context uniqueness check FAILED: "
            f"{duplicates.height} duplicate team-game records found."
        )

    bad_totals = context.filter(
        (
            pl.col("team_implied_points")
            + pl.col("opponent_implied_points")
            - pl.col("total_line")
        ).abs()
        > 0.001
    )

    if bad_totals.height > 0:
        raise ValueError(
            "Implied-points validation FAILED: "
            f"{bad_totals.height} rows do not equal the game total."
        )

    print("Team-game context validation: PASSED")


def build_offensive_player_games(player_stats, team_game_context):
    """Join offensive player-game statistics to pregame team-game context."""

    print("\nBuilding offensive player-game modeling dataset...")

    # We only need the context fields that do not already exist
    # in the player-statistics dataset.
    context_columns = team_game_context.select(
        [
            "game_id",
            "team",
            "gameday",
            "gametime",
            "opponent",
            "is_home",
            "is_neutral",
            "team_rest",
            "opponent_rest",
            "rest_differential",
            "team_moneyline",
            "opponent_moneyline",
            "team_spread",
            "total_line",
            "team_implied_points",
            "opponent_implied_points",
            "roof",
            "surface",
            "temp",
            "wind",
        ]
    )

    # Match every player-game to the context for that player's team
    # in that exact NFL game.
    player_games = player_stats.join(
        context_columns,
        on=["game_id", "team"],
        how="left",
    )

    # Sort by season/week/game/team/player so the output is easy to inspect.
    player_games = player_games.sort(
        ["season", "week", "game_id", "team", "player_id"]
    )

    print(
        f"Built {player_games.height:,} enriched "
        f"offensive player-game records."
    )

    return player_games


def validate_offensive_player_games(player_stats, player_games):
    """Make sure the context join did not lose, add, or duplicate players."""

    if player_games.height != player_stats.height:
        raise ValueError(
            "Offensive player-game row-count check FAILED: "
            f"expected {player_stats.height}, found {player_games.height}."
        )

    duplicates = (
        player_games.group_by(
            ["player_id", "season", "week", "game_id"]
        )
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "Offensive player-game uniqueness check FAILED: "
            f"{duplicates.height} duplicate player-game records found."
        )

    missing_context = player_games.filter(
        pl.col("opponent").is_null()
    )

    if missing_context.height > 0:
        raise ValueError(
            "Offensive player-game context check FAILED: "
            f"{missing_context.height} player-game records "
            f"have no matching context."
        )

    print("Offensive player-game row-count check: PASSED")
    print("Offensive player-game uniqueness check: PASSED")
    print("Offensive player-game context check: PASSED")


def add_offensive_dfs_scoring(player_games):
    """Calculate historical DraftKings and FanDuel points from football stats."""

    print("\nCalculating historical offensive DFS scoring...")

    base_points = (
        pl.col("passing_yards").fill_null(0) * 0.04
        + pl.col("passing_tds").fill_null(0) * 4
        - pl.col("passing_interceptions").fill_null(0)
        + pl.col("rushing_yards").fill_null(0) * 0.1
        + pl.col("rushing_tds").fill_null(0) * 6
        + pl.col("receiving_yards").fill_null(0) * 0.1
        + pl.col("receiving_tds").fill_null(0) * 6
        + (
            pl.col("passing_2pt_conversions").fill_null(0)
            + pl.col("rushing_2pt_conversions").fill_null(0)
            + pl.col("receiving_2pt_conversions").fill_null(0)
        ) * 2
        + pl.col("special_teams_tds").fill_null(0) * 6
        + pl.col("fumble_recovery_tds").fill_null(0) * 6
    )

    dk_points = (
        base_points
        + pl.col("receptions").fill_null(0)
        - pl.col("fumbles_lost_total").fill_null(0)
        + pl.when(pl.col("passing_yards").fill_null(0) >= 300)
        .then(3.0)
        .otherwise(0.0)
        + pl.when(pl.col("rushing_yards").fill_null(0) >= 100)
        .then(3.0)
        .otherwise(0.0)
        + pl.when(pl.col("receiving_yards").fill_null(0) >= 100)
        .then(3.0)
        .otherwise(0.0)
    ).alias("actual_dk_points")

    fd_points = (
        base_points
        + pl.col("receptions").fill_null(0) * 0.5
        - pl.col("fumbles_lost_total").fill_null(0) * 2
        + pl.when(pl.col("passing_yards").fill_null(0) >= 300)
        .then(3.0)
        .otherwise(0.0)
        + pl.when(pl.col("rushing_yards").fill_null(0) >= 100)
        .then(3.0)
        .otherwise(0.0)
        + pl.when(pl.col("receiving_yards").fill_null(0) >= 100)
        .then(3.0)
        .otherwise(0.0)
    ).alias("actual_fd_points")

    scored = player_games.with_columns(
        dk_points,
        fd_points,
    )

    print(
        f"Calculated DraftKings and FanDuel scoring for "
        f"{scored.height:,} offensive player-game records."
    )

    return scored


def validate_offensive_dfs_scoring(player_games, scored_games):
    """Validate that DFS scoring preserves the player-game dataset."""

    if scored_games.height != player_games.height:
        raise ValueError(
            "Offensive DFS scoring row-count check FAILED: "
            f"expected {player_games.height}, found {scored_games.height}."
        )

    missing_scores = scored_games.filter(
        pl.col("actual_dk_points").is_null()
        | pl.col("actual_fd_points").is_null()
    )

    if missing_scores.height > 0:
        raise ValueError(
            "Offensive DFS scoring null check FAILED: "
            f"{missing_scores.height} player-game records have null scores."
        )

    duplicates = (
        scored_games.group_by(
            ["player_id", "season", "week", "game_id"]
        )
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "Offensive DFS scoring uniqueness check FAILED: "
            f"{duplicates.height} duplicate player-game records found."
        )

    print("Offensive DFS scoring row-count check: PASSED")
    print("Offensive DFS scoring null check: PASSED")
    print("Offensive DFS scoring uniqueness check: PASSED")


def load_play_by_play_data(season):
    """Load raw regular-season play-by-play used for exact DST scoring."""

    print(f"\nLoading {season} NFL play-by-play data...")

    raw_data = nfl.load_pbp([season])

    save_raw_dataset(
        raw_data,
        f"play_by_play_{season}.parquet",
    )

    data = raw_data.filter(pl.col("season_type") == "REG")

    # Normalize the team identifiers used by DST scoring and joins.
    data = normalize_franchise_columns(
        data,
        ["posteam", "defteam"],
    )

    print(f"Loaded {data.height:,} regular-season play-by-play records.")

    return data


def build_points_allowed(play_by_play, team_game_context):
    """Reconstruct DFS points allowed from finalized play-by-play score changes."""

    print("\nReconstructing DST points allowed from play-by-play...")

    pbp = play_by_play.with_columns(
        [
            (
                pl.col("posteam_score_post")
                .fill_null(pl.col("posteam_score"))
                .fill_null(0)
                - pl.col("posteam_score").fill_null(0)
            )
            .clip(lower_bound=0)
            .alias("posteam_delta"),
            (
                pl.col("defteam_score_post")
                .fill_null(pl.col("defteam_score"))
                .fill_null(0)
                - pl.col("defteam_score").fill_null(0)
            )
            .clip(lower_bound=0)
            .alias("defteam_delta"),
        ]
    )

    posteam_points = (
        pbp.filter(
            (pl.col("posteam_delta") > 0)
            & pl.col("defteam").is_not_null()
        )
        .select(
            [
                "game_id",
                pl.col("defteam").alias("team"),
                pl.col("posteam_delta").alias("points_allowed"),
            ]
        )
    )

    # Defensive score changes normally do not count against the offense's DST.
    # The exception is a special-teams TD scored against the kicking team.
    special_teams_defteam_points = (
        pbp.filter(
            (pl.col("defteam_delta") == 6)
            & pl.col("posteam").is_not_null()
            & (
                (pl.col("kickoff_attempt") == 1)
                | (pl.col("punt_attempt") == 1)
                | (
                    (pl.col("field_goal_attempt") == 1)
                    & (pl.col("field_goal_result") == "blocked")
                )
            )
        )
        .select(
            [
                "game_id",
                pl.col("posteam").alias("team"),
                pl.col("defteam_delta").alias("points_allowed"),
            ]
        )
    )

    scoring_points = pl.concat(
        [posteam_points, special_teams_defteam_points],
        how="vertical",
    )

    allowed_totals = scoring_points.group_by(
        ["game_id", "team"]
    ).agg(
        pl.col("points_allowed").sum().alias("points_allowed")
    )

    points_allowed = (
        team_game_context.select(["game_id", "team"])
        .join(
            allowed_totals,
            on=["game_id", "team"],
            how="left",
        )
        .with_columns(
            pl.col("points_allowed").fill_null(0).cast(pl.Int64)
        )
        .sort(["game_id", "team"])
    )

    print(
        f"Reconstructed points allowed for "
        f"{points_allowed.height:,} team-game records."
    )

    return points_allowed


def validate_points_allowed(points_allowed, team_game_context):
    """Validate the reconstructed points-allowed dataset."""

    if points_allowed.height != team_game_context.height:
        raise ValueError(
            "DST points-allowed row-count check FAILED: "
            f"expected {team_game_context.height}, "
            f"found {points_allowed.height}."
        )

    duplicates = (
        points_allowed.group_by(["game_id", "team"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "DST points-allowed uniqueness check FAILED: "
            f"{duplicates.height} duplicate team-game records found."
        )

    invalid = points_allowed.filter(pl.col("points_allowed") < 0)

    if invalid.height > 0:
        raise ValueError(
            "DST points-allowed validation FAILED: "
            f"{invalid.height} negative values found."
        )

    print("DST points-allowed row-count check: PASSED")
    print("DST points-allowed uniqueness check: PASSED")
    print("DST points-allowed nonnegative check: PASSED")


def points_allowed_fantasy_points():
    """Return the shared DraftKings/FanDuel points-allowed tier expression."""

    return (
        pl.when(pl.col("points_allowed") == 0)
        .then(10.0)
        .when(pl.col("points_allowed") <= 6)
        .then(7.0)
        .when(pl.col("points_allowed") <= 13)
        .then(4.0)
        .when(pl.col("points_allowed") <= 20)
        .then(1.0)
        .when(pl.col("points_allowed") <= 27)
        .then(0.0)
        .when(pl.col("points_allowed") <= 34)
        .then(-1.0)
        .otherwise(-4.0)
    )


def build_dst_scoring_dataset(
    team_stats,
    team_game_context,
    points_allowed,
    play_by_play,
):
    """Build team DST records and calculate historical DK/FD fantasy points."""

    print("\nBuilding historical DST scoring dataset...")

    context_columns = team_game_context.select(
        [
            "game_id",
            "team",
            "gameday",
            "gametime",
            "opponent",
            "is_home",
            "is_neutral",
            "team_rest",
            "opponent_rest",
            "rest_differential",
            "team_moneyline",
            "opponent_moneyline",
            "team_spread",
            "total_line",
            "team_implied_points",
            "opponent_implied_points",
            "roof",
            "surface",
            "temp",
            "wind",
        ]
    )

    dst = (
        team_stats.join(
            context_columns,
            on=["game_id", "team"],
            how="left",
        )
        .join(
            points_allowed,
            on=["game_id", "team"],
            how="left",
        )
    )

    # Successful defensive PAT/two-point returns are worth two DST points.
    defensive_conversion_returns = (
        play_by_play.filter(
            (
                (pl.col("defensive_two_point_conv") == 1)
                | (pl.col("defensive_extra_point_conv") == 1)
            )
            & pl.col("defteam").is_not_null()
        )
        .group_by(["game_id", pl.col("defteam").alias("team")])
        .len()
        .rename({"len": "defensive_conversion_returns"})
    )

    dst = (
        dst.join(
            defensive_conversion_returns,
            on=["game_id", "team"],
            how="left",
        )
        .with_columns(
            pl.col("defensive_conversion_returns").fill_null(0)
        )
    )

    blocked_kicks = (
        pl.col("def_punt_blocks").fill_null(0)
        + pl.col("def_pat_blocks").fill_null(0)
        + pl.col("def_fg_blocks").fill_null(0)
    )

    # nflverse def_tds already includes non-special-teams defensive TDs,
    # including defensive fumble-return TDs. Add special-teams TDs once.
    return_tds = (
        pl.col("def_tds").fill_null(0)
        + pl.col("special_teams_tds").fill_null(0)
    )

    common_dst_points = (
        pl.col("def_sacks").fill_null(0)
        + pl.col("def_interceptions").fill_null(0) * 2
        + pl.col("fumble_recovery_opp").fill_null(0) * 2
        + pl.col("def_safeties").fill_null(0) * 2
        + blocked_kicks * 2
        + return_tds * 6
        + pl.col("defensive_conversion_returns").fill_null(0) * 2
        + points_allowed_fantasy_points()
    )

    dst = dst.with_columns(
        [
            common_dst_points.alias("actual_dk_points"),
            common_dst_points.alias("actual_fd_points"),
        ]
    )

    print(
        f"Calculated DraftKings and FanDuel DST scoring for "
        f"{dst.height:,} team-game records."
    )

    return dst.sort(["season", "week", "game_id", "team"])


def validate_dst_scoring(team_stats, scored_dst):
    """Validate the completed historical DST scoring dataset."""

    if scored_dst.height != team_stats.height:
        raise ValueError(
            "DST scoring row-count check FAILED: "
            f"expected {team_stats.height}, found {scored_dst.height}."
        )

    duplicates = (
        scored_dst.group_by(["team", "season", "week", "game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicates.height > 0:
        raise ValueError(
            "DST scoring uniqueness check FAILED: "
            f"{duplicates.height} duplicate team-game records found."
        )

    missing = scored_dst.filter(
        pl.col("opponent").is_null()
        | pl.col("points_allowed").is_null()
        | pl.col("actual_dk_points").is_null()
        | pl.col("actual_fd_points").is_null()
    )

    if missing.height > 0:
        raise ValueError(
            "DST scoring null/context check FAILED: "
            f"{missing.height} incomplete team-game records found."
        )

    print("DST scoring row-count check: PASSED")
    print("DST scoring uniqueness check: PASSED")
    print("DST scoring null/context check: PASSED")


if __name__ == "__main__":
    verify_directories()

    seasons = list(range(START_SEASON, END_SEASON + 1))
    season_label = f"{START_SEASON}_{END_SEASON}"

    print(
        f"\nBuilding historical NFL datasets for "
        f"{START_SEASON}-{END_SEASON}..."
    )

    # Load and combine offensive player-game data.
    player_stats = pl.concat(
        [load_offensive_player_stats(season) for season in seasons],
        how="diagonal_relaxed",
    )
    validate_player_game_uniqueness(player_stats)

    # Load and combine team-game data.
    team_stats = pl.concat(
        [load_team_stats(season) for season in seasons],
        how="diagonal_relaxed",
    )
    validate_team_game_uniqueness(team_stats)

    # Load and combine schedule/results/game-environment data.
    schedule_data = pl.concat(
        [load_schedule_data(season) for season in seasons],
        how="diagonal_relaxed",
    )
    validate_schedule_uniqueness(schedule_data)

    # Verify that team statistics connect correctly to scheduled games.
    validate_team_schedule_matches(team_stats, schedule_data)

    # Build and validate one context record per NFL team per game.
    team_game_context = build_team_game_context(schedule_data)
    validate_team_game_context(team_game_context, schedule_data)

    save_processed_dataset(
        team_game_context,
        f"team_game_context_{season_label}.parquet",
    )

    # Join every offensive player-game to its team's game context.
    offensive_player_games = build_offensive_player_games(
        player_stats,
        team_game_context,
    )

    validate_offensive_player_games(
        player_stats,
        offensive_player_games,
    )

    save_processed_dataset(
        offensive_player_games,
        f"offensive_player_games_{season_label}.parquet",
    )

    # Calculate historical DraftKings and FanDuel offensive fantasy points.
    offensive_player_games_scored = add_offensive_dfs_scoring(
        offensive_player_games,
    )

    validate_offensive_dfs_scoring(
        offensive_player_games,
        offensive_player_games_scored,
    )

    save_processed_dataset(
        offensive_player_games_scored,
        f"offensive_player_games_scored_{season_label}.parquet",
    )

    # Load and combine play-by-play for DST scoring.
    play_by_play = pl.concat(
        [load_play_by_play_data(season) for season in seasons],
        how="diagonal_relaxed",
    )

    # Reconstruct the scoreboard points that count against each DST.
    points_allowed = build_points_allowed(
        play_by_play,
        team_game_context,
    )

    validate_points_allowed(
        points_allowed,
        team_game_context,
    )

    # Build, score, validate, and save the historical DST dataset.
    dst_scored = build_dst_scoring_dataset(
        team_stats,
        team_game_context,
        points_allowed,
        play_by_play,
    )

    validate_dst_scoring(
        team_stats,
        dst_scored,
    )

    save_processed_dataset(
        dst_scored,
        f"dst_scored_{season_label}.parquet",
    )