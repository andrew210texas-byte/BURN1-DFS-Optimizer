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
            f"Team-game uniqueness check FAILED: "
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
            f"Schedule game uniqueness check FAILED: "
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

    # A left join should preserve the exact number of offensive player rows.
    if player_games.height != player_stats.height:
        raise ValueError(
            "Offensive player-game row-count check FAILED: "
            f"expected {player_stats.height}, found {player_games.height}."
        )

    # Each player should still have only one row for each NFL game.
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

    # If the join worked, every player row should have an opponent.
    # The opponent field comes directly from our team-game context table,
    # so a null opponent would indicate that the context join failed.
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


if __name__ == "__main__":
    verify_directories()

    # Load, preserve, and validate individual offensive player-game data.
    player_stats = load_offensive_player_stats(2025)
    validate_player_game_uniqueness(player_stats)

    # Load, preserve, and validate team-game data for defense/special teams.
    team_stats = load_team_stats(2025)
    validate_team_game_uniqueness(team_stats)

    # Load, preserve, and validate schedule/results/game-environment data.
    schedule_data = load_schedule_data(2025)
    validate_schedule_uniqueness(schedule_data)

    # Verify that team statistics connect correctly to scheduled games.
    validate_team_schedule_matches(team_stats, schedule_data)

    # Build and validate one pregame-context record per NFL team per game.
    team_game_context = build_team_game_context(schedule_data)
    validate_team_game_context(team_game_context, schedule_data)

    # Save the processed team-game context dataset.
    save_processed_dataset(
        team_game_context,
        "team_game_context_2025.parquet",
    )

    # Join every offensive player-game to its team's game context.
    offensive_player_games = build_offensive_player_games(
        player_stats,
        team_game_context,
    )

    # Verify that the join preserved all 6,037 player-game records
    # without creating duplicates or leaving unmatched context.
    validate_offensive_player_games(
        player_stats,
        offensive_player_games,
    )

    # Save the completed Step 5 player modeling dataset.
    save_processed_dataset(
        offensive_player_games,
        "offensive_player_games_2025.parquet",
    )