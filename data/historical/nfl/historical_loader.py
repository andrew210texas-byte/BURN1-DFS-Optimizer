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


def load_offensive_player_stats(season):
    """Load regular-season QB, RB, WR, and TE player-game statistics."""

    print(f"\nLoading {season} NFL player statistics...")

    # Load player statistics for the requested NFL season.
    data = nfl.load_player_stats([season])

    # Keep only regular-season games.
    data = data.filter(pl.col("season_type") == "REG")

    # Keep only the offensive positions used by our DFS player model.
    data = data.filter(pl.col("position").is_in(OFFENSIVE_POSITIONS))

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
    """Load regular-season team-game statistics for the DST model."""

    print(f"\nLoading {season} NFL team statistics...")

    # Load team-level statistics for the requested NFL season.
    data = nfl.load_team_stats([season])

    # Keep only regular-season games.
    data = data.filter(pl.col("season_type") == "REG")

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
    """Load regular-season schedule, results, and game-environment data."""

    print(f"\nLoading {season} NFL schedule data...")

    # Load schedule and game information for the requested season.
    data = nfl.load_schedules([season])

    # Keep only regular-season games.
    data = data.filter(pl.col("game_type") == "REG")

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


if __name__ == "__main__":
    verify_directories()

    # Load and validate individual offensive player-game data.
    player_stats = load_offensive_player_stats(2025)
    validate_player_game_uniqueness(player_stats)

    # Load and validate team-game data for defense/special teams.
    team_stats = load_team_stats(2025)
    validate_team_game_uniqueness(team_stats)

    # Load and validate schedule/results/game-environment data.
    schedule_data = load_schedule_data(2025)
    validate_schedule_uniqueness(schedule_data)

    # Verify that team statistics connect correctly to scheduled games.
    validate_team_schedule_matches(team_stats, schedule_data)