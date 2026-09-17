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

    # Download/load player statistics for the requested NFL season.
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


if __name__ == "__main__":
    verify_directories()

    player_stats = load_offensive_player_stats(2025)

    validate_player_game_uniqueness(player_stats)