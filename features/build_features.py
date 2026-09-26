from pathlib import Path

import pandas as pd

from nfl_features import build_offensive_features


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


print()
print("=" * 80)
print("STEP 9A - SHARED HISTORICAL FEATURE BUILD")
print("=" * 80)

print(f"\nLoading:\n{INPUT_FILE}")

source = pd.read_parquet(
    INPUT_FILE
)

print(
    f"\nLoaded {len(source):,} "
    "offensive player-game records."
)

features = build_offensive_features(
    source
)


print("\nRunning feature validations...")

if len(features) != len(source):
    raise ValueError(
        "Feature row-count validation FAILED: "
        f"expected {len(source):,}, "
        f"found {len(features):,}."
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
        f"{duplicate_count:,} duplicates found."
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
        f"{target_nulls:,} missing targets."
    )

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
        "Temporal leakage validation FAILED."
    )

print("Feature row-count check: PASSED")
print("Feature uniqueness check: PASSED")
print("DFS target null check: PASSED")
print("First-game temporal leakage check: PASSED")


features.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print(f"\nSaved:\n{OUTPUT_FILE}")

print()
print(
    f"Rows:    {len(features):,}"
)

print(
    f"Columns: {len(features.columns):,}"
)

print()
print("=" * 80)
print("STEP 9A SHARED FEATURE BUILD COMPLETE")
print("=" * 80)
