from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features.nfl_features import build_offensive_features


SOURCE_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_player_games_scored_2016_2025.parquet"
)

FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_features_v1_2016_2025.parquet"
)

MANIFEST_PATH = (
    BASE_DIR
    / "modeling"
    / "artifacts"
    / "production_model_manifest_v1.json"
)

TARGET_SEASON = 2025
TARGET_WEEK = 18

PERFORMANCE_COLUMNS_TO_NULL = [
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
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
    "racr",
    "actual_dk_points",
    "actual_fd_points",
]


def main():
    print("=" * 80)
    print("STEP 9B - LIVE FEATURE PARITY TEST")
    print("=" * 80)

    source = pd.read_parquet(SOURCE_PATH)
    reference = pd.read_parquet(FEATURE_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    feature_columns = manifest["feature_columns"]

    if len(feature_columns) != manifest["feature_count"]:
        raise ValueError(
            "Production manifest feature-count mismatch."
        )

    print(f"\nProduction predictors: {len(feature_columns)}")
    print(
        f"Fake live target: {TARGET_SEASON} Week {TARGET_WEEK}"
    )

    history_mask = (
        (source["season"] < TARGET_SEASON)
        | (
            (source["season"] == TARGET_SEASON)
            & (source["week"] < TARGET_WEEK)
        )
    )

    target_mask = (
        (source["season"] == TARGET_SEASON)
        & (source["week"] == TARGET_WEEK)
    )

    history = source.loc[history_mask].copy()
    target = source.loc[target_mask].copy()

    if target.empty:
        raise ValueError(
            "No target rows found for the fake live week."
        )

    print(f"Historical rows before target: {len(history):,}")
    print(f"Fake live target rows:        {len(target):,}")

    missing_null_columns = [
        column
        for column in PERFORMANCE_COLUMNS_TO_NULL
        if column not in target.columns
    ]

    if missing_null_columns:
        raise ValueError(
            "Expected performance columns are missing: "
            + ", ".join(missing_null_columns)
        )

    target[PERFORMANCE_COLUMNS_TO_NULL] = np.nan

    if target["actual_dk_points"].notna().any():
        raise ValueError(
            "Fake live DK outcomes were not fully erased."
        )

    if target["actual_fd_points"].notna().any():
        raise ValueError(
            "Fake live FD outcomes were not fully erased."
        )

    live_source = pd.concat(
        [history, target],
        ignore_index=True,
        sort=False,
    )

    live_features = build_offensive_features(
        live_source,
        allow_missing_targets=True,
    )

    live_target = live_features.loc[
        (live_features["season"] == TARGET_SEASON)
        & (live_features["week"] == TARGET_WEEK)
    ].copy()

    reference_target = reference.loc[
        (reference["season"] == TARGET_SEASON)
        & (reference["week"] == TARGET_WEEK)
    ].copy()

    key_columns = [
        "player_id",
        "season",
        "week",
        "game_id",
    ]

    live_target = live_target.sort_values(
        key_columns
    ).reset_index(drop=True)

    reference_target = reference_target.sort_values(
        key_columns
    ).reset_index(drop=True)

    if len(live_target) != len(reference_target):
        raise ValueError(
            "Live target row-count parity FAILED: "
            f"{len(live_target):,} live vs "
            f"{len(reference_target):,} reference."
        )

    print(
        f"\nTarget row-count parity: PASSED "
        f"({len(live_target):,} rows)"
    )

    if not live_target[key_columns].equals(
        reference_target[key_columns]
    ):
        raise ValueError(
            "Player-game key parity FAILED."
        )

    print("Player-game key parity: PASSED")

    missing_live_features = [
        column
        for column in feature_columns
        if column not in live_target.columns
    ]

    if missing_live_features:
        raise ValueError(
            "Live feature table is missing production predictors: "
            + ", ".join(missing_live_features)
        )

    live_matrix = (
        live_target[feature_columns]
        .astype(float)
        .to_numpy()
    )

    reference_matrix = (
        reference_target[feature_columns]
        .astype(float)
        .to_numpy()
    )

    matches = np.isclose(
        live_matrix,
        reference_matrix,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    if not matches.all():
        bad_rows, bad_columns = np.where(~matches)

        first_bad_row = bad_rows[0]
        first_bad_column = bad_columns[0]

        feature_name = feature_columns[first_bad_column]

        player_id = live_target.iloc[first_bad_row][
            "player_id"
        ]

        live_value = live_matrix[
            first_bad_row,
            first_bad_column,
        ]

        reference_value = reference_matrix[
            first_bad_row,
            first_bad_column,
        ]

        raise ValueError(
            "Live predictor parity FAILED.\n"
            f"Player: {player_id}\n"
            f"Feature: {feature_name}\n"
            f"Live value: {live_value}\n"
            f"Reference value: {reference_value}"
        )

    print(
        f"Production predictor parity: PASSED "
        f"({len(feature_columns)} predictors)"
    )

    if live_target["dk_points_current_rules"].notna().any():
        raise ValueError(
            "Live DK target isolation FAILED."
        )

    if live_target["fd_points_current_rules"].notna().any():
        raise ValueError(
            "Live FD target isolation FAILED."
        )

    print("Unknown-target isolation: PASSED")

    print("\n" + "=" * 80)
    print("STEP 9B LIVE FEATURE PARITY TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
