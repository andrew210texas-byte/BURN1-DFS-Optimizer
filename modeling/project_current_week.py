from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = BASE_DIR / "modeling" / "artifacts"
LIVE_DATA_DIR = BASE_DIR / "data" / "live" / "nfl"
OUTPUT_DIR = BASE_DIR / "modeling" / "output"

MANIFEST_PATH = (
    ARTIFACT_DIR
    / "production_model_manifest_v1.json"
)

POSITIONS = ["QB", "RB", "WR", "TE"]
SITES = ["dk", "fd"]


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    feature_columns = manifest["feature_columns"]

    if len(feature_columns) != manifest["feature_count"]:
        raise ValueError(
            "Production manifest feature-count mismatch."
        )

    return manifest, feature_columns


def load_live_features(season, week):
    path = (
        LIVE_DATA_DIR
        / f"offensive_features_{season}_week_{week}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Live feature file not found: {path}"
        )

    data = pd.read_parquet(path)

    return data, path


def validate_live_data(
    data,
    season,
    week,
    feature_columns,
):
    if data.empty:
        raise ValueError(
            "Live feature table is empty."
        )

    if not (
        (data["season"] == season)
        & (data["week"] == week)
    ).all():
        raise ValueError(
            "Live feature table contains rows outside "
            "the requested season/week."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Live feature table is missing production predictors: "
            + ", ".join(missing_features)
        )

    duplicates = data.duplicated(
        [
            "player_id",
            "season",
            "week",
            "game_id",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Live player-game uniqueness check FAILED."
        )

    if data["dk_points_current_rules"].notna().any():
        raise ValueError(
            "Live DK outcomes are unexpectedly known."
        )

    if data["fd_points_current_rules"].notna().any():
        raise ValueError(
            "Live FD outcomes are unexpectedly known."
        )

    numeric = data[feature_columns].select_dtypes(
        include="number"
    )

    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(
            "Production predictor matrix contains infinity."
        )

    print("Live season/week check: PASSED")
    print("Player-game uniqueness: PASSED")
    print(
        f"Production model contract: PASSED "
        f"({len(feature_columns)} predictors)"
    )
    print("Unknown-target isolation: PASSED")
    print("Infinite-value check: PASSED")


def load_model(site, position):
    filename = (
        f"{site}_{position.lower()}_"
        f"projection_model_v1.joblib"
    )

    path = ARTIFACT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Production model not found: {path}"
        )

    return joblib.load(path)


def project_site_position(
    data,
    site,
    position,
    feature_columns,
):
    mask = data["position"].eq(position)

    position_data = data.loc[mask].copy()

    if position_data.empty:
        raise ValueError(
            f"No {position} rows available for {site.upper()}."
        )

    model = load_model(
        site=site,
        position=position,
    )

    matrix = position_data[
        feature_columns
    ].copy()

    predictions = model.predict(matrix)

    if len(predictions) != len(position_data):
        raise ValueError(
            f"{site.upper()} {position} prediction "
            "row-count mismatch."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"{site.upper()} {position} produced "
            "non-finite predictions."
        )

    return position_data.index, predictions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate current-week DraftKings and FanDuel "
            "offensive projections from production models."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    season = args.season
    week = args.week

    print("=" * 80)
    print("STEP 9C - CURRENT-WEEK DFS PROJECTIONS")
    print("=" * 80)

    print(f"\nSeason: {season}")
    print(f"Week:   {week}")

    manifest, feature_columns = load_manifest()

    data, input_path = load_live_features(
        season=season,
        week=week,
    )

    print(f"\nLoaded:")
    print(input_path)
    print(f"Rows: {len(data):,}")

    print("\nValidating live model inputs...")
    validate_live_data(
        data=data,
        season=season,
        week=week,
        feature_columns=feature_columns,
    )

    output = data[
        [
            "player_id",
            "player_name",
            "player_display_name",
            "position",
            "season",
            "week",
            "game_id",
            "gameday",
            "gametime",
            "team",
            "opponent",
            "is_home",
            "is_neutral",
            "team_spread",
            "total_line",
            "team_implied_points",
            "opponent_implied_points",
            "prior_games",
            "prior_games_this_season",
        ]
    ].copy()

    output["dk_projection_v1"] = np.nan
    output["fd_projection_v1"] = np.nan

    print("\nGenerating projections...")

    for site in SITES:
        projection_column = (
            "dk_projection_v1"
            if site == "dk"
            else "fd_projection_v1"
        )

        for position in POSITIONS:
            indices, predictions = project_site_position(
                data=data,
                site=site,
                position=position,
                feature_columns=feature_columns,
            )

            output.loc[
                indices,
                projection_column,
            ] = predictions

            print(
                f"{site.upper()} {position}: "
                f"{len(predictions):,} projections"
            )

    if output[
        [
            "dk_projection_v1",
            "fd_projection_v1",
        ]
    ].isna().any().any():
        raise ValueError(
            "Projection completeness check FAILED."
        )

    if not np.isfinite(
        output[
            [
                "dk_projection_v1",
                "fd_projection_v1",
            ]
        ].to_numpy()
    ).all():
        raise ValueError(
            "Projection finite-value check FAILED."
        )

    print("\nProjection completeness: PASSED")
    print("Projection finite-value check: PASSED")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        OUTPUT_DIR
        / f"projections_{season}_week_{week}_v1.parquet"
    )

    csv_path = (
        OUTPUT_DIR
        / f"projections_{season}_week_{week}_v1.csv"
    )

    output = output.sort_values(
        [
            "gameday",
            "gametime",
            "game_id",
            "team",
            "position",
            "player_display_name",
        ]
    ).reset_index(drop=True)

    output.to_parquet(
        parquet_path,
        index=False,
    )

    output.to_csv(
        csv_path,
        index=False,
    )

    print("\nSaved:")
    print(parquet_path)
    print(csv_path)

    print("\nPROJECTION SUMMARY")
    print("-" * 80)

    for position in POSITIONS:
        position_data = output.loc[
            output["position"].eq(position)
        ]

        print(
            f"{position}: "
            f"{len(position_data):,} players | "
            f"DK mean {position_data['dk_projection_v1'].mean():.2f} | "
            f"DK max {position_data['dk_projection_v1'].max():.2f} | "
            f"FD mean {position_data['fd_projection_v1'].mean():.2f} | "
            f"FD max {position_data['fd_projection_v1'].max():.2f}"
        )

    print("\nTOP 10 DK PROJECTIONS")
    print("-" * 80)

    dk_top = output.nlargest(
        10,
        "dk_projection_v1",
    )[
        [
            "player_display_name",
            "position",
            "team",
            "opponent",
            "dk_projection_v1",
            "prior_games",
            "prior_games_this_season",
        ]
    ]

    print(
        dk_top.to_string(
            index=False,
            formatters={
                "dk_projection_v1": "{:.2f}".format,
            },
        )
    )

    print("\nTOP 10 FD PROJECTIONS")
    print("-" * 80)

    fd_top = output.nlargest(
        10,
        "fd_projection_v1",
    )[
        [
            "player_display_name",
            "position",
            "team",
            "opponent",
            "fd_projection_v1",
            "prior_games",
            "prior_games_this_season",
        ]
    ]

    print(
        fd_top.to_string(
            index=False,
            formatters={
                "fd_projection_v1": "{:.2f}".format,
            },
        )
    )

    print("\n" + "=" * 80)
    print("STEP 9C CURRENT-WEEK PROJECTIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
