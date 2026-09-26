from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl

from features.build_live_features import (
    build_team_game_context,
    completed_game_ids,
    build_target_schedule,
    load_slate_teams,
    normalize_team_series,
)
from data.historical.nfl.historical_loader import (
    build_points_allowed,
    build_dst_scoring_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

HISTORICAL_DST = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "dst_scored_2016_2025.parquet"
)

DST_MANIFEST = (
    ROOT_DIR
    / "modeling"
    / "artifacts"
    / "dst_production_model_manifest_v1.json"
)

ARTIFACT_DIR = ROOT_DIR / "modeling" / "artifacts"
OUTPUT_DIR = ROOT_DIR / "data" / "live" / "nfl"
MODEL_OUTPUT_DIR = ROOT_DIR / "modeling" / "output"

FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "JAC": "JAX",
}

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


def to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.copy()

    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()

    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()

    return pd.DataFrame(frame)


def to_polars(frame):
    if isinstance(frame, pl.DataFrame):
        return frame

    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)

    return pl.DataFrame(frame)


def normalize_polars_team(frame, columns):
    data = to_pandas(frame)

    for column in columns:
        if column in data.columns:
            data[column] = (
                data[column]
                .replace(FRANCHISE_ALIASES)
            )

    return pl.from_pandas(data)


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


def build_feature_frame(data):
    data = data.copy()

    data["gameday"] = pd.to_datetime(
        data["gameday"]
    )

    data = (
        data
        .sort_values(
            ["team", "gameday", "game_id"]
        )
        .reset_index(drop=True)
    )

    grouped = data.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    for column in ROLLING_STATS:
        if column not in data.columns:
            data[column] = np.nan

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
                        3,
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
                        3,
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
        if column.startswith(
            "opp_allowed_"
        )
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

    data["dk_dst_points_current_rules"] = (
        data["actual_dk_points"]
    )

    data["fd_dst_points_current_rules"] = (
        data["actual_fd_points"]
    )

    return data


def main():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--salary-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--slate",
        required=True,
    )

    args = parser.parse_args()

    season = args.season
    week = args.week
    salary_file = args.salary_file
    slate = args.slate

    print()
    print("=" * 80)
    print("LIVE DST FEATURE + PROJECTION PIPELINE")
    print("=" * 80)

    historical = pd.read_parquet(
        HISTORICAL_DST
    )

    historical["team"] = (
        historical["team"]
        .replace(FRANCHISE_ALIASES)
    )

    historical["opponent"] = (
        historical["opponent"]
        .replace(FRANCHISE_ALIASES)
    )

    print(
        f"\nHistorical DST rows: "
        f"{len(historical):,}"
    )

    print("\nLoading current NFL schedule...")

    schedule = to_pandas(
        nfl.load_schedules([season])
    )

    schedule["home_team"] = (
        normalize_team_series(
            schedule["home_team"]
        )
    )

    schedule["away_team"] = (
        normalize_team_series(
            schedule["away_team"]
        )
    )

    slate_site, slate_teams, salary_rows = (
        load_slate_teams(
            salary_file
        )
    )

    completed_ids = completed_game_ids(
        schedule
    )

    target_schedule = build_target_schedule(
        schedule=schedule,
        season=season,
        week=week,
        slate_teams=slate_teams,
        completed_ids=completed_ids,
    )

    target_game_ids = set(
        target_schedule["game_id"]
    )

    context = build_team_game_context(
        schedule
    )

    target_context = context.loc[
        context["game_id"].isin(
            target_game_ids
        )
    ].copy()

    if len(target_context) != (
        len(target_schedule) * 2
    ):
        raise ValueError(
            "Live DST target context "
            "row-count validation FAILED."
        )

    print(
        f"\nTarget DST rows: "
        f"{len(target_context)}"
    )

    print(
        f"Target games: "
        f"{len(target_schedule)}"
    )

    print("\nLoading current team stats...")

    team_stats = nfl.load_team_stats(
        [season]
    )

    team_stats = normalize_polars_team(
        team_stats,
        ["team"],
    )

    if "season_type" in team_stats.columns:
        team_stats = team_stats.filter(
            pl.col("season_type") == "REG"
        )

    team_stats = team_stats.filter(
        pl.col("game_id").is_in(
            list(completed_ids)
        )
        & ~pl.col("game_id").is_in(
            list(target_game_ids)
        )
    )

    print(
        f"Completed current-season "
        f"team-game rows: "
        f"{team_stats.height:,}"
    )

    print("\nLoading current play-by-play...")

    pbp = nfl.load_pbp([season])

    pbp = normalize_polars_team(
        pbp,
        [
            "posteam",
            "defteam",
        ],
    )

    pbp = pbp.filter(
        pl.col("game_id").is_in(
            list(completed_ids)
        )
        & ~pl.col("game_id").is_in(
            list(target_game_ids)
        )
    )

    current_context = context.loc[
        context["game_id"].isin(
            completed_ids
        )
        & ~context["game_id"].isin(
            target_game_ids
        )
    ].copy()

    current_context_pl = pl.from_pandas(
        current_context
    )

    points_allowed = build_points_allowed(
        pbp,
        current_context_pl,
    )

    current_dst = build_dst_scoring_dataset(
        team_stats,
        current_context_pl,
        points_allowed,
        pbp,
    )

    current_dst = to_pandas(
        current_dst
    )

    current_dst["team"] = (
        current_dst["team"]
        .replace(FRANCHISE_ALIASES)
    )

    current_dst["opponent"] = (
        current_dst["opponent"]
        .replace(FRANCHISE_ALIASES)
    )

    print(
        f"\nCurrent completed DST rows: "
        f"{len(current_dst):,}"
    )

    # Target rows deliberately contain no actual results.
    target = target_context.copy()

    historical_columns = historical.columns

    skeleton = pd.DataFrame(
        index=target.index,
        columns=historical_columns,
    )

    for column in historical_columns:
        if column in target.columns:
            skeleton[column] = target[column]

    skeleton["season"] = season
    skeleton["week"] = week
    skeleton["game_id"] = target["game_id"]
    skeleton["gameday"] = target["gameday"]
    skeleton["gametime"] = target["gametime"]
    skeleton["team"] = target["team"]
    skeleton["opponent"] = target["opponent"]

    skeleton["actual_dk_points"] = np.nan
    skeleton["actual_fd_points"] = np.nan

    combined = pd.concat(
        [
            historical,
            current_dst.reindex(
                columns=historical_columns
            ),
            skeleton,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "season",
            "game_id",
            "team",
        ],
        keep="last",
    )

    features = build_feature_frame(
        combined
    )

    live = features.loc[
        (features["season"] == season)
        & (features["week"] == week)
        & features["game_id"].isin(
            target_game_ids
        )
    ].copy()

    if len(live) != len(target_context):
        raise ValueError(
            "Live DST feature row-count "
            "validation FAILED."
        )

    if live.duplicated(
        ["season", "game_id", "team"]
    ).any():
        raise ValueError(
            "Live DST uniqueness FAILED."
        )

    if live[
        "actual_dk_points"
    ].notna().any():
        raise ValueError(
            "Live DST target leakage detected."
        )

    with open(
        DST_MANIFEST,
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    feature_columns = manifest[
        "feature_columns"
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in live.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing live DST model features: "
            + ", ".join(missing_features)
        )

    X = live[
        feature_columns
    ].copy()

    for column in feature_columns:
        if pd.api.types.is_bool_dtype(
            X[column]
        ):
            X[column] = (
                X[column].astype(int)
            )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"\nDST production contract: "
        f"{len(feature_columns)} predictors PASSED"
    )

    for site in ["DK", "FD"]:
        model_info = manifest[
            "models"
        ][f"{site}_DST"]

        model = joblib.load(
            ARTIFACT_DIR
            / model_info["artifact"]
        )

        prediction = model.predict(X)

        if not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                f"{site} DST produced "
                "non-finite predictions."
            )

        live[
            f"{site.lower()}_projection"
        ] = prediction

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_output = (
        OUTPUT_DIR
        / (
            f"dst_features_{season}_"
            f"week_{week}_{slate}.parquet"
        )
    )

    projection_output = (
        MODEL_OUTPUT_DIR
        / (
            f"dst_projections_{season}_"
            f"week_{week}_{slate}_v1.csv"
        )
    )

    live.to_parquet(
        feature_output,
        index=False,
    )

    projection_columns = [
        "season",
        "week",
        "game_id",
        "gameday",
        "gametime",
        "team",
        "opponent",
        "team_spread",
        "total_line",
        "team_implied_points",
        "opponent_implied_points",
        "dk_projection",
        "fd_projection",
    ]

    projections = (
        live[
            projection_columns
        ]
        .sort_values(
            "dk_projection",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    projections.to_csv(
        projection_output,
        index=False,
    )

    print()
    print("=" * 80)
    print("LIVE DST PROJECTIONS")
    print("=" * 80)

    print(
        projections[
            [
                "team",
                "opponent",
                "dk_projection",
                "fd_projection",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print()
    print(
        f"Rows:       {len(projections)}"
    )
    print(
        f"Games:      "
        f"{projections['game_id'].nunique()}"
    )
    print(
        f"Teams:      "
        f"{projections['team'].nunique()}"
    )

    if len(projections) != 26:
        raise ValueError(
            "Expected exactly 26 DST projections "
            "for the 13-game slate."
        )

    if (
        projections["team"].nunique()
        != 26
    ):
        raise ValueError(
            "Expected 26 unique DST teams."
        )

    print()
    print(f"Features saved:\n{feature_output}")
    print()
    print(
        f"Projections saved:\n"
        f"{projection_output}"
    )

    print()
    print("Live DST leakage check: PASSED")
    print("Live DST row count:     PASSED")
    print("Live DST uniqueness:    PASSED")
    print("DST model contract:     PASSED")

    print()
    print("=" * 80)
    print(
        "LIVE DST FEATURE + "
        "PROJECTION PIPELINE COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
