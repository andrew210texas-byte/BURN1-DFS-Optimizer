from pathlib import Path
import argparse
import json
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features.nfl_features import build_offensive_features


HISTORICAL_SOURCE_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_player_games_scored_2016_2025.parquet"
)

MANIFEST_PATH = (
    BASE_DIR
    / "modeling"
    / "artifacts"
    / "production_model_manifest_v1.json"
)

OUTPUT_DIR = BASE_DIR / "data" / "live" / "nfl"

OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE"]

FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
}

CONTEXT_COLUMNS = [
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

SCORING_COLUMNS = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "special_teams_tds",
    "fumble_recovery_tds",
    "receptions",
    "fumbles_lost_total",
]


def to_pandas(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    raise TypeError(
        f"Unsupported dataframe type: {type(data).__name__}"
    )


def normalize_team_series(series):
    return series.replace(FRANCHISE_ALIASES)


def build_team_game_context(schedule):
    schedule = schedule.copy()

    schedule["home_team"] = normalize_team_series(
        schedule["home_team"]
    )
    schedule["away_team"] = normalize_team_series(
        schedule["away_team"]
    )

    away = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "season": schedule["season"],
            "week": schedule["week"],
            "gameday": schedule["gameday"],
            "gametime": schedule["gametime"],
            "team": schedule["away_team"],
            "opponent": schedule["home_team"],
            "is_home": False,
            "is_neutral": schedule["location"].eq("Neutral"),
            "team_rest": schedule["away_rest"],
            "opponent_rest": schedule["home_rest"],
            "rest_differential": (
                schedule["away_rest"]
                - schedule["home_rest"]
            ),
            "team_moneyline": schedule["away_moneyline"],
            "opponent_moneyline": schedule["home_moneyline"],
            "team_spread": schedule["spread_line"],
            "total_line": schedule["total_line"],
            "team_implied_points": (
                schedule["total_line"]
                - schedule["spread_line"]
            ) / 2,
            "opponent_implied_points": (
                schedule["total_line"]
                + schedule["spread_line"]
            ) / 2,
            "roof": schedule["roof"],
            "surface": schedule["surface"],
            "temp": (
                schedule["temp"]
                if "temp" in schedule.columns
                else np.nan
            ),
            "wind": (
                schedule["wind"]
                if "wind" in schedule.columns
                else np.nan
            ),
        }
    )

    home = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "season": schedule["season"],
            "week": schedule["week"],
            "gameday": schedule["gameday"],
            "gametime": schedule["gametime"],
            "team": schedule["home_team"],
            "opponent": schedule["away_team"],
            "is_home": True,
            "is_neutral": schedule["location"].eq("Neutral"),
            "team_rest": schedule["home_rest"],
            "opponent_rest": schedule["away_rest"],
            "rest_differential": (
                schedule["home_rest"]
                - schedule["away_rest"]
            ),
            "team_moneyline": schedule["home_moneyline"],
            "opponent_moneyline": schedule["away_moneyline"],
            "team_spread": -schedule["spread_line"],
            "total_line": schedule["total_line"],
            "team_implied_points": (
                schedule["total_line"]
                + schedule["spread_line"]
            ) / 2,
            "opponent_implied_points": (
                schedule["total_line"]
                - schedule["spread_line"]
            ) / 2,
            "roof": schedule["roof"],
            "surface": schedule["surface"],
            "temp": (
                schedule["temp"]
                if "temp" in schedule.columns
                else np.nan
            ),
            "wind": (
                schedule["wind"]
                if "wind" in schedule.columns
                else np.nan
            ),
        }
    )

    context = pd.concat(
        [away, home],
        ignore_index=True,
    )

    context["gameday"] = pd.to_datetime(
        context["gameday"]
    )

    duplicates = context.duplicated(
        ["game_id", "team"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Team-game context uniqueness FAILED."
        )

    return context.sort_values(
        ["season", "week", "game_id", "team"]
    ).reset_index(drop=True)


def add_current_rules_scoring(data):
    data = data.copy()

    missing = [
        column
        for column in SCORING_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "2026 player statistics are missing scoring columns: "
            + ", ".join(missing)
        )

    def zero(column):
        return pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(0)

    base = (
        zero("passing_yards") * 0.04
        + zero("passing_tds") * 4
        - zero("passing_interceptions")
        + zero("rushing_yards") * 0.1
        + zero("rushing_tds") * 6
        + zero("receiving_yards") * 0.1
        + zero("receiving_tds") * 6
        + (
            zero("passing_2pt_conversions")
            + zero("rushing_2pt_conversions")
            + zero("receiving_2pt_conversions")
        ) * 2
        + zero("special_teams_tds") * 6
        + zero("fumble_recovery_tds") * 6
    )

    bonuses = (
        (zero("passing_yards") >= 300).astype(float) * 3
        + (zero("rushing_yards") >= 100).astype(float) * 3
        + (zero("receiving_yards") >= 100).astype(float) * 3
    )

    data["actual_dk_points"] = (
        base
        + zero("receptions")
        - zero("fumbles_lost_total")
        + bonuses
    )

    data["actual_fd_points"] = (
        base
        + zero("receptions") * 0.5
        - zero("fumbles_lost_total") * 2
        + bonuses
    )

    return data


def attach_schedule_context(player_stats, context):
    data = player_stats.copy()

    data["team"] = normalize_team_series(
        data["team"]
    )

    payload_columns = [
        column
        for column in CONTEXT_COLUMNS
        if column not in ["game_id", "team"]
    ]

    removable = [
        column
        for column in payload_columns
        if column in data.columns
    ]

    if removable:
        data = data.drop(columns=removable)

    joined = data.merge(
        context[CONTEXT_COLUMNS],
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    missing_context = joined["opponent"].isna()

    if missing_context.any():
        sample = joined.loc[
            missing_context,
            ["game_id", "team", "player_id"],
        ].head(10)

        raise ValueError(
            "Player-to-schedule context join FAILED.\n"
            + sample.to_string(index=False)
        )

    return joined


def build_target_roster(
    rosters,
    target_context,
    season,
    week,
    historical_columns,
):
    roster = rosters.copy()

    if "week" not in roster.columns:
        raise ValueError(
            "Roster source has no week column."
        )

    roster = roster.loc[
        roster["week"].notna()
        & (roster["week"] < week)
    ].copy()

    if roster.empty:
        raise ValueError(
            "No roster snapshot exists before target week."
        )

    latest_roster_week = int(roster["week"].max())

    roster = roster.loc[
        roster["week"] == latest_roster_week
    ].copy()

    roster["team"] = normalize_team_series(
        roster["team"]
    )

    roster = roster.loc[
        roster["position"].isin(OFFENSIVE_POSITIONS)
        & roster["gsis_id"].notna()
        & roster["team"].isin(
            target_context["team"].unique()
        )
    ].copy()

    roster = roster.drop_duplicates(
        ["gsis_id", "team"],
        keep="last",
    )

    multi_team = (
        roster.groupby("gsis_id")["team"]
        .nunique()
        .loc[lambda values: values > 1]
    )

    if not multi_team.empty:
        raise ValueError(
            "Roster contains players assigned to multiple "
            "teams in the same latest snapshot: "
            + ", ".join(multi_team.index.astype(str))
        )

    roster = roster.drop_duplicates(
        ["gsis_id"],
        keep="last",
    )

    target = roster.merge(
        target_context[
            [
                "game_id",
                "season",
                "week",
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
        ],
        on="team",
        how="left",
        validate="many_to_one",
    )

    if target["game_id"].isna().any():
        raise ValueError(
            "Target roster contains a team with no target-week game."
        )

    skeleton = pd.DataFrame(
        index=target.index,
        columns=historical_columns,
    )

    skeleton["player_id"] = target["gsis_id"]
    skeleton["player_name"] = target["full_name"]
    skeleton["player_display_name"] = target["full_name"]
    skeleton["position"] = target["position"]
    skeleton["position_group"] = target["position"]
    skeleton["season"] = season
    skeleton["week"] = week
    skeleton["game_id"] = target["game_id"]
    skeleton["team"] = target["team"]

    for column in [
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
    ]:
        if column in skeleton.columns:
            skeleton[column] = target[column]

    skeleton["actual_dk_points"] = np.nan
    skeleton["actual_fd_points"] = np.nan

    duplicates = skeleton.duplicated(
        ["player_id", "season", "week", "game_id"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Target player-game uniqueness FAILED."
        )

    return (
        skeleton.reset_index(drop=True),
        latest_roster_week,
    )


def print_target_schedule(schedule, week):
    target = schedule.loc[
        schedule["week"] == week
    ].copy()

    if target.empty:
        raise ValueError(
            f"No schedule games found for Week {week}."
        )

    target = target.sort_values(
        ["gameday", "gametime", "game_id"]
    )

    print("\nTARGET-WEEK NFL SCHEDULE")
    print("-" * 80)

    for row in target.itertuples():
        print(
            f"{row.gameday}  {row.gametime}  "
            f"{row.away_team} @ {row.home_team}  "
            f"[{row.game_id}]"
        )

    print("-" * 80)
    print(f"Games: {len(target)}")

    return target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe live NFL offensive features "
            "for one explicit target week."
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
    print("STEP 9B - LIVE NFL FEATURE BUILD")
    print("=" * 80)
    print(f"\nTarget season: {season}")
    print(f"Target week:   {week}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical = pd.read_parquet(
        HISTORICAL_SOURCE_PATH
    )

    historical_columns = historical.columns.tolist()

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    model_features = manifest["feature_columns"]

    print(
        f"Historical archive: "
        f"{len(historical):,} player-games"
    )

    print("\nLoading current NFL schedule...")
    schedule = to_pandas(
        nfl.load_schedules([season])
    )

    schedule["home_team"] = normalize_team_series(
        schedule["home_team"]
    )
    schedule["away_team"] = normalize_team_series(
        schedule["away_team"]
    )

    target_schedule = print_target_schedule(
        schedule,
        week,
    )

    expected_game_count = len(target_schedule)

    if target_schedule["game_id"].nunique() != expected_game_count:
        raise ValueError(
            "Target schedule game-ID uniqueness FAILED."
        )

    if (
        target_schedule["home_team"]
        == target_schedule["away_team"]
    ).any():
        raise ValueError(
            "Target schedule contains a team playing itself."
        )

    target_team_count = pd.concat(
        [
            target_schedule["home_team"],
            target_schedule["away_team"],
        ]
    ).nunique()

    if target_team_count != expected_game_count * 2:
        raise ValueError(
            "A team appears more than once in the target week."
        )

    print("Target schedule integrity: PASSED")

    context = build_team_game_context(schedule)

    target_context = context.loc[
        (context["season"] == season)
        & (context["week"] == week)
    ].copy()

    if len(target_context) != expected_game_count * 2:
        raise ValueError(
            "Target team-game context row count FAILED."
        )

    print(
        f"Target team-game context: PASSED "
        f"({len(target_context)} team rows)"
    )

    print("\nLoading current-season player statistics...")
    current_stats = to_pandas(
        nfl.load_player_stats([season])
    )

    current_stats = current_stats.loc[
        current_stats["season_type"].eq("REG")
        & current_stats["position"].isin(
            OFFENSIVE_POSITIONS
        )
        & current_stats["player_id"].notna()
        & (current_stats["week"] < week)
    ].copy()

    current_stats["team"] = normalize_team_series(
        current_stats["team"]
    )

    current_stats = attach_schedule_context(
        current_stats,
        context,
    )

    current_stats = add_current_rules_scoring(
        current_stats
    )

    duplicate_current = current_stats.duplicated(
        ["player_id", "season", "week", "game_id"],
        keep=False,
    )

    if duplicate_current.any():
        raise ValueError(
            "Current-season player-game uniqueness FAILED."
        )

    current_stats = current_stats.reindex(
        columns=historical_columns
    )

    completed_weeks = sorted(
        pd.to_numeric(
            current_stats["week"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Current completed/available offensive "
        f"player-games: {len(current_stats):,}"
    )
    print(
        f"Stat weeks currently available before target: "
        f"{completed_weeks}"
    )

    print("\nLoading current roster snapshots...")
    rosters = to_pandas(
        nfl.load_rosters(seasons=[season])
    )

    target_skeleton, roster_week = build_target_roster(
        rosters=rosters,
        target_context=target_context,
        season=season,
        week=week,
        historical_columns=historical_columns,
    )

    print(
        f"Roster snapshot used: Week {roster_week}"
    )
    print(
        f"Target offensive roster rows: "
        f"{len(target_skeleton):,}"
    )

    combined = pd.concat(
        [
            historical,
            current_stats,
            target_skeleton,
        ],
        ignore_index=True,
        sort=False,
    )

    combined["gameday"] = pd.to_datetime(
        combined["gameday"]
    )

    print(
        f"\nCombined historical/live source rows: "
        f"{len(combined):,}"
    )

    features = build_offensive_features(
        combined,
        allow_missing_targets=True,
    )

    target_features = features.loc[
        (features["season"] == season)
        & (features["week"] == week)
    ].copy()

    if len(target_features) != len(target_skeleton):
        raise ValueError(
            "Live feature row-count validation FAILED."
        )

    if target_features[
        "dk_points_current_rules"
    ].notna().any():
        raise ValueError(
            "Live DK target isolation FAILED."
        )

    if target_features[
        "fd_points_current_rules"
    ].notna().any():
        raise ValueError(
            "Live FD target isolation FAILED."
        )

    missing_model_features = [
        column
        for column in model_features
        if column not in target_features.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Live output is missing production model features: "
            + ", ".join(missing_model_features)
        )

    schedule_lookup = target_context[
        [
            "game_id",
            "team",
            "opponent",
            "gameday",
            "gametime",
            "is_home",
        ]
    ].copy()

    schedule_check = target_features[
        [
            "game_id",
            "team",
            "opponent",
            "gameday",
            "gametime",
            "is_home",
        ]
    ].merge(
        schedule_lookup,
        on=["game_id", "team"],
        how="left",
        suffixes=("_feature", "_schedule"),
        validate="many_to_one",
    )

    if schedule_check["opponent_schedule"].isna().any():
        raise ValueError(
            "Live schedule mapping FAILED."
        )

    for column in [
        "opponent",
        "gameday",
        "gametime",
        "is_home",
    ]:
        left = schedule_check[
            f"{column}_feature"
        ].astype(str)

        right = schedule_check[
            f"{column}_schedule"
        ].astype(str)

        if not left.equals(right):
            raise ValueError(
                f"Live schedule {column} parity FAILED."
            )

    print("\nLIVE VALIDATIONS")
    print("-" * 80)
    print("Target row-count check: PASSED")
    print("Unknown-target isolation: PASSED")
    print(
        f"Production model contract: PASSED "
        f"({len(model_features)} predictors)"
    )
    print(
        "Game ID / team / opponent / date / "
        "time / home-away mapping: PASSED"
    )

    output_path = (
        OUTPUT_DIR
        / f"offensive_features_{season}_week_{week}.parquet"
    )

    target_features.to_parquet(
        output_path,
        index=False,
    )

    schedule_output = (
        OUTPUT_DIR
        / f"schedule_{season}_week_{week}.csv"
    )

    target_schedule.to_csv(
        schedule_output,
        index=False,
    )

    print("\nSaved:")
    print(output_path)
    print(schedule_output)

    print(
        f"\nLive offensive feature rows: "
        f"{len(target_features):,}"
    )
    print(
        f"Feature columns: "
        f"{len(target_features.columns):,}"
    )

    print("\n" + "=" * 80)
    print("STEP 9B LIVE NFL FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
