from pathlib import Path

import nflreadpy as nfl
import polars as pl
import pandas as pd

from data.historical.nfl.historical_loader import (
    OFFENSIVE_POSITIONS,
    normalize_franchise_columns,
    build_team_game_context,
    build_points_allowed,
    build_dst_scoring_dataset,
    add_offensive_dfs_scoring,
)


SEASON = 2026
WEEK = 3

SLATE_PATH = Path(
    "backtest/data/dk/2026_week3_sunday_main.csv"
)

OUTPUT_PATH = Path(
    "backtest/data/dk/2026_week3_actuals.csv"
)


print("===== LOADING SLATE TEAMS =====")

slate = pd.read_csv(SLATE_PATH)

slate_teams = sorted(
    set(slate["team"].dropna().astype(str))
)

print("Slate teams:", len(slate_teams))
print(", ".join(slate_teams))


print("\n===== LOADING 2026 PLAYER STATS =====")

player_stats = nfl.load_player_stats([SEASON])

player_stats = (
    player_stats
    .filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") == WEEK)
        & pl.col("position").is_in(OFFENSIVE_POSITIONS)
    )
)

player_stats = normalize_franchise_columns(
    player_stats,
    ["team"],
)

player_stats = player_stats.filter(
    pl.col("team").is_in(slate_teams)
)

print("Week 3 offensive rows:", player_stats.height)


print("\n===== SCORING OFFENSIVE PLAYERS =====")

offensive_scored = add_offensive_dfs_scoring(
    player_stats
)

offensive_actuals = offensive_scored.select(
    [
        pl.col("player_id").cast(pl.Utf8),
        pl.col("player_name").cast(pl.Utf8),
        pl.col("position").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
        pl.col("actual_dk_points").cast(pl.Float64),
    ]
)


print("\n===== LOADING 2026 SCHEDULE =====")

schedule = nfl.load_schedules([SEASON])

schedule = schedule.filter(
    (pl.col("game_type") == "REG")
    & (pl.col("week") == WEEK)
)

schedule = normalize_franchise_columns(
    schedule,
    ["home_team", "away_team"],
)

schedule = schedule.filter(
    pl.col("home_team").is_in(slate_teams)
    & pl.col("away_team").is_in(slate_teams)
)

print("Sunday-main games:", schedule.height)


print("\n===== BUILDING TEAM CONTEXT =====")

team_context = build_team_game_context(
    schedule
)

team_context = team_context.filter(
    pl.col("team").is_in(slate_teams)
)

print("Team-context rows:", team_context.height)


print("\n===== LOADING 2026 TEAM STATS =====")

team_stats = nfl.load_team_stats([SEASON])

team_stats = team_stats.filter(
    (pl.col("season_type") == "REG")
    & (pl.col("week") == WEEK)
)

team_stats = normalize_franchise_columns(
    team_stats,
    ["team"],
)

team_stats = team_stats.filter(
    pl.col("team").is_in(slate_teams)
)

print("Week 3 team rows:", team_stats.height)


print("\n===== LOADING WEEK 3 PLAY-BY-PLAY =====")

pbp = nfl.load_pbp([SEASON])

pbp = pbp.filter(
    (pl.col("season_type") == "REG")
    & (pl.col("week") == WEEK)
)

pbp = normalize_franchise_columns(
    pbp,
    ["posteam", "defteam"],
)

game_ids = schedule["game_id"].to_list()

pbp = pbp.filter(
    pl.col("game_id").is_in(game_ids)
)

print("Sunday-main PBP rows:", pbp.height)


print("\n===== RECONSTRUCTING DST POINTS ALLOWED =====")

points_allowed = build_points_allowed(
    pbp,
    team_context,
)


print("\n===== SCORING DST =====")

dst_scored = build_dst_scoring_dataset(
    team_stats,
    team_context,
    points_allowed,
    pbp,
)

dst_actuals = dst_scored.select(
    [
        pl.lit("").alias("player_id"),
        pl.col("team").cast(pl.Utf8).alias("player_name"),
        pl.lit("DST").alias("position"),
        pl.col("team").cast(pl.Utf8),
        pl.col("actual_dk_points").cast(pl.Float64),
    ]
)


print("\n===== COMBINING ACTUAL RESULTS =====")

actuals = pl.concat(
    [
        offensive_actuals,
        dst_actuals,
    ],
    how="vertical",
)

actuals = actuals.sort(
    ["position", "team", "player_name"]
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

actuals.write_csv(OUTPUT_PATH)


print("\n===== WEEK 3 ACTUALS CREATED =====")
print("Path:", OUTPUT_PATH)
print("Rows:", actuals.height)

print("\nPosition counts:")
print(
    actuals
    .group_by("position")
    .len()
    .sort("position")
)

print("\nActual DK points range:")
print(
    actuals.select(
        [
            pl.col("actual_dk_points").min().alias("min"),
            pl.col("actual_dk_points").max().alias("max"),
            pl.col("actual_dk_points").null_count().alias("nulls"),
        ]
    )
)

print("\nTop 15 actual DK scores:")
print(
    actuals
    .sort("actual_dk_points", descending=True)
    .head(15)
)
