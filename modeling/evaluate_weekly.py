from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT_DIR
    / "modeling"
    / "output"
    / "walk_forward_predictions_v1.parquet"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "modeling"
    / "output"
    / "weekly_dfs_evaluation_v1.csv"
)


def spearman_rank(actual, predicted):
    actual_rank = pd.Series(actual).rank(method="average")
    predicted_rank = pd.Series(predicted).rank(method="average")
    return actual_rank.corr(predicted_rank)


def top_n_overlap(group, actual_column, prediction_column, n):
    usable = group[
        [actual_column, prediction_column]
    ].dropna()

    if len(usable) < n:
        return np.nan

    actual_top = set(
        usable.nlargest(
            n,
            actual_column,
        ).index
    )

    predicted_top = set(
        usable.nlargest(
            n,
            prediction_column,
        ).index
    )

    return len(actual_top & predicted_top) / n


print()
print("=" * 80)
print("STEP 8B - WEEKLY DFS MODEL EVALUATION")
print("=" * 80)

predictions = pd.read_parquet(PREDICTIONS_FILE)

print(f"\nLoaded {len(predictions):,} walk-forward predictions.")

results = []

for site in ["DK", "FD"]:

    if site == "DK":
        actual_column = "dk_points_current_rules"
    else:
        actual_column = "fd_points_current_rules"

    site_data = predictions[
        predictions["site"] == site
    ].copy()

    for (
        season,
        week,
        position,
    ), group in site_data.groupby(
        [
            "season",
            "week",
            "position",
        ]
    ):

        valid_model = group[
            [
                actual_column,
                "model_projection",
            ]
        ].dropna()

        valid_baseline = group[
            [
                actual_column,
                "baseline_projection",
            ]
        ].dropna()

        model_rank = (
            spearman_rank(
                valid_model[actual_column],
                valid_model["model_projection"],
            )
            if len(valid_model) >= 3
            else np.nan
        )

        baseline_rank = (
            spearman_rank(
                valid_baseline[actual_column],
                valid_baseline["baseline_projection"],
            )
            if len(valid_baseline) >= 3
            else np.nan
        )

        results.append(
            {
                "site": site,
                "season": season,
                "week": week,
                "position": position,
                "players": len(group),
                "model_weekly_rank": model_rank,
                "baseline_weekly_rank": baseline_rank,
                "model_top3_overlap": top_n_overlap(
                    group,
                    actual_column,
                    "model_projection",
                    3,
                ),
                "baseline_top3_overlap": top_n_overlap(
                    group,
                    actual_column,
                    "baseline_projection",
                    3,
                ),
                "model_top5_overlap": top_n_overlap(
                    group,
                    actual_column,
                    "model_projection",
                    5,
                ),
                "baseline_top5_overlap": top_n_overlap(
                    group,
                    actual_column,
                    "baseline_projection",
                    5,
                ),
            }
        )

results = pd.DataFrame(results)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 80)
print("AVERAGE WEEKLY RESULTS BY POSITION")
print("=" * 80)

summary = (
    results.groupby(
        ["site", "position"]
    )
    .agg(
        weeks=("week", "count"),
        model_rank=("model_weekly_rank", "mean"),
        baseline_rank=("baseline_weekly_rank", "mean"),
        model_top3=("model_top3_overlap", "mean"),
        baseline_top3=("baseline_top3_overlap", "mean"),
        model_top5=("model_top5_overlap", "mean"),
        baseline_top5=("baseline_top5_overlap", "mean"),
    )
    .reset_index()
)

summary["rank_gain"] = (
    summary["model_rank"]
    - summary["baseline_rank"]
)

summary["top3_gain"] = (
    summary["model_top3"]
    - summary["baseline_top3"]
)

summary["top5_gain"] = (
    summary["model_top5"]
    - summary["baseline_top5"]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print()
print(summary.round(3).to_string(index=False))


print()
print("=" * 80)
print("OVERALL WEEKLY RESULTS")
print("=" * 80)

overall = (
    results.groupby("site")
    .agg(
        model_rank=("model_weekly_rank", "mean"),
        baseline_rank=("baseline_weekly_rank", "mean"),
        model_top3=("model_top3_overlap", "mean"),
        baseline_top3=("baseline_top3_overlap", "mean"),
        model_top5=("model_top5_overlap", "mean"),
        baseline_top5=("baseline_top5_overlap", "mean"),
    )
    .reset_index()
)

overall["rank_gain"] = (
    overall["model_rank"]
    - overall["baseline_rank"]
)

overall["top3_gain"] = (
    overall["model_top3"]
    - overall["baseline_top3"]
)

overall["top5_gain"] = (
    overall["model_top5"]
    - overall["baseline_top5"]
)

print()
print(overall.round(3).to_string(index=False))

print()
print(f"Saved:\n{OUTPUT_FILE}")

print()
print("=" * 80)
print("STEP 8B WEEKLY DFS EVALUATION COMPLETE")
print("=" * 80)
