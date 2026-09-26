from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_features_v1_2016_2025.parquet"
)

OUTPUT_DIR = ROOT_DIR / "modeling" / "output"

METRICS_FILE = OUTPUT_DIR / "walk_forward_metrics_v1.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "walk_forward_predictions_v1.parquet"


VALIDATION_SEASONS = [2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]

TARGETS = {
    "dk": "dk_points_current_rules",
    "fd": "fd_points_current_rules",
}

BASELINES = {
    "dk": "actual_dk_points_roll3",
    "fd": "actual_fd_points_roll3",
}


def rank_correlation(actual, predicted):
    """
    Spearman-style rank correlation without requiring scipy.

    DFS cares about ranking players correctly, not only point error.
    """
    actual_rank = pd.Series(actual).rank(method="average")
    predicted_rank = pd.Series(predicted).rank(method="average")

    return actual_rank.corr(predicted_rank)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "rank_corr": np.nan,
        }

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    rank_corr = rank_correlation(actual, predicted)

    return {
        "rows": len(actual),
        "mae": mae,
        "rmse": rmse,
        "rank_corr": rank_corr,
    }


print()
print("=" * 80)
print("STEP 8 - WALK-FORWARD DFS PROJECTION MODEL V1")
print("=" * 80)

print(f"\nLoading feature dataset:\n{FEATURE_FILE}")

data = pd.read_parquet(FEATURE_FILE)

print(f"\nRows:    {len(data):,}")
print(f"Columns: {len(data.columns):,}")


# ---------------------------------------------------------------------
# SELECT ONLY PREDICTOR COLUMNS THAT WERE BUILT TO BE LEAKAGE-SAFE
# ---------------------------------------------------------------------

historical_suffixes = (
    "_lag1",
    "_roll3",
    "_ewm",
    "_season_to_date",
    "_prior_season",
)

historical_features = [
    column
    for column in data.columns
    if column.endswith(historical_suffixes)
]

opponent_features = [
    column
    for column in data.columns
    if column.startswith("opp_pos_")
]

current_game_context = [
    "is_home",
    "is_neutral",
    "team_rest",
    "opponent_rest",
    "rest_differential",
    "team_spread",
    "total_line",
    "team_implied_points",
    "opponent_implied_points",
    "is_indoor",
    "days_since_last_game",
    "prior_games",
    "prior_games_this_season",
]

candidate_features = (
    historical_features
    + opponent_features
    + current_game_context
)

candidate_features = list(dict.fromkeys(candidate_features))

feature_columns = []

for column in candidate_features:
    if column not in data.columns:
        continue

    if (
        pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
    ):
        feature_columns.append(column)


print(f"\nModel predictor columns: {len(feature_columns):,}")

if not feature_columns:
    raise ValueError("No usable model features were found.")


# ---------------------------------------------------------------------
# CLEAN NUMERIC INPUTS
# ---------------------------------------------------------------------

for column in feature_columns:
    if pd.api.types.is_bool_dtype(data[column]):
        data[column] = data[column].astype(int)

data[feature_columns] = (
    data[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
)


# ---------------------------------------------------------------------
# WALK-FORWARD VALIDATION
# ---------------------------------------------------------------------

metrics_rows = []
prediction_rows = []

for site, target_column in TARGETS.items():

    baseline_column = BASELINES[site]

    print()
    print("-" * 80)
    print(f"{site.upper()} MODEL")
    print("-" * 80)

    for validation_season in VALIDATION_SEASONS:

        print(f"\nValidation season: {validation_season}")

        for position in POSITIONS:

            train = data[
                (data["season"] < validation_season)
                & (data["position"] == position)
            ].copy()

            test = data[
                (data["season"] == validation_season)
                & (data["position"] == position)
            ].copy()

            if train.empty or test.empty:
                continue

            X_train = train[feature_columns]
            y_train = train[target_column]

            X_test = test[feature_columns]
            y_test = test[target_column]

            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.5,
                early_stopping=True,
                random_state=42,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(X_test)

            model_metrics = calculate_metrics(
                y_test,
                predictions,
            )

            baseline_values = test[baseline_column]

            baseline_metrics = calculate_metrics(
                y_test,
                baseline_values,
            )

            metrics_rows.append(
                {
                    "site": site.upper(),
                    "season": validation_season,
                    "position": position,
                    "model_rows": model_metrics["rows"],
                    "model_mae": model_metrics["mae"],
                    "model_rmse": model_metrics["rmse"],
                    "model_rank_corr": model_metrics["rank_corr"],
                    "baseline_rows": baseline_metrics["rows"],
                    "baseline_mae": baseline_metrics["mae"],
                    "baseline_rmse": baseline_metrics["rmse"],
                    "baseline_rank_corr": baseline_metrics["rank_corr"],
                }
            )

            output = test[
                [
                    "player_id",
                    "player_display_name",
                    "position",
                    "season",
                    "week",
                    "game_id",
                    "team",
                    "opponent",
                    target_column,
                ]
            ].copy()

            output["site"] = site.upper()
            output["model_projection"] = predictions
            output["baseline_projection"] = baseline_values.values

            prediction_rows.append(output)

            print(
                f"{position}: "
                f"Train {len(train):,} | "
                f"Test {len(test):,} | "
                f"RMSE {model_metrics['rmse']:.3f} | "
                f"MAE {model_metrics['mae']:.3f} | "
                f"Rank {model_metrics['rank_corr']:.3f}"
            )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

metrics = pd.DataFrame(metrics_rows)

predictions = pd.concat(
    prediction_rows,
    ignore_index=True,
)

metrics.to_csv(
    METRICS_FILE,
    index=False,
)

predictions.to_parquet(
    PREDICTIONS_FILE,
    index=False,
)


# ---------------------------------------------------------------------
# OVERALL RESULTS
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("WALK-FORWARD RESULTS")
print("=" * 80)

summary = (
    metrics.groupby(["site", "position"])
    .agg(
        model_mae=("model_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
        model_rmse=("model_rmse", "mean"),
        baseline_rmse=("baseline_rmse", "mean"),
        model_rank_corr=("model_rank_corr", "mean"),
        baseline_rank_corr=("baseline_rank_corr", "mean"),
    )
    .reset_index()
)

summary["rmse_improvement"] = (
    summary["baseline_rmse"]
    - summary["model_rmse"]
)

summary["rank_improvement"] = (
    summary["model_rank_corr"]
    - summary["baseline_rank_corr"]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print()
print(summary.round(3).to_string(index=False))


print()
print("=" * 80)
print("OVERALL BY SITE")
print("=" * 80)

for site in ["DK", "FD"]:

    site_predictions = predictions[
        predictions["site"] == site
    ].copy()

    if site == "DK":
        actual_column = "dk_points_current_rules"
    else:
        actual_column = "fd_points_current_rules"

    model_metrics = calculate_metrics(
        site_predictions[actual_column],
        site_predictions["model_projection"],
    )

    baseline_metrics = calculate_metrics(
        site_predictions[actual_column],
        site_predictions["baseline_projection"],
    )

    print()
    print(site)
    print(
        f"Model RMSE:    {model_metrics['rmse']:.3f}"
    )
    print(
        f"Baseline RMSE: {baseline_metrics['rmse']:.3f}"
    )
    print(
        f"Model MAE:     {model_metrics['mae']:.3f}"
    )
    print(
        f"Baseline MAE:  {baseline_metrics['mae']:.3f}"
    )
    print(
        f"Model Rank:    {model_metrics['rank_corr']:.3f}"
    )
    print(
        f"Baseline Rank: {baseline_metrics['rank_corr']:.3f}"
    )


print()
print(f"Metrics saved:\n{METRICS_FILE}")
print()
print(f"Predictions saved:\n{PREDICTIONS_FILE}")

print()
print("=" * 80)
print("STEP 8 WALK-FORWARD MODEL V1 COMPLETE")
print("=" * 80)
