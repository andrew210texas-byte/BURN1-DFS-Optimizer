from pathlib import Path
import json

import joblib
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
    / "dst_features_v1_2016_2025.parquet"
)

OUTPUT_DIR = ROOT_DIR / "modeling" / "output"
ARTIFACT_DIR = ROOT_DIR / "modeling" / "artifacts"

METRICS_FILE = (
    OUTPUT_DIR
    / "dst_walk_forward_metrics_v1.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "dst_walk_forward_predictions_v1.parquet"
)

MANIFEST_FILE = (
    ARTIFACT_DIR
    / "dst_production_model_manifest_v1.json"
)

VALIDATION_SEASONS = [2023, 2024, 2025]

TARGETS = {
    "DK": "dk_dst_points_current_rules",
    "FD": "fd_dst_points_current_rules",
}

BASELINES = {
    "DK": "actual_dk_points_roll3",
    "FD": "actual_fd_points_roll3",
}

MODEL_PARAMS_VALIDATION = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.5,
    "early_stopping": True,
    "random_state": 42,
}

MODEL_PARAMS_PRODUCTION = {
    **MODEL_PARAMS_VALIDATION,
    "early_stopping": False,
}


def rank_correlation(actual, predicted):
    actual_rank = pd.Series(actual).rank(method="average")
    predicted_rank = pd.Series(predicted).rank(method="average")

    return actual_rank.corr(predicted_rank)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "rank_corr": np.nan,
        }

    return {
        "rows": len(actual),
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "rank_corr": rank_correlation(
            actual,
            predicted,
        ),
    }


print()
print("=" * 80)
print("DST PROJECTION MODEL V1")
print("=" * 80)

print(f"\nLoading:\n{FEATURE_FILE}")

data = pd.read_parquet(FEATURE_FILE)

print(f"\nRows:    {len(data):,}")
print(f"Columns: {len(data.columns):,}")


# ---------------------------------------------------------------------
# SELECT LEAKAGE-SAFE DST FEATURES
# ---------------------------------------------------------------------

historical_suffixes = (
    "_lag1",
    "_roll3",
    "_ewm",
    "_season_to_date",
)

historical_features = [
    column
    for column in data.columns
    if column.endswith(historical_suffixes)
]

opponent_features = [
    column
    for column in data.columns
    if column.startswith("opp_allowed_")
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

candidate_features = list(
    dict.fromkeys(
        historical_features
        + opponent_features
        + current_game_context
    )
)

feature_columns = []

for column in candidate_features:
    if column not in data.columns:
        continue

    if (
        pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
    ):
        feature_columns.append(column)

if not feature_columns:
    raise ValueError(
        "No usable DST predictors found."
    )

print(
    f"\nDST predictor columns: "
    f"{len(feature_columns):,}"
)

for column in feature_columns:
    if pd.api.types.is_bool_dtype(
        data[column]
    ):
        data[column] = (
            data[column].astype(int)
        )

data[feature_columns] = (
    data[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


# ---------------------------------------------------------------------
# WALK-FORWARD VALIDATION
# ---------------------------------------------------------------------

metrics_rows = []
prediction_rows = []

print()
print("=" * 80)
print("DST WALK-FORWARD VALIDATION")
print("=" * 80)

for site, target_column in TARGETS.items():

    baseline_column = BASELINES[site]

    print()
    print("-" * 80)
    print(f"{site} DST")
    print("-" * 80)

    for validation_season in VALIDATION_SEASONS:

        train = data[
            data["season"] < validation_season
        ].copy()

        test = data[
            data["season"] == validation_season
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"No train/test data for "
                f"{site} {validation_season}."
            )

        X_train = train[feature_columns]
        y_train = train[target_column]

        X_test = test[feature_columns]
        y_test = test[target_column]

        model = HistGradientBoostingRegressor(
            **MODEL_PARAMS_VALIDATION
        )

        model.fit(
            X_train,
            y_train,
        )

        model_predictions = model.predict(
            X_test
        )

        model_metrics = calculate_metrics(
            y_test,
            model_predictions,
        )

        baseline_metrics = calculate_metrics(
            y_test,
            test[baseline_column],
        )

        metrics_rows.append(
            {
                "site": site,
                "season": validation_season,
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
                "season",
                "week",
                "game_id",
                "gameday",
                "team",
                "opponent",
                target_column,
            ]
        ].copy()

        output["site"] = site
        output[
            "model_projection"
        ] = model_predictions

        output[
            "baseline_projection"
        ] = test[
            baseline_column
        ].to_numpy()

        prediction_rows.append(output)

        print(
            f"{validation_season}: "
            f"Train {len(train):,} | "
            f"Test {len(test):,} | "
            f"RMSE {model_metrics['rmse']:.3f} | "
            f"Baseline {baseline_metrics['rmse']:.3f} | "
            f"MAE {model_metrics['mae']:.3f} | "
            f"Rank {model_metrics['rank_corr']:.3f}"
        )


metrics = pd.DataFrame(metrics_rows)

predictions = pd.concat(
    prediction_rows,
    ignore_index=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# WALK-FORWARD SUMMARY
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("DST WALK-FORWARD SUMMARY")
print("=" * 80)

summary = (
    metrics
    .groupby("site")
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

print()
print(
    summary
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# TRAIN FULL-HISTORY PRODUCTION DST MODELS
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("TRAIN FULL-HISTORY DST PRODUCTION MODELS")
print("=" * 80)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

manifest = {
    "version": "v1",
    "training_seasons": [2016, 2025],
    "feature_count": len(feature_columns),
    "feature_columns": feature_columns,
    "model_parameters": MODEL_PARAMS_PRODUCTION,
    "models": {},
}

for site, target_column in TARGETS.items():

    print()
    print(f"{site} DST")

    X_train = data[feature_columns]
    y_train = data[target_column]

    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS_PRODUCTION
    )

    model.fit(
        X_train,
        y_train,
    )

    filename = (
        f"{site.lower()}_dst_model_v1.joblib"
    )

    model_path = (
        ARTIFACT_DIR
        / filename
    )

    joblib.dump(
        model,
        model_path,
    )

    reloaded = joblib.load(
        model_path
    )

    sample_size = min(
        25,
        len(X_train),
    )

    smoke_predictions = (
        reloaded.predict(
            X_train.tail(sample_size)
        )
    )

    if (
        len(smoke_predictions)
        != sample_size
    ):
        raise ValueError(
            f"{site} DST reload "
            "smoke test FAILED."
        )

    if not np.isfinite(
        smoke_predictions
    ).all():
        raise ValueError(
            f"{site} DST produced "
            "non-finite predictions."
        )

    manifest["models"][
        f"{site}_DST"
    ] = {
        "site": site,
        "position": "DST",
        "target": target_column,
        "training_rows": int(len(data)),
        "first_season": int(
            data["season"].min()
        ),
        "last_season": int(
            data["season"].max()
        ),
        "artifact": filename,
    }

    print(
        f"{len(data):,} rows | "
        f"saved {filename} | "
        "reload test PASSED"
    )


with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


# ---------------------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------------------

expected_artifacts = [
    ARTIFACT_DIR
    / "dk_dst_model_v1.joblib",
    ARTIFACT_DIR
    / "fd_dst_model_v1.joblib",
]

for path in expected_artifacts:
    if not path.exists():
        raise ValueError(
            f"Missing DST artifact: "
            f"{path.name}"
        )

if len(
    manifest["models"]
) != 2:
    raise ValueError(
        "DST manifest validation FAILED."
    )

print()
print(f"Metrics saved:\n{METRICS_FILE}")

print()
print(
    f"Walk-forward predictions saved:\n"
    f"{PREDICTIONS_FILE}"
)

print()
print(
    f"DST manifest saved:\n"
    f"{MANIFEST_FILE}"
)

print()
print("DST model serialization: PASSED")
print("DST manifest validation: PASSED")

print()
print("=" * 80)
print("DST MODEL V1 COMPLETE")
print("=" * 80)
