from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_features_v1_2016_2025.parquet"
)

ARTIFACT_DIR = ROOT_DIR / "modeling" / "artifacts"
MANIFEST_FILE = ARTIFACT_DIR / "production_model_manifest_v1.json"

POSITIONS = ["QB", "RB", "WR", "TE"]

TARGETS = {
    "DK": "dk_points_current_rules",
    "FD": "fd_points_current_rules",
}

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.5,
    "early_stopping": False,
    "random_state": 42,
}


print()
print("=" * 80)
print("STEP 8C - TRAIN FULL-HISTORY PRODUCTION MODELS")
print("=" * 80)

print(f"\nLoading:\n{FEATURE_FILE}")

data = pd.read_parquet(FEATURE_FILE)

print(f"\nRows:    {len(data):,}")
print(f"Columns: {len(data.columns):,}")


# ---------------------------------------------------------------------
# USE THE SAME LEAKAGE-SAFE FEATURE RULES AS WALK-FORWARD V1
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
    raise ValueError("No production model features found.")

print(f"\nProduction predictor columns: {len(feature_columns):,}")


# ---------------------------------------------------------------------
# CLEAN MODEL INPUT TYPES
# ---------------------------------------------------------------------

for column in feature_columns:
    if pd.api.types.is_bool_dtype(data[column]):
        data[column] = data[column].astype(int)

data[feature_columns] = (
    data[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
)


# ---------------------------------------------------------------------
# TRAIN 8 MODELS:
# DK QB/RB/WR/TE
# FD QB/RB/WR/TE
# ---------------------------------------------------------------------

manifest = {
    "version": "v1",
    "training_seasons": [2016, 2025],
    "feature_count": len(feature_columns),
    "feature_columns": feature_columns,
    "model_parameters": MODEL_PARAMS,
    "models": {},
}

for site, target_column in TARGETS.items():

    print()
    print("-" * 80)
    print(f"{site} PRODUCTION MODELS")
    print("-" * 80)

    for position in POSITIONS:

        train = data[
            data["position"] == position
        ].copy()

        X_train = train[feature_columns]
        y_train = train[target_column]

        model = HistGradientBoostingRegressor(
            **MODEL_PARAMS
        )

        model.fit(
            X_train,
            y_train,
        )

        filename = (
            f"{site.lower()}_{position.lower()}_projection_model_v1.joblib"
        )

        model_path = ARTIFACT_DIR / filename

        joblib.dump(
            model,
            model_path,
        )

        # Reload immediately so we prove the serialized artifact works.
        reloaded_model = joblib.load(model_path)

        sample_size = min(25, len(X_train))

        smoke_predictions = reloaded_model.predict(
            X_train.tail(sample_size)
        )

        if len(smoke_predictions) != sample_size:
            raise ValueError(
                f"{site} {position} serialization smoke test FAILED."
            )

        if not np.isfinite(smoke_predictions).all():
            raise ValueError(
                f"{site} {position} produced invalid smoke predictions."
            )

        model_key = f"{site}_{position}"

        manifest["models"][model_key] = {
            "site": site,
            "position": position,
            "target": target_column,
            "training_rows": int(len(train)),
            "first_season": int(train["season"].min()),
            "last_season": int(train["season"].max()),
            "artifact": filename,
        }

        print(
            f"{position}: "
            f"{len(train):,} rows | "
            f"saved {filename} | "
            f"reload test PASSED"
        )


# ---------------------------------------------------------------------
# SAVE MODEL CONTRACT
# ---------------------------------------------------------------------

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

print()
print(f"Manifest saved:\n{MANIFEST_FILE}")


# ---------------------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------------------

expected_models = len(POSITIONS) * len(TARGETS)

joblib_files = list(
    ARTIFACT_DIR.glob("*_projection_model_v1.joblib")
)

if len(joblib_files) != expected_models:
    raise ValueError(
        "Artifact-count validation FAILED: "
        f"expected {expected_models}, found {len(joblib_files)}."
    )

if len(manifest["models"]) != expected_models:
    raise ValueError(
        "Manifest validation FAILED."
    )

print()
print("Production model artifact count: PASSED")
print("Production model manifest:       PASSED")
print("Serialization/reload tests:      PASSED")

print()
print("=" * 80)
print("STEP 8C PRODUCTION MODEL TRAINING COMPLETE")
print("=" * 80)
