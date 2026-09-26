from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

REFERENCE_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "reference"
    / "offensive_features_v1_reference.parquet"
)

CANDIDATE_FILE = (
    ROOT_DIR
    / "data"
    / "historical"
    / "nfl"
    / "processed"
    / "offensive_features_v1_2016_2025.parquet"
)

KEY_COLUMNS = [
    "player_id",
    "season",
    "week",
    "game_id",
]

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-10


print()
print("=" * 80)
print("STEP 9A - FEATURE PARITY HARNESS")
print("=" * 80)

reference = pd.read_parquet(REFERENCE_FILE)
candidate = pd.read_parquet(CANDIDATE_FILE)

print(f"\nReference rows:    {len(reference):,}")
print(f"Candidate rows:    {len(candidate):,}")
print(f"Reference columns: {len(reference.columns):,}")
print(f"Candidate columns: {len(candidate.columns):,}")


# ---------------------------------------------------------------------
# SCHEMA VALIDATION
# ---------------------------------------------------------------------

if len(reference) != len(candidate):
    raise ValueError(
        "PARITY FAILED: row counts differ."
    )

reference_columns = list(reference.columns)
candidate_columns = list(candidate.columns)

if reference_columns != candidate_columns:
    missing = [
        column
        for column in reference_columns
        if column not in candidate_columns
    ]

    extra = [
        column
        for column in candidate_columns
        if column not in reference_columns
    ]

    raise ValueError(
        "PARITY FAILED: schemas differ.\n"
        f"Missing columns: {missing}\n"
        f"Extra columns: {extra}"
    )

print("\nSchema parity: PASSED")


# ---------------------------------------------------------------------
# KEY VALIDATION
# ---------------------------------------------------------------------

for column in KEY_COLUMNS:
    if column not in reference.columns:
        raise ValueError(
            f"PARITY FAILED: missing key column {column}."
        )

if reference.duplicated(KEY_COLUMNS).any():
    raise ValueError(
        "PARITY FAILED: reference contains duplicate player-game keys."
    )

if candidate.duplicated(KEY_COLUMNS).any():
    raise ValueError(
        "PARITY FAILED: candidate contains duplicate player-game keys."
    )

reference = reference.sort_values(
    KEY_COLUMNS,
    kind="stable",
).reset_index(drop=True)

candidate = candidate.sort_values(
    KEY_COLUMNS,
    kind="stable",
).reset_index(drop=True)

if not reference[KEY_COLUMNS].equals(
    candidate[KEY_COLUMNS]
):
    raise ValueError(
        "PARITY FAILED: player-game keys differ."
    )

print("Player-game key parity: PASSED")


# ---------------------------------------------------------------------
# VALUE VALIDATION
# ---------------------------------------------------------------------

numeric_columns = [
    column
    for column in reference.columns
    if pd.api.types.is_numeric_dtype(reference[column])
]

non_numeric_columns = [
    column
    for column in reference.columns
    if column not in numeric_columns
]

numeric_failures = []
largest_difference = 0.0
largest_difference_column = None

for column in numeric_columns:

    left = reference[column].to_numpy()
    right = candidate[column].to_numpy()

    equal = np.allclose(
        left,
        right,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        equal_nan=True,
    )

    if not equal:
        numeric_failures.append(column)

        finite = (
            np.isfinite(left)
            & np.isfinite(right)
        )

        if finite.any():
            difference = np.max(
                np.abs(
                    left[finite]
                    - right[finite]
                )
            )

            if difference > largest_difference:
                largest_difference = float(difference)
                largest_difference_column = column


non_numeric_failures = []

for column in non_numeric_columns:

    left = reference[column].astype("string")
    right = candidate[column].astype("string")

    equal = (
        left.fillna("<NULL>")
        .equals(
            right.fillna("<NULL>")
        )
    )

    if not equal:
        non_numeric_failures.append(column)


if numeric_failures or non_numeric_failures:

    raise ValueError(
        "PARITY FAILED: feature values changed.\n"
        f"Numeric failures: {numeric_failures}\n"
        f"Non-numeric failures: {non_numeric_failures}\n"
        f"Largest numeric difference: {largest_difference}\n"
        f"Largest-difference column: {largest_difference_column}"
    )


print(
    f"Numeric value parity: PASSED "
    f"({len(numeric_columns):,} columns)"
)

print(
    f"Non-numeric value parity: PASSED "
    f"({len(non_numeric_columns):,} columns)"
)


# ---------------------------------------------------------------------
# MODEL CONTRACT VALIDATION
# ---------------------------------------------------------------------

manifest_file = (
    ROOT_DIR
    / "modeling"
    / "artifacts"
    / "production_model_manifest_v1.json"
)

if not manifest_file.exists():
    raise ValueError(
        "PARITY FAILED: production model manifest is missing."
    )

import json

with open(
    manifest_file,
    "r",
    encoding="utf-8",
) as file:
    manifest = json.load(file)

model_features = manifest["feature_columns"]

missing_model_features = [
    column
    for column in model_features
    if column not in candidate.columns
]

if missing_model_features:
    raise ValueError(
        "PARITY FAILED: candidate table is missing model features:\n"
        f"{missing_model_features}"
    )

if len(model_features) != manifest["feature_count"]:
    raise ValueError(
        "PARITY FAILED: manifest feature count does not match "
        "its feature-column list."
    )

print(
    f"Production model contract: PASSED "
    f"({len(model_features):,} predictors)"
)


print()
print("=" * 80)
print("STEP 9A PARITY HARNESS PASSED")
print("=" * 80)
