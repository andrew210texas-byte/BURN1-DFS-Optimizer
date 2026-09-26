from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


TEAM_ALIASES = {
    "JAC": "JAX",
}

NAME_ALIASES = {
    "nicksingleton": "nicholassingleton",
    "joshuapalmer": "joshpalmer",
    "mitchtinsley": "mitchelltinsley",
    "drewogletree": "andrewogletree",
    "matthibner": "matthewhibner",
}

NON_OFFENSIVE_DK_IDS = {
    "44221300",  # William Wagner - LS
    "44221346",  # James Winchester - LS
    "44284123",  # Luke Basso - LS
    "44221374",  # Andrew DePaola - LS
    "44221336",  # Cal Adomitis - LS
    "44221202",  # Zach Wood - LS
    "44221248",  # Ben Mann - LS
    "44221364",  # Evan Deckers - LS
    "44221236",  # Tyler Ott - LS
}

UNRESOLVED_DK_IDS = {
    "44220538",  # Al-Jay Henderson - no GSIS ID
    "44221384",  # Ty Pezza - DK TE / NFL WR mismatch
    "44284113",  # Ihmir Smith-Marsette - DK/NFL team mismatch
    "44220882",  # River Cracraft - no current roster/model match
}


def normalize_team(value):
    value = str(value).strip().upper()
    return TEAM_ALIASES.get(value, value)


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = value.lower()
    value = re.sub(r"[^a-z0-9 ]", "", value)

    value = re.sub(
        r"\s+(jr|sr|ii|iii|iv|v)$",
        "",
        value,
    )

    value = re.sub(r"[^a-z0-9]", "", value)

    return NAME_ALIASES.get(value, value)


def integrate_draftkings_salary(
    salary_path,
    projection_path,
    output_path,
):
    salary = pd.read_csv(
        salary_path,
        dtype={"ID": str},
    )

    projections = pd.read_csv(
        projection_path,
        dtype={"player_id": str},
    )

    salary["dk_id"] = salary["ID"].astype(str)

    salary["team_norm"] = (
        salary["TeamAbbrev"]
        .map(normalize_team)
    )

    salary["name_norm"] = (
        salary["Name"]
        .map(normalize_name)
    )

    projections["team_norm"] = (
        projections["team"]
        .map(normalize_team)
    )

    projections["name_norm"] = (
        projections["player_display_name"]
        .map(normalize_name)
    )

    offensive = salary[
        salary["Position"].isin(
            ["QB", "RB", "WR", "TE"]
        )
    ].copy()

    offensive["match_key"] = (
        offensive["name_norm"]
        + "|"
        + offensive["team_norm"]
        + "|"
        + offensive["Position"]
    )

    projections["match_key"] = (
        projections["name_norm"]
        + "|"
        + projections["team_norm"]
        + "|"
        + projections["position"]
    )

    projection_counts = (
        projections["match_key"]
        .value_counts()
    )

    offensive["projection_match_count"] = (
        offensive["match_key"]
        .map(projection_counts)
        .fillna(0)
        .astype(int)
    )

    if (
        offensive["projection_match_count"] > 1
    ).any():
        bad = offensive.loc[
            offensive[
                "projection_match_count"
            ] > 1,
            [
                "Name",
                "Position",
                "TeamAbbrev",
                "ID",
                "projection_match_count",
            ],
        ]

        raise ValueError(
            "Ambiguous DraftKings identity matches:\n"
            + bad.to_string(index=False)
        )

    projection_lookup = (
        projections
        .drop_duplicates("match_key")
        .set_index("match_key")
    )

    mapped_columns = {
        "player_id": "gsis_player_id",
        "player_display_name": "model_player_name",
        "position": "model_position",
        "team": "model_team",
        "opponent": "model_opponent",
        "game_id": "model_game_id",
        "dk_projection_v1": "projection",
        "fd_projection_v1": "fd_projection_reference",
        "prior_games": "prior_games",
        "prior_games_this_season": "prior_games_this_season",
    }

    for source, target in mapped_columns.items():
        offensive[target] = (
            offensive["match_key"]
            .map(projection_lookup[source])
        )

    offensive["integration_status"] = np.where(
        offensive[
            "projection_match_count"
        ].eq(1),
        "MATCHED",
        "UNRESOLVED",
    )

    offensive["integration_reason"] = np.where(
        offensive[
            "projection_match_count"
        ].eq(1),
        "unique_name_team_position_match",
        "no_unique_projection_match",
    )

    non_offensive_mask = (
        offensive["dk_id"]
        .isin(NON_OFFENSIVE_DK_IDS)
    )

    offensive.loc[
        non_offensive_mask,
        "integration_status",
    ] = "EXCLUDED"

    offensive.loc[
        non_offensive_mask,
        "integration_reason",
    ] = "nfl_roster_position_long_snapper"

    unresolved_mask = (
        offensive["dk_id"]
        .isin(UNRESOLVED_DK_IDS)
    )

    offensive.loc[
        unresolved_mask,
        "integration_status",
    ] = "UNRESOLVED"

    offensive.loc[
        unresolved_mask,
        "integration_reason",
    ] = "identity_or_roster_source_conflict"

    unavailable_mask = (
        offensive["Status"]
        .fillna("")
        .str.upper()
        .isin(["OUT", "IR"])
    )

    offensive.loc[
        unavailable_mask,
        "integration_status",
    ] = "EXCLUDED"

    offensive.loc[
        unavailable_mask,
        "integration_reason",
    ] = "draftkings_status_out_or_ir"

    matched = offensive[
        offensive[
            "integration_status"
        ].eq("MATCHED")
    ]

    if matched["projection"].isna().any():
        raise ValueError(
            "Matched rows contain missing projections."
        )

    if not np.isfinite(
        matched["projection"].to_numpy()
    ).all():
        raise ValueError(
            "Matched projections contain non-finite values."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    offensive.to_csv(
        output_path,
        index=False,
    )

    print("=" * 80)
    print("STEP 10 - DRAFTKINGS SLATE INTEGRATION")
    print("=" * 80)

    print(
        f"\nOffensive salary rows: "
        f"{len(offensive):,}"
    )

    summary = (
        offensive[
            "integration_status"
        ]
        .value_counts()
        .reindex(
            [
                "MATCHED",
                "EXCLUDED",
                "UNRESOLVED",
            ],
            fill_value=0,
        )
    )

    for status, count in summary.items():
        print(
            f"{status:<10}: {count:,}"
        )

    print(
        f"\nUsable projected players: "
        f"{len(matched):,}"
    )

    print(
        "\nUNRESOLVED PLAYERS"
    )
    print("-" * 80)

    unresolved = offensive[
        offensive[
            "integration_status"
        ].eq("UNRESOLVED")
    ]

    if unresolved.empty:
        print("NONE")
    else:
        print(
            unresolved[
                [
                    "Name",
                    "Position",
                    "TeamAbbrev",
                    "Salary",
                    "Status",
                    "ID",
                    "integration_reason",
                ]
            ]
            .sort_values(
                [
                    "Position",
                    "TeamAbbrev",
                    "Name",
                ]
            )
            .to_string(index=False)
        )

    print(
        "\nEXCLUDED PLAYERS"
    )
    print("-" * 80)

    excluded = offensive[
        offensive[
            "integration_status"
        ].eq("EXCLUDED")
    ]

    if excluded.empty:
        print("NONE")
    else:
        print(
            excluded[
                [
                    "Name",
                    "Position",
                    "TeamAbbrev",
                    "Salary",
                    "Status",
                    "ID",
                    "integration_reason",
                ]
            ]
            .sort_values(
                [
                    "Position",
                    "TeamAbbrev",
                    "Name",
                ]
            )
            .to_string(index=False)
        )

    print(
        f"\nSaved:\n{output_path}"
    )

    print(
        "\nDraftKings salary/projection "
        "integration: PASSED"
    )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]

    integrate_draftkings_salary(
        salary_path=(
            BASE_DIR
            / "data"
            / "salaries"
            / "dk_2026_week3_sunday_main.csv"
        ),
        projection_path=(
            BASE_DIR
            / "modeling"
            / "output"
            / "projections_2026_week_3_"
              "dk_2026_week3_sunday_main_v1.csv"
        ),
        output_path=(
            BASE_DIR
            / "data"
            / "live"
            / "nfl"
            / "dk_2026_week3_sunday_main_integrated.csv"
        ),
    )
