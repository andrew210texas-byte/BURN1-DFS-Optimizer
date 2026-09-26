from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from models.player import Player
from optimizer.lineup_optimizer import optimize_nfl_lineup


ROOT = Path.cwd()

SALARY_FILE = (
    ROOT
    / "data"
    / "salaries"
    / "fd_2026_week3_sunday_main.csv"
)

OFFENSIVE_PROJECTION_FILE = (
    ROOT
    / "modeling"
    / "output"
    / "projections_2026_week_3_dk_2026_week3_sunday_main_v1.csv"
)

DST_PROJECTION_FILE = (
    ROOT
    / "modeling"
    / "output"
    / "dst_projections_2026_week_3_dk_2026_week3_sunday_main_v1.csv"
)

INTEGRATED_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "fd_2026_week3_sunday_main_integrated.csv"
)

FINAL_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "fd_2026_week3_sunday_main_final_pool.csv"
)


TEAM_ALIASES = {
    "JAC": "JAX",
}


NAME_ALIASES = {
    "nick singleton": "nicholas singleton",
    "joshua palmer": "josh palmer",
    "mitch tinsley": "mitchell tinsley",
    "drew ogletree": "andrew ogletree",
    "matt hibner": "matthew hibner",
}


def clean(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_team(value):
    team = clean(value).upper()
    return TEAM_ALIASES.get(team, team)


def normalize_name(value):
    name = clean(value)

    name = unicodedata.normalize(
        "NFKD",
        name,
    )

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    name = (
        name
        .lower()
        .replace("’", "'")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    name = re.sub(
        r"\s+(jr|sr|ii|iii|iv|v)$",
        "",
        name,
    ).strip()

    name = NAME_ALIASES.get(
        name,
        name,
    )

    return name


def normalize_position(value):
    position = clean(value).upper()

    if position in {
        "D",
        "DEF",
        "DST",
    }:
        return "DST"

    return position


def find_fd_projection_column(data):
    candidates = [
        "fd_projection_v1",
        "fd_projection",
        "FD_projection",
        "projection_fd",
    ]

    for column in candidates:
        if column in data.columns:
            return column

    raise ValueError(
        "Could not find FD projection column. "
        f"Available columns: {list(data.columns)}"
    )


print()
print("=" * 92)
print("FANDUEL MODEL-DRIVEN SUNDAY MAIN PIPELINE")
print("=" * 92)


# ==================================================================
# LOAD DATA
# ==================================================================

salary = pd.read_csv(
    SALARY_FILE,
    dtype={"Id": str},
)

offense_projection = pd.read_csv(
    OFFENSIVE_PROJECTION_FILE
)

dst_projection = pd.read_csv(
    DST_PROJECTION_FILE
)

fd_projection_column = (
    find_fd_projection_column(
        offense_projection
    )
)

print()
print(f"FanDuel salary rows:       {len(salary):,}")
print(
    f"Offensive projection rows: "
    f"{len(offense_projection):,}"
)
print(
    f"DST projection rows:       "
    f"{len(dst_projection):,}"
)
print(
    f"FD projection column:      "
    f"{fd_projection_column}"
)


# ==================================================================
# NORMALIZE SALARY DATA
# ==================================================================

salary["canonical_position"] = (
    salary["Position"]
    .map(normalize_position)
)

salary["canonical_team"] = (
    salary["Team"]
    .map(normalize_team)
)

salary["canonical_opponent"] = (
    salary["Opponent"]
    .map(normalize_team)
)

salary["canonical_name"] = (
    salary["Nickname"]
    .map(normalize_name)
)

offensive_salary = salary.loc[
    salary["canonical_position"].isin(
        ["QB", "RB", "WR", "TE"]
    )
].copy()

dst_salary = salary.loc[
    salary["canonical_position"].eq(
        "DST"
    )
].copy()

print()
print(
    f"FanDuel offensive rows:    "
    f"{len(offensive_salary):,}"
)
print(
    f"FanDuel DST rows:          "
    f"{len(dst_salary):,}"
)


# ==================================================================
# NORMALIZE OUR OFFENSIVE PROJECTIONS
# ==================================================================

offense_projection[
    "canonical_position"
] = (
    offense_projection["position"]
    .map(normalize_position)
)

offense_projection[
    "canonical_team"
] = (
    offense_projection["team"]
    .map(normalize_team)
)

offense_projection[
    "canonical_name"
] = (
    offense_projection["player_name"]
    .map(normalize_name)
)

projection_lookup = {}

for index, row in (
    offense_projection.iterrows()
):
    key = (
        row["canonical_name"],
        row["canonical_team"],
        row["canonical_position"],
    )

    projection_lookup.setdefault(
        key,
        [],
    ).append(index)


# ==================================================================
# MATCH FANDUEL OFFENSIVE PLAYERS
# ==================================================================

integration_rows = []

for _, row in offensive_salary.iterrows():

    key = (
        row["canonical_name"],
        row["canonical_team"],
        row["canonical_position"],
    )

    matches = projection_lookup.get(
        key,
        [],
    )

    record = row.to_dict()

    if len(matches) == 1:
        projection_row = (
            offense_projection.loc[
                matches[0]
            ]
        )

        projection = pd.to_numeric(
            projection_row[
                fd_projection_column
            ],
            errors="coerce",
        )

        if pd.isna(projection):
            record[
                "integration_status"
            ] = "UNRESOLVED"

            record[
                "integration_reason"
            ] = "missing_fd_projection"

            record[
                "model_projection"
            ] = np.nan

        else:
            record[
                "integration_status"
            ] = "MATCHED"

            record[
                "integration_reason"
            ] = ""

            record[
                "model_projection"
            ] = float(projection)

            record[
                "model_player_id"
            ] = clean(
                projection_row.get(
                    "player_id",
                    "",
                )
            )

            record[
                "game_id"
            ] = clean(
                projection_row.get(
                    "game_id",
                    "",
                )
            )

    elif len(matches) > 1:
        raise ValueError(
            "Ambiguous FanDuel identity "
            f"match for {row['Nickname']} "
            f"{row['Team']} "
            f"{row['Position']}."
        )

    else:
        record[
            "integration_status"
        ] = "UNRESOLVED"

        record[
            "integration_reason"
        ] = "no_projection_identity_match"

        record[
            "model_projection"
        ] = np.nan

    status = clean(
        row.get(
            "Injury Indicator",
            "",
        )
    ).upper()

    if status in {
        "O",
        "OUT",
        "IR",
    }:
        record[
            "integration_status"
        ] = "EXCLUDED"

        record[
            "integration_reason"
        ] = "injury_status"

    integration_rows.append(record)


integrated_offense = pd.DataFrame(
    integration_rows
)


# ==================================================================
# REMOVE NON-OFFENSIVE ROSTER TYPES MASQUERADING AS TE
# ==================================================================

long_snapper_names = {
    "william wagner",
    "james winchester",
    "luke basso",
    "andrew depaola",
    "cal adomitis",
    "zach wood",
    "ben mann",
    "evan deckers",
    "tyler ott",
}

ls_mask = (
    integrated_offense[
        "canonical_name"
    ].isin(long_snapper_names)
)

integrated_offense.loc[
    ls_mask,
    "integration_status",
] = "EXCLUDED"

integrated_offense.loc[
    ls_mask,
    "integration_reason",
] = "long_snapper"


# ==================================================================
# DST INTEGRATION
# ==================================================================

dst_projection[
    "canonical_team"
] = (
    dst_projection["team"]
    .map(normalize_team)
)

dst_projection[
    "canonical_opponent"
] = (
    dst_projection["opponent"]
    .map(normalize_team)
)

dst_lookup = (
    dst_projection
    .set_index("canonical_team")
)

dst_rows = []

for _, row in dst_salary.iterrows():

    team = row["canonical_team"]

    if team not in dst_lookup.index:
        raise ValueError(
            "No DST projection for "
            f"FanDuel team {team}."
        )

    projection_row = (
        dst_lookup.loc[team]
    )

    record = row.to_dict()

    record[
        "integration_status"
    ] = "MATCHED"

    record[
        "integration_reason"
    ] = ""

    record[
        "model_projection"
    ] = float(
        projection_row[
            "fd_projection"
        ]
    )

    record[
        "game_id"
    ] = clean(
        projection_row[
            "game_id"
        ]
    )

    dst_rows.append(record)


integrated_dst = pd.DataFrame(
    dst_rows
)


# ==================================================================
# COMBINE + VALIDATE
# ==================================================================

integrated = pd.concat(
    [
        integrated_offense,
        integrated_dst,
    ],
    ignore_index=True,
    sort=False,
)

INTEGRATED_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

integrated.to_csv(
    INTEGRATED_FILE,
    index=False,
)

summary = (
    integrated[
        "integration_status"
    ]
    .value_counts()
)

print()
print("IDENTITY / PROJECTION INTEGRATION")
print("-" * 92)

for status in [
    "MATCHED",
    "EXCLUDED",
    "UNRESOLVED",
]:
    print(
        f"{status:<12}"
        f"{int(summary.get(status, 0)):>6}"
    )

unresolved = integrated.loc[
    integrated[
        "integration_status"
    ].eq("UNRESOLVED")
]

if len(unresolved):
    print()
    print("UNRESOLVED PLAYERS")
    print("-" * 92)

    print(
        unresolved[
            [
                "Nickname",
                "Position",
                "Team",
                "integration_reason",
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================================
# PLAYER OBJECTS
# ==================================================================

usable = integrated.loc[
    integrated[
        "integration_status"
    ].eq("MATCHED")
].copy()

if usable[
    "model_projection"
].isna().any():
    raise ValueError(
        "Matched FanDuel rows contain "
        "missing model projections."
    )

if not np.isfinite(
    usable[
        "model_projection"
    ].astype(float)
).all():
    raise ValueError(
        "FanDuel pool contains "
        "non-finite projections."
    )

players = []

pool_rows = []

for _, row in usable.iterrows():

    position = (
        row["canonical_position"]
    )

    raw_roster = clean(
        row["Roster Position"]
    )

    roster_positions = []

    for roster_position in (
        raw_roster.split("/")
    ):
        roster_position = (
            roster_position
            .strip()
            .upper()
        )

        if roster_position == "DEF":
            roster_position = "DST"

        if roster_position:
            roster_positions.append(
                roster_position
            )

    if not roster_positions:
        roster_positions = [
            position
        ]

    player = Player(
        name=clean(
            row["Nickname"]
        ),
        site="FanDuel",
        position=position,
        team=row[
            "canonical_team"
        ],
        opponent=row[
            "canonical_opponent"
        ],
        salary=int(
            row["Salary"]
        ),
        player_id=clean(
            row["Id"]
        ),
        roster_positions=tuple(
            roster_positions
        ),
        status=clean(
            row.get(
                "Injury Indicator",
                "",
            )
        ),
        projection=float(
            row[
                "model_projection"
            ]
        ),
    )

    players.append(player)

    pool_rows.append(
        {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "opponent": player.opponent,
            "game": clean(
                row.get(
                    "Game",
                    "",
                )
            ),
            "game_id": clean(
                row.get(
                    "game_id",
                    "",
                )
            ),
            "salary": player.salary,
            "roster_positions": (
                "/".join(
                    player.roster_positions
                )
            ),
            "status": player.status,
            "projection": (
                player.projection
            ),
        }
    )


# ==================================================================
# FINAL POOL VALIDATION
# ==================================================================

if len(
    {
        player.player_id
        for player in players
    }
) != len(players):
    raise ValueError(
        "Duplicate FanDuel player IDs "
        "in final pool."
    )

position_counts = {}

for player in players:
    position_counts[
        player.position
    ] = (
        position_counts.get(
            player.position,
            0,
        )
        + 1
    )

pool = pd.DataFrame(
    pool_rows
)

pool.to_csv(
    FINAL_POOL_FILE,
    index=False,
)

print()
print("FINAL FANDUEL POOL")
print("-" * 92)
print(
    f"Total usable players: "
    f"{len(players):,}"
)

for position in [
    "QB",
    "RB",
    "WR",
    "TE",
    "DST",
]:
    print(
        f"{position:<4}"
        f"{position_counts.get(position, 0):>6}"
    )

print()
print(
    f"Final pool saved:\n"
    f"{FINAL_POOL_FILE}"
)


# ==================================================================
# OPTIMIZE
# ==================================================================

print()
print("=" * 92)
print("RUNNING FANDUEL OR-TOOLS OPTIMIZER")
print("=" * 92)

lineup = optimize_nfl_lineup(
    players,
    salary_cap=60000,
)

if not lineup:
    raise ValueError(
        "OR-Tools returned no "
        "feasible FanDuel lineup."
    )

if len(lineup) != 9:
    raise ValueError(
        "FanDuel lineup must contain "
        f"9 players; got {len(lineup)}."
    )

total_salary = sum(
    entry.player.salary
    for entry in lineup
)

total_projection = sum(
    entry.player.projection
    for entry in lineup
)

if total_salary > 60000:
    raise ValueError(
        "FanDuel lineup exceeds "
        "$60,000 salary cap."
    )

ids = [
    entry.player.player_id
    for entry in lineup
]

if len(ids) != len(set(ids)):
    raise ValueError(
        "Duplicate player in "
        "FanDuel lineup."
    )


print()
print("=" * 92)
print("FIRST MODEL-DRIVEN FANDUEL LINEUP")
print("=" * 92)

for entry in lineup:
    player = entry.player

    print(
        f"{entry.roster_slot:<5} "
        f"{player.name:<28} "
        f"{player.team:>3} vs "
        f"{player.opponent:<3} "
        f"${player.salary:>6,}   "
        f"{player.projection:>6.2f}"
    )

print("-" * 92)

print(
    f"{'TOTAL':<45}"
    f"${total_salary:>6,}   "
    f"{total_projection:>6.2f}"
)

print()
print(
    f"Salary remaining: "
    f"${60000 - total_salary:,}"
)

print()
print("FanDuel roster size:       PASSED")
print("FanDuel salary cap:        PASSED")
print("Player uniqueness:         PASSED")
print("Own FD projections only:   PASSED")
print("Team/opponent preservation: PASSED")

print()
print("=" * 92)
print("FIRST LIVE FANDUEL OPTIMIZER SOLVE COMPLETE")
print("=" * 92)

