from pathlib import Path
import pandas as pd
import numpy as np

from models.player import Player
from optimizer.lineup_optimizer import optimize_nfl_lineup


ROOT = Path.cwd()

SALARY_FILE = (
    ROOT
    / "data"
    / "salaries"
    / "dk_2026_week3_sunday_main.csv"
)

OFFENSE_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "dk_2026_week3_sunday_main_integrated.csv"
)

DST_FILE = (
    ROOT
    / "modeling"
    / "output"
    / "dst_projections_2026_week_3_dk_2026_week3_sunday_main_v1.csv"
)

FINAL_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "dk_2026_week3_sunday_main_final_pool.csv"
)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


print()
print("=" * 88)
print("DRAFTKINGS FINAL MODEL-DRIVEN SLATE")
print("=" * 88)

salary = pd.read_csv(
    SALARY_FILE,
    dtype={"ID": str},
)

offense = pd.read_csv(
    OFFENSE_FILE,
    dtype={"ID": str},
)

dst = pd.read_csv(DST_FILE)

print(f"\nDK salary rows:       {len(salary):,}")
print(f"Integrated offense:   {len(offense):,}")
print(f"DST projections:      {len(dst):,}")


# ------------------------------------------------------------------
# OFFENSE
# ------------------------------------------------------------------

if "integration_status" not in offense.columns:
    raise ValueError(
        "Offensive integration file is missing integration_status."
    )

projection_candidates = [
    "dk_projection",
    "DK_projection",
    "projection",
]

offense_projection_column = next(
    (
        column
        for column in projection_candidates
        if column in offense.columns
    ),
    None,
)

if offense_projection_column is None:
    raise ValueError(
        "Could not locate DK model projection column "
        f"in offense file. Columns: {list(offense.columns)}"
    )

usable_offense = offense.loc[
    offense["integration_status"].eq("MATCHED")
].copy()

usable_offense["model_projection"] = pd.to_numeric(
    usable_offense[offense_projection_column],
    errors="coerce",
)

if usable_offense["model_projection"].isna().any():
    raise ValueError(
        "Matched offensive rows contain missing projections."
    )

if not np.isfinite(
    usable_offense["model_projection"]
).all():
    raise ValueError(
        "Matched offensive rows contain non-finite projections."
    )

print(
    f"Usable offensive rows: "
    f"{len(usable_offense):,}"
)


# ------------------------------------------------------------------
# DST
# ------------------------------------------------------------------

salary_position = salary[
    "Position"
].astype(str).str.upper()

dst_salary = salary.loc[
    salary_position.isin(
        ["DST", "D", "DEF"]
    )
].copy()

if len(dst_salary) != 26:
    raise ValueError(
        f"Expected 26 DK DST salary rows, found {len(dst_salary)}."
    )

dst_salary["canonical_team"] = (
    dst_salary["TeamAbbrev"]
    .astype(str)
    .str.strip()
    .replace({"JAC": "JAX"})
)

dst["canonical_team"] = (
    dst["team"]
    .astype(str)
    .str.strip()
    .replace({"JAC": "JAX"})
)

dst_payload = dst[
    [
        "canonical_team",
        "opponent",
        "dk_projection",
    ]
].copy()

dst_integrated = dst_salary.merge(
    dst_payload,
    on="canonical_team",
    how="left",
    validate="one_to_one",
)

if dst_integrated["dk_projection"].isna().any():
    missing = dst_integrated.loc[
        dst_integrated["dk_projection"].isna(),
        ["Name", "TeamAbbrev"],
    ]

    raise ValueError(
        "Missing DST projections:\n"
        + missing.to_string(index=False)
    )

dst_integrated["model_projection"] = pd.to_numeric(
    dst_integrated["dk_projection"],
    errors="coerce",
)

if not np.isfinite(
    dst_integrated["model_projection"]
).all():
    raise ValueError(
        "DST contains non-finite projections."
    )

print(
    f"Usable DST rows:       "
    f"{len(dst_integrated):,}"
)


# ------------------------------------------------------------------
# BUILD PLAYER OBJECTS
# ------------------------------------------------------------------

players = []


for _, row in usable_offense.iterrows():

    position = clean(row["Position"]).upper()

    roster_position = clean(
        row.get(
            "Roster Position",
            position,
        )
    )

    roster_positions = tuple(
        part.strip().upper()
        for part in roster_position.split("/")
        if part.strip()
    )

    if not roster_positions:
        roster_positions = (position,)

    player = Player(
        name=clean(row["Name"]),
        site="DraftKings",
        position=position,
        team=clean(row["TeamAbbrev"]),
        opponent=clean(
            row.get("opponent", "")
        ),
        salary=int(row["Salary"]),
        player_id=clean(row["ID"]),
        roster_positions=roster_positions,
        status=clean(row.get("Status", "")),
        projection=float(
            row["model_projection"]
        ),
    )

    players.append(player)


for _, row in dst_integrated.iterrows():

    player = Player(
        name=clean(row["Name"]),
        site="DraftKings",
        position="DST",
        team=clean(row["canonical_team"]),
        opponent=clean(row["opponent"]),
        salary=int(row["Salary"]),
        player_id=clean(row["ID"]),
        roster_positions=("DST",),
        status=clean(row.get("Status", "")),
        projection=float(
            row["model_projection"]
        ),
    )

    players.append(player)


# ------------------------------------------------------------------
# VALIDATE FINAL POOL
# ------------------------------------------------------------------

if len(players) != (
    len(usable_offense)
    + len(dst_integrated)
):
    raise ValueError(
        "Final player-pool row count FAILED."
    )

if len(
    {
        player.player_id
        for player in players
    }
) != len(players):
    raise ValueError(
        "Duplicate DraftKings IDs in final player pool."
    )

position_counts = {}

for player in players:
    position_counts[player.position] = (
        position_counts.get(
            player.position,
            0,
        )
        + 1
    )

print()
print("FINAL POOL")
print("-" * 88)
print(f"Total players: {len(players):,}")

for position in [
    "QB",
    "RB",
    "WR",
    "TE",
    "DST",
]:
    print(
        f"{position:<4} "
        f"{position_counts.get(position, 0):>4}"
    )


# ------------------------------------------------------------------
# SAVE AUDITABLE FINAL POOL
# ------------------------------------------------------------------

pool_rows = []

for player in players:
    pool_rows.append(
        {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "opponent": player.opponent,
            "salary": player.salary,
            "roster_positions": "/".join(
                player.roster_positions
            ),
            "status": player.status,
            "projection": player.projection,
        }
    )

pool = pd.DataFrame(pool_rows)

pool.to_csv(
    FINAL_POOL_FILE,
    index=False,
)

print()
print(f"Final pool saved:\n{FINAL_POOL_FILE}")


# ------------------------------------------------------------------
# OR-TOOLS OPTIMIZATION
# ------------------------------------------------------------------

print()
print("=" * 88)
print("RUNNING OR-TOOLS")
print("=" * 88)

lineup = optimize_nfl_lineup(
    players,
    salary_cap=50000,
)

if not lineup:
    raise ValueError(
        "OR-Tools returned no feasible lineup."
    )

if len(lineup) != 9:
    raise ValueError(
        f"Expected 9 lineup slots, got {len(lineup)}."
    )

total_salary = sum(
    entry.player.salary
    for entry in lineup
)

total_projection = sum(
    entry.player.projection
    for entry in lineup
)

if total_salary > 50000:
    raise ValueError(
        "Optimized lineup exceeds DK salary cap."
    )

selected_ids = [
    entry.player.player_id
    for entry in lineup
]

if len(selected_ids) != len(set(selected_ids)):
    raise ValueError(
        "Optimized lineup contains duplicate players."
    )


print()
print("=" * 88)
print("FIRST MODEL-DRIVEN DRAFTKINGS LINEUP")
print("=" * 88)

for entry in lineup:
    player = entry.player

    print(
        f"{entry.roster_slot:<5} "
        f"{player.name:<28} "
        f"{player.team:>3} vs "
        f"{player.opponent:<3} "
        f"${player.salary:>5,}   "
        f"{player.projection:>6.2f}"
    )

print("-" * 88)
print(
    f"{'TOTAL':<42}"
    f"${total_salary:>5,}   "
    f"{total_projection:>6.2f}"
)

print()
print(
    f"Salary remaining: "
    f"${50000 - total_salary:,}"
)

print()
print("Final pool uniqueness: PASSED")
print("Roster size:           PASSED")
print("Salary cap:            PASSED")
print("Player uniqueness:     PASSED")
print("Model projections only: PASSED")

print()
print("=" * 88)
print("FIRST LIVE DK OPTIMIZER SOLVE COMPLETE")
print("=" * 88)
