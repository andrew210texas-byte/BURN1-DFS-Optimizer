from pathlib import Path

import pandas as pd

from models.player import Player
from optimizer.lineup_optimizer import (
    optimize_nfl_lineup,
)


ROOT = Path.cwd()


def clean(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_team(value):
    team = clean(value).upper()

    if team == "JAC":
        return "JAX"

    return team


def dk_opponent(
    team,
    game_info,
):
    team = normalize_team(team)

    matchup = clean(
        game_info
    ).split()[0]

    if "@" not in matchup:
        return ""

    away, home = matchup.split(
        "@",
        1,
    )

    away = normalize_team(away)
    home = normalize_team(home)

    if team == away:
        return home

    if team == home:
        return away

    return ""


def load_dk_players():

    pool_path = (
        ROOT
        / "data"
        / "live"
        / "nfl"
        / "dk_2026_week3_sunday_main_final_pool.csv"
    )

    salary_path = (
        ROOT
        / "data"
        / "salaries"
        / "dk_2026_week3_sunday_main.csv"
    )

    pool = pd.read_csv(
        pool_path,
        dtype={
            "player_id": str,
        },
    )

    salary = pd.read_csv(
        salary_path,
        dtype={
            "ID": str,
        },
    )

    salary_lookup = (
        salary
        .set_index("ID")
    )

    players = []

    for _, row in pool.iterrows():

        player_id = clean(
            row["player_id"]
        )

        team = normalize_team(
            row["team"]
        )

        opponent = clean(
            row.get(
                "opponent",
                "",
            )
        )

        if (
            not opponent
            and player_id
            in salary_lookup.index
        ):
            salary_row = (
                salary_lookup.loc[
                    player_id
                ]
            )

            opponent = dk_opponent(
                team,
                salary_row[
                    "Game Info"
                ],
            )

        opponent = normalize_team(
            opponent
        )

        roster_positions = tuple(
            value.strip().upper()
            for value
            in clean(
                row[
                    "roster_positions"
                ]
            ).split("/")
            if value.strip()
        )

        players.append(
            Player(
                name=clean(
                    row["name"]
                ),
                site="DraftKings",
                position=clean(
                    row["position"]
                ).upper(),
                team=team,
                opponent=opponent,
                salary=int(
                    row["salary"]
                ),
                player_id=player_id,
                roster_positions=(
                    roster_positions
                ),
                status=clean(
                    row.get(
                        "status",
                        "",
                    )
                ),
                projection=float(
                    row[
                        "projection"
                    ]
                ),
            )
        )

    missing_opponents = [
        player.name
        for player in players
        if not player.opponent
    ]

    if missing_opponents:
        raise ValueError(
            "DraftKings players missing "
            "opponents: "
            + ", ".join(
                missing_opponents[:20]
            )
        )

    return players


def load_fd_players():

    pool_path = (
        ROOT
        / "data"
        / "live"
        / "nfl"
        / "fd_2026_week3_sunday_main_final_pool.csv"
    )

    pool = pd.read_csv(
        pool_path,
        dtype={
            "player_id": str,
        },
    )

    players = []

    for _, row in pool.iterrows():

        roster_positions = tuple(
            value.strip().upper()
            for value
            in clean(
                row[
                    "roster_positions"
                ]
            ).split("/")
            if value.strip()
        )

        players.append(
            Player(
                name=clean(
                    row["name"]
                ),
                site="FanDuel",
                position=clean(
                    row["position"]
                ).upper(),
                team=normalize_team(
                    row["team"]
                ),
                opponent=normalize_team(
                    row["opponent"]
                ),
                salary=int(
                    row["salary"]
                ),
                player_id=clean(
                    row[
                        "player_id"
                    ]
                ),
                roster_positions=(
                    roster_positions
                ),
                status=clean(
                    row.get(
                        "status",
                        "",
                    )
                ),
                projection=float(
                    row[
                        "projection"
                    ]
                ),
            )
        )

    missing_opponents = [
        player.name
        for player in players
        if not player.opponent
    ]

    if missing_opponents:
        raise ValueError(
            "FanDuel players missing "
            "opponents: "
            + ", ".join(
                missing_opponents[:20]
            )
        )

    return players


def validate_correlations(
    lineup,
):
    quarterback = next(
        entry.player
        for entry in lineup
        if entry.player.position
        == "QB"
    )

    dst = next(
        entry.player
        for entry in lineup
        if entry.player.position
        == "DST"
    )

    skill_players = [
        entry.player
        for entry in lineup
        if entry.player.position
        in {
            "RB",
            "WR",
            "TE",
        }
    ]

    qb_teammates = [
        player
        for player in skill_players
        if player.team
        == quarterback.team
    ]

    bring_backs = [
        player
        for player in skill_players
        if player.team
        == quarterback.opponent
    ]

    dst_rbs = [
        player
        for player in skill_players
        if (
            player.position == "RB"
            and player.team
            == dst.team
        )
    ]

    if not qb_teammates:
        raise ValueError(
            "QB stack validation FAILED."
        )

    if not bring_backs:
        raise ValueError(
            "QB bring-back validation FAILED."
        )

    if not dst_rbs:
        raise ValueError(
            "RB + DST validation FAILED."
        )

    if (
        dst.team
        == quarterback.opponent
    ):
        raise ValueError(
            "QB vs opposing DST "
            "validation FAILED."
        )

    return (
        quarterback,
        qb_teammates,
        bring_backs,
        dst,
        dst_rbs,
    )


def run_site(
    site_name,
    players,
    salary_cap,
):

    print()
    print("=" * 96)
    print(
        f"{site_name.upper()} "
        "GPP CORRELATION OPTIMIZER"
    )
    print("=" * 96)

    lineup = optimize_nfl_lineup(
        players,
        salary_cap=salary_cap,
        gpp_mode=True,
        qb_stack_min=1,
        bring_back_min=1,
        rb_dst_stack=True,
    )

    if not lineup:
        raise ValueError(
            f"No feasible {site_name} "
            "GPP lineup."
        )

    (
        quarterback,
        qb_teammates,
        bring_backs,
        dst,
        dst_rbs,
    ) = validate_correlations(
        lineup
    )

    total_salary = sum(
        entry.player.salary
        for entry in lineup
    )

    total_projection = sum(
        entry.player.projection
        for entry in lineup
    )

    if total_salary > salary_cap:
        raise ValueError(
            "Salary cap validation FAILED."
        )

    print()

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

    print("-" * 96)

    print(
        f"{'TOTAL':<45}"
        f"${total_salary:>6,}   "
        f"{total_projection:>6.2f}"
    )

    print()
    print(
        f"Salary remaining: "
        f"${salary_cap - total_salary:,}"
    )

    print()
    print("CORRELATION CHECK")
    print("-" * 96)

    print(
        f"QB: "
        f"{quarterback.name} "
        f"({quarterback.team})"
    )

    print(
        "QB teammate stack: "
        + ", ".join(
            f"{player.name} "
            f"({player.position})"
            for player
            in qb_teammates
        )
    )

    print(
        "Opponent bring-back: "
        + ", ".join(
            f"{player.name} "
            f"({player.position})"
            for player
            in bring_backs
        )
    )

    print(
        f"DST: "
        f"{dst.name} "
        f"({dst.team})"
    )

    print(
        "Same-team RB + DST: "
        + ", ".join(
            player.name
            for player
            in dst_rbs
        )
    )

    print()
    print(
        "QB stack:             PASSED"
    )
    print(
        "Opponent bring-back:  PASSED"
    )
    print(
        "RB + DST stack:       PASSED"
    )
    print(
        "QB vs opposing DST:   PASSED"
    )
    print(
        "Salary cap:           PASSED"
    )

    output_rows = []

    for entry in lineup:
        player = entry.player

        output_rows.append(
            {
                "site": site_name,
                "slot": (
                    entry.roster_slot
                ),
                "player_id": (
                    player.player_id
                ),
                "name": player.name,
                "position": (
                    player.position
                ),
                "team": player.team,
                "opponent": (
                    player.opponent
                ),
                "salary": (
                    player.salary
                ),
                "projection": (
                    player.projection
                ),
            }
        )

    output_path = (
        ROOT
        / "modeling"
        / "output"
        / (
            "gpp_lineup_"
            + site_name.lower()
            + "_2026_week3_sunday_main.csv"
        )
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved:\n{output_path}"
    )


def main():
    dk_players = load_dk_players()
    fd_players = load_fd_players()

    run_site(
        "DraftKings",
        dk_players,
        50000,
    )

    run_site(
        "FanDuel",
        fd_players,
        60000,
    )

    print()
    print("=" * 96)
    print(
        "SHARED DK + FD GPP "
        "CORRELATION ENGINE PASSED"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
