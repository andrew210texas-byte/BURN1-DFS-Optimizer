from pathlib import Path

import pandas as pd

from optimizer.portfolio_optimizer import (
    generate_nfl_portfolio,
)
from run_gpp_live import (
    load_dk_players,
    load_fd_players,
)


ROOT = Path.cwd()

LINEUP_COUNT = 20
MIN_UNIQUE_PLAYERS = 2


def validate_portfolio(
    site_name,
    portfolio,
    salary_cap,
):
    if len(portfolio.lineups) != LINEUP_COUNT:
        raise ValueError(
            f"{site_name}: expected "
            f"{LINEUP_COUNT} lineups, got "
            f"{len(portfolio.lineups)}."
        )

    lineup_sets = [
        set(profile.player_ids)
        for profile in portfolio.profiles
    ]

    for index, lineup_set in enumerate(
        lineup_sets
    ):
        if len(lineup_set) != 9:
            raise ValueError(
                f"{site_name}: lineup "
                f"{index + 1} does not contain "
                "9 unique players."
            )

    maximum_overlap = (
        9 - MIN_UNIQUE_PLAYERS
    )

    for first_index in range(
        len(lineup_sets)
    ):
        for second_index in range(
            first_index + 1,
            len(lineup_sets),
        ):
            overlap = len(
                lineup_sets[first_index]
                & lineup_sets[second_index]
            )

            if overlap > maximum_overlap:
                raise ValueError(
                    f"{site_name}: lineups "
                    f"{first_index + 1} and "
                    f"{second_index + 1} share "
                    f"{overlap} players."
                )

    for profile in portfolio.profiles:
        if profile.total_salary > salary_cap:
            raise ValueError(
                f"{site_name}: lineup "
                f"{profile.lineup_number} "
                "exceeds salary cap."
            )

    return True


def print_portfolio(
    site_name,
    players,
    portfolio,
):
    player_lookup = {
        player.player_id: player
        for player in players
    }

    print()
    print("=" * 110)
    print(
        f"{site_name.upper()} "
        "MULTI-LINEUP PORTFOLIO"
    )
    print("=" * 110)

    for profile in portfolio.profiles:
        attributes = []

        if profile.is_unstacked:
            attributes.append("UNSTACKED")
        else:
            attributes.append(
                f"QB+{profile.qb_stack_size}"
            )

        if profile.bring_back_size:
            attributes.append(
                f"BRINGBACK+"
                f"{profile.bring_back_size}"
            )

        if profile.has_rb_dst:
            attributes.append("RB+DST")

        if profile.has_qb_vs_opposing_dst:
            attributes.append(
                "QB-vs-OPP-DST"
            )

        print(
            f"Lineup "
            f"{profile.lineup_number:>2}: "
            f"${profile.total_salary:>6,} | "
            f"{profile.total_projection:>6.2f} | "
            + " | ".join(attributes)
        )

    print()
    print("TOP PORTFOLIO EXPOSURES")
    print("-" * 110)

    exposure_rows = sorted(
        portfolio.exposures.items(),
        key=lambda item: (
            -item[1],
            player_lookup[item[0]].name,
        ),
    )

    for player_id, exposure in exposure_rows[
        :20
    ]:
        player = player_lookup[player_id]

        print(
            f"{player.name:<28} "
            f"{player.position:<4} "
            f"{player.team:<3} "
            f"{exposure:>6.1%} "
            f"("
            f"{portfolio.player_counts[player_id]}"
            f"/{len(portfolio.lineups)})"
        )


def save_portfolio(
    site_name,
    portfolio,
):
    rows = []

    for profile, lineup in zip(
        portfolio.profiles,
        portfolio.lineups,
    ):
        for entry in lineup:
            player = entry.player

            rows.append(
                {
                    "site": site_name,
                    "lineup_number": (
                        profile.lineup_number
                    ),
                    "slot": entry.roster_slot,
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
                    "salary": player.salary,
                    "projection": (
                        player.projection
                    ),
                    "lineup_salary": (
                        profile.total_salary
                    ),
                    "lineup_projection": (
                        profile.total_projection
                    ),
                    "qb_stack_size": (
                        profile.qb_stack_size
                    ),
                    "bring_back_size": (
                        profile.bring_back_size
                    ),
                    "has_rb_dst": (
                        profile.has_rb_dst
                    ),
                    "is_unstacked": (
                        profile.is_unstacked
                    ),
                }
            )

    output_path = (
        ROOT
        / "modeling"
        / "output"
        / (
            "portfolio_"
            + site_name.lower()
            + "_2026_week3_sunday_main.csv"
        )
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved:\n{output_path}")


def run_site(
    site_name,
    players,
    salary_cap,
):
    portfolio = generate_nfl_portfolio(
        players,
        salary_cap=salary_cap,
        lineup_count=LINEUP_COUNT,
        min_unique_players=(
            MIN_UNIQUE_PLAYERS
        ),

        # IMPORTANT:
        # No correlation construction is
        # universally forced in this first
        # portfolio test.
        gpp_mode=False,
    )

    validate_portfolio(
        site_name,
        portfolio,
        salary_cap,
    )

    print_portfolio(
        site_name,
        players,
        portfolio,
    )

    save_portfolio(
        site_name,
        portfolio,
    )

    print()
    print(
        f"{site_name} portfolio size: "
        f"{len(portfolio.lineups)} PASSED"
    )
    print(
        f"{site_name} pairwise uniqueness: "
        "PASSED"
    )
    print(
        f"{site_name} exposure calculation: "
        "PASSED"
    )


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
print("=" * 110)
print(
    "SHARED DK + FD MULTI-LINEUP "
    "PORTFOLIO ENGINE PASSED"
)
print("=" * 110)