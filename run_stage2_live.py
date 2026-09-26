from pathlib import Path

import pandas as pd

from models.player import Player
from optimizer.portfolio_optimizer import (
    Stage2InfeasibleError,
    generate_nfl_candidates,
    select_nfl_portfolio,
)


ROOT = Path(__file__).resolve().parent

DK_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "dk_2026_week3_sunday_main_final_pool.csv"
)

FD_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "fd_2026_week3_sunday_main_final_pool.csv"
)


def clean(value):
    if value is None:
        return ""

    return str(value).strip()


def load_final_pool(
    file_path,
    site_name,
):
    data = pd.read_csv(
        file_path,
        keep_default_na=False,
    )

    players = []

    for _, row in data.iterrows():

        roster_positions = tuple(
            part.strip().upper()
            for part in clean(
                row["roster_positions"]
            ).split("/")
            if part.strip()
        )

        player = Player(
            name=clean(row["name"]),
            site=site_name,
            position=clean(
                row["position"]
            ).upper(),
            team=clean(row["team"]),
            opponent=clean(
                row["opponent"]
            ),
            salary=int(row["salary"]),
            player_id=clean(
                row["player_id"]
            ),
            roster_positions=(
                roster_positions
            ),
            status=clean(
                row.get("status", "")
            ),
            projection=float(
                row["projection"]
            ),
        )

        players.append(player)

    return players


def print_stage2_result(
    site_name,
    result,
):
    print()
    print("=" * 88)
    print(site_name)
    print("=" * 88)

    print(
        f"Solver status:           "
        f"{result.solver_status}"
    )

    print(
        f"Candidate pool:          "
        f"{result.candidate_count}"
    )

    print(
        f"Selected lineups:        "
        f"{result.selected_lineup_count}/"
        f"{result.requested_lineup_count}"
    )

    print(
        f"Portfolio projection:    "
        f"{result.total_projection:.2f}"
    )

    print()
    print("TOP REALIZED EXPOSURES")
    print("-" * 88)

    top_exposures = sorted(
        result.exposures.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
        reverse=True,
    )[:10]

    for player_id, exposure in top_exposures:
        count = result.player_counts[
            player_id
        ]

        print(
            f"{player_id:<28} "
            f"{count:>2}/"
            f"{result.selected_lineup_count:<2} "
            f"{exposure:>6.0%}"
        )


def find_feasible_max_exposure_test(
    candidates,
    unconstrained,
    lineup_count,
    maximum_exposure,
):
    ranked_players = sorted(
        unconstrained.exposures.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for player_id, original_exposure in ranked_players:

        if original_exposure <= maximum_exposure:
            continue

        try:
            result = select_nfl_portfolio(
                candidates,
                lineup_count=lineup_count,
                max_exposures={
                    player_id: maximum_exposure,
                },
            )

        except Stage2InfeasibleError:
            continue

        return (
            player_id,
            original_exposure,
            result,
        )

    raise Stage2InfeasibleError(
        "No player above the requested "
        "maximum exposure had a feasible "
        "20-lineup Stage 2 solution in "
        "the current candidate pool."
    )


def test_site(
    site_name,
    players,
    salary_cap,
):
    print()
    print("=" * 88)
    print(
        f"{site_name.upper()} "
        "STAGE 1 CANDIDATE GENERATION"
    )
    print("=" * 88)

    candidates = generate_nfl_candidates(
        players,
        salary_cap=salary_cap,
        candidate_count=200,
        min_unique_players=2,
        gpp_mode=False,
    )

    print(
        f"Generated: "
        f"{candidates.generated_count}/"
        f"{candidates.requested_count}"
    )

    if candidates.generated_count < 20:
        raise ValueError(
            f"{site_name} did not generate "
            "enough candidates for a "
            "20-lineup portfolio."
        )

    unconstrained = select_nfl_portfolio(
        candidates,
        lineup_count=20,
    )

    if (
        unconstrained.selected_lineup_count
        != 20
    ):
        raise ValueError(
            f"{site_name} Stage 2 did not "
            "select exactly 20 lineups."
        )

    print_stage2_result(
        f"{site_name.upper()} "
        "UNCONSTRAINED STAGE 2",
        unconstrained,
    )

    maximum_exposure = 0.50

    (
        max_player,
        original_exposure,
        constrained,
    ) = find_feasible_max_exposure_test(
        candidates,
        unconstrained,
        lineup_count=20,
        maximum_exposure=maximum_exposure,
    )

    constrained_exposure = (
        constrained.exposures.get(
            max_player,
            0.0,
        )
    )

    if (
        constrained_exposure
        > maximum_exposure
    ):
        raise ValueError(
            f"{site_name} maximum exposure "
            "constraint FAILED."
        )

    print()
    print(
        f"{site_name} MAX EXPOSURE TEST"
    )
    print("-" * 88)

    print(
        f"Player ID:             "
        f"{max_player}"
    )

    print(
        f"Before constraint:     "
        f"{original_exposure:.0%}"
    )

    print(
        f"Maximum allowed:       "
        f"{maximum_exposure:.0%}"
    )

    print(
        f"After constraint:      "
        f"{constrained_exposure:.0%}"
    )

    print_stage2_result(
        f"{site_name.upper()} "
        "CONSTRAINED STAGE 2",
        constrained,
    )

    minimum_player = max(
        unconstrained.exposures,
        key=unconstrained.exposures.get,
    )

    minimum_required = 0.50

    minimum_test = select_nfl_portfolio(
        candidates,
        lineup_count=20,
        min_exposures={
            minimum_player: minimum_required,
        },
    )

    minimum_realized = (
        minimum_test.exposures.get(
            minimum_player,
            0.0,
        )
    )

    if (
        minimum_realized
        < minimum_required
    ):
        raise ValueError(
            f"{site_name} minimum exposure "
            "constraint FAILED."
        )

    print()
    print(
        f"{site_name} MIN EXPOSURE TEST"
    )
    print("-" * 88)

    print(
        f"Player ID:             "
        f"{minimum_player}"
    )

    print(
        f"Minimum required:      "
        f"{minimum_required:.0%}"
    )

    print(
        f"Realized exposure:     "
        f"{minimum_realized:.0%}"
    )

    impossible_player_id = (
        "__BURN1_IMPOSSIBLE_PLAYER__"
    )

    try:
        select_nfl_portfolio(
            candidates,
            lineup_count=20,
            min_exposures={
                impossible_player_id: 0.50,
            },
        )

    except Stage2InfeasibleError as exc:
        print()
        print(
            f"{site_name} INFEASIBILITY "
            "TEST PASSED"
        )
        print("-" * 88)
        print(str(exc))

    else:
        raise ValueError(
            f"{site_name} Stage 2 "
            "infeasibility test FAILED."
        )

    print()
    print(
        f"{site_name} STAGE 2 TEST PASSED"
    )


def main():
    dk_players = load_final_pool(
        DK_POOL_FILE,
        "DraftKings",
    )

    fd_players = load_final_pool(
        FD_POOL_FILE,
        "FanDuel",
    )

    print()
    print(
        f"Loaded DraftKings players: "
        f"{len(dk_players)}"
    )

    print(
        f"Loaded FanDuel players:    "
        f"{len(fd_players)}"
    )

    test_site(
        "DraftKings",
        dk_players,
        salary_cap=50000,
    )

    test_site(
        "FanDuel",
        fd_players,
        salary_cap=60000,
    )

    print()
    print("=" * 88)
    print(
        "SHARED DK + FD STAGE 2 "
        "GLOBAL PORTFOLIO SELECTION PASSED"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()