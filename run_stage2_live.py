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

LINEUP_COUNT = 20
CANDIDATE_COUNT = 200


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


def print_section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_stage2_result(
    title,
    result,
):
    print_section(title)

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
                lineup_count=LINEUP_COUNT,
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
        "Stage 2 solution."
    )


def find_forced_minimum_test(
    candidates,
    unconstrained,
):
    candidate_counts = {}

    for profile in candidates.profiles:
        for player_id in set(
            profile.player_ids
        ):
            candidate_counts[player_id] = (
                candidate_counts.get(
                    player_id,
                    0,
                )
                + 1
            )

    options = []

    for player_id, candidate_count in (
        candidate_counts.items()
    ):
        original_count = (
            unconstrained.player_counts.get(
                player_id,
                0,
            )
        )

        maximum_possible_count = min(
            candidate_count,
            LINEUP_COUNT,
        )

        if (
            original_count
            >= maximum_possible_count
        ):
            continue

        target_count = original_count + 1

        if target_count < 2:
            target_count = 2

        if (
            target_count
            > maximum_possible_count
        ):
            continue

        target_exposure = (
            target_count
            / LINEUP_COUNT
        )

        options.append(
            (
                original_count,
                player_id,
                target_exposure,
            )
        )

    options.sort(
        reverse=True,
    )

    for (
        original_count,
        player_id,
        target_exposure,
    ) in options:
        try:
            result = select_nfl_portfolio(
                candidates,
                lineup_count=LINEUP_COUNT,
                min_exposures={
                    player_id: target_exposure,
                },
            )

        except Stage2InfeasibleError:
            continue

        realized_count = (
            result.player_counts.get(
                player_id,
                0,
            )
        )

        if realized_count > original_count:
            return (
                player_id,
                original_count,
                target_exposure,
                result,
            )

    raise Stage2InfeasibleError(
        "Could not find a player whose "
        "minimum exposure could be forced "
        "above the unconstrained result."
    )


def find_feasible_lock_test(
    candidates,
    unconstrained,
):
    candidate_counts = {}

    for profile in candidates.profiles:
        for player_id in set(
            profile.player_ids
        ):
            candidate_counts[player_id] = (
                candidate_counts.get(
                    player_id,
                    0,
                )
                + 1
            )

    ranked_players = sorted(
        candidate_counts.items(),
        key=lambda item: (
            item[1],
            unconstrained.exposures.get(
                item[0],
                0.0,
            ),
        ),
        reverse=True,
    )

    for player_id, candidate_count in ranked_players:
        original_exposure = (
            unconstrained.exposures.get(
                player_id,
                0.0,
            )
        )

        if original_exposure >= 1.0:
            continue

        if candidate_count < LINEUP_COUNT:
            continue

        try:
            result = select_nfl_portfolio(
                candidates,
                lineup_count=LINEUP_COUNT,
                locked_player_ids={
                    player_id,
                },
            )

        except Stage2InfeasibleError:
            continue

        realized = result.exposures.get(
            player_id,
            0.0,
        )

        if realized == 1.0:
            return (
                player_id,
                original_exposure,
                result,
            )

    raise Stage2InfeasibleError(
        "Could not find a player that can "
        "be locked into all selected lineups."
    )


def find_feasible_exclude_test(
    candidates,
    unconstrained,
):
    ranked_players = sorted(
        unconstrained.exposures.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for player_id, original_exposure in ranked_players:
        if original_exposure <= 0.0:
            continue

        try:
            result = select_nfl_portfolio(
                candidates,
                lineup_count=LINEUP_COUNT,
                excluded_player_ids={
                    player_id,
                },
            )

        except Stage2InfeasibleError:
            continue

        realized = result.exposures.get(
            player_id,
            0.0,
        )

        if realized == 0.0:
            return (
                player_id,
                original_exposure,
                result,
            )

    raise Stage2InfeasibleError(
        "Could not find a player that can "
        "be excluded while retaining a "
        "feasible portfolio."
    )


def strategy_matches(
    profile,
    strategy_name,
):
    if strategy_name == "stacked":
        return not profile.is_unstacked

    if strategy_name == "unstacked":
        return profile.is_unstacked

    if strategy_name == "qb_stack_1_plus":
        return profile.qb_stack_size >= 1

    if strategy_name == "qb_stack_2_plus":
        return profile.qb_stack_size >= 2

    if strategy_name == "bring_back_1_plus":
        return profile.bring_back_size >= 1

    if strategy_name == "rb_dst":
        return profile.has_rb_dst

    if strategy_name == "qb_vs_opposing_dst":
        return profile.has_qb_vs_opposing_dst

    raise ValueError(
        f"Unknown strategy: {strategy_name}"
    )


def strategy_count(
    profiles,
    strategy_name,
):
    return sum(
        1
        for profile in profiles
        if strategy_matches(
            profile,
            strategy_name,
        )
    )


def find_strategy_control_test(
    candidates,
    unconstrained,
):
    strategy_names = (
        "stacked",
        "unstacked",
        "qb_stack_1_plus",
        "qb_stack_2_plus",
        "bring_back_1_plus",
        "rb_dst",
    )

    for strategy_name in strategy_names:
        original_count = strategy_count(
            unconstrained.profiles,
            strategy_name,
        )

        candidate_count = strategy_count(
            candidates.profiles,
            strategy_name,
        )

        maximum_possible = min(
            candidate_count,
            LINEUP_COUNT,
        )

        if original_count < maximum_possible:
            target_count = (
                original_count + 1
            )

            target_exposure = (
                target_count
                / LINEUP_COUNT
            )

            try:
                result = select_nfl_portfolio(
                    candidates,
                    lineup_count=LINEUP_COUNT,
                    min_strategy_exposures={
                        strategy_name: (
                            target_exposure
                        ),
                    },
                )

            except Stage2InfeasibleError:
                pass

            else:
                realized_count = (
                    strategy_count(
                        result.profiles,
                        strategy_name,
                    )
                )

                if (
                    realized_count
                    > original_count
                ):
                    return (
                        strategy_name,
                        "minimum",
                        original_count,
                        target_count,
                        realized_count,
                        result,
                    )

        if original_count > 0:
            target_count = (
                original_count - 1
            )

            target_exposure = (
                target_count
                / LINEUP_COUNT
            )

            try:
                result = select_nfl_portfolio(
                    candidates,
                    lineup_count=LINEUP_COUNT,
                    max_strategy_exposures={
                        strategy_name: (
                            target_exposure
                        ),
                    },
                )

            except Stage2InfeasibleError:
                continue

            realized_count = strategy_count(
                result.profiles,
                strategy_name,
            )

            if realized_count < original_count:
                return (
                    strategy_name,
                    "maximum",
                    original_count,
                    target_count,
                    realized_count,
                    result,
                )

    raise Stage2InfeasibleError(
        "Could not find a feasible strategy "
        "constraint that changes the "
        "unconstrained portfolio."
    )



def find_combined_control_test(
    candidates,
    unconstrained,
):
    lock_options = []
    candidate_counts = {}

    for profile in candidates.profiles:
        for player_id in set(profile.player_ids):
            candidate_counts[player_id] = (
                candidate_counts.get(player_id, 0) + 1
            )

    for player_id, candidate_count in candidate_counts.items():
        original_exposure = unconstrained.exposures.get(
            player_id,
            0.0,
        )

        if (
            original_exposure < 1.0
            and candidate_count >= LINEUP_COUNT
        ):
            lock_options.append(
                (original_exposure, player_id)
            )

    lock_options.sort(reverse=True)

    exclude_options = sorted(
        unconstrained.exposures.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    strategy_names = (
        "stacked",
        "unstacked",
        "qb_stack_1_plus",
        "qb_stack_2_plus",
        "bring_back_1_plus",
        "rb_dst",
    )

    for lock_before, lock_player in lock_options:
        for exclude_player, exclude_before in exclude_options:
            if exclude_player == lock_player:
                continue

            for strategy_name in strategy_names:
                original_strategy_count = strategy_count(
                    unconstrained.profiles,
                    strategy_name,
                )

                candidate_strategy_count = strategy_count(
                    candidates.profiles,
                    strategy_name,
                )

                if (
                    original_strategy_count
                    < min(candidate_strategy_count, LINEUP_COUNT)
                ):
                    target_count = original_strategy_count + 1
                    strategy_kwargs = {
                        "min_strategy_exposures": {
                            strategy_name: target_count / LINEUP_COUNT,
                        }
                    }
                    boundary_type = "minimum"
                elif original_strategy_count > 0:
                    target_count = original_strategy_count - 1
                    strategy_kwargs = {
                        "max_strategy_exposures": {
                            strategy_name: target_count / LINEUP_COUNT,
                        }
                    }
                    boundary_type = "maximum"
                else:
                    continue

                try:
                    result = select_nfl_portfolio(
                        candidates,
                        lineup_count=LINEUP_COUNT,
                        locked_player_ids={lock_player},
                        excluded_player_ids={exclude_player},
                        **strategy_kwargs,
                    )
                except Stage2InfeasibleError:
                    continue

                lock_after = result.exposures.get(
                    lock_player,
                    0.0,
                )
                exclude_after = result.exposures.get(
                    exclude_player,
                    0.0,
                )
                strategy_after = strategy_count(
                    result.profiles,
                    strategy_name,
                )

                strategy_ok = (
                    strategy_after >= target_count
                    if boundary_type == "minimum"
                    else strategy_after <= target_count
                )

                if (
                    lock_after == 1.0
                    and exclude_after == 0.0
                    and strategy_ok
                    and result.selected_lineup_count == LINEUP_COUNT
                ):
                    return {
                        "lock_player": lock_player,
                        "lock_before": lock_before,
                        "lock_after": lock_after,
                        "exclude_player": exclude_player,
                        "exclude_before": exclude_before,
                        "exclude_after": exclude_after,
                        "strategy_name": strategy_name,
                        "strategy_type": boundary_type,
                        "strategy_target": target_count,
                        "strategy_after": strategy_after,
                        "result": result,
                    }

    raise Stage2InfeasibleError(
        "Could not find a feasible combined lock, exclude, "
        "and strategy-control portfolio."
    )

def test_site(
    site_name,
    players,
    salary_cap,
):
    print_section(
        f"{site_name.upper()} "
        "STAGE 1 CANDIDATE GENERATION"
    )

    candidates = generate_nfl_candidates(
        players,
        salary_cap=salary_cap,
        candidate_count=CANDIDATE_COUNT,
        min_unique_players=2,
        gpp_mode=False,
    )

    print(
        f"Generated: "
        f"{candidates.generated_count}/"
        f"{candidates.requested_count}"
    )

    if (
        candidates.generated_count
        < LINEUP_COUNT
    ):
        raise ValueError(
            f"{site_name} did not generate "
            "enough candidates."
        )

    unconstrained = select_nfl_portfolio(
        candidates,
        lineup_count=LINEUP_COUNT,
    )

    if (
        unconstrained.selected_lineup_count
        != LINEUP_COUNT
    ):
        raise ValueError(
            f"{site_name} Stage 2 did not "
            "select exactly "
            f"{LINEUP_COUNT} lineups."
        )

    print_stage2_result(
        f"{site_name.upper()} "
        "UNCONSTRAINED STAGE 2",
        unconstrained,
    )

    (
        max_player,
        original_exposure,
        max_result,
    ) = find_feasible_max_exposure_test(
        candidates,
        unconstrained,
        maximum_exposure=0.50,
    )

    max_realized = (
        max_result.exposures.get(
            max_player,
            0.0,
        )
    )

    if max_realized > 0.50:
        raise ValueError(
            f"{site_name} maximum exposure "
            "constraint FAILED."
        )

    print()
    print(
        f"{site_name} MAX EXPOSURE TEST PASSED"
    )
    print("-" * 88)
    print(
        f"{max_player}: "
        f"{original_exposure:.0%} -> "
        f"{max_realized:.0%}"
    )

    (
        min_player,
        original_count,
        minimum_required,
        min_result,
    ) = find_forced_minimum_test(
        candidates,
        unconstrained,
    )

    min_realized_count = (
        min_result.player_counts.get(
            min_player,
            0,
        )
    )

    min_realized = (
        min_result.exposures.get(
            min_player,
            0.0,
        )
    )

    if (
        min_realized_count
        <= original_count
    ):
        raise ValueError(
            f"{site_name} forced minimum "
            "exposure test FAILED."
        )

    if min_realized < minimum_required:
        raise ValueError(
            f"{site_name} minimum exposure "
            "constraint FAILED."
        )

    print()
    print(
        f"{site_name} FORCED MIN EXPOSURE "
        "TEST PASSED"
    )
    print("-" * 88)
    print(
        f"{min_player}: "
        f"{original_count}/{LINEUP_COUNT} -> "
        f"{min_realized_count}/{LINEUP_COUNT} "
        f"(minimum "
        f"{minimum_required:.0%})"
    )

    (
        lock_player,
        lock_before,
        lock_result,
    ) = find_feasible_lock_test(
        candidates,
        unconstrained,
    )

    lock_after = (
        lock_result.exposures.get(
            lock_player,
            0.0,
        )
    )

    if lock_after != 1.0:
        raise ValueError(
            f"{site_name} lock test FAILED."
        )

    print()
    print(
        f"{site_name} LOCK TEST PASSED"
    )
    print("-" * 88)
    print(
        f"{lock_player}: "
        f"{lock_before:.0%} -> "
        f"{lock_after:.0%}"
    )

    (
        exclude_player,
        exclude_before,
        exclude_result,
    ) = find_feasible_exclude_test(
        candidates,
        unconstrained,
    )

    exclude_after = (
        exclude_result.exposures.get(
            exclude_player,
            0.0,
        )
    )

    if exclude_after != 0.0:
        raise ValueError(
            f"{site_name} exclude test FAILED."
        )

    print()
    print(
        f"{site_name} EXCLUDE TEST PASSED"
    )
    print("-" * 88)
    print(
        f"{exclude_player}: "
        f"{exclude_before:.0%} -> "
        f"{exclude_after:.0%}"
    )

    (
        strategy_name,
        strategy_type,
        strategy_before,
        strategy_target,
        strategy_after,
        strategy_result,
    ) = find_strategy_control_test(
        candidates,
        unconstrained,
    )

    if strategy_type == "minimum":
        if (
            strategy_after
            < strategy_target
        ):
            raise ValueError(
                f"{site_name} strategy "
                "minimum test FAILED."
            )
    else:
        if (
            strategy_after
            > strategy_target
        ):
            raise ValueError(
                f"{site_name} strategy "
                "maximum test FAILED."
            )

    print()
    print(
        f"{site_name} STRATEGY CONTROL "
        "TEST PASSED"
    )
    print("-" * 88)
    print(
        f"Strategy:              "
        f"{strategy_name}"
    )
    print(
        f"Constraint type:       "
        f"{strategy_type}"
    )
    print(
        f"Before:                "
        f"{strategy_before}/"
        f"{LINEUP_COUNT}"
    )
    print(
        f"Target boundary:       "
        f"{strategy_target}/"
        f"{LINEUP_COUNT}"
    )
    print(
        f"After:                 "
        f"{strategy_after}/"
        f"{LINEUP_COUNT}"
    )
    print(
        f"Portfolio projection:  "
        f"{strategy_result.total_projection:.2f}"
    )

    combined = find_combined_control_test(
        candidates,
        unconstrained,
    )

    combined_result = combined["result"]

    if combined_result.selected_lineup_count != LINEUP_COUNT:
        raise ValueError(
            f"{site_name} combined control test did not "
            f"return exactly {LINEUP_COUNT} lineups."
        )

    print()
    print(
        f"{site_name} COMBINED CONTROL TEST PASSED"
    )
    print("-" * 88)
    print(
        f"Lock:                  "
        f"{combined['lock_player']} "
        f"{combined['lock_before']:.0%} -> "
        f"{combined['lock_after']:.0%}"
    )
    print(
        f"Exclude:               "
        f"{combined['exclude_player']} "
        f"{combined['exclude_before']:.0%} -> "
        f"{combined['exclude_after']:.0%}"
    )
    print(
        f"Strategy:              "
        f"{combined['strategy_name']} "
        f"({combined['strategy_type']})"
    )
    print(
        f"Strategy boundary:     "
        f"{combined['strategy_target']}/{LINEUP_COUNT}"
    )
    print(
        f"Strategy realized:     "
        f"{combined['strategy_after']}/{LINEUP_COUNT}"
    )
    print(
        f"Selected lineups:      "
        f"{combined_result.selected_lineup_count}/"
        f"{LINEUP_COUNT}"
    )

    impossible_player_id = (
        "__BURN1_IMPOSSIBLE_PLAYER__"
    )

    try:
        select_nfl_portfolio(
            candidates,
            lineup_count=LINEUP_COUNT,
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

    try:
        select_nfl_portfolio(
            candidates,
            lineup_count=LINEUP_COUNT,
            locked_player_ids={
                "__BURN1_CONFLICT__",
            },
            excluded_player_ids={
                "__BURN1_CONFLICT__",
            },
        )

    except ValueError as exc:
        print()
        print(
            f"{site_name} LOCK/EXCLUDE "
            "CONFLICT TEST PASSED"
        )
        print("-" * 88)
        print(str(exc))

    else:
        raise ValueError(
            f"{site_name} lock/exclude "
            "conflict test FAILED."
        )

    print()
    print(
        f"{site_name} EXPANDED STAGE 2 "
        "CONTROL SUITE PASSED"
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

    print_section(
        "SHARED DK + FD EXPANDED STAGE 2 "
        "CONTROL SUITE PASSED"
    )


if __name__ == "__main__":
    main()