from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import pandas as pd

from optimizer.portfolio_optimizer import (
    CandidatePoolResult,
    _player_ids,
    _profile_lineup,
    generate_nfl_candidates,
    optimize_nfl_lineup,
)


@dataclass(frozen=True)
class CandidateDiversityStats:
    requested_count: int
    generated_count: int
    stage1_player_cap: float | None
    max_player_appearances: int | None
    unique_players: int
    actual_max_candidate_exposure: float


def _candidate_gpp_flags(
    candidate_count: int,
    gpp_mode: bool,
    candidate_gpp_fraction: float | None,
) -> list[bool]:

    if candidate_gpp_fraction is None:
        return [
            gpp_mode
            for _ in range(candidate_count)
        ]

    if not 0.0 <= candidate_gpp_fraction <= 1.0:
        raise ValueError(
            "candidate_gpp_fraction must be between 0 and 1."
        )

    gpp_target_count = int(
        math.floor(
            candidate_count
            * candidate_gpp_fraction
            + 0.5
        )
    )

    flags = []
    assigned = 0

    for candidate_number in range(
        1,
        candidate_count + 1,
    ):
        target_through_candidate = int(
            math.floor(
                candidate_number
                * gpp_target_count
                / candidate_count
                + 0.5
            )
        )

        use_gpp = (
            target_through_candidate
            > assigned
        )

        flags.append(use_gpp)

        if use_gpp:
            assigned += 1

    return flags


def generate_nfl_candidates_with_player_cap(
    players,
    salary_cap: int,
    candidate_count: int = 200,
    min_unique_players: int = 2,
    stage1_player_cap: float | None = None,
    gpp_mode: bool = False,
    qb_stack_min: int = 1,
    bring_back_min: int = 0,
    rb_dst_stack: bool = False,
    candidate_gpp_fraction: float | None = None,
    candidate_solver_time_limit_seconds: float | None = 1.0,
    candidate_solver_relative_gap_limit: float | None = 0.001,
    progress_callback=None,
):
    """
    Experimental Stage-1 generator.

    stage1_player_cap controls the maximum number of candidate lineups
    in which any individual player may appear.

    Example:
        candidate_count=200
        stage1_player_cap=0.80

    means no player may appear in more than 160 generated candidates.

    The production optimizer is NOT modified by this experiment.
    """

    if stage1_player_cap is None:
        pool = generate_nfl_candidates(
            players=players,
            salary_cap=salary_cap,
            candidate_count=candidate_count,
            min_unique_players=min_unique_players,
            gpp_mode=gpp_mode,
            qb_stack_min=qb_stack_min,
            bring_back_min=bring_back_min,
            rb_dst_stack=rb_dst_stack,
            candidate_gpp_fraction=(
                candidate_gpp_fraction
            ),
            candidate_solver_time_limit_seconds=(
                candidate_solver_time_limit_seconds
            ),
            candidate_solver_relative_gap_limit=(
                candidate_solver_relative_gap_limit
            ),
            progress_callback=progress_callback,
        )

        return pool

    if not 0.0 < stage1_player_cap <= 1.0:
        raise ValueError(
            "stage1_player_cap must be > 0 and <= 1."
        )

    maximum_appearances = int(
        math.floor(
            candidate_count
            * stage1_player_cap
            + 1e-9
        )
    )

    maximum_appearances = max(
        1,
        maximum_appearances,
    )

    candidate_gpp_flags = (
        _candidate_gpp_flags(
            candidate_count,
            gpp_mode,
            candidate_gpp_fraction,
        )
    )

    lineups = []
    profiles = []
    excluded_lineups = []

    appearance_counts = Counter()

    first_solve_by_mode = {
        False: True,
        True: True,
    }

    for candidate_number in range(
        1,
        candidate_count + 1,
    ):
        capped_ids = {
            player_id
            for (
                player_id,
                appearances,
            ) in appearance_counts.items()
            if appearances
            >= maximum_appearances
        }

        available_players = [
            player
            for player in players
            if str(player.player_id)
            not in capped_ids
        ]

        candidate_uses_gpp = (
            candidate_gpp_flags[
                candidate_number - 1
            ]
        )

        first_solve = (
            first_solve_by_mode[
                candidate_uses_gpp
            ]
        )

        lineup = optimize_nfl_lineup(
            available_players,
            salary_cap=salary_cap,
            gpp_mode=candidate_uses_gpp,
            qb_stack_min=qb_stack_min,
            bring_back_min=bring_back_min,
            rb_dst_stack=rb_dst_stack,
            excluded_lineups=excluded_lineups,
            min_unique_players=(
                min_unique_players
            ),
            solver_time_limit_seconds=(
                None
                if first_solve
                else candidate_solver_time_limit_seconds
            ),
            solver_relative_gap_limit=(
                None
                if first_solve
                else candidate_solver_relative_gap_limit
            ),
        )

        first_solve_by_mode[
            candidate_uses_gpp
        ] = False

        if not lineup:
            break

        ids = [
            str(player_id)
            for player_id
            in _player_ids(lineup)
        ]

        lineups.append(lineup)
        excluded_lineups.append(ids)

        profiles.append(
            _profile_lineup(
                lineup,
                candidate_number,
            )
        )

        for player_id in set(ids):
            appearance_counts[
                player_id
            ] += 1

        if progress_callback is not None:
            progress_callback(
                candidate_number,
                candidate_count,
            )

    return CandidatePoolResult(
        lineups=lineups,
        profiles=profiles,
        requested_count=candidate_count,
        generated_count=len(lineups),
    )


def candidate_exposure_report(
    candidate_pool,
    players,
) -> pd.DataFrame:

    player_lookup = {
        str(player.player_id): player
        for player in players
    }

    counts = Counter()

    for profile in candidate_pool.profiles:
        for player_id in set(
            profile.player_ids
        ):
            counts[
                str(player_id)
            ] += 1

    total = len(
        candidate_pool.lineups
    )

    rows = []

    for player_id, count in counts.items():
        player = player_lookup.get(
            str(player_id)
        )

        if player is None:
            continue

        rows.append(
            {
                "player_id":
                    str(player_id),
                "player_name":
                    player.name,
                "position":
                    player.position,
                "team":
                    player.team,
                "salary":
                    player.salary,
                "projection":
                    round(
                        float(
                            player.projection
                        ),
                        4,
                    ),
                "candidate_count":
                    count,
                "candidate_exposure":
                    (
                        count / total
                        if total
                        else 0.0
                    ),
                "candidate_exposure_pct":
                    (
                        round(
                            count
                            / total
                            * 100,
                            1,
                        )
                        if total
                        else 0.0
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "candidate_count",
                "projection",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )
