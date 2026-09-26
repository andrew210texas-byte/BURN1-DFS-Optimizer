from collections import Counter
from dataclasses import dataclass
import math

from ortools.sat.python import cp_model

from optimizer.lineup_optimizer import optimize_nfl_lineup


@dataclass
class LineupProfile:
    lineup_number: int
    total_salary: int
    total_projection: float
    qb_stack_size: int
    bring_back_size: int
    has_rb_dst: bool
    has_qb_vs_opposing_dst: bool
    is_unstacked: bool
    player_ids: tuple[str, ...]


@dataclass
class PortfolioResult:
    lineups: list
    profiles: list[LineupProfile]
    exposures: dict[str, float]
    player_counts: dict[str, int]


@dataclass
class CandidatePoolResult:
    lineups: list
    profiles: list[LineupProfile]
    requested_count: int
    generated_count: int


@dataclass
class PortfolioSelectionResult:
    lineups: list
    profiles: list[LineupProfile]
    exposures: dict[str, float]
    player_counts: dict[str, int]
    selected_candidate_indices: list[int]
    requested_lineup_count: int
    selected_lineup_count: int
    candidate_count: int
    total_projection: float
    solver_status: str


class Stage2InfeasibleError(ValueError):
    pass


def _player_ids(lineup):
    return tuple(
        entry.player.player_id
        for entry in lineup
    )


def _profile_lineup(
    lineup,
    lineup_number,
):
    players = [
        entry.player
        for entry in lineup
    ]

    quarterback = next(
        player
        for player in players
        if player.position == "QB"
    )

    dst = next(
        player
        for player in players
        if player.position == "DST"
    )

    skill_players = [
        player
        for player in players
        if player.position in {
            "RB",
            "WR",
            "TE",
        }
    ]

    qb_teammates = [
        player
        for player in skill_players
        if player.team == quarterback.team
    ]

    bring_backs = [
        player
        for player in skill_players
        if player.team == quarterback.opponent
    ]

    same_team_dst_rbs = [
        player
        for player in skill_players
        if (
            player.position == "RB"
            and player.team == dst.team
        )
    ]

    total_salary = sum(
        player.salary
        for player in players
    )

    total_projection = sum(
        player.projection
        for player in players
    )

    return LineupProfile(
        lineup_number=lineup_number,
        total_salary=total_salary,
        total_projection=total_projection,
        qb_stack_size=len(qb_teammates),
        bring_back_size=len(bring_backs),
        has_rb_dst=bool(same_team_dst_rbs),
        has_qb_vs_opposing_dst=(
            dst.team == quarterback.opponent
        ),
        is_unstacked=(
            len(qb_teammates) == 0
        ),
        player_ids=_player_ids(lineup),
    )


def _calculate_exposures(lineups):
    counts = Counter()

    for lineup in lineups:
        counts.update(
            set(_player_ids(lineup))
        )

    lineup_count = len(lineups)

    if lineup_count == 0:
        return {}, {}

    exposures = {
        player_id: count / lineup_count
        for player_id, count
        in counts.items()
    }

    return exposures, dict(counts)


def _calculate_profile_exposures(profiles):
    counts = Counter()

    for profile in profiles:
        counts.update(
            set(profile.player_ids)
        )

    lineup_count = len(profiles)

    if lineup_count == 0:
        return {}, {}

    exposures = {
        player_id: count / lineup_count
        for player_id, count
        in counts.items()
    }

    return exposures, dict(counts)


def _validate_exposure_value(
    player_id,
    exposure,
    exposure_type,
):
    if not 0.0 <= exposure <= 1.0:
        raise ValueError(
            f"{exposure_type} exposure for "
            f"{player_id} must be between "
            "0.0 and 1.0."
        )


def _minimum_exposure_count(
    exposure,
    lineup_count,
):
    epsilon = 1e-9

    return math.ceil(
        exposure * lineup_count
        - epsilon
    )


def _maximum_exposure_count(
    exposure,
    lineup_count,
):
    epsilon = 1e-9

    return math.floor(
        exposure * lineup_count
        + epsilon
    )


def generate_nfl_portfolio(
    players,
    salary_cap,
    lineup_count,
    min_unique_players=2,
    gpp_mode=False,
    qb_stack_min=1,
    bring_back_min=0,
    rb_dst_stack=False,
):
    if lineup_count < 1:
        raise ValueError(
            "lineup_count must be at least 1."
        )

    if min_unique_players < 1:
        raise ValueError(
            "min_unique_players must be at least 1."
        )

    lineups = []
    profiles = []
    excluded_lineups = []

    for lineup_number in range(
        1,
        lineup_count + 1,
    ):
        lineup = optimize_nfl_lineup(
            players,
            salary_cap=salary_cap,
            gpp_mode=gpp_mode,
            qb_stack_min=qb_stack_min,
            bring_back_min=bring_back_min,
            rb_dst_stack=rb_dst_stack,
            excluded_lineups=excluded_lineups,
            min_unique_players=min_unique_players,
        )

        if not lineup:
            break

        player_ids = _player_ids(lineup)

        lineups.append(lineup)
        excluded_lineups.append(player_ids)

        profiles.append(
            _profile_lineup(
                lineup,
                lineup_number,
            )
        )

    exposures, player_counts = (
        _calculate_exposures(lineups)
    )

    return PortfolioResult(
        lineups=lineups,
        profiles=profiles,
        exposures=exposures,
        player_counts=player_counts,
    )


def generate_nfl_candidates(
    players,
    salary_cap,
    candidate_count=200,
    min_unique_players=2,
    gpp_mode=False,
    qb_stack_min=1,
    bring_back_min=0,
    rb_dst_stack=False,
):
    if candidate_count < 1:
        raise ValueError(
            "candidate_count must be at least 1."
        )

    if min_unique_players < 1:
        raise ValueError(
            "min_unique_players must be at least 1."
        )

    lineups = []
    profiles = []
    excluded_lineups = []

    for candidate_number in range(
        1,
        candidate_count + 1,
    ):
        lineup = optimize_nfl_lineup(
            players,
            salary_cap=salary_cap,
            gpp_mode=gpp_mode,
            qb_stack_min=qb_stack_min,
            bring_back_min=bring_back_min,
            rb_dst_stack=rb_dst_stack,
            excluded_lineups=excluded_lineups,
            min_unique_players=min_unique_players,
        )

        if not lineup:
            break

        player_ids = _player_ids(lineup)

        lineups.append(lineup)
        excluded_lineups.append(player_ids)

        profiles.append(
            _profile_lineup(
                lineup,
                candidate_number,
            )
        )

    return CandidatePoolResult(
        lineups=lineups,
        profiles=profiles,
        requested_count=candidate_count,
        generated_count=len(lineups),
    )


def select_nfl_portfolio(
    candidate_pool,
    lineup_count,
    min_exposures=None,
    max_exposures=None,
):
    if lineup_count < 1:
        raise ValueError(
            "lineup_count must be at least 1."
        )

    if not isinstance(
        candidate_pool,
        CandidatePoolResult,
    ):
        raise TypeError(
            "candidate_pool must be a "
            "CandidatePoolResult."
        )

    candidate_count = len(
        candidate_pool.lineups
    )

    if candidate_count == 0:
        raise Stage2InfeasibleError(
            "Stage 2 cannot run because "
            "the candidate pool is empty."
        )

    if len(candidate_pool.profiles) != candidate_count:
        raise ValueError(
            "Candidate lineups and profiles "
            "must have the same length."
        )

    if lineup_count > candidate_count:
        raise Stage2InfeasibleError(
            "Stage 2 cannot select "
            f"{lineup_count} lineups from only "
            f"{candidate_count} candidates."
        )

    min_exposures = (
        {}
        if min_exposures is None
        else dict(min_exposures)
    )

    max_exposures = (
        {}
        if max_exposures is None
        else dict(max_exposures)
    )

    constrained_player_ids = (
        set(min_exposures)
        | set(max_exposures)
    )

    minimum_counts = {}
    maximum_counts = {}

    for player_id in constrained_player_ids:
        minimum_exposure = min_exposures.get(
            player_id,
            0.0,
        )

        maximum_exposure = max_exposures.get(
            player_id,
            1.0,
        )

        _validate_exposure_value(
            player_id,
            minimum_exposure,
            "Minimum",
        )

        _validate_exposure_value(
            player_id,
            maximum_exposure,
            "Maximum",
        )

        if minimum_exposure > maximum_exposure:
            raise ValueError(
                f"Minimum exposure for "
                f"{player_id} cannot exceed "
                "maximum exposure."
            )

        minimum_counts[player_id] = (
            _minimum_exposure_count(
                minimum_exposure,
                lineup_count,
            )
        )

        maximum_counts[player_id] = (
            _maximum_exposure_count(
                maximum_exposure,
                lineup_count,
            )
        )

    player_candidate_indices = {}

    for candidate_index, profile in enumerate(
        candidate_pool.profiles
    ):
        for player_id in set(
            profile.player_ids
        ):
            player_candidate_indices.setdefault(
                player_id,
                [],
            ).append(candidate_index)

    if candidate_pool.lineups:
        roster_size = len(
            candidate_pool.lineups[0]
        )
    else:
        roster_size = 0

    required_slots = sum(
        minimum_counts.values()
    )

    available_slots = (
        lineup_count
        * roster_size
    )

    if required_slots > available_slots:
        raise Stage2InfeasibleError(
            "Stage 2 exposure minimums are "
            "structurally infeasible: "
            f"{required_slots} required player "
            "appearances exceed "
            f"{available_slots} available "
            "portfolio roster slots."
        )

    for player_id, minimum_count in (
        minimum_counts.items()
    ):
        available_candidate_count = len(
            player_candidate_indices.get(
                player_id,
                [],
            )
        )

        if (
            minimum_count
            > available_candidate_count
        ):
            raise Stage2InfeasibleError(
                "Stage 2 minimum exposure for "
                f"{player_id} requires "
                f"{minimum_count} lineups, but "
                "that player appears in only "
                f"{available_candidate_count} "
                "Stage 1 candidates."
            )

    model = cp_model.CpModel()

    selected = [
        model.NewBoolVar(
            f"candidate_{candidate_index}"
        )
        for candidate_index
        in range(candidate_count)
    ]

    model.Add(
        sum(selected)
        == lineup_count
    )

    for player_id in constrained_player_ids:
        candidate_indices = (
            player_candidate_indices.get(
                player_id,
                [],
            )
        )

        appearances = sum(
            selected[candidate_index]
            for candidate_index
            in candidate_indices
        )

        minimum_count = minimum_counts[
            player_id
        ]

        maximum_count = maximum_counts[
            player_id
        ]

        model.Add(
            appearances >= minimum_count
        )

        model.Add(
            appearances <= maximum_count
        )

    projection_scale = 1000

    model.Maximize(
        sum(
            round(
                profile.total_projection
                * projection_scale
            )
            * selected[candidate_index]
            for candidate_index, profile
            in enumerate(
                candidate_pool.profiles
            )
        )
    )

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        raise Stage2InfeasibleError(
            "Stage 2 portfolio selection is "
            "infeasible with the current "
            "candidate pool and exposure "
            "constraints."
        )

    selected_candidate_indices = [
        candidate_index
        for candidate_index
        in range(candidate_count)
        if solver.Value(
            selected[candidate_index]
        )
        == 1
    ]

    selected_lineups = [
        candidate_pool.lineups[
            candidate_index
        ]
        for candidate_index
        in selected_candidate_indices
    ]

    selected_profiles = [
        candidate_pool.profiles[
            candidate_index
        ]
        for candidate_index
        in selected_candidate_indices
    ]

    exposures, player_counts = (
        _calculate_profile_exposures(
            selected_profiles
        )
    )

    total_projection = sum(
        profile.total_projection
        for profile in selected_profiles
    )

    if status == cp_model.OPTIMAL:
        solver_status = "OPTIMAL"
    else:
        solver_status = "FEASIBLE"

    return PortfolioSelectionResult(
        lineups=selected_lineups,
        profiles=selected_profiles,
        exposures=exposures,
        player_counts=player_counts,
        selected_candidate_indices=(
            selected_candidate_indices
        ),
        requested_lineup_count=lineup_count,
        selected_lineup_count=len(
            selected_lineups
        ),
        candidate_count=candidate_count,
        total_projection=(
            total_projection
        ),
        solver_status=solver_status,
    )