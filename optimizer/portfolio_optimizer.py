from collections import Counter
from dataclasses import dataclass

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

@dataclass
class CandidatePoolResult:
    lineups: list
    profiles: list[LineupProfile]
    requested_count: int
    generated_count: int


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
