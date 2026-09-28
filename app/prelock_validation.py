from __future__ import annotations

import math
from collections import Counter

from models.player import Player


NFL_POSITIONS = {
    "QB",
    "RB",
    "WR",
    "TE",
    "DST",
}

NFL_ROSTER_SLOTS = {
    "QB",
    "RB",
    "WR",
    "TE",
    "FLEX",
    "DST",
}

UNAVAILABLE_STATUSES = {
    "OUT",
    "IR",
    "O",
}

EXPECTED_DISPLAY_SLOTS = Counter(
    {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "FLEX": 1,
        "DST": 1,
    }
)


class PreLockValidationError(ValueError):
    pass


def _clean(value):
    if value is None:
        return ""

    return str(value).strip()


def validate_player_pool(
    players: list[Player],
    *,
    site: str,
    salary_cap: int,
    locked_player_ids=None,
    excluded_player_ids=None,
    min_player_exposures=None,
    max_player_exposures=None,
):
    locked_player_ids = set(
        locked_player_ids or set()
    )

    excluded_player_ids = set(
        excluded_player_ids or set()
    )

    min_player_exposures = dict(
        min_player_exposures or {}
    )

    max_player_exposures = dict(
        max_player_exposures or {}
    )

    if not players:
        raise PreLockValidationError(
            "Pre-lock validation failed: "
            "player pool is empty."
        )

    seen_ids = set()

    for index, player in enumerate(
        players,
        start=1,
    ):
        player_id = _clean(
            player.player_id
        )

        name = _clean(
            player.name
        )

        team = _clean(
            player.team
        )

        opponent = _clean(
            player.opponent
        )

        position = _clean(
            player.position
        ).upper()

        status = _clean(
            player.status
        ).upper()

        if not player_id:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"player row {index} has no player_id."
            )

        if player_id in seen_ids:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"duplicate player_id {player_id}."
            )

        seen_ids.add(
            player_id
        )

        if not name:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"player {player_id} has no name."
            )

        if not team:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} ({player_id}) has no team."
            )

        if not opponent:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} ({player_id}) has no opponent."
            )

        if team == opponent:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has the same team and opponent "
                f"({team})."
            )

        if position not in NFL_POSITIONS:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has unsupported position "
                f"{position!r}."
            )

        roster_positions = {
            _clean(value).upper()
            for value in (
                player.roster_positions or ()
            )
            if _clean(value)
        }

        if not roster_positions:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has no roster eligibility."
            )

        invalid_roster_positions = (
            roster_positions
            - NFL_ROSTER_SLOTS
        )

        if invalid_roster_positions:
            invalid = sorted(
                invalid_roster_positions
            )[0]

            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has unsupported roster "
                f"eligibility {invalid!r}."
            )

        try:
            salary = int(
                player.salary
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has invalid salary "
                f"{player.salary!r}."
            ) from exc

        if salary <= 0:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has non-positive salary "
                f"{salary}."
            )

        if salary > salary_cap:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} salary {salary} exceeds "
                f"{site} salary cap {salary_cap}."
            )

        try:
            projection = float(
                player.projection
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has invalid projection "
                f"{player.projection!r}."
            ) from exc

        if not math.isfinite(
            projection
        ):
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has a non-finite projection."
            )

        if projection < 0:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"{name} has negative projection "
                f"{projection}."
            )

        if (
            player_id in locked_player_ids
            and status in UNAVAILABLE_STATUSES
        ):
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"locked player {name} is unavailable "
                f"with status {status}."
            )

    conflicts = (
        locked_player_ids
        & excluded_player_ids
    )

    if conflicts:
        player_id = sorted(
            conflicts
        )[0]

        raise PreLockValidationError(
            "Pre-lock validation failed: "
            f"player {player_id} is both locked "
            "and excluded."
        )

    referenced_ids = (
        locked_player_ids
        | excluded_player_ids
        | set(
            min_player_exposures
        )
        | set(
            max_player_exposures
        )
    )

    missing_references = (
        referenced_ids
        - seen_ids
    )

    if missing_references:
        player_id = sorted(
            missing_references
        )[0]

        raise PreLockValidationError(
            "Pre-lock validation failed: "
            f"constraint references unknown "
            f"player_id {player_id}."
        )

    eligible_players = [
        player
        for player in players
        if _clean(
            player.status
        ).upper()
        not in UNAVAILABLE_STATUSES
        and _clean(
            player.player_id
        )
        not in excluded_player_ids
    ]

    eligible_counts = Counter(
        _clean(
            player.position
        ).upper()
        for player in eligible_players
    )

    minimum_position_counts = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "DST": 1,
    }

    for position, required in (
        minimum_position_counts.items()
    ):
        available = eligible_counts.get(
            position,
            0,
        )

        if available < required:
            raise PreLockValidationError(
                "Pre-lock validation failed: "
                f"only {available} eligible {position} "
                f"players remain; at least {required} "
                "are required."
            )

    return {
        "players": len(
            players
        ),
        "eligible_players": len(
            eligible_players
        ),
        "unavailable_players": (
            len(players)
            - len(eligible_players)
        ),
        "zero_projection_players": sum(
            1
            for player in players
            if float(
                player.projection
            )
            == 0.0
        ),
    }


def validate_final_lineups(
    lineups,
    *,
    site: str,
    salary_cap: int,
    expected_lineup_count: int,
):
    if len(lineups) != expected_lineup_count:
        raise PreLockValidationError(
            "Final validation failed: "
            f"expected {expected_lineup_count} "
            f"lineups but received {len(lineups)}."
        )

    seen_lineups = set()

    for lineup_number, lineup in enumerate(
        lineups,
        start=1,
    ):
        players = lineup.get(
            "players",
            []
        )

        if len(players) != 9:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} has "
                f"{len(players)} players instead of 9."
            )

        player_ids = [
            _clean(
                player.get(
                    "player_id"
                )
            )
            for player in players
        ]

        if any(
            not player_id
            for player_id in player_ids
        ):
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} contains "
                "a missing player_id."
            )

        if len(
            set(player_ids)
        ) != 9:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} contains "
                "a duplicate player."
            )

        lineup_key = tuple(
            sorted(
                player_ids
            )
        )

        if lineup_key in seen_lineups:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} duplicates "
                "another final lineup."
            )

        seen_lineups.add(
            lineup_key
        )

        roster_slots = Counter(
            _clean(
                player.get(
                    "roster_slot"
                )
            ).upper()
            for player in players
        )

        if roster_slots != EXPECTED_DISPLAY_SLOTS:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} has invalid "
                f"roster construction {dict(roster_slots)}."
            )

        salary = sum(
            int(
                player.get(
                    "salary",
                    0,
                )
            )
            for player in players
        )

        if salary > salary_cap:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} salary "
                f"{salary} exceeds {site} cap "
                f"{salary_cap}."
            )

        reported_salary = int(
            lineup.get(
                "total_salary",
                salary,
            )
        )

        if reported_salary != salary:
            raise PreLockValidationError(
                "Final validation failed: "
                f"lineup {lineup_number} salary "
                "does not match its serialized total."
            )

        for player in players:
            projection = float(
                player.get(
                    "projection",
                    0.0,
                )
            )

            if (
                not math.isfinite(
                    projection
                )
                or projection < 0
            ):
                raise PreLockValidationError(
                    "Final validation failed: "
                    f"lineup {lineup_number} contains "
                    "an invalid projection."
                )

    return {
        "lineups": len(
            lineups
        ),
        "unique_lineups": len(
            seen_lineups
        ),
    }
