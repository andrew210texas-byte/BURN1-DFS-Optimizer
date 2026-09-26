from ortools.sat.python import cp_model

from models.lineup_player import LineupPlayer


SKILL_POSITIONS = {
    "RB",
    "WR",
    "TE",
}


def optimize_nfl_lineup(
    players,
    salary_cap,
    gpp_mode=False,
    qb_stack_min=1,
    bring_back_min=0,
    rb_dst_stack=False,
    excluded_lineups=None,
    min_unique_players=1,
):
    eligible_players = [
        player
        for player in players
        if player.status.upper() not in {
            "OUT",
            "IR",
            "O",
        }
    ]

    model = cp_model.CpModel()

    roster_slots = [
        "QB",
        "RB1",
        "RB2",
        "WR1",
        "WR2",
        "WR3",
        "TE",
        "FLEX",
        "DST",
    ]

    slot_position = {
        "QB": "QB",
        "RB1": "RB",
        "RB2": "RB",
        "WR1": "WR",
        "WR2": "WR",
        "WR3": "WR",
        "TE": "TE",
        "FLEX": "FLEX",
        "DST": "DST",
    }

    selected = {}

    for player_index, player in enumerate(
        eligible_players
    ):
        for slot in roster_slots:
            required_position = (
                slot_position[slot]
            )

            if required_position in (
                player.roster_positions
            ):
                selected[
                    player_index,
                    slot,
                ] = model.new_bool_var(
                    f"player_{player_index}_{slot}"
                )

    # Every roster slot must contain exactly one player.
    for slot in roster_slots:
        model.add(
            sum(
                selected[
                    player_index,
                    slot,
                ]
                for player_index, player
                in enumerate(
                    eligible_players
                )
                if (
                    player_index,
                    slot,
                ) in selected
            )
            == 1
        )

    # One player can occupy at most one roster slot.
    for player_index, player in enumerate(
        eligible_players
    ):
        model.add(
            sum(
                selected[
                    player_index,
                    slot,
                ]
                for slot in roster_slots
                if (
                    player_index,
                    slot,
                ) in selected
            )
            <= 1
        )

    # Build a simple "is this player used?" variable.
    used = {}

    for player_index, player in enumerate(
        eligible_players
    ):
        used[player_index] = (
            model.new_bool_var(
                f"used_{player_index}"
            )
        )

        model.add(
            used[player_index]
            == sum(
                selected[
                    player_index,
                    slot,
                ]
                for slot in roster_slots
                if (
                    player_index,
                    slot,
                ) in selected
            )
        )

    # Salary cap.
    model.add(
        sum(
            selected[
                player_index,
                slot,
            ]
            * player.salary
            for player_index, player
            in enumerate(
                eligible_players
            )
            for slot in roster_slots
            if (
                player_index,
                slot,
            ) in selected
        )
        <= salary_cap
    )

    # ==============================================================
    # GPP CORRELATION RULES
    # ==============================================================

    if gpp_mode:

        quarterback_indexes = [
            index
            for index, player
            in enumerate(
                eligible_players
            )
            if player.position == "QB"
        ]

        dst_indexes = [
            index
            for index, player
            in enumerate(
                eligible_players
            )
            if player.position == "DST"
        ]

        # ----------------------------------------------------------
        # QB STACK
        #
        # If a QB is selected, require at least N same-team
        # RB/WR/TE players.
        # ----------------------------------------------------------

        if qb_stack_min > 0:

            for qb_index in (
                quarterback_indexes
            ):
                qb = eligible_players[
                    qb_index
                ]

                teammate_indexes = [
                    index
                    for index, player
                    in enumerate(
                        eligible_players
                    )
                    if (
                        player.team == qb.team
                        and player.position
                        in SKILL_POSITIONS
                    )
                ]

                if (
                    len(teammate_indexes)
                    < qb_stack_min
                ):
                    model.add(
                        used[qb_index] == 0
                    )

                else:
                    model.add(
                        sum(
                            used[index]
                            for index
                            in teammate_indexes
                        )
                        >= (
                            qb_stack_min
                            * used[qb_index]
                        )
                    )

        # ----------------------------------------------------------
        # BRING-BACK
        #
        # If a QB is selected, require at least N opposing
        # RB/WR/TE players.
        # ----------------------------------------------------------

        if bring_back_min > 0:

            for qb_index in (
                quarterback_indexes
            ):
                qb = eligible_players[
                    qb_index
                ]

                opponent_indexes = [
                    index
                    for index, player
                    in enumerate(
                        eligible_players
                    )
                    if (
                        player.team
                        == qb.opponent
                        and player.position
                        in SKILL_POSITIONS
                    )
                ]

                if (
                    len(opponent_indexes)
                    < bring_back_min
                ):
                    model.add(
                        used[qb_index] == 0
                    )

                else:
                    model.add(
                        sum(
                            used[index]
                            for index
                            in opponent_indexes
                        )
                        >= (
                            bring_back_min
                            * used[qb_index]
                        )
                    )

        # ----------------------------------------------------------
        # RB + DST
        #
        # If a DST is selected, require at least one RB from
        # that same team.
        # ----------------------------------------------------------

        if rb_dst_stack:

            for dst_index in (
                dst_indexes
            ):
                dst = eligible_players[
                    dst_index
                ]

                rb_indexes = [
                    index
                    for index, player
                    in enumerate(
                        eligible_players
                    )
                    if (
                        player.position == "RB"
                        and player.team
                        == dst.team
                    )
                ]

                if not rb_indexes:
                    model.add(
                        used[dst_index] == 0
                    )

                else:
                    model.add(
                        sum(
                            used[index]
                            for index
                            in rb_indexes
                        )
                        >= used[dst_index]
                    )

        # ----------------------------------------------------------
        # QB vs OPPOSING DST
        #
        # Do not roster a QB against the defense he is facing.
        # ----------------------------------------------------------

        for qb_index in (
            quarterback_indexes
        ):
            qb = eligible_players[
                qb_index
            ]

            for dst_index in (
                dst_indexes
            ):
                dst = eligible_players[
                    dst_index
                ]

                if (
                    dst.team
                    == qb.opponent
                ):
                    model.add(
                        used[qb_index]
                        + used[dst_index]
                        <= 1
                    )

    # ==============================================================
    # PORTFOLIO UNIQUENESS
    # ==============================================================
    #
    # Each previously generated lineup is represented by its player
    # IDs. A new lineup may overlap with it by at most:
    #
    #     roster size - minimum unique players
    #
    # With a nine-player NFL roster and min_unique_players=2, for
    # example, the next lineup can share at most seven players with
    # any previous lineup.
    #
    # This is inactive for existing single-lineup calls.

    if excluded_lineups:
        if min_unique_players < 1:
            raise ValueError(
                "min_unique_players must be at least 1."
            )

        roster_size = len(roster_slots)

        if min_unique_players > roster_size:
            raise ValueError(
                "min_unique_players cannot exceed roster size."
            )

        maximum_overlap = (
            roster_size - min_unique_players
        )

        for previous_lineup in excluded_lineups:
            previous_ids = set(previous_lineup)

            overlap_indexes = [
                index
                for index, player
                in enumerate(eligible_players)
                if player.player_id in previous_ids
            ]

            if overlap_indexes:
                model.add(
                    sum(
                        used[index]
                        for index in overlap_indexes
                    )
                    <= maximum_overlap
                )

    # ==============================================================
    # OBJECTIVE
    # ==============================================================

    projection_scale = 100

    model.maximize(
        sum(
            selected[
                player_index,
                slot,
            ]
            * int(
                round(
                    player.projection
                    * projection_scale
                )
            )
            for player_index, player
            in enumerate(
                eligible_players
            )
            for slot in roster_slots
            if (
                player_index,
                slot,
            ) in selected
        )
    )

    solver = cp_model.CpSolver()

    status = solver.solve(model)

    if status not in {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }:
        return []

    lineup = []

    for slot in roster_slots:
        for player_index, player in enumerate(
            eligible_players
        ):
            if (
                player_index,
                slot,
            ) not in selected:
                continue

            if (
                solver.value(
                    selected[
                        player_index,
                        slot,
                    ]
                )
                == 1
            ):
                display_slot = slot

                if slot in {
                    "RB1",
                    "RB2",
                }:
                    display_slot = "RB"

                elif slot in {
                    "WR1",
                    "WR2",
                    "WR3",
                }:
                    display_slot = "WR"

                lineup.append(
                    LineupPlayer(
                        roster_slot=(
                            display_slot
                        ),
                        player=player,
                    )
                )

                break

    return lineup
