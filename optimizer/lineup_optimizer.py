from ortools.sat.python import cp_model

from models.lineup_player import LineupPlayer


def optimize_nfl_lineup(players, salary_cap):
    eligible_players = [
        player
        for player in players
        if player.status.upper() not in {"OUT", "IR"}
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

    slot_eligibility = {
        "QB": {"QB"},
        "RB1": {"RB"},
        "RB2": {"RB"},
        "WR1": {"WR"},
        "WR2": {"WR"},
        "WR3": {"WR"},
        "TE": {"TE"},
        "FLEX": {"RB", "WR", "TE"},
        "DST": {"DST"},
    }

    selected = {}

    for player_index, player in enumerate(eligible_players):
        for slot in roster_slots:
            if player.position in slot_eligibility[slot]:
                selected[player_index, slot] = model.new_bool_var(
                    f"player_{player_index}_{slot}"
                )

    # Every roster slot must contain exactly one player.
    for slot in roster_slots:
        model.add(
            sum(
                selected[player_index, slot]
                for player_index, player in enumerate(eligible_players)
                if (player_index, slot) in selected
            )
            == 1
        )

    # A player can only appear once in a lineup.
    for player_index, player in enumerate(eligible_players):
        model.add(
            sum(
                selected[player_index, slot]
                for slot in roster_slots
                if (player_index, slot) in selected
            )
            <= 1
        )

    # Total salary cannot exceed the site's salary cap.
    model.add(
        sum(
            selected[player_index, slot] * player.salary
            for player_index, player in enumerate(eligible_players)
            for slot in roster_slots
            if (player_index, slot) in selected
        )
        <= salary_cap
    )

    # CP-SAT optimizes integers, so scale decimal fantasy points by 100.
    projection_scale = 100

    model.maximize(
        sum(
            selected[player_index, slot]
            * int(round(player.projection * projection_scale))
            for player_index, player in enumerate(eligible_players)
            for slot in roster_slots
            if (player_index, slot) in selected
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
        for player_index, player in enumerate(eligible_players):
            if (player_index, slot) not in selected:
                continue

            if solver.value(selected[player_index, slot]) == 1:
                display_slot = slot

                if slot in {"RB1", "RB2"}:
                    display_slot = "RB"
                elif slot in {"WR1", "WR2", "WR3"}:
                    display_slot = "WR"

                lineup.append(
                    LineupPlayer(
                        roster_slot=display_slot,
                        player=player,
                    )
                )

                break

    return lineup