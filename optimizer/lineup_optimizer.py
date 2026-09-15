from ortools.sat.python import cp_model


def optimize_nfl_lineup(players, salary_cap):
    eligible_players = [
        player
        for player in players
        if player.status.upper() not in {"OUT", "IR"}
    ]

    model = cp_model.CpModel()

    selected = {}

    for index, player in enumerate(eligible_players):
        selected[index] = model.new_bool_var(f"player_{index}")

    # Exactly 9 players in an NFL lineup.
    model.add(
        sum(selected.values()) == 9
    )

    # Salary cap.
    model.add(
        sum(
            selected[index] * player.salary
            for index, player in enumerate(eligible_players)
        )
        <= salary_cap
    )

    # Exactly 1 quarterback.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "QB"
        )
        == 1
    )

    # Exactly 1 defense/special teams.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "DST"
        )
        == 1
    )

    # Running backs:
    # 2 required + possibly 1 FLEX.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "RB"
        )
        >= 2
    )

    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "RB"
        )
        <= 3
    )

    # Wide receivers:
    # 3 required + possibly 1 FLEX.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "WR"
        )
        >= 3
    )

    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "WR"
        )
        <= 4
    )

    # Tight ends:
    # 1 required + possibly 1 FLEX.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "TE"
        )
        >= 1
    )

    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position == "TE"
        )
        <= 2
    )

    # RB + WR + TE must total exactly 7.
    # This accounts for:
    # 2 RB + 3 WR + 1 TE + 1 FLEX.
    model.add(
        sum(
            selected[index]
            for index, player in enumerate(eligible_players)
            if player.position in {"RB", "WR", "TE"}
        )
        == 7
    )

    # CP-SAT works with integers.
    # Multiply projections by 100 so decimal fantasy points
    # can be optimized as integers.
    projection_scale = 100

    model.maximize(
        sum(
            selected[index]
            * int(round(player.projection * projection_scale))
            for index, player in enumerate(eligible_players)
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

    for index, player in enumerate(eligible_players):
        if solver.value(selected[index]) == 1:
            lineup.append(player)

    return lineup