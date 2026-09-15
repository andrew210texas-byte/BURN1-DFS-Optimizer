import pandas as pd

from models.player import Player


def load_draftkings_players(csv_path):
    df = pd.read_csv(
        csv_path,
        dtype={"ID": str},
    )

    players = []

    for _, row in df.iterrows():
        game_info = str(row["Game Info"])
        matchup = game_info.split(" ")[0]

        teams = matchup.split("@")

        if len(teams) == 2:
            away_team = teams[0]
            home_team = teams[1]

            if row["TeamAbbrev"] == away_team:
                opponent = home_team
            elif row["TeamAbbrev"] == home_team:
                opponent = away_team
            else:
                opponent = ""
        else:
            opponent = ""

        if pd.isna(row["AvgPointsPerGame"]):
            projection = 0.0
        else:
            projection = float(row["AvgPointsPerGame"])

        raw_roster_positions = str(row["Roster Position"])

        roster_positions = tuple(
            position.strip()
            for position in raw_roster_positions.split("/")
            if position.strip()
        )

        player = Player(
            name=str(row["Name"]),
            site="DraftKings",
            position=str(row["Position"]),
            team=str(row["TeamAbbrev"]),
            opponent=opponent,
            salary=int(row["Salary"]),
            player_id=str(row["ID"]),
            roster_positions=roster_positions,
            status="" if pd.isna(row["Status"]) else str(row["Status"]),
            projection=projection,
        )

        players.append(player)

    return players