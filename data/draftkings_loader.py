import pandas as pd

from models.player import Player


def load_draftkings_players(csv_path):
    df = pd.read_csv(csv_path)

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
            else:
                opponent = away_team
        else:
            opponent = ""

        if pd.isna(row["AvgPointsPerGame"]):
            projection = 0.0
        else:
            projection = float(row["AvgPointsPerGame"])

        player = Player(
            name=row["Name"],
            site="DraftKings",
            position=row["Position"],
            team=row["TeamAbbrev"],
            opponent=opponent,
            salary=int(row["Salary"]),
            status="" if pd.isna(row["Status"]) else str(row["Status"]),
            projection=projection,
        )

        players.append(player)

    return players