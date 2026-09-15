import pandas as pd

from models.player import Player


def load_fanduel_players(csv_path):
    df = pd.read_csv(csv_path)

    players = []

    for _, row in df.iterrows():
        position = row["Position"]

        if position == "D":
            position = "DST"

        player = Player(
    name=row["Nickname"],
    site="FanDuel",
    position=position,
    team=row["Team"],
    opponent=row["Opponent"],
    salary=int(row["Salary"]),
    status=(
        ""
        if pd.isna(row["Injury Indicator"])
        else str(row["Injury Indicator"])
    ),
)

        players.append(player)

    return players