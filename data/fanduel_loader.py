import pandas as pd

from models.player import Player


def load_fanduel_players(csv_path):
    df = pd.read_csv(
        csv_path,
        dtype={"Id": str},
    )

    players = []

    for _, row in df.iterrows():
        position = str(row["Position"])

        if position == "D":
            position = "DST"

        if pd.isna(row["FPPG"]):
            projection = 0.0
        else:
            projection = float(row["FPPG"])

        raw_roster_positions = str(row["Roster Position"])

        roster_positions = []

        for roster_position in raw_roster_positions.split("/"):
            roster_position = roster_position.strip()

            if roster_position == "DEF":
                roster_position = "DST"

            if roster_position:
                roster_positions.append(roster_position)

        player = Player(
            name=str(row["Nickname"]),
            site="FanDuel",
            position=position,
            team=str(row["Team"]),
            opponent=str(row["Opponent"]),
            salary=int(row["Salary"]),
            player_id=str(row["Id"]),
            roster_positions=tuple(roster_positions),
            status=(
                ""
                if pd.isna(row["Injury Indicator"])
                else str(row["Injury Indicator"])
            ),
            projection=projection,
        )

        players.append(player)

    return players