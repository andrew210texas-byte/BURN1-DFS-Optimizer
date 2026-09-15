from data.draftkings_loader import load_draftkings_players
from data.fanduel_loader import load_fanduel_players


draftkings_players = load_draftkings_players(
    "data/salaries/DKSalaries.csv"
)

fanduel_players = load_fanduel_players(
    "data/salaries/FanDuel-NFL-2026 CDT-09 CDT-20 CDT-134251-players-list.csv"
)


def print_player_pool(site_name, players):
    print(f"{site_name} NFL Player Pool")
    print(f"Players loaded: {len(players)}")

    unavailable_players = [
        player for player in players
        if player.status.upper() in {"OUT", "IR"}
    ]

    questionable_players = [
        player for player in players
        if player.status.upper() in {"Q", "D"}
    ]

    print(f"Unavailable (OUT/IR): {len(unavailable_players)}")
    print(f"Questionable/Doubtful: {len(questionable_players)}")
    print()

    positions = {}

    for player in players:
        if player.position not in positions:
            positions[player.position] = 0

        positions[player.position] += 1

    for position, count in positions.items():
        print(f"{position}: {count}")

    print()
    print("First 10 players:")
    print()

    for player in players[:10]:
        print(
            f"{player.name} | "
            f"{player.position} | "
            f"{player.team} vs {player.opponent} | "
            f"${player.salary:,}"
        )

    print()


print("DFS Optimizer")
print()

print_player_pool("DraftKings", draftkings_players)
print_player_pool("FanDuel", fanduel_players)