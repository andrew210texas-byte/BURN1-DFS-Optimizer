from data.draftkings_loader import load_draftkings_players
from data.fanduel_loader import load_fanduel_players
from optimizer.lineup_optimizer import optimize_nfl_lineup


draftkings_players = load_draftkings_players(
    "data/salaries/DKSalaries.csv"
)

fanduel_players = load_fanduel_players(
    "data/salaries/FanDuel-NFL-2026 CDT-09 CDT-20 CDT-134251-players-list.csv"
)


def print_lineup(site_name, lineup, salary_cap):
    if not lineup:
        print(f"No valid {site_name} lineup found.")
        print()
        return

    position_order = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 4,
        "DST": 5,
    }

    lineup = sorted(
        lineup,
        key=lambda player: position_order.get(player.position, 99)
    )

    total_salary = sum(player.salary for player in lineup)
    total_projection = sum(player.projection for player in lineup)

    print(f"{site_name} Optimized NFL Lineup")
    print("-" * 70)

    for player in lineup:
        print(
            f"{player.position:<3} | "
            f"{player.name:<25} | "
            f"{player.team} vs {player.opponent:<3} | "
            f"${player.salary:>5,} | "
            f"{player.projection:>6.2f} pts"
        )

    print("-" * 70)
    print(f"Players:          {len(lineup)}")
    print(f"Total Salary:     ${total_salary:,}")
    print(f"Salary Remaining: ${salary_cap - total_salary:,}")
    print(f"Total Points:     {total_projection:.2f}")
    print()


draftkings_lineup = optimize_nfl_lineup(
    draftkings_players,
    salary_cap=50000,
)

fanduel_lineup = optimize_nfl_lineup(
    fanduel_players,
    salary_cap=60000,
)


print()
print("DFS OPTIMIZER")
print("=" * 70)
print()

print_lineup(
    "DraftKings",
    draftkings_lineup,
    salary_cap=50000,
)

print_lineup(
    "FanDuel",
    fanduel_lineup,
    salary_cap=60000,
)