from models.player import Player


player = Player(
    name="Test Quarterback",
    site="DraftKings",
    position="QB",
    team="KC",
    opponent="DEN",
    salary=7600,
)

print("DFS Optimizer")
print()
print(f"Player: {player.name}")
print(f"Site: {player.site}")
print(f"Position: {player.position}")
print(f"Team: {player.team}")
print(f"Opponent: {player.opponent}")
print(f"Salary: ${player.salary:,}")
print(f"Projection: {player.projection}")
print(f"Ceiling: {player.ceiling}")
print(f"Ownership: {player.ownership}")