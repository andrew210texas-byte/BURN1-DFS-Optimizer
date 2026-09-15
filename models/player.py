from dataclasses import dataclass


@dataclass
class Player:
    name: str
    site: str
    position: str
    team: str
    opponent: str
    salary: int

    player_id: str = ""
    roster_positions: tuple[str, ...] = ()

    status: str = ""
    projection: float = 0.0
    ceiling: float = 0.0
    ownership: float = 0.0