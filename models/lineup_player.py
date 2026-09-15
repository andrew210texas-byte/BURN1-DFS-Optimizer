from dataclasses import dataclass

from models.player import Player


@dataclass
class LineupPlayer:
    roster_slot: str
    player: Player