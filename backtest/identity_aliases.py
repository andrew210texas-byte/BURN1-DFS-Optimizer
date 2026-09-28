"""
Explicit BURN1 player-name aliases.

Global suffix cleanup such as Jr., Sr., II, III, IV, and V lives in
utils.player_identity.

This file is only for deterministic provider/name differences that
cannot safely be inferred globally.

Do NOT add fuzzy matches here.
Every alias should be deliberate and auditable.
"""

DEFAULT_PLAYER_ALIASES = {
    "joshuapalmer": "joshpalmer",
    "nicksingleton": "nicholassingleton",
    "mitchtinsley": "mitchelltinsley",
    "drewogletree": "andrewogletree",
    "matthibner": "matthewhibner",
}
