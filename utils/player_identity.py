"""
Shared BURN1 player identity utilities.

Used across NFL, NBA, MLB, and future sports for matching
salary data, feature data, projections, results, and provider data.

Important:
- Common suffixes are normalized globally.
- Nicknames / provider-specific aliases are NOT guessed globally.
- Explicit aliases should be supplied by the caller so they remain
  sport/provider specific and auditable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping


COMMON_SUFFIXES = {
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
}


def normalize_player_name(
    value: object,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """
    Convert a player name into a deterministic comparison key.

    Examples:
        James Cook III  -> jamescook
        Aaron Jones Sr. -> aaronjones
        Travis Etienne Jr. -> travisetienne

    Punctuation and accents are also normalized.

    Optional aliases must use normalized keys:
        {
            "joshuapalmer": "joshpalmer",
            "nicksingleton": "nicholassingleton",
        }

    Aliases are intentionally explicit rather than fuzzy/automatic.
    """

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    while (
        tokens
        and tokens[-1] in COMMON_SUFFIXES
    ):
        tokens.pop()

    key = "".join(tokens)

    if aliases:
        key = aliases.get(
            key,
            key,
        )

    return key


def normalize_team(value: object) -> str:
    """
    Normalize a team/provider abbreviation.
    """

    if value is None:
        return ""

    return str(value).upper().strip()


def normalize_position(value: object) -> str:
    """
    Normalize a roster or statistical position.
    """

    if value is None:
        return ""

    return str(value).upper().strip()


def player_identity_key(
    name: object,
    team: object,
    position: object | None = None,
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """
    Produce a deterministic cross-source player identity key.

    With position:
        (normalized_name, team, position)

    Without position:
        (normalized_name, team)

    Position may be omitted when roster eligibility differs from
    statistical/model position.
    """

    key = (
        normalize_player_name(
            name,
            aliases=aliases,
        ),
        normalize_team(team),
    )

    if position is not None:
        key += (
            normalize_position(position),
        )

    return key
