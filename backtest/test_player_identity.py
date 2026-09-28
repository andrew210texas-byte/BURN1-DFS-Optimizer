from utils.player_identity import (
    normalize_player_name,
    player_identity_key,
)


TESTS = {
    "James Cook III": "jamescook",
    "Aaron Jones Sr.": "aaronjones",
    "Michael Pittman Jr.": "michaelpittman",
    "Gardner Minshew II": "gardnerminshew",
    "Travis Etienne Jr.": "travisetienne",
    "Odell Beckham Jr.": "odellbeckham",
    "Calvin Austin III": "calvinaustin",
}


for raw, expected in TESTS.items():
    actual = normalize_player_name(raw)

    assert actual == expected, (
        f"{raw}: expected {expected}, got {actual}"
    )


aliases = {
    "joshuapalmer": "joshpalmer",
    "nicksingleton": "nicholassingleton",
    "mitchtinsley": "mitchelltinsley",
    "drewogletree": "andrewogletree",
    "matthibner": "matthewhibner",
}


assert (
    normalize_player_name(
        "Joshua Palmer",
        aliases,
    )
    == "joshpalmer"
)

assert (
    player_identity_key(
        "James Cook III",
        "buf",
        "rb",
    )
    == (
        "jamescook",
        "BUF",
        "RB",
    )
)


print("SUCCESS: Shared BURN1 identity normalization passed.")
print()
print("Suffix normalization:")
for raw, expected in TESTS.items():
    print(
        f"{raw:<25} -> {expected}"
    )

print()
print(
    "Explicit alias example:",
    normalize_player_name(
        "Joshua Palmer",
        aliases,
    ),
)
