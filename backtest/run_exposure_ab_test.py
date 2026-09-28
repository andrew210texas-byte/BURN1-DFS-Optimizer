from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio_safe,
)
from backtest.player_adapter import historical_frame_to_players
from utils.player_identity import (
    normalize_player_name,
    normalize_position,
    normalize_team,
)


SLATE_PATH = Path(
    "backtest/data/dk/2026_week3_burn1_v1.csv"
)

ACTUALS_PATH = Path(
    "backtest/data/dk/2026_week3_actuals.csv"
)

RESULT_DIR = Path(
    "backtest/results/exposure_ab"
)

REPORT_DIR = Path(
    "backtest/reports/exposure_ab"
)

LINEUP_COUNT = 20
CANDIDATE_COUNT = 200


@dataclass(frozen=True)
class Scenario:
    name: str
    max_exposure: float | None


SCENARIOS = [
    Scenario(
        name="uncapped",
        max_exposure=None,
    ),
    Scenario(
        name="max80",
        max_exposure=0.80,
    ),
    Scenario(
        name="max60",
        max_exposure=0.60,
    ),
]


# Explicit provider/name aliases.
#
# Common Jr/Sr/II/III/etc suffixes are already handled
# globally by utils.player_identity.
ALIASES = {
    "joshuapalmer": "joshpalmer",
    "nicksingleton": "nicholassingleton",
    "mitchtinsley": "mitchelltinsley",
    "drewogletree": "andrewogletree",
    "matthibner": "matthewhibner",
}


def identity_name(value):
    return normalize_player_name(
        value,
        aliases=ALIASES,
    )


def load_actuals():
    actuals = pd.read_csv(
        ACTUALS_PATH
    )

    actuals["name_key"] = (
        actuals["player_name"]
        .map(identity_name)
    )

    actuals["team_key"] = (
        actuals["team"]
        .map(normalize_team)
    )

    actuals["position_key"] = (
        actuals["position"]
        .map(normalize_position)
    )

    # Verified Week 3 zero:
    # Joshua/Josh Palmer had no Week 3 player-stat row
    # and no play-by-play references.
    palmer_key = identity_name(
        "Joshua Palmer"
    )

    palmer_exists = (
        (
            actuals["name_key"]
            == palmer_key
        )
        &
        (
            actuals["team_key"]
            == "BUF"
        )
        &
        (
            actuals["position_key"]
            == "WR"
        )
    ).any()

    if not palmer_exists:
        actuals = pd.concat(
            [
                actuals,
                pd.DataFrame(
                    [
                        {
                            "player_name":
                                "Joshua Palmer",
                            "team":
                                "BUF",
                            "position":
                                "WR",
                            "actual_dk_points":
                                0.0,
                            "name_key":
                                palmer_key,
                            "team_key":
                                "BUF",
                            "position_key":
                                "WR",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return actuals


def score_portfolio(
    result,
    actuals,
):
    rows = []
    unmatched = []

    for lineup in result["lineups"]:
        actual_total = 0.0
        matched = 0

        names = []

        for player in lineup["players"]:
            name = str(
                player["name"]
            )

            team = normalize_team(
                player["team"]
            )

            position = normalize_position(
                player["position"]
            )

            names.append(name)

            if position == "DST":
                match = actuals[
                    (
                        actuals["team_key"]
                        == team
                    )
                    &
                    (
                        actuals["position_key"]
                        == "DST"
                    )
                ]

            else:
                match = actuals[
                    (
                        actuals["name_key"]
                        == identity_name(name)
                    )
                    &
                    (
                        actuals["team_key"]
                        == team
                    )
                    &
                    (
                        actuals["position_key"]
                        == position
                    )
                ]

            if len(match) == 1:
                actual_total += float(
                    match.iloc[0][
                        "actual_dk_points"
                    ]
                )

                matched += 1

            else:
                unmatched.append(
                    {
                        "lineup_number":
                            lineup[
                                "lineup_number"
                            ],
                        "name":
                            name,
                        "team":
                            team,
                        "position":
                            position,
                        "matches_found":
                            len(match),
                    }
                )

        rows.append(
            {
                "lineup_number":
                    lineup[
                        "lineup_number"
                    ],
                "candidate_lineup_number":
                    lineup[
                        "candidate_lineup_number"
                    ],
                "total_salary":
                    lineup[
                        "total_salary"
                    ],
                "burn1_projection":
                    lineup[
                        "total_projection"
                    ],
                "actual_dk_points":
                    round(
                        actual_total,
                        2,
                    ),
                "matched_players":
                    matched,
                "players":
                    " | ".join(names),
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(unmatched),
    )


def summarize_exposures(
    result,
):
    exposures = pd.DataFrame(
        result["exposures"]
    )

    if exposures.empty:
        return {
            "unique_players": 0,
            "max_exposure": 0.0,
            "players_100_pct": 0,
            "players_80_plus": 0,
            "players_60_plus": 0,
        }

    return {
        "unique_players":
            len(exposures),
        "max_exposure":
            float(
                exposures[
                    "exposure"
                ].max()
            ),
        "players_100_pct":
            int(
                (
                    exposures[
                        "exposure"
                    ]
                    >= 1.0
                ).sum()
            ),
        "players_80_plus":
            int(
                (
                    exposures[
                        "exposure"
                    ]
                    >= 0.80
                ).sum()
            ),
        "players_60_plus":
            int(
                (
                    exposures[
                        "exposure"
                    ]
                    >= 0.60
                ).sum()
            ),
    }


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    slate = pd.read_csv(
        SLATE_PATH
    )

    players = (
        historical_frame_to_players(
            slate
        )
    )

    actuals = load_actuals()

    player_ids = [
        str(player.player_id)
        for player in players
        if str(
            player.player_id
        ).strip()
    ]

    print(
        "===== BURN1 EXPOSURE A/B TEST ====="
    )

    print(
        "Slate players:",
        len(players),
    )

    print(
        "Players with IDs:",
        len(player_ids),
    )

    print(
        "Scenarios:",
        ", ".join(
            scenario.name
            for scenario in SCENARIOS
        ),
    )

    comparison_rows = []

    for scenario in SCENARIOS:
        print()
        print(
            "=" * 70
        )

        print(
            "SCENARIO:",
            scenario.name,
        )

        print(
            "=" * 70
        )

        if (
            scenario.max_exposure
            is None
        ):
            max_exposures = {}

            cap_display = (
                "UNCAPPED"
            )

        else:
            max_exposures = {
                player_id:
                    scenario.max_exposure
                for player_id
                in player_ids
            }

            cap_display = (
                f"{scenario.max_exposure:.0%}"
            )

        print(
            "Player exposure cap:",
            cap_display,
        )

        config = Burn1PortfolioConfig(
            site="DraftKings",
            lineup_count=(
                LINEUP_COUNT
            ),
            candidate_count=(
                CANDIDATE_COUNT
            ),
            max_player_exposures=(
                max_exposures
            ),
        )

        response = (
            run_burn1_nfl_portfolio_safe(
                players,
                config,
            )
        )

        payload = (
            response.to_dict()
        )

        result_path = (
            RESULT_DIR
            / (
                "2026_week3_"
                f"{scenario.name}.json"
            )
        )

        result_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "OK:",
            payload["ok"],
        )

        if not payload["ok"]:
            print(
                "ERROR:",
                payload["error"],
            )

            comparison_rows.append(
                {
                    "scenario":
                        scenario.name,
                    "exposure_cap":
                        scenario.max_exposure,
                    "ok":
                        False,
                    "error":
                        payload["error"],
                }
            )

            continue

        result = payload["result"]

        print(
            "Candidates:",
            result[
                "generated_candidates"
            ],
        )

        print(
            "Lineups:",
            result[
                "generated_lineups"
            ],
        )

        print(
            "Solver:",
            result[
                "solver_status"
            ],
        )

        report, unmatched = (
            score_portfolio(
                result,
                actuals,
            )
        )

        report_path = (
            REPORT_DIR
            / (
                "2026_week3_"
                f"{scenario.name}_"
                "scored.csv"
            )
        )

        report.to_csv(
            report_path,
            index=False,
        )

        if len(unmatched):
            unmatched_path = (
                REPORT_DIR
                / (
                    "2026_week3_"
                    f"{scenario.name}_"
                    "unmatched.csv"
                )
            )

            unmatched.to_csv(
                unmatched_path,
                index=False,
            )

        exposure_stats = (
            summarize_exposures(
                result
            )
        )

        best = float(
            report[
                "actual_dk_points"
            ].max()
        )

        average = float(
            report[
                "actual_dk_points"
            ].mean()
        )

        median = float(
            report[
                "actual_dk_points"
            ].median()
        )

        worst = float(
            report[
                "actual_dk_points"
            ].min()
        )

        average_projection = float(
            report[
                "burn1_projection"
            ].mean()
        )

        total_projection = float(
            report[
                "burn1_projection"
            ].sum()
        )

        count_150 = int(
            (
                report[
                    "actual_dk_points"
                ]
                >= 150
            ).sum()
        )

        count_160 = int(
            (
                report[
                    "actual_dk_points"
                ]
                >= 160
            ).sum()
        )

        count_170 = int(
            (
                report[
                    "actual_dk_points"
                ]
                >= 170
            ).sum()
        )

        fully_matched = bool(
            (
                report[
                    "matched_players"
                ]
                == 9
            ).all()
        )

        print()
        print(
            "Best:",
            round(
                best,
                2,
            ),
        )

        print(
            "Average:",
            round(
                average,
                2,
            ),
        )

        print(
            "Median:",
            round(
                median,
                2,
            ),
        )

        print(
            "Worst:",
            round(
                worst,
                2,
            ),
        )

        print(
            "150+:",
            count_150,
        )

        print(
            "160+:",
            count_160,
        )

        print(
            "170+:",
            count_170,
        )

        print(
            "Unique players:",
            exposure_stats[
                "unique_players"
            ],
        )

        print(
            "Actual max exposure:",
            f"{exposure_stats['max_exposure']:.0%}",
        )

        print(
            "All lineups 9/9 matched:",
            fully_matched,
        )

        comparison_rows.append(
            {
                "scenario":
                    scenario.name,
                "exposure_cap":
                    (
                        1.0
                        if scenario.max_exposure
                        is None
                        else scenario.max_exposure
                    ),
                "ok":
                    True,
                "best_actual":
                    round(best, 2),
                "average_actual":
                    round(
                        average,
                        2,
                    ),
                "median_actual":
                    round(
                        median,
                        2,
                    ),
                "worst_actual":
                    round(
                        worst,
                        2,
                    ),
                "lineups_150_plus":
                    count_150,
                "lineups_160_plus":
                    count_160,
                "lineups_170_plus":
                    count_170,
                "average_projection":
                    round(
                        average_projection,
                        4,
                    ),
                "total_projection":
                    round(
                        total_projection,
                        4,
                    ),
                "unique_players":
                    exposure_stats[
                        "unique_players"
                    ],
                "actual_max_exposure":
                    round(
                        exposure_stats[
                            "max_exposure"
                        ],
                        4,
                    ),
                "players_100_pct":
                    exposure_stats[
                        "players_100_pct"
                    ],
                "players_80_plus":
                    exposure_stats[
                        "players_80_plus"
                    ],
                "players_60_plus":
                    exposure_stats[
                        "players_60_plus"
                    ],
                "all_lineups_matched":
                    fully_matched,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        REPORT_DIR
        / "2026_week3_exposure_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "FINAL EXPOSURE COMPARISON"
    )

    print(
        "=" * 90
    )

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        comparison_path,
    )


if __name__ == "__main__":
    main()
