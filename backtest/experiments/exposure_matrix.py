from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.experiments.stage1_diversity import (
    candidate_exposure_report,
    generate_nfl_candidates_with_player_cap,
)
from backtest.player_adapter import (
    historical_frame_to_players,
)
from backtest.scoring import (
    prepare_actuals,
    score_lineups,
    summarize_scores,
)
from optimizer.portfolio_optimizer import (
    Stage2InfeasibleError,
    select_nfl_portfolio,
)


SLATE_PATH = Path(
    "backtest/data/dk/"
    "2026_week3_burn1_v1.csv"
)

ACTUALS_PATH = Path(
    "backtest/data/dk/"
    "2026_week3_actuals.csv"
)

OVERRIDES_PATH = Path(
    "backtest/data/"
    "manual_outcome_overrides.csv"
)

OUTPUT_ROOT = Path(
    "backtest/experiments/output/"
    "week3_exposure_matrix"
)

LINEUP_COUNT = 20
CANDIDATE_COUNT = 200
SALARY_CAP = 50000


@dataclass(frozen=True)
class Scenario:
    name: str
    stage1_cap: float | None
    stage2_cap: float | None


SCENARIOS = [
    Scenario(
        name="baseline_uncapped",
        stage1_cap=None,
        stage2_cap=None,
    ),
    Scenario(
        name="stage1_90_stage2_80",
        stage1_cap=0.90,
        stage2_cap=0.80,
    ),
    Scenario(
        name="stage1_80_stage2_80",
        stage1_cap=0.80,
        stage2_cap=0.80,
    ),
    Scenario(
        name="stage1_80_stage2_60",
        stage1_cap=0.80,
        stage2_cap=0.60,
    ),
    Scenario(
        name="stage1_70_stage2_60",
        stage1_cap=0.70,
        stage2_cap=0.60,
    ),
]


def final_exposure_report(
    lineups,
    players,
) -> pd.DataFrame:

    player_lookup = {
        str(player.player_id): player
        for player in players
    }

    counts = {}

    for lineup in lineups:
        seen = set()

        for item in lineup:
            player = item.player

            player_id = str(
                player.player_id
            )

            if player_id in seen:
                continue

            seen.add(player_id)

            counts[player_id] = (
                counts.get(
                    player_id,
                    0,
                )
                + 1
            )

    rows = []

    for player_id, count in counts.items():
        player = player_lookup.get(
            player_id
        )

        if player is None:
            continue

        rows.append(
            {
                "player_id":
                    player_id,
                "player_name":
                    player.name,
                "position":
                    player.position,
                "team":
                    player.team,
                "lineup_count":
                    count,
                "exposure":
                    count
                    / len(lineups),
                "exposure_pct":
                    round(
                        count
                        / len(lineups)
                        * 100,
                        1,
                    ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "lineup_count",
                "player_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def main():
    OUTPUT_ROOT.mkdir(
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

    actuals = prepare_actuals(
        ACTUALS_PATH,
        OVERRIDES_PATH,
        season=2026,
        week=3,
        site="DK",
    )

    all_player_ids = [
        str(player.player_id)
        for player in players
        if str(
            player.player_id
        ).strip()
    ]

    print(
        "============================================================"
    )
    print(
        "BURN1 WEEK 3 STAGE-1 / STAGE-2 EXPOSURE MATRIX"
    )
    print(
        "============================================================"
    )

    print(
        "Slate players:",
        len(players),
    )

    print(
        "Candidate target:",
        CANDIDATE_COUNT,
    )

    print(
        "Final lineups:",
        LINEUP_COUNT,
    )

    comparison_rows = []

    for scenario in SCENARIOS:
        print()
        print(
            "=" * 76
        )
        print(
            "SCENARIO:",
            scenario.name,
        )
        print(
            "=" * 76
        )

        print(
            "Stage 1 cap:",
            (
                "UNCAPPED"
                if scenario.stage1_cap
                is None
                else f"{scenario.stage1_cap:.0%}"
            ),
        )

        print(
            "Stage 2 cap:",
            (
                "UNCAPPED"
                if scenario.stage2_cap
                is None
                else f"{scenario.stage2_cap:.0%}"
            ),
        )

        scenario_dir = (
            OUTPUT_ROOT
            / scenario.name
        )

        scenario_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_pool = (
            generate_nfl_candidates_with_player_cap(
                players=players,
                salary_cap=SALARY_CAP,
                candidate_count=(
                    CANDIDATE_COUNT
                ),
                min_unique_players=2,
                stage1_player_cap=(
                    scenario.stage1_cap
                ),
                gpp_mode=False,
                qb_stack_min=1,
                bring_back_min=0,
                rb_dst_stack=False,
            )
        )

        generated = len(
            candidate_pool.lineups
        )

        candidate_report = (
            candidate_exposure_report(
                candidate_pool,
                players,
            )
        )

        candidate_report.to_csv(
            scenario_dir
            / "candidate_exposures.csv",
            index=False,
        )

        stage1_actual_max = (
            float(
                candidate_report[
                    "candidate_exposure"
                ].max()
            )
            if not candidate_report.empty
            else 0.0
        )

        print(
            "Candidates generated:",
            generated,
        )

        print(
            "Stage 1 actual max exposure:",
            f"{stage1_actual_max:.0%}",
        )

        if generated < LINEUP_COUNT:
            error = (
                "Stage 1 generated fewer candidates "
                "than required final lineups."
            )

            print(
                "FAILED:",
                error,
            )

            comparison_rows.append(
                {
                    "scenario":
                        scenario.name,
                    "stage1_cap":
                        scenario.stage1_cap,
                    "stage2_cap":
                        scenario.stage2_cap,
                    "candidates_generated":
                        generated,
                    "stage1_actual_max":
                        stage1_actual_max,
                    "stage2_feasible":
                        False,
                    "error":
                        error,
                }
            )

            continue

        max_exposures = (
            {}
            if scenario.stage2_cap
            is None
            else {
                player_id:
                    scenario.stage2_cap
                for player_id
                in all_player_ids
            }
        )

        try:
            portfolio = (
                select_nfl_portfolio(
                    candidate_pool=(
                        candidate_pool
                    ),
                    lineup_count=(
                        LINEUP_COUNT
                    ),
                    max_exposures=(
                        max_exposures
                    ),
                )
            )

        except Stage2InfeasibleError as exc:
            error = str(exc)

            print(
                "Stage 2 feasible: False"
            )

            print(
                "ERROR:",
                error,
            )

            comparison_rows.append(
                {
                    "scenario":
                        scenario.name,
                    "stage1_cap":
                        scenario.stage1_cap,
                    "stage2_cap":
                        scenario.stage2_cap,
                    "candidates_generated":
                        generated,
                    "stage1_actual_max":
                        round(
                            stage1_actual_max,
                            4,
                        ),
                    "stage2_feasible":
                        False,
                    "error":
                        error,
                }
            )

            continue

        print(
            "Stage 2 feasible: True"
        )

        scored, unmatched = (
            score_lineups(
                portfolio.lineups,
                actuals,
            )
        )

        scored.to_csv(
            scenario_dir
            / "scored_lineups.csv",
            index=False,
        )

        if not unmatched.empty:
            unmatched.to_csv(
                scenario_dir
                / "unmatched.csv",
                index=False,
            )

        final_exposures = (
            final_exposure_report(
                portfolio.lineups,
                players,
            )
        )

        final_exposures.to_csv(
            scenario_dir
            / "final_exposures.csv",
            index=False,
        )

        metrics = summarize_scores(
            scored
        )

        final_actual_max = (
            float(
                final_exposures[
                    "exposure"
                ].max()
            )
            if not final_exposures.empty
            else 0.0
        )

        unique_players = int(
            len(final_exposures)
        )

        players_100 = int(
            (
                final_exposures[
                    "exposure"
                ]
                >= 1.0
            ).sum()
        )

        players_80 = int(
            (
                final_exposures[
                    "exposure"
                ]
                >= 0.80
            ).sum()
        )

        players_60 = int(
            (
                final_exposures[
                    "exposure"
                ]
                >= 0.60
            ).sum()
        )

        print()
        print(
            "Best actual:",
            metrics[
                "best_actual"
            ],
        )

        print(
            "Average actual:",
            metrics[
                "average_actual"
            ],
        )

        print(
            "Median actual:",
            metrics[
                "median_actual"
            ],
        )

        print(
            "Worst actual:",
            metrics[
                "worst_actual"
            ],
        )

        print(
            "150+:",
            metrics[
                "lineups_150_plus"
            ],
        )

        print(
            "160+:",
            metrics[
                "lineups_160_plus"
            ],
        )

        print(
            "170+:",
            metrics[
                "lineups_170_plus"
            ],
        )

        print(
            "Unique final players:",
            unique_players,
        )

        print(
            "Final actual max exposure:",
            f"{final_actual_max:.0%}",
        )

        print(
            "Average projection:",
            metrics[
                "average_projection"
            ],
        )

        print(
            "All lineups matched:",
            metrics[
                "all_lineups_matched"
            ],
        )

        metadata = {
            "scenario":
                scenario.name,
            "stage1_cap":
                scenario.stage1_cap,
            "stage2_cap":
                scenario.stage2_cap,
            "candidate_target":
                CANDIDATE_COUNT,
            "candidates_generated":
                generated,
            "lineup_count":
                LINEUP_COUNT,
            "stage1_actual_max":
                stage1_actual_max,
            "final_actual_max":
                final_actual_max,
            **metrics,
        }

        (
            scenario_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        comparison_rows.append(
            {
                "scenario":
                    scenario.name,
                "stage1_cap":
                    (
                        1.0
                        if scenario.stage1_cap
                        is None
                        else scenario.stage1_cap
                    ),
                "stage2_cap":
                    (
                        1.0
                        if scenario.stage2_cap
                        is None
                        else scenario.stage2_cap
                    ),
                "candidates_generated":
                    generated,
                "stage1_actual_max":
                    round(
                        stage1_actual_max,
                        4,
                    ),
                "stage2_feasible":
                    True,
                "best_actual":
                    metrics[
                        "best_actual"
                    ],
                "average_actual":
                    metrics[
                        "average_actual"
                    ],
                "median_actual":
                    metrics[
                        "median_actual"
                    ],
                "worst_actual":
                    metrics[
                        "worst_actual"
                    ],
                "lineups_150_plus":
                    metrics[
                        "lineups_150_plus"
                    ],
                "lineups_160_plus":
                    metrics[
                        "lineups_160_plus"
                    ],
                "lineups_170_plus":
                    metrics[
                        "lineups_170_plus"
                    ],
                "average_projection":
                    metrics[
                        "average_projection"
                    ],
                "total_projection":
                    metrics[
                        "total_projection"
                    ],
                "unique_final_players":
                    unique_players,
                "final_actual_max":
                    round(
                        final_actual_max,
                        4,
                    ),
                "players_100_pct":
                    players_100,
                "players_80_plus":
                    players_80,
                "players_60_plus":
                    players_60,
                "all_lineups_matched":
                    metrics[
                        "all_lineups_matched"
                    ],
                "error":
                    "",
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        OUTPUT_ROOT
        / "matrix_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "FINAL BURN1 EXPOSURE MATRIX"
    )

    print(
        "=" * 110
    )

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved comparison:",
        comparison_path,
    )


if __name__ == "__main__":
    main()
