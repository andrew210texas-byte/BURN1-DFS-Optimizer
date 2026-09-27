from __future__ import annotations

import csv
import json
from statistics import mean, median
from pathlib import Path
from typing import Any

from app.intelligence.foundation import (
    BACKTEST_ROOT,
    RUN_ROOT,
    _atomic_json_write,
    utc_now_iso,
)


def load_actuals(
    actuals_file: Path,
) -> dict[str, dict[str, float | None]]:
    actuals_file = Path(actuals_file)

    with actuals_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "player_id",
            "actual_points",
        }

        columns = set(
            reader.fieldnames or []
        )

        missing = required - columns

        if missing:
            raise ValueError(
                "Actual-results CSV is missing "
                "required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        actuals = {}

        for row in reader:
            player_id = str(
                row["player_id"]
            ).strip()

            if not player_id:
                continue

            ownership_raw = str(
                row.get(
                    "actual_ownership",
                    "",
                )
            ).strip()

            actuals[player_id] = {
                "actual_points": float(
                    row["actual_points"]
                ),
                "actual_ownership": (
                    float(ownership_raw)
                    if ownership_raw
                    else None
                ),
            }

    return actuals


def score_archived_run(
    run_id: str,
    actuals_file: Path,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id

    result_path = (
        run_dir / "result.json"
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"No archived result found "
            f"for run {run_id}."
        )

    result = json.loads(
        result_path.read_text(
            encoding="utf-8",
        )
    )

    actuals = load_actuals(
        actuals_file
    )

    scored_lineups = []
    missing_player_ids = set()

    for lineup in result.get(
        "lineups",
        [],
    ):
        actual_score = 0.0
        complete = True

        for player in lineup.get(
            "players",
            [],
        ):
            player_id = str(
                player["player_id"]
            )

            actual = actuals.get(
                player_id
            )

            if actual is None:
                missing_player_ids.add(
                    player_id
                )
                complete = False
                continue

            actual_score += float(
                actual["actual_points"]
            )

        scored_lineups.append(
            {
                "lineup_number": lineup.get(
                    "lineup_number"
                ),
                "projection": float(
                    lineup.get(
                        "total_projection",
                        0.0,
                    )
                ),
                "actual_score": round(
                    actual_score,
                    4,
                ),
                "projection_error": (
                    round(
                        actual_score
                        - float(
                            lineup.get(
                                "total_projection",
                                0.0,
                            )
                        ),
                        4,
                    )
                    if complete
                    else None
                ),
                "complete_actuals": (
                    complete
                ),
            }
        )

    completed_scores = [
        row["actual_score"]
        for row in scored_lineups
        if row["complete_actuals"]
    ]

    summary = {
        "run_id": run_id,
        "scored_at": utc_now_iso(),
        "actuals_file": str(
            Path(actuals_file)
        ),
        "lineup_count": len(
            scored_lineups
        ),
        "complete_lineup_count": len(
            completed_scores
        ),
        "missing_player_ids": sorted(
            missing_player_ids
        ),
        "actual_score_mean": (
            round(
                mean(completed_scores),
                4,
            )
            if completed_scores
            else None
        ),
        "actual_score_median": (
            round(
                median(completed_scores),
                4,
            )
            if completed_scores
            else None
        ),
        "actual_score_max": (
            round(
                max(completed_scores),
                4,
            )
            if completed_scores
            else None
        ),
        "actual_score_min": (
            round(
                min(completed_scores),
                4,
            )
            if completed_scores
            else None
        ),
        "lineups": scored_lineups,
    }

    BACKTEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        BACKTEST_ROOT
        / f"{run_id}_score.json"
    )

    _atomic_json_write(
        output,
        summary,
    )

    return summary
