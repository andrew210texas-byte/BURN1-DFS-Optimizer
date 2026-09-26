from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio_safe,
)
from models.player import Player


ROOT = Path(__file__).resolve().parent

DEFAULT_SLATE = "2026_week3_sunday_main"

SITE_CONFIG = {
    "dk": {
        "site_name": "DraftKings",
        "pool_file": ROOT / "data" / "live" / "nfl" / "dk_2026_week3_sunday_main_final_pool.csv",
        "output_slug": "draftkings",
    },
    "fd": {
        "site_name": "FanDuel",
        "pool_file": ROOT / "data" / "live" / "nfl" / "fd_2026_week3_sunday_main_final_pool.csv",
        "output_slug": "fanduel",
    },
}


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_players(pool_file: Path, site_name: str) -> list[Player]:
    if not pool_file.exists():
        raise FileNotFoundError(
            f"Final player pool not found: {pool_file}"
        )

    data = pd.read_csv(
        pool_file,
        keep_default_na=False,
    )

    required_columns = {
        "player_id",
        "name",
        "position",
        "team",
        "opponent",
        "salary",
        "roster_positions",
        "status",
        "projection",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Final player pool is missing required columns: "
            + ", ".join(sorted(missing))
        )

    players = []

    for _, row in data.iterrows():
        roster_positions = tuple(
            part.strip().upper()
            for part in clean(
                row["roster_positions"]
            ).split("/")
            if part.strip()
        )

        position = clean(row["position"]).upper()

        if not roster_positions:
            roster_positions = (position,)

        players.append(
            Player(
                name=clean(row["name"]),
                site=site_name,
                position=position,
                team=clean(row["team"]),
                opponent=clean(row["opponent"]),
                salary=int(row["salary"]),
                player_id=clean(row["player_id"]),
                roster_positions=roster_positions,
                status=clean(row["status"]),
                projection=float(row["projection"]),
            )
        )

    if not players:
        raise ValueError(
            f"{site_name} final player pool is empty."
        )

    if len(
        {
            player.player_id
            for player in players
        }
    ) != len(players):
        raise ValueError(
            f"{site_name} final player pool contains duplicate player IDs."
        )

    return players


def status_printer(site_name: str):
    def callback(event):
        print(
            f"[{site_name}] "
            f"{event.status}: "
            f"{event.message}",
            flush=True,
        )

    return callback


def build_lineup_rows(result_dict: dict) -> list[dict]:
    rows = []

    for lineup in result_dict["lineups"]:
        for player in lineup["players"]:
            rows.append(
                {
                    "lineup_number": lineup["lineup_number"],
                    "roster_slot": player["roster_slot"],
                    "player_id": player["player_id"],
                    "name": player["name"],
                    "position": player["position"],
                    "team": player["team"],
                    "opponent": player["opponent"],
                    "salary": player["salary"],
                    "projection": player["projection"],
                    "lineup_salary": lineup["total_salary"],
                    "lineup_projection": lineup["total_projection"],
                    "qb_stack_size": lineup["qb_stack_size"],
                    "bring_back_size": lineup["bring_back_size"],
                    "has_rb_dst": lineup["has_rb_dst"],
                    "has_qb_vs_opposing_dst": lineup[
                        "has_qb_vs_opposing_dst"
                    ],
                    "is_unstacked": lineup["is_unstacked"],
                }
            )

    return rows


def build_wide_lineup_rows(result_dict: dict) -> list[dict]:
    rows = []

    for lineup in result_dict["lineups"]:
        row = {
            "lineup_number": lineup["lineup_number"],
            "total_salary": lineup["total_salary"],
            "total_projection": lineup["total_projection"],
            "qb_stack_size": lineup["qb_stack_size"],
            "bring_back_size": lineup["bring_back_size"],
            "has_rb_dst": lineup["has_rb_dst"],
            "has_qb_vs_opposing_dst": lineup[
                "has_qb_vs_opposing_dst"
            ],
            "is_unstacked": lineup["is_unstacked"],
        }

        counters = {
            "QB": 0,
            "RB": 0,
            "WR": 0,
            "TE": 0,
            "FLEX": 0,
            "DST": 0,
        }

        for player in lineup["players"]:
            slot = player["roster_slot"].upper()
            counters[slot] = counters.get(slot, 0) + 1

            if slot in {"RB", "WR"}:
                column_slot = f"{slot}{counters[slot]}"
            else:
                column_slot = slot

            row[f"{column_slot}_id"] = player["player_id"]
            row[f"{column_slot}_name"] = player["name"]
            row[f"{column_slot}_team"] = player["team"]
            row[f"{column_slot}_salary"] = player["salary"]
            row[f"{column_slot}_projection"] = player["projection"]

        rows.append(row)

    return rows


def save_outputs(
    result_dict: dict,
    site_slug: str,
    slate_slug: str,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"burn1_{site_slug}_{slate_slug}"
    )

    lineup_long_file = (
        output_dir
        / f"{prefix}_lineups_long.csv"
    )
    lineup_wide_file = (
        output_dir
        / f"{prefix}_lineups_wide.csv"
    )
    exposure_file = (
        output_dir
        / f"{prefix}_exposures.csv"
    )
    strategy_file = (
        output_dir
        / f"{prefix}_strategies.csv"
    )
    summary_file = (
        output_dir
        / f"{prefix}_summary.json"
    )

    pd.DataFrame(
        build_lineup_rows(result_dict)
    ).to_csv(
        lineup_long_file,
        index=False,
    )

    pd.DataFrame(
        build_wide_lineup_rows(result_dict)
    ).to_csv(
        lineup_wide_file,
        index=False,
    )

    pd.DataFrame(
        result_dict["exposures"]
    ).to_csv(
        exposure_file,
        index=False,
    )

    strategy_rows = [
        {
            "strategy": strategy_name,
            "count": metrics["count"],
            "exposure": metrics["exposure"],
        }
        for strategy_name, metrics
        in result_dict["strategy_summary"].items()
    ]

    pd.DataFrame(
        strategy_rows
    ).to_csv(
        strategy_file,
        index=False,
    )

    summary = {
        "status": result_dict["status"],
        "site": result_dict["site"],
        "slate": slate_slug,
        "requested_lineups": result_dict[
            "requested_lineups"
        ],
        "generated_lineups": result_dict[
            "generated_lineups"
        ],
        "requested_candidates": result_dict[
            "requested_candidates"
        ],
        "generated_candidates": result_dict[
            "generated_candidates"
        ],
        "solver_status": result_dict[
            "solver_status"
        ],
        "total_projection": result_dict[
            "total_projection"
        ],
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
        "message": result_dict["message"],
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "lineups_long": lineup_long_file,
        "lineups_wide": lineup_wide_file,
        "exposures": exposure_file,
        "strategies": strategy_file,
        "summary": summary_file,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the production BURN1 NFL "
            "two-stage portfolio optimizer."
        )
    )

    parser.add_argument(
        "--site",
        choices=("dk", "fd"),
        required=True,
        help="DFS site: dk or fd.",
    )

    parser.add_argument(
        "--lineups",
        type=int,
        default=20,
        help="Final portfolio lineup count.",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Stage 1 candidate count.",
    )

    parser.add_argument(
        "--min-unique",
        type=int,
        default=2,
        help="Minimum unique players between Stage 1 candidates.",
    )

    parser.add_argument(
        "--slate",
        default=DEFAULT_SLATE,
        help="Output slate label/slug.",
    )

    parser.add_argument(
        "--pool-file",
        type=Path,
        default=None,
        help=(
            "Optional final-pool CSV override. "
            "Defaults to the current DK/FD Week 3 Sunday Main pool."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "modeling"
            / "output"
            / "burn1"
        ),
        help="Directory for BURN1 production outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    site = SITE_CONFIG[args.site]
    site_name = site["site_name"]
    site_slug = site["output_slug"]

    pool_file = (
        args.pool_file
        if args.pool_file is not None
        else site["pool_file"]
    )

    print()
    print("=" * 92)
    print(
        f"BURN1 DFS — {site_name.upper()} "
        "PRODUCTION PORTFOLIO RUN"
    )
    print("=" * 92)
    print(f"Pool file:         {pool_file}")
    print(f"Final lineups:     {args.lineups}")
    print(f"Stage 1 candidates:{args.candidates:>7}")
    print(f"Minimum unique:    {args.min_unique}")

    players = load_players(
        pool_file,
        site_name,
    )

    print(
        f"Loaded players:    {len(players):>7}"
    )

    config = Burn1PortfolioConfig(
        site=site_name,
        lineup_count=args.lineups,
        candidate_count=args.candidates,
        min_unique_players=args.min_unique,
    )

    started = perf_counter()

    response = run_burn1_nfl_portfolio_safe(
        players,
        config,
        status_callback=status_printer(
            site_name
        ),
    )

    elapsed = perf_counter() - started

    response_dict = response.to_dict()

    if not response.ok:
        error = response_dict["error"]

        print()
        print("=" * 92)
        print("BURN1 PRODUCTION RUN FAILED")
        print("=" * 92)
        print(
            f"Code:    {error['code']}"
        )
        print(
            f"Stage:   {error['stage']}"
        )
        print(
            f"Message: {error['message']}"
        )

        if error["details"]:
            print(
                f"Details: {error['details']}"
            )

        print(
            f"Elapsed: {elapsed:.2f}s"
        )

        return 1

    result = response_dict["result"]

    if result["generated_lineups"] != args.lineups:
        raise ValueError(
            "BURN1 production output does not "
            "contain the requested lineup count."
        )

    files = save_outputs(
        result,
        site_slug=site_slug,
        slate_slug=args.slate,
        output_dir=args.output_dir,
        elapsed_seconds=elapsed,
    )

    print()
    print("=" * 92)
    print("BURN1 PRODUCTION PORTFOLIO COMPLETE")
    print("=" * 92)
    print(
        f"Solver status:       "
        f"{result['solver_status']}"
    )
    print(
        f"Candidates generated:"
        f" {result['generated_candidates']}/"
        f"{result['requested_candidates']}"
    )
    print(
        f"Final lineups:       "
        f"{result['generated_lineups']}/"
        f"{result['requested_lineups']}"
    )
    print(
        f"Portfolio projection:"
        f" {result['total_projection']:.2f}"
    )
    print(
        f"Elapsed seconds:     "
        f"{elapsed:.2f}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 92)

    for label, file_path in files.items():
        print(
            f"{label:<16} {file_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
