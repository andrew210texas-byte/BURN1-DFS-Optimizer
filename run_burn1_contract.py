from pathlib import Path

import pandas as pd

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio,
)
from models.player import Player


ROOT = Path(__file__).resolve().parent

DK_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "dk_2026_week3_sunday_main_final_pool.csv"
)

FD_POOL_FILE = (
    ROOT
    / "data"
    / "live"
    / "nfl"
    / "fd_2026_week3_sunday_main_final_pool.csv"
)


def _clean(value):
    if value is None:
        return ""
    return str(value).strip()


def _load_players(file_path, site_name):
    data = pd.read_csv(
        file_path,
        keep_default_na=False,
    )

    players = []

    for _, row in data.iterrows():
        roster_positions = tuple(
            part.strip().upper()
            for part in _clean(
                row["roster_positions"]
            ).split("/")
            if part.strip()
        )

        players.append(
            Player(
                name=_clean(row["name"]),
                site=site_name,
                position=_clean(
                    row["position"]
                ).upper(),
                team=_clean(row["team"]),
                opponent=_clean(row["opponent"]),
                salary=int(row["salary"]),
                player_id=_clean(row["player_id"]),
                roster_positions=roster_positions,
                status=_clean(
                    row.get("status", "")
                ),
                projection=float(
                    row["projection"]
                ),
            )
        )

    return players


def _test_site(site_name, file_path):
    players = _load_players(
        file_path,
        site_name,
    )

    config = Burn1PortfolioConfig(
        site=site_name,
        lineup_count=20,
        candidate_count=200,
        min_unique_players=2,
    )

    result = run_burn1_nfl_portfolio(
        players,
        config,
    )

    payload = result.to_dict()

    if payload["status"] != "complete":
        raise ValueError(
            f"{site_name} contract status FAILED."
        )

    if payload["generated_lineups"] != 20:
        raise ValueError(
            f"{site_name} contract lineup count FAILED."
        )

    if payload["generated_candidates"] < payload["requested_lineups"]:
        raise ValueError(
            f"{site_name} contract candidate count FAILED."
        )

    if len(payload["lineups"]) != 20:
        raise ValueError(
            f"{site_name} serialized lineups FAILED."
        )

    if not payload["exposures"]:
        raise ValueError(
            f"{site_name} exposure payload FAILED."
        )

    if "stacked" not in payload["strategy_summary"]:
        raise ValueError(
            f"{site_name} strategy payload FAILED."
        )

    first_lineup = payload["lineups"][0]

    if len(first_lineup["players"]) != 9:
        raise ValueError(
            f"{site_name} serialized roster FAILED."
        )

    print()
    print("=" * 88)
    print(
        f"{site_name.upper()} "
        "BURN1 APPLICATION CONTRACT PASSED"
    )
    print("=" * 88)
    print(
        f"Status:               "
        f"{payload['status']}"
    )
    print(
        f"Solver status:        "
        f"{payload['solver_status']}"
    )
    print(
        f"Candidates:           "
        f"{payload['generated_candidates']}/"
        f"{payload['requested_candidates']}"
    )
    print(
        f"Final lineups:        "
        f"{payload['generated_lineups']}/"
        f"{payload['requested_lineups']}"
    )
    print(
        f"Portfolio projection: "
        f"{payload['total_projection']:.2f}"
    )
    print(
        f"Exposure rows:        "
        f"{len(payload['exposures'])}"
    )
    print(
        f"Strategy metrics:     "
        f"{len(payload['strategy_summary'])}"
    )


def main():
    _test_site(
        "DraftKings",
        DK_POOL_FILE,
    )

    _test_site(
        "FanDuel",
        FD_POOL_FILE,
    )

    print()
    print("=" * 88)
    print(
        "SHARED DK + FD BURN1 APPLICATION "
        "CONTRACT PASSED"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
