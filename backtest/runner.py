import json
from pathlib import Path

import pandas as pd

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio_safe,
)
from backtest.player_adapter import historical_frame_to_players


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_backtest(
    slate_csv: str | Path,
    config_path: str | Path,
):
    slate_csv = Path(slate_csv)
    config_path = Path(config_path)

    df = pd.read_csv(slate_csv)

    players = historical_frame_to_players(df)

    raw_config = load_config(config_path)

    site_value = str(raw_config["site"]).strip().lower()

    if site_value in {"dk", "draftkings"}:
        site_name = "DraftKings"
    elif site_value in {"fd", "fanduel"}:
        site_name = "FanDuel"
    else:
        raise ValueError(
            f"Unsupported site in backtest config: {raw_config['site']}"
        )

    config = Burn1PortfolioConfig(
        site=site_name,
        lineup_count=int(raw_config["lineup_count"]),
        candidate_count=int(raw_config["candidate_count"]),
    )

    response = run_burn1_nfl_portfolio_safe(
        players,
        config,
    )

    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a BURN1 historical DFS backtest."
    )

    parser.add_argument(
        "--slate",
        required=True,
        help="Canonical BURN1 historical slate CSV.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Backtest JSON configuration.",
    )

    args = parser.parse_args()

    result = run_backtest(
        slate_csv=args.slate,
        config_path=args.config,
    )

    print(json.dumps(result.to_dict(), indent=2))
