from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from app.intelligence.backtest import (
    score_archived_run,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score an archived BURN1 run "
            "against actual fantasy results."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--actuals",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    summary = score_archived_run(
        args.run_id,
        args.actuals,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
