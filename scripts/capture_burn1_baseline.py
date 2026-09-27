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


from app.intelligence.foundation import (
    capture_slate_snapshot,
)
from run_burn1_live import (
    DEFAULT_SLATE,
    SITE_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture an immutable BURN1 "
            "slate baseline."
        )
    )

    parser.add_argument(
        "--label",
        default="manual",
    )

    parser.add_argument(
        "--slate",
        default=DEFAULT_SLATE,
    )

    args = parser.parse_args()

    manifest = capture_slate_snapshot(
        SITE_CONFIG,
        slate=args.slate,
        label=args.label,
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
